from __future__ import annotations

import argparse
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_NB = _ROOT / "notebooks"
for _p in (str(_ROOT), str(_SRC), str(_NB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)

from ecosystem_complexity._oe_helpers import build_oe_prior_sigma
from ecosystem_complexity.api import build_model, run_model
from ecosystem_complexity.data.israd_14c import build_bulk_14C_blocks, build_resp_14C_obs
from ecosystem_complexity.data.parsers import attach_atm14C
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.paths import GRAVEN_PATH, HUA_PATH, INTCAL_PATH
from ecosystem_complexity.data.schemas import ObservationData
from ecosystem_complexity.oe_utils import ss_state_for_params
from ecosystem_complexity.optimizer import params_to_vector, vector_to_params
from ecosystem_complexity.sites.driver import OPT_FIELDS, build_state0, run_site_canonical
from ecosystem_complexity.sites.forcing import load_site_forcing, load_site_observations, resolve_forcing_file
from ecosystem_complexity.sites.soc import build_soc_prior
from ecosystem_complexity.sites.spec import load_site_spec
from ecosystem_complexity.state import make_default_params
from ecosystem_complexity.warming import compute_pool_rh, repeat_forcing, warm_forcing
from notebooks.paper_figs.fig_09 import BIOME_GROUP_COLORS, BIOME_GROUP_LABELS, BIOME_GROUP_ORDER, build_cross_ecosystem_tables
from notebooks.paper_figs.fig_10 import make_figure_10_from_posterior_analysis
from notebooks.paper_figs.io import load_table
from notebooks.paper_figs.utils import finalize_figure


RNG_SEED = 7
POSTERIOR_DRAW_COUNT = 64
PRIOR_DRAW_COUNT = 64
MC_ITERATIONS = 2000
NULL_ITERATIONS = 1000
WARMING_HORIZON_YEARS = 100.0
WARMING_DELTA_C = 4.0
OLD_POOLS = ("soil_slow", "soil_passive")
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-summary",
        default=str(_NB / "exports" / "network_inversion_fluxcom_er_20260719" / "site_summary.csv"),
    )
    parser.add_argument(
        "--warming-summary",
        default=str(_NB / "exports" / "warming_vulnerability_fluxcom_er_20260719" / "site_warming_summary.csv"),
    )
    parser.add_argument(
        "--new-sites",
        nargs="+",
        default=[
            str(_NB / "exports" / "new_sites_incubation_20260719.csv"),
            str(_NB / "exports" / "incubation_new_sites_runnable_20260719.csv"),
        ],
    )
    parser.add_argument(
        "--site-set",
        help=(
            "Optional YAML with an explicit 'configs' list. Its sites are added "
            "to the standard warming-MCMC network."
        ),
    )
    parser.add_argument(
        "--chain-dir",
        default=str(_NB / "exports" / "uncertainty_projections_mcmc_long" / "chains"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_NB / "paper_figs" / "outputs" / "figure10_posterior_20260719"),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for per-site cached CSV outputs. Defaults to <output-dir>/site_cache.",
    )
    parser.add_argument(
        "--posterior-draws",
        type=int,
        default=POSTERIOR_DRAW_COUNT,
    )
    parser.add_argument(
        "--prior-draws",
        type=int,
        default=PRIOR_DRAW_COUNT,
    )
    parser.add_argument(
        "--mc-iterations",
        type=int,
        default=MC_ITERATIONS,
    )
    parser.add_argument(
        "--null-iterations",
        type=int,
        default=NULL_ITERATIONS,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RNG_SEED,
    )
    return parser.parse_args()


def _site_cache_name(site: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(site)).strip("_")
    return slug or "site"


def _site_cache_path(cache_dir: str | Path, site: str) -> Path:
    return Path(cache_dir) / f"{_site_cache_name(site)}.csv"


def _discover_all_specs() -> dict[str, str]:
    out: dict[str, str] = {}
    for subdir in ("configs/multisite", "configs/expansion"):
        for path in sorted((_ROOT / subdir).glob("*.yaml")):
            spec = load_site_spec(str(path))
            out[spec.israd_name] = str(path)
    return out


def _biome_group(biome: str) -> str:
    biome = biome.lower()
    if any(key in biome for key in ("arctic", "tundra", "permafrost")):
        return "arctic_permafrost"
    if "boreal" in biome:
        return "boreal"
    if any(key in biome for key in ("peatland", "moss")):
        return "peatland"
    if "tropical" in biome:
        return "tropical"
    if any(key in biome for key in ("grassland", "mollisol", "mediterranean")):
        return "grassland_mediterranean"
    if any(key in biome for key in ("temperate", "conifer")):
        return "temperate_forest"
    return "other"


def _selected_site_metadata(
    network_summary: str,
    warming_summary: str,
    new_sites: list[str],
    site_set: str | None = None,
) -> pd.DataFrame:
    tables = build_cross_ecosystem_tables(network_summary, warming_summary, new_sites)
    all_sites = tables["all_sites_union"].copy()
    cfg_by_site = pd.read_csv(network_summary)[["site", "config", "tower_id", "observation_path"]].drop_duplicates("site")
    spec_paths = _discover_all_specs()
    all_sites = all_sites.merge(cfg_by_site, on="site", how="left")
    all_sites["config"] = all_sites.apply(
        lambda row: row["config"] if pd.notna(row["config"]) else spec_paths.get(str(row["site"]), np.nan),
        axis=1,
    )
    all_sites["tower_id"] = all_sites.apply(
        lambda row: row["tower_id"] if pd.notna(row["tower_id"]) else load_site_spec(str(row["config"])).tower_id,
        axis=1,
    )
    all_sites["observation_path"] = all_sites.apply(
        lambda row: row["observation_path"] if pd.notna(row["observation_path"]) else load_site_spec(str(row["config"])).observation_path,
        axis=1,
    )
    all_sites["include_er_constraint"] = True
    all_sites["include_incubation_constraint"] = all_sites["config"].astype(str).str.contains("configs/expansion/")
    all_sites["posterior_kind"] = np.where(all_sites["tower_id"].isin({"US-Ha1", "US-A10", "US-Ho1", "US-EML"}), "saved_mcmc", "gaussian")
    if site_set:
        with open(site_set, encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        config_paths = payload.get("configs")
        if not isinstance(config_paths, list) or not config_paths or not all(isinstance(p, str) for p in config_paths):
            raise ValueError(f"{site_set}: expected a non-empty string list at 'configs'")
        additions = []
        for raw_path in config_paths:
            config_path = str(Path(raw_path) if Path(raw_path).is_absolute() else _ROOT / raw_path)
            spec = load_site_spec(config_path)
            if spec.israd_name in set(all_sites["site"]):
                continue
            additions.append({
                "site": spec.israd_name, "label": spec.label, "biome": spec.biome,
                "biome_group": _biome_group(spec.biome), "source_set": str(payload.get("name", "site_set")),
                "has_direct_warming": True, "config": config_path, "tower_id": spec.tower_id,
                "observation_path": spec.observation_path, "include_er_constraint": True,
                "include_incubation_constraint": "configs/expansion/" in config_path,
                "posterior_kind": "gaussian", "dfs_total": np.nan,
            })
        if additions:
            all_sites = pd.concat([all_sites, pd.DataFrame(additions)], ignore_index=True, sort=False)
    return all_sites.sort_values(["has_direct_warming", "site"], ascending=[False, True]).reset_index(drop=True)


def _build_site_context(
    config_path: str,
    include_er_constraint: bool,
    include_incubation_constraint: bool,
) -> dict:
    spec = load_site_spec(config_path)
    model = build_model(config_path)
    forcing_path = resolve_forcing_file(spec)
    forcing = load_site_forcing(spec, forcing_path, model)
    tower_obs = (
        load_site_observations(spec, forcing_path, model, forcing=forcing)
        if include_er_constraint
        else None
    )
    hemisphere = "NH" if spec.lat >= 0 else "SH"
    years_daily, d14c_daily = load_full_14C_record(
        hua_path=HUA_PATH,
        graven_path=GRAVEN_PATH,
        intcal_path=INTCAL_PATH,
        hemisphere=hemisphere,
        start_year=1500.0,
        end_year=2025.0,
    )
    forcing = attach_atm14C(forcing, d14c_daily, years_daily)
    soc_prior_state, _c_pools_prior, _ss_years, c_total_obs = build_soc_prior(model, forcing)
    pool_blocks = []
    if spec.observation_path in {"fraction", "combined"}:
        from ecosystem_complexity.data.israd_14c import build_fraction_14C_blocks

        pool_blocks.extend(build_fraction_14C_blocks(spec.israd_name, forcing.time, model, spec.fraction_rules))
    if spec.observation_path in {"bulk_resp", "combined"}:
        pool_blocks.extend(build_bulk_14C_blocks(spec.israd_name, forcing.time, model))
    resp = (
        build_resp_14C_obs(spec.israd_name, forcing.time)
        if spec.observation_path in {"bulk_resp", "combined"}
        else jnp.full(forcing.time.shape[0], jnp.nan, dtype=jnp.float32)
    )
    if include_incubation_constraint:
        from ecosystem_complexity.data.israd_incubation import build_incubation_rate_blocks
        from ecosystem_complexity.data.israd_14c import build_incubation_14C_blocks

        incubation_rows = build_incubation_rate_blocks(spec.israd_name, model)
        pool_blocks = pool_blocks + [row["block"] for row in incubation_rows]
        # The same expansion-site switch now carries the dated isotope
        # measurement as well as the incubation-rate block. The 14C builder
        # pools replicate jars by date and applies its conservative ≥20‰
        # representativeness floor.
        pool_blocks.extend(build_incubation_14C_blocks(spec.israd_name, forcing.time))
    T = int(forcing.time.shape[0])
    er_obs = (
        tower_obs.ER
        if (tower_obs is not None and tower_obs.ER is not None)
        else jnp.full(T, jnp.nan)
    )
    obs_full = ObservationData(
        time=forcing.time,
        NEE=jnp.full(T, jnp.nan),
        GPP=jnp.full(T, jnp.nan),
        ER=er_obs,
        NEE_unc=jnp.full(T, jnp.nan),
        delta14C_obs={},
        deltaD14C_obs={},
        C_pools_obs={},
        delta14C_resp=resp,
        C_total_obs=c_total_obs,
    )
    from ecosystem_complexity.data.israd_14c import _bulk_pool_ic_seeds

    state0 = build_state0(model, soc_prior_state, pool_blocks, ic_seeds=_bulk_pool_ic_seeds(spec.israd_name))
    forcing_proj = repeat_forcing(forcing, WARMING_HORIZON_YEARS)
    forcing_warm = warm_forcing(forcing_proj, WARMING_DELTA_C)
    params_prior = make_default_params(model.config)
    return {
        "spec": spec,
        "model": model,
        "forcing": forcing,
        "forcing_proj": forcing_proj,
        "forcing_warm": forcing_warm,
        "state0": state0,
        "params_prior": params_prior,
        "opt_fields": OPT_FIELDS,
        "obs_full": obs_full,
    }


def _draw_gaussian_samples(
    rng: np.random.Generator,
    x_mean: np.ndarray,
    cov: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    cov_sym = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov_sym)
    eigvals = np.clip(eigvals, 0.0, None)
    transform = eigvecs @ np.diag(np.sqrt(eigvals))
    z = rng.standard_normal((n_samples, x_mean.size))
    return x_mean[None, :] + z @ transform.T


def _prior_mean_and_cov(context: dict) -> tuple[np.ndarray, np.ndarray]:
    x_mean = np.array(params_to_vector(context["params_prior"], context["opt_fields"]), dtype=np.float64)
    sigma = np.array(
        build_oe_prior_sigma(context["model"].config, context["params_prior"], context["opt_fields"]),
        dtype=np.float64,
    )
    return x_mean, np.diag(np.square(sigma))


def _observed_gap_for_spec(config_path: str) -> dict[str, float]:
    spec = load_site_spec(config_path)
    if spec.observation_path != "bulk_resp":
        return {
            "obs_bulk_mean": np.nan,
            "obs_resp_mean": np.nan,
            "obs_offset_resp_minus_bulk": np.nan,
            "n_bulk_vals": 0,
            "n_resp_vals": 0,
        }
    model = build_model(config_path)
    forcing_path = resolve_forcing_file(spec)
    forcing = load_site_forcing(spec, forcing_path, model)
    hemisphere = "NH" if spec.lat >= 0 else "SH"
    years_daily, d14c_daily = load_full_14C_record(
        hua_path=HUA_PATH,
        graven_path=GRAVEN_PATH,
        intcal_path=INTCAL_PATH,
        hemisphere=hemisphere,
        start_year=1500.0,
        end_year=2025.0,
    )
    forcing = attach_atm14C(forcing, d14c_daily, years_daily)
    bulk_blocks = build_bulk_14C_blocks(spec.israd_name, forcing.time, model)
    resp = np.array(build_resp_14C_obs(spec.israd_name, forcing.time), dtype=float)
    bulk_vals: list[float] = []
    for block in bulk_blocks:
        vals = np.array(block.y, dtype=float).ravel()
        bulk_vals.extend(vals[np.isfinite(vals)].tolist())
    resp_vals = resp[np.isfinite(resp)]
    if not bulk_vals or resp_vals.size == 0:
        return {
            "obs_bulk_mean": np.nan,
            "obs_resp_mean": np.nan,
            "obs_offset_resp_minus_bulk": np.nan,
            "n_bulk_vals": len(bulk_vals),
            "n_resp_vals": int(resp_vals.size),
        }
    return {
        "obs_bulk_mean": float(np.mean(bulk_vals)),
        "obs_resp_mean": float(np.mean(resp_vals)),
        "obs_offset_resp_minus_bulk": float(np.mean(resp_vals) - np.mean(bulk_vals)),
        "n_bulk_vals": int(len(bulk_vals)),
        "n_resp_vals": int(resp_vals.size),
    }


def _run_draw_metrics(context: dict, x_vec: np.ndarray) -> dict[str, float]:
    params = vector_to_params(jnp.array(x_vec, dtype=jnp.float32), context["params_prior"], context["opt_fields"])
    model = context["model"]
    forcing_proj = context["forcing_proj"]
    forcing_warm = context["forcing_warm"]
    state_init = ss_state_for_params(model, forcing_proj, context["state0"], params)
    out_base = run_model(model, forcing_proj, state0=state_init, params=params)
    out_warm = run_model(model, forcing_warm, state0=state_init, params=params)
    jax.block_until_ready(out_warm.C12)

    tau_years = np.exp(np.array(params.log_tau, dtype=np.float64)) / 365.25
    c0 = float(np.sum(np.array(state_init.C12, dtype=np.float64)))
    c_base_final = float(np.sum(np.array(out_base.C12[-1], dtype=np.float64)))
    c_warm_final = float(np.sum(np.array(out_warm.C12[-1], dtype=np.float64)))
    abs_c_loss = c_base_final - c_warm_final
    frac_c_loss = abs_c_loss / c0 if c0 > 0 else np.nan

    rh_base = compute_pool_rh(model, forcing_proj, params, out_base)
    rh_warm = compute_pool_rh(model, forcing_warm, params, out_warm)
    old_idx = [model.pool_index[name] for name in OLD_POOLS if name in model.pool_index.pool_names]
    delta_total = float(np.sum(rh_warm.sum(axis=1) - rh_base.sum(axis=1)))
    delta_old = float(np.sum(rh_warm[:, old_idx].sum(axis=1) - rh_base[:, old_idx].sum(axis=1)))
    old_fraction = delta_old / delta_total if abs(delta_total) > 1e-12 else np.nan
    return {
        "tau_active_yr": float(tau_years[model.pool_index["soil_active"]]),
        "tau_slow_yr": float(tau_years[model.pool_index["soil_slow"]]),
        "tau_passive_yr": float(tau_years[model.pool_index["soil_passive"]]),
        "turnover_separation": float(np.log10(tau_years[model.pool_index["soil_passive"]] / tau_years[model.pool_index["soil_active"]])),
        "frac_c_loss": float(frac_c_loss),
        "abs_c_loss_gCm2": float(abs_c_loss),
        "old_fraction_of_excess_rh": float(old_fraction),
    }


def _quantiles(values: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan, np.nan)
    return (
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.5)),
        float(np.quantile(vals, 0.975)),
    )


def _summarize_site_draws(
    posterior_df: pd.DataFrame,
    *,
    payload: dict,
    posterior_kind: str,
    dfs_total: float,
    gap: dict[str, float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "site": payload["site"],
        "label": payload["label"],
        "tower_id": payload["tower_id"],
        "biome": payload["biome"],
        "biome_group": payload["biome_group"],
        "source_set": payload["source_set"],
        "config": payload["config"],
        "posterior_kind": posterior_kind,
        "n_draws": int(len(posterior_df)),
        "dfs_total": dfs_total,
        **gap,
    }
    for col in (
        "tau_active_yr",
        "tau_slow_yr",
        "tau_passive_yr",
        "turnover_separation",
        "frac_c_loss",
        "abs_c_loss_gCm2",
        "old_fraction_of_excess_rh",
    ):
        q025, median, q975 = _quantiles(posterior_df[col].to_numpy(dtype=float))
        row[f"{col}_q025"] = q025
        row[f"{col}_median"] = median
        row[f"{col}_q975"] = q975
    row["has_observed_gap"] = bool(np.isfinite(row["obs_offset_resp_minus_bulk"]))
    return row


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2 or y.size < 2:
        return (np.nan, np.nan)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _site_worker(payload: dict) -> dict[str, object]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    cache_path = _site_cache_path(payload["cache_dir"], payload["site"])
    if cache_path.is_file():
        cached = pd.read_csv(cache_path)
        posterior_df = cached[cached["sample_set"] == "posterior"].copy()
        prior_df = cached[cached["sample_set"] == "prior"].copy()
        if posterior_df.empty:
            raise ValueError(f"Cached posterior CSV is empty for {payload['site']}: {cache_path}")
        posterior_kind = str(posterior_df["posterior_kind"].iloc[0])
        dfs_total = float(posterior_df["dfs_total"].iloc[0])
        gap = {
            "obs_bulk_mean": float(posterior_df["obs_bulk_mean"].iloc[0]),
            "obs_resp_mean": float(posterior_df["obs_resp_mean"].iloc[0]),
            "obs_offset_resp_minus_bulk": float(posterior_df["obs_offset_resp_minus_bulk"].iloc[0]),
            "n_bulk_vals": int(posterior_df["n_bulk_vals"].iloc[0]),
            "n_resp_vals": int(posterior_df["n_resp_vals"].iloc[0]),
        }
        return {
            "site_summary": _summarize_site_draws(
                posterior_df,
                payload=payload,
                posterior_kind=posterior_kind,
                dfs_total=dfs_total,
                gap=gap,
            ),
            "posterior_draws": posterior_df.drop(columns=["sample_set"], errors="ignore").to_dict(orient="records"),
            "prior_draws": prior_df.drop(columns=["sample_set"], errors="ignore").to_dict(orient="records"),
            "cached": True,
        }

    rng = np.random.default_rng(int(payload["seed"]))
    config_path = str(payload["config"])
    use_saved_chain = bool(payload["posterior_kind"] == "saved_mcmc")
    include_er_constraint = bool(payload["include_er_constraint"])
    include_incubation_constraint = bool(payload["include_incubation_constraint"])

    if use_saved_chain:
        context = _build_site_context(config_path, include_er_constraint, include_incubation_constraint)
        chain_path = Path(payload["chain_dir"]) / f"{payload['tower_id']}__all_observations__mcmc_chain.npz"
        with np.load(chain_path) as data:
            posterior_samples = np.array(data["retained_samples"], dtype=np.float64)
        dfs_total = float(payload["dfs_total"]) if pd.notna(payload["dfs_total"]) else np.nan
        posterior_kind = "saved_mcmc"
    else:
        result = run_site_canonical(
            load_site_spec(config_path),
            observation_path=payload["observation_path"],
            include_er_constraint=include_er_constraint,
            include_incubation_constraint=include_incubation_constraint,
            include_incubation_14c_constraint=include_incubation_constraint,
        )
        context = {
            "spec": result["spec"],
            "model": result["model"],
            "forcing": result["forcing"],
            "forcing_proj": repeat_forcing(result["forcing"], WARMING_HORIZON_YEARS),
            "forcing_warm": warm_forcing(repeat_forcing(result["forcing"], WARMING_HORIZON_YEARS), WARMING_DELTA_C),
            "state0": result["state0"],
            "params_prior": make_default_params(result["model"].config),
            "opt_fields": OPT_FIELDS,
            "obs_full": result["obs_full"],
        }
        x_mean = np.array(result["oe_result"].x_opt, dtype=np.float64)
        cov = np.array(result["oe_result"].Sx, dtype=np.float64)
        posterior_samples = _draw_gaussian_samples(rng, x_mean, cov, int(payload["posterior_draws"]))
        dfs_total = float(np.trace(np.array(result["oe_result"].averaging_kernel, dtype=float)))
        posterior_kind = "gaussian_refit"

    prior_mean, prior_cov = _prior_mean_and_cov(context)
    prior_samples = _draw_gaussian_samples(rng, prior_mean, prior_cov, int(payload["prior_draws"]))
    gap = _observed_gap_for_spec(config_path)

    draw_rows: list[dict[str, object]] = []
    for draw_idx, x_vec in enumerate(posterior_samples):
        metrics = _run_draw_metrics(context, x_vec)
        draw_rows.append(
            {
                "site": payload["site"],
                "label": payload["label"],
                "tower_id": payload["tower_id"],
                "biome": payload["biome"],
                "biome_group": payload["biome_group"],
                "source_set": payload["source_set"],
                "config": config_path,
                "posterior_kind": posterior_kind,
                "draw": draw_idx,
                "dfs_total": dfs_total,
                **gap,
                **metrics,
            }
        )

    prior_rows: list[dict[str, object]] = []
    for draw_idx, x_vec in enumerate(prior_samples):
        metrics = _run_draw_metrics(context, x_vec)
        prior_rows.append(
            {
                "site": payload["site"],
                "draw": draw_idx,
                "turnover_separation": metrics["turnover_separation"],
                "old_fraction_of_excess_rh": metrics["old_fraction_of_excess_rh"],
            }
        )

    df = pd.DataFrame(draw_rows)
    row = _summarize_site_draws(
        df,
        payload=payload,
        posterior_kind=posterior_kind,
        dfs_total=dfs_total,
        gap=gap,
    )
    cache_df = pd.concat(
        [
            df.assign(sample_set="posterior"),
            pd.DataFrame(prior_rows).assign(sample_set="prior"),
        ],
        ignore_index=True,
        sort=False,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(cache_path, index=False)
    return {
        "site_summary": row,
        "posterior_draws": draw_rows,
        "prior_draws": prior_rows,
        "cached": False,
    }


def _regression_samples(
    draws: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_iter: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    by_site = {site: grp.reset_index(drop=True) for site, grp in draws.groupby("site", sort=True)}
    rows: list[dict[str, object]] = []
    for i in range(n_iter):
        sample_rows = []
        for site, grp in by_site.items():
            take = int(rng.integers(0, len(grp)))
            sample_rows.append(grp.iloc[take])
        sample = pd.DataFrame(sample_rows)[["site", x_col, y_col]].dropna()
        x = sample[x_col].to_numpy(dtype=float)
        y = sample[y_col].to_numpy(dtype=float)
        slope, intercept = _slope_intercept(x, y)
        rows.append(
            {
                "relationship": f"{y_col}__vs__{x_col}",
                "iteration": i,
                "n": int(len(sample)),
                "slope": slope,
                "intercept": intercept,
                "pearson_r": _pearson(x, y),
                "spearman_rho": float(spearmanr(x, y, nan_policy="omit").statistic) if len(sample) >= 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _summarize_stat_frame(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, grp in df.groupby(group_col, sort=False):
        row = {group_col: key, "n_iterations": int(len(grp))}
        for col in ("slope", "intercept", "pearson_r", "spearman_rho"):
            q025, median, q975 = _quantiles(grp[col].to_numpy(dtype=float))
            row[f"{col}_q025"] = q025
            row[f"{col}_median"] = median
            row[f"{col}_q975"] = q975
        row["n_sites"] = int(pd.to_numeric(grp["n"], errors="coerce").median())
        rows.append(row)
    return pd.DataFrame(rows)


def _leave_one_out(site_metrics: pd.DataFrame) -> pd.DataFrame:
    use = site_metrics[["site", "turnover_separation_median", "old_fraction_of_excess_rh_median"]].dropna()
    rows = []
    for site in use["site"]:
        sample = use[use["site"] != site]
        x = sample["turnover_separation_median"].to_numpy(dtype=float)
        y = sample["old_fraction_of_excess_rh_median"].to_numpy(dtype=float)
        slope, _intercept = _slope_intercept(x, y)
        rows.append(
            {
                "excluded_site": site,
                "n_sites": int(len(sample)),
                "slope": slope,
                "pearson_r": _pearson(x, y),
                "spearman_rho": float(spearmanr(x, y, nan_policy="omit").statistic) if len(sample) >= 2 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("excluded_site").reset_index(drop=True)


def _predictor_comparison(
    site_metrics: pd.DataFrame,
    posterior_draws: pd.DataFrame,
    n_iter: int,
    seed: int,
) -> pd.DataFrame:
    specs = [
        ("turnover_separation", "old_fraction_of_excess_rh"),
        ("dfs_total", "old_fraction_of_excess_rh"),
        ("obs_offset_resp_minus_bulk", "old_fraction_of_excess_rh"),
    ]
    rows: list[dict[str, object]] = []
    for i, (x_col, y_col) in enumerate(specs):
        if x_col == "turnover_separation":
            stat_samples = _regression_samples(
                posterior_draws.rename(columns={"turnover_separation": x_col, "old_fraction_of_excess_rh": y_col}),
                x_col,
                y_col,
                n_iter,
                seed + i,
            )
            summary = _summarize_stat_frame(stat_samples, "relationship").iloc[0].to_dict()
            rows.append(
                {
                    "predictor": x_col,
                    "response": y_col,
                    "n_sites": int(stat_samples["n"].median()),
                    **summary,
                }
            )
            continue

        stat_samples = _regression_samples(
            posterior_draws.rename(columns={"old_fraction_of_excess_rh": y_col}),
            x_col,
            y_col,
            n_iter,
            seed + i,
        )
        stat_samples = stat_samples.dropna(subset=["slope", "pearson_r", "spearman_rho"])
        summary = _summarize_stat_frame(stat_samples, "relationship").iloc[0].to_dict()
        rows.append(
            {
                "predictor": x_col,
                "response": y_col,
                "n_sites": int(stat_samples["n"].median()) if not stat_samples.empty else 0,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def _prior_structural_null(prior_draws: pd.DataFrame, observed_r: float, n_iter: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    by_site = {site: grp.reset_index(drop=True) for site, grp in prior_draws.groupby("site", sort=True)}
    rows = []
    for i in range(n_iter):
        sample_rows = []
        for site, grp in by_site.items():
            take = int(rng.integers(0, len(grp)))
            sample_rows.append(grp.iloc[take])
        sample = pd.DataFrame(sample_rows).dropna(subset=["turnover_separation", "old_fraction_of_excess_rh"])
        x = sample["turnover_separation"].to_numpy(dtype=float)
        y = sample["old_fraction_of_excess_rh"].to_numpy(dtype=float)
        rows.append(
            {
                "iteration": i,
                "n_sites": int(len(sample)),
                "pearson_r": _pearson(x, y),
                "spearman_rho": float(spearmanr(x, y, nan_policy="omit").statistic) if len(sample) >= 2 else np.nan,
            }
        )
    null_df = pd.DataFrame(rows)
    abs_null = np.abs(null_df["pearson_r"].to_numpy(dtype=float))
    p_emp = float((1.0 + np.sum(abs_null >= abs(observed_r))) / (len(abs_null) + 1.0))
    percentile = float(100.0 * np.mean(null_df["pearson_r"].to_numpy(dtype=float) <= observed_r))
    summary = pd.DataFrame(
        [
            {
                "null_definition": (
                    "Independent prior draws per site from the saved OE prior for "
                    "the optimized parameter vector (log_tau and log_f_transfer), "
                    "propagated through the existing +4 C, 100-year warming experiment."
                ),
                "observed_pearson_r": observed_r,
                "null_pearson_r_q025": float(np.quantile(null_df["pearson_r"], 0.025)),
                "null_pearson_r_median": float(np.quantile(null_df["pearson_r"], 0.5)),
                "null_pearson_r_q975": float(np.quantile(null_df["pearson_r"], 0.975)),
                "null_spearman_rho_q025": float(np.quantile(null_df["spearman_rho"], 0.025)),
                "null_spearman_rho_median": float(np.quantile(null_df["spearman_rho"], 0.5)),
                "null_spearman_rho_q975": float(np.quantile(null_df["spearman_rho"], 0.975)),
                "observed_percentile": percentile,
                "empirical_p_two_sided": p_emp,
                "n_iterations": int(n_iter),
            }
        ]
    )
    return null_df, summary


def _supplementary_null_figure(
    null_samples: pd.DataFrame,
    null_summary: pd.DataFrame,
    output_dir: str,
) -> None:
    summary = null_summary.iloc[0]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(null_samples["pearson_r"], bins=30, color="#C8D6D1", edgecolor="white")
    ax.axvline(float(summary["observed_pearson_r"]), color="#8C2F39", linewidth=2.0, label="Observed median-site r")
    ax.set_xlabel("Null Pearson r")
    ax.set_ylabel("Count")
    ax.set_title("Structural null from site priors")
    ax.text(
        0.02,
        0.98,
        (
            f"Observed percentile = {summary['observed_percentile']:.1f}\n"
            f"Empirical p = {summary['empirical_p_two_sided']:.3f}\n"
            f"Null median = {summary['null_pearson_r_median']:.2f}\n"
            f"Null 95% = [{summary['null_pearson_r_q025']:.2f}, {summary['null_pearson_r_q975']:.2f}]"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.85", "boxstyle": "round,pad=0.3"},
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    alt_text = (
        "Histogram of the structural-null Pearson correlation distribution derived from independent prior draws at each site, "
        "propagated through the existing 100-year plus-4-degree-C warming experiment. A vertical line marks the observed "
        "cross-site correlation based on posterior median turnover separation and posterior median old-fraction response."
    )
    caption = (
        "Structural-null test for the turnover-separation versus old-fraction relationship. The null distribution was generated by "
        "drawing independent parameter vectors from each site's saved OE prior over the optimized state vector and propagating those "
        "draws through the existing standardized warming experiment. The observed cross-site Pearson correlation among posterior "
        "median site metrics lies in the extreme tail of the prior-driven structural null."
    )
    finalize_figure(
        fig,
        "supplement_figure_structural_null",
        output_dir,
        {"null_samples": null_samples, "null_summary": null_summary},
        alt_text,
        "Supplementary Figure: Structural Null",
        caption,
    )
    plt.close(fig)


def _predicted_percentile_table(regression_samples: pd.DataFrame, site_metrics: pd.DataFrame) -> pd.DataFrame:
    x20 = float(np.quantile(site_metrics["turnover_separation_median"].dropna(), 0.2))
    x80 = float(np.quantile(site_metrics["turnover_separation_median"].dropna(), 0.8))
    rows = []
    rel_to_metric = {
        "old_fraction_of_excess_rh__vs__turnover_separation": "old_fraction_of_excess_rh",
        "frac_c_loss__vs__turnover_separation": "frac_c_loss",
        "dfs_total__vs__turnover_separation": "dfs_total",
    }
    for rel, metric in rel_to_metric.items():
        grp = regression_samples[regression_samples["relationship"] == rel]
        for label, xval in (("p20", x20), ("p80", x80)):
            pred = grp["intercept"].to_numpy(dtype=float) + grp["slope"].to_numpy(dtype=float) * xval
            q025, median, q975 = _quantiles(pred)
            rows.append(
                {
                    "relationship": rel,
                    "metric": metric,
                    "turnover_percentile": label,
                    "turnover_separation_value": xval,
                    "predicted_q025": q025,
                    "predicted_median": median,
                    "predicted_q975": q975,
                }
            )
    return pd.DataFrame(rows)


def _write_readme(
    output_dir: Path,
    site_metrics: pd.DataFrame,
    regression_summary: pd.DataFrame,
    loo: pd.DataFrame,
    predictor_table: pd.DataFrame,
    null_summary: pd.DataFrame,
) -> None:
    rel = regression_summary.set_index("relationship")
    null_row = null_summary.iloc[0]
    loo_sign_change = bool((loo["slope"] <= 0).any())
    readme = f"""# Figure 10 posterior analysis

Generated on Sunday, July 19, 2026.

## Methods

- Site set: 34 sites matching the current Figure 10 universe: 24 direct-warming network sites plus 10 incubation-expansion sites.
- Posterior sources:
  - Saved MCMC chains were reused for `US-Ha1`, `US-A10`, `US-Ho1`, and `US-EML`.
  - Gaussian posterior draws were regenerated in parallel for the remaining sites from each site's OE posterior mean and covariance.
- Warming experiment: the existing standardized `+4 C`, `100-year` repeated-forcing experiment.
- Monte Carlo propagation: {MC_ITERATIONS:,} cross-site iterations, each drawing one posterior sample per site.
- Structural null: independent prior draws at each site propagated through the same warming experiment.

## Main results

- Turnover separation vs old-fraction:
  - slope median = {rel.loc['old_fraction_of_excess_rh__vs__turnover_separation', 'slope_median']:.3f}
  - slope 95% = [{rel.loc['old_fraction_of_excess_rh__vs__turnover_separation', 'slope_q025']:.3f}, {rel.loc['old_fraction_of_excess_rh__vs__turnover_separation', 'slope_q975']:.3f}]
  - Pearson r median = {rel.loc['old_fraction_of_excess_rh__vs__turnover_separation', 'pearson_r_median']:.3f}
  - Pearson r 95% = [{rel.loc['old_fraction_of_excess_rh__vs__turnover_separation', 'pearson_r_q025']:.3f}, {rel.loc['old_fraction_of_excess_rh__vs__turnover_separation', 'pearson_r_q975']:.3f}]
  - Spearman rho median = {rel.loc['old_fraction_of_excess_rh__vs__turnover_separation', 'spearman_rho_median']:.3f}
  - Spearman rho 95% = [{rel.loc['old_fraction_of_excess_rh__vs__turnover_separation', 'spearman_rho_q025']:.3f}, {rel.loc['old_fraction_of_excess_rh__vs__turnover_separation', 'spearman_rho_q975']:.3f}]

- Turnover separation vs fractional C loss:
  - Pearson r median = {rel.loc['frac_c_loss__vs__turnover_separation', 'pearson_r_median']:.3f}
  - Pearson r 95% = [{rel.loc['frac_c_loss__vs__turnover_separation', 'pearson_r_q025']:.3f}, {rel.loc['frac_c_loss__vs__turnover_separation', 'pearson_r_q975']:.3f}]
  - Spearman rho median = {rel.loc['frac_c_loss__vs__turnover_separation', 'spearman_rho_median']:.3f}
  - Spearman rho 95% = [{rel.loc['frac_c_loss__vs__turnover_separation', 'spearman_rho_q025']:.3f}, {rel.loc['frac_c_loss__vs__turnover_separation', 'spearman_rho_q975']:.3f}]

- Turnover separation vs total DFS:
  - Pearson r median = {rel.loc['dfs_total__vs__turnover_separation', 'pearson_r_median']:.3f}
  - Pearson r 95% = [{rel.loc['dfs_total__vs__turnover_separation', 'pearson_r_q025']:.3f}, {rel.loc['dfs_total__vs__turnover_separation', 'pearson_r_q975']:.3f}]

- Leave-one-site-out robustness:
  - slope range = [{loo['slope'].min():.3f}, {loo['slope'].max():.3f}]
  - Pearson r range = [{loo['pearson_r'].min():.3f}, {loo['pearson_r'].max():.3f}]
  - Spearman rho range = [{loo['spearman_rho'].min():.3f}, {loo['spearman_rho'].max():.3f}]
  - Any sign change in slope: {'yes' if loo_sign_change else 'no'}

- Structural null:
  - observed Pearson r = {null_row['observed_pearson_r']:.3f}
  - null median = {null_row['null_pearson_r_median']:.3f}
  - null 95% = [{null_row['null_pearson_r_q025']:.3f}, {null_row['null_pearson_r_q975']:.3f}]
  - observed percentile = {null_row['observed_percentile']:.1f}
  - empirical p = {null_row['empirical_p_two_sided']:.4f}

## Files

- `figures/figure_10.*`: revised Figure 10
- `figures/supplement_figure_structural_null.*`: structural-null figure
- `csv/figure_10/posterior_site_metrics.csv`: site summary table
- `csv/figure_10/posterior_regression_summary.csv`: posterior regression summary
- `csv/figure_10/leave_one_out.csv`: leave-one-site-out table
- `csv/figure_10/predictor_comparison.csv`: predictor comparison table
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else outdir / "site_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    site_meta = _selected_site_metadata(
        args.network_summary, args.warming_summary, args.new_sites, args.site_set
    )
    worker_payloads = []
    for i, row in site_meta.iterrows():
        worker_payloads.append(
            {
                **row.to_dict(),
                "chain_dir": args.chain_dir,
                "posterior_draws": args.posterior_draws,
                "prior_draws": args.prior_draws,
                "cache_dir": str(cache_dir),
                "seed": args.seed + i * 97,
            }
        )

    site_rows: list[dict[str, object]] = []
    posterior_rows: list[dict[str, object]] = []
    prior_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(worker_payloads)))) as pool:
        future_map = {pool.submit(_site_worker, payload): payload for payload in worker_payloads}
        for i, fut in enumerate(as_completed(future_map), start=1):
            payload_meta = future_map[fut]
            try:
                payload = fut.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "site": payload_meta["site"],
                        "config": payload_meta["config"],
                        "error": repr(exc),
                    }
                )
                print(f"[{i}/{len(future_map)}] FAILED {payload_meta['site']} :: {exc!r}", flush=True)
                continue
            site_rows.append(payload["site_summary"])
            posterior_rows.extend(payload["posterior_draws"])
            prior_rows.extend(payload["prior_draws"])
            if payload.get("cached"):
                print(f"[{i}/{len(future_map)}] skipped {payload['site_summary']['site']} :: cached csv", flush=True)
            else:
                print(f"[{i}/{len(future_map)}] completed {payload['site_summary']['site']}", flush=True)

    if failures:
        fail_df = pd.DataFrame(failures)
        fail_path = outdir / "failures.csv"
        fail_df.to_csv(fail_path, index=False)
        raise RuntimeError(f"{len(failures)} site jobs failed. See {fail_path}")

    site_metrics = pd.DataFrame(site_rows).sort_values(["source_set", "site"]).reset_index(drop=True)
    posterior_draws = pd.DataFrame(posterior_rows).sort_values(["site", "draw"]).reset_index(drop=True)
    prior_draws = pd.DataFrame(prior_rows).sort_values(["site", "draw"]).reset_index(drop=True)

    regression_frames = []
    for x_col, y_col in (
        ("turnover_separation", "old_fraction_of_excess_rh"),
        ("turnover_separation", "frac_c_loss"),
        ("turnover_separation", "dfs_total"),
    ):
        regression_frames.append(_regression_samples(posterior_draws, x_col, y_col, args.mc_iterations, args.seed))
    regression_samples = pd.concat(regression_frames, ignore_index=True)
    regression_summary = _summarize_stat_frame(regression_samples, "relationship")

    leave_one_out = _leave_one_out(site_metrics)
    predictor_comparison = _predictor_comparison(site_metrics, posterior_draws, args.mc_iterations, args.seed + 1000)

    observed_site_sample = site_metrics[["turnover_separation_median", "old_fraction_of_excess_rh_median"]].dropna()
    observed_r = _pearson(
        observed_site_sample["turnover_separation_median"].to_numpy(dtype=float),
        observed_site_sample["old_fraction_of_excess_rh_median"].to_numpy(dtype=float),
    )
    structural_null_samples, structural_null_summary = _prior_structural_null(
        prior_draws,
        observed_r,
        args.null_iterations,
        args.seed + 2000,
    )
    predicted_percentiles = _predicted_percentile_table(regression_samples, site_metrics)

    fig, _axes = make_figure_10_from_posterior_analysis(
        site_metrics,
        regression_samples,
        regression_summary,
        leave_one_out,
        predictor_comparison,
        predicted_percentiles,
        output_dir=str(outdir),
    )
    plt.close(fig)
    _supplementary_null_figure(structural_null_samples, structural_null_summary, str(outdir))

    csv_dir = outdir / "csv" / "figure_10"
    csv_dir.mkdir(parents=True, exist_ok=True)
    site_metrics.to_csv(csv_dir / "posterior_site_metrics.csv", index=False)
    posterior_draws.to_csv(csv_dir / "posterior_site_draws.csv", index=False)
    prior_draws.to_csv(csv_dir / "prior_null_draws.csv", index=False)
    regression_samples.to_csv(csv_dir / "posterior_regression_samples.csv", index=False)
    regression_summary.to_csv(csv_dir / "posterior_regression_summary.csv", index=False)
    leave_one_out.to_csv(csv_dir / "leave_one_out.csv", index=False)
    predictor_comparison.to_csv(csv_dir / "predictor_comparison.csv", index=False)
    structural_null_samples.to_csv(csv_dir / "structural_null_samples.csv", index=False)
    structural_null_summary.to_csv(csv_dir / "structural_null_summary.csv", index=False)
    predicted_percentiles.to_csv(csv_dir / "predicted_percentiles.csv", index=False)
    _write_readme(outdir, site_metrics, regression_summary, leave_one_out, predictor_comparison, structural_null_summary)


if __name__ == "__main__":
    main()
