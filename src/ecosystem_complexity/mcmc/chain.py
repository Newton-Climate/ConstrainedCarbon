"""Per-site chain worker and top-level driver for the MCMC pipeline.

The driver was previously ``apps/sample_mcmc.py::main``; it now runs as
``run_from_args(args)`` so the dispatcher can pass a resolved args
namespace directly, without patching ``sys.argv``.
"""
from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ecosystem_complexity import mcmc as _mcmc
from ecosystem_complexity.mcmc.plots import _supplementary_null_figure, _write_readme
from ecosystem_complexity.mcmc.posterior_analysis import (
    _leave_one_out,
    _pearson,
    _predicted_percentile_table,
    _predictor_comparison,
    _prior_structural_null,
    _regression_samples,
    _summarize_site_draws,
    _summarize_stat_frame,
)
from ecosystem_complexity.mcmc.priors import (
    _build_site_context,
    _draw_gaussian_samples,
    _observed_gap_for_spec,
    _prior_mean_and_cov,
)
from ecosystem_complexity.inference.utilities import ss_state_for_params
from ecosystem_complexity.inference.parameters import vector_to_params
from ecosystem_complexity.sites.driver import OPT_FIELDS, run_site_canonical
from ecosystem_complexity.sites.spec import load_site_spec
from ecosystem_complexity.model.state import make_default_params
from ecosystem_complexity.model.api import run_model
from ecosystem_complexity.synthesis.biomes import biome_group as _biome_group
from ecosystem_complexity.visualize.cross_ecosystem import build_cross_ecosystem_tables
from ecosystem_complexity.visualize.figure_10 import make_figure_10_from_posterior_analysis
from ecosystem_complexity.synthesis.warming import compute_pool_rh, repeat_forcing, warm_forcing

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NB = _REPO_ROOT / "notebooks"


def _site_cache_name(site: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(site)).strip("_")
    return slug or "site"


def _site_cache_path(cache_dir: str | Path, site: str) -> Path:
    return Path(cache_dir) / f"{_site_cache_name(site)}.csv"


def _discover_all_specs() -> dict[str, str]:
    out: dict[str, str] = {}
    for subdir in ("configs/multisite", "configs/expansion"):
        for path in sorted((_REPO_ROOT / subdir).glob("*.yaml")):
            spec = load_site_spec(str(path))
            out[spec.israd_name] = str(path)
    return out


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
            config_path = str(Path(raw_path) if Path(raw_path).is_absolute() else _REPO_ROOT / raw_path)
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
    old_idx = [model.pool_index[name] for name in _mcmc.OLD_POOLS if name in model.pool_index.pool_names]
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
            "forcing_proj": repeat_forcing(result["forcing"], _mcmc.WARMING_HORIZON_YEARS),
            "forcing_warm": warm_forcing(repeat_forcing(result["forcing"], _mcmc.WARMING_HORIZON_YEARS), _mcmc.WARMING_DELTA_C),
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


def _default_paths() -> dict[str, object]:
    return {
        "network_summary": str(_NB / "exports" / "network_inversion_fluxcom_er_20260719" / "site_summary.csv"),
        "warming_summary": str(_NB / "exports" / "warming_vulnerability_fluxcom_er_20260719" / "site_warming_summary.csv"),
        "new_sites": [
            str(_NB / "exports" / "new_sites_incubation_20260719.csv"),
            str(_NB / "exports" / "incubation_new_sites_runnable_20260719.csv"),
        ],
        "chain_dir": str(_NB / "exports" / "uncertainty_projections_mcmc_long" / "chains"),
        "output_dir": str(_NB / "paper_figs" / "outputs" / "figure10_posterior_20260719"),
    }


def build_args_namespace(overrides: dict[str, object] | None = None) -> argparse.Namespace:
    """Build the runtime args namespace ``run_from_args`` expects.

    Values default to the package-level constants and the historical
    export paths; ``overrides`` (typically the union of the YAML
    ``mcmc:`` block and CLI flags) win where provided.
    """
    defaults = _default_paths()
    ns = argparse.Namespace(
        network_summary=defaults["network_summary"],
        warming_summary=defaults["warming_summary"],
        new_sites=defaults["new_sites"],
        chain_dir=defaults["chain_dir"],
        output_dir=defaults["output_dir"],
        cache_dir=None,
        site_set=None,
        posterior_draws=_mcmc.POSTERIOR_DRAW_COUNT,
        prior_draws=_mcmc.PRIOR_DRAW_COUNT,
        mc_iterations=_mcmc.MC_ITERATIONS,
        null_iterations=_mcmc.NULL_ITERATIONS,
        workers=max(1, min(8, (os.cpu_count() or 2) - 1)),
        seed=_mcmc.RNG_SEED,
    )
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                setattr(ns, k, v)
    return ns


def run_from_args(args: argparse.Namespace) -> None:
    """End-to-end MCMC posterior-warming pipeline (was ``sample_mcmc.main``)."""
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
