from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
_NB = _ROOT / "notebooks"
for _p in (str(_SRC), str(_NB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)

from ecosystem_complexity.api import run_model
from ecosystem_complexity.optimizer import vector_to_params
from ecosystem_complexity.oe_utils import ss_state_for_params

from clm.cmip_global import load_clm_targets
from clm.fit_clm import SITE_RUNTIME, fit_one_site
from paper_figs.fig_01 import make_figure_01
from paper_figs.fig_02 import make_figure_02
from paper_figs.fig_03 import make_figure_03
from paper_figs.fig_04 import make_figure_04
from paper_figs.fig_05 import make_figure_05
from paper_figs.fig_06 import make_figure_06
from paper_figs.fig_07 import make_figure_07
from paper_figs.fig_08 import make_figure_08
from paper_figs.utils import close_or_show
from uncertainty_projections import (
    SITE_ORDER,
    SITE_SPECS,
    SUBSETS,
    SUBSET_TO_LABEL,
    _compute_pool_rh,
    _draw_gaussian_samples,
    _prior_mean_and_cov,
    _repeat_forcing,
    _warm_forcing,
)


SITE_ID_TO_CMIP_KEY = {
    "US-Ha1": "harvard_forest",
    "US-A10": "barrow",
    "US-Ho1": "howland_forest",
    "US-EML": "eight_mile_lake",
}

SITE_ID_TO_ECOSYSTEM = {
    "US-Ha1": "Harvard Forest",
    "US-A10": "Barrow",
    "US-Ho1": "Howland Forest",
    "US-EML": "Eight-mile Lake",
}

SITE_ID_TO_BIOME = {
    "US-Ha1": "temperate_forest",
    "US-A10": "arctic_tundra",
    "US-Ho1": "boreal_forest",
    "US-EML": "arctic_tundra",
}

SUBSET_ORDER = [key for key, _ in SUBSETS]
SUBSET_LABELS = {
    "prior_only": "Prior",
    "stocks_only": "Stocks",
    "soil14c_only": "Soil 14C",
    "resp14c_only": "Respired 14C",
    "stocks_soil14c": "Stocks + Soil 14C",
    "stocks_resp14c": "Stocks + Respired 14C",
    "all_observations": "All observations",
}

WARMING_YEARS = [10, 50, 100]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manuscript-style figure inputs from the current saved results and generate the figures."
    )
    parser.add_argument(
        "--chain-dir",
        default=str(_NB / "exports" / "uncertainty_projections_mcmc_long" / "chains"),
    )
    parser.add_argument(
        "--fit-summary",
        default=str(_NB / "exports" / "uncertainty_projections_mcmc_long" / "uncertainty_projection_fit_summary.csv"),
    )
    parser.add_argument(
        "--averaging-kernel-glob",
        default=str(_NB / "exports" / "*_averaging_kernel_matrix.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_NB / "paper_figs" / "outputs" / "current_results"),
    )
    parser.add_argument(
        "--pathway-information",
        default=str(_NB / "exports" / "israd_14c_pathway_information.csv"),
    )
    parser.add_argument(
        "--n-prior-samples",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )
    return parser.parse_args()


def _pool_to_mode(pool_name: str) -> str:
    lower = pool_name.lower()
    if "active" in lower or "fast" in lower or "litter" in lower:
        return "fast"
    if "passive" in lower or "stable" in lower:
        return "slow"
    return "intermediate"


def _subset_label(subset_key: str) -> str:
    return SUBSET_LABELS.get(subset_key, SUBSET_TO_LABEL.get(subset_key, subset_key).replace("\n", " "))


def _observation_rows(site_id: str, site_data: dict) -> list[dict]:
    ecosystem = SITE_ID_TO_ECOSYSTEM[site_id]
    biome = SITE_ID_TO_BIOME[site_id]
    rows: list[dict] = []
    years = np.array(site_data["time_years"], dtype=float)

    for pool_name, arr in site_data.get("delta14C_obs", {}).items():
        vals = np.array(arr, dtype=float)
        for idx in np.where(np.isfinite(vals))[0]:
            rows.append(
                {
                    "ecosystem": ecosystem,
                    "biome": biome,
                    "site": site_id,
                    "observation_type": "bulk_soil",
                    "delta14c_permil": float(vals[idx]),
                    "delta14c_uncertainty_permil": np.nan,
                    "soil_depth_top_cm": np.nan,
                    "soil_depth_bottom_cm": np.nan,
                    "fraction_name": pool_name,
                    "sampling_date": float(years[idx]),
                }
            )

    resp = np.array(site_data.get("delta14C_resp"), dtype=float)
    if resp.size:
        for idx in np.where(np.isfinite(resp))[0]:
            rows.append(
                {
                    "ecosystem": ecosystem,
                    "biome": biome,
                    "site": site_id,
                    "observation_type": "respired_carbon",
                    "delta14c_permil": float(resp[idx]),
                    "delta14c_uncertainty_permil": np.nan,
                    "soil_depth_top_cm": np.nan,
                    "soil_depth_bottom_cm": np.nan,
                    "fraction_name": np.nan,
                    "sampling_date": float(years[idx]),
                }
            )

    for block in site_data.get("extra_blocks", []):
        name = str(block.name).lower()
        if any(token in name for token in ("bulk", "layer")):
            continue
        if not any(token in name for token in ("israd", "density", "macrofossil", "fraction")):
            continue
        vals = np.array(block.y, dtype=float).ravel()
        sigma = np.sqrt(np.array(block.Se, dtype=float).ravel())
        for idx, value in enumerate(vals):
            rows.append(
                {
                    "ecosystem": ecosystem,
                    "biome": biome,
                    "site": site_id,
                    "observation_type": "fraction_specific_soil",
                    "delta14c_permil": float(value),
                    "delta14c_uncertainty_permil": float(sigma[idx]) if idx < sigma.size else np.nan,
                    "soil_depth_top_cm": np.nan,
                    "soil_depth_bottom_cm": np.nan,
                    "fraction_name": block.name,
                    "sampling_date": np.nan,
                }
            )
    return rows


def _load_chain_samples(
    chain_dir: str,
    site_data: dict,
    subset_key: str,
    n_prior_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if subset_key == "prior_only":
        x_mean, cov = _prior_mean_and_cov(site_data)
        return _draw_gaussian_samples(rng, x_mean, cov, n_prior_samples)

    path = os.path.join(chain_dir, f"{site_data['site_id']}__{subset_key}__mcmc_chain.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing saved MCMC chain for {site_data['site_id']} subset {subset_key}: {path}")
    with np.load(path) as data:
        return np.array(data["retained_samples"], dtype=np.float64)


def _year_index_map(forcing_proj, years: list[int]) -> dict[int, int]:
    n_days = int(forcing_proj.time.shape[0])
    out: dict[int, int] = {}
    for year in years:
        out[year] = min(n_days - 1, max(0, int(round(year * 365.25)) - 1))
    return out


def _extract_log_tau_samples(samples_x: np.ndarray, n_modes: int) -> np.ndarray:
    return np.array(samples_x[:, :n_modes], dtype=np.float64)


def _info_rows(
    site_id: str,
    subset_key: str,
    samples_x: np.ndarray,
    site_data: dict,
) -> list[dict]:
    ecosystem = SITE_ID_TO_ECOSYSTEM[site_id]
    subset_label = _subset_label(subset_key)
    n_modes = len(site_data["idx"].pool_names)
    prior_mean, prior_cov = _prior_mean_and_cov(site_data)
    del prior_mean
    prior_sd = np.sqrt(np.diag(prior_cov)[:n_modes])
    log_tau = _extract_log_tau_samples(samples_x, n_modes)
    post_sd = np.std(log_tau, axis=0, ddof=1) if log_tau.shape[0] > 1 else np.zeros(n_modes)

    rows = []
    for i, pool_name in enumerate(site_data["idx"].pool_names):
        prior_var = float(prior_sd[i] ** 2)
        post_var = float(max(post_sd[i] ** 2, 1e-12))
        ak_diag = float(np.clip(1.0 - post_var / prior_var, 0.0, 1.0)) if prior_var > 0 else np.nan
        rows.append(
            {
                "ecosystem": ecosystem,
                "observation_subset": subset_key,
                "observation_subset_label": subset_label,
                "mode": _pool_to_mode(pool_name),
                "degrees_of_freedom": ak_diag,
                "averaging_kernel_diagonal": ak_diag,
                "posterior_sd": float(max(post_sd[i], 1e-12)),
                "prior_sd": float(prior_sd[i]),
                "uncertainty_reduction_fraction": ak_diag,
                "information_gain_nats": float(0.5 * np.log(prior_var / post_var)) if prior_var > 0 else np.nan,
            }
        )
    return rows


def _project_subset_samples(
    site_id: str,
    subset_key: str,
    samples_x: np.ndarray,
    site_data: dict,
    forcing_proj,
    forcing_warm,
) -> tuple[list[dict], list[dict], dict[int, dict]]:
    ecosystem = SITE_ID_TO_ECOSYSTEM[site_id]
    subset_label = _subset_label(subset_key)
    year_idx = _year_index_map(forcing_proj, WARMING_YEARS)
    posterior_rows: list[dict] = []
    warming_rows: list[dict] = []
    draw_diag: dict[int, dict] = {}
    total_days = int(forcing_proj.time.shape[0])
    first_year_end = min(total_days - 1, 364)

    for draw_idx, x_vec in enumerate(samples_x):
        params = vector_to_params(
            jnp.array(x_vec, dtype=jnp.float32),
            site_data["params_prior"],
            tuple(site_data["opt_fields"]),
        )
        state_init = ss_state_for_params(
            site_data["model"],
            forcing_proj,
            site_data["state0_obs"],
            params,
        )
        out_base = run_model(site_data["model"], forcing_proj, state0=state_init, params=params)
        out_warm = run_model(site_data["model"], forcing_warm, state0=state_init, params=params)
        jax.block_until_ready(out_warm.C12)

        rh_base = _compute_pool_rh(site_data["model"], forcing_proj, params, out_base)
        rh_warm = _compute_pool_rh(site_data["model"], forcing_warm, params, out_warm)
        base_total_resp = float(np.sum(rh_base))
        init_c = np.array(state_init.C12, dtype=np.float64)
        tau_years = np.exp(np.array(params.log_tau, dtype=np.float64)) / 365.25
        annual_base_resp = rh_base[: first_year_end + 1].sum(axis=0)
        day100 = year_idx[max(WARMING_YEARS)]
        c_warm_100 = np.array(out_warm.C12[day100], dtype=np.float64)
        cum_base_100 = rh_base[: day100 + 1].sum(axis=0)
        cum_warm_100 = rh_warm[: day100 + 1].sum(axis=0)
        excess_total_100 = float(np.sum(cum_warm_100 - cum_base_100))

        slow_idx = [
            i
            for i, pool_name in enumerate(site_data["idx"].pool_names)
            if _pool_to_mode(pool_name) == "slow"
        ]
        slow_fraction = float(np.sum((cum_warm_100 - cum_base_100)[slow_idx]) / excess_total_100) if abs(excess_total_100) > 1e-12 else np.nan
        draw_diag[draw_idx] = {
            "turnover_time_years": tau_years.copy(),
            "carbon_stock": init_c.copy(),
            "baseline_respiration": annual_base_resp.copy(),
            "cumulative_carbon_loss": (init_c - c_warm_100).copy(),
            "slow_fraction": slow_fraction,
        }

        for pool_i, pool_name in enumerate(site_data["idx"].pool_names):
            mode = _pool_to_mode(pool_name)
            posterior_rows.append(
                {
                    "ecosystem": ecosystem,
                    "observation_subset": subset_key,
                    "observation_subset_label": subset_label,
                    "draw": draw_idx,
                    "mode": mode,
                    "turnover_time_years": float(tau_years[pool_i]),
                    "decomposition_rate_per_year": float(1.0 / tau_years[pool_i]),
                    "carbon_stock": float(init_c[pool_i]),
                    "respiration_fraction": float(np.sum(rh_base[:, pool_i]) / base_total_resp) if base_total_resp > 0 else np.nan,
                }
            )

            for year in WARMING_YEARS:
                day_idx = year_idx[year]
                c_base = float(np.array(out_base.C12[day_idx], dtype=np.float64)[pool_i])
                c_warm = float(np.array(out_warm.C12[day_idx], dtype=np.float64)[pool_i])
                cum_base = float(rh_base[: day_idx + 1, pool_i].sum())
                cum_warm = float(rh_warm[: day_idx + 1, pool_i].sum())
                warming_rows.append(
                    {
                        "ecosystem": ecosystem,
                        "observation_subset": subset_key,
                        "observation_subset_label": subset_label,
                        "draw": draw_idx,
                        "q10": 2.0,
                        "delta_temperature_c": 4.0,
                        "year": year,
                        "mode": mode,
                        "control_carbon": c_base,
                        "warm_carbon": c_warm,
                        "control_respiration": cum_base,
                        "warm_respiration": cum_warm,
                        "cumulative_control_respiration": cum_base,
                        "cumulative_warm_respiration": cum_warm,
                    }
                )
    return posterior_rows, warming_rows, draw_diag


def _build_cesm_rows(
    site_id: str,
    site_data: dict,
    obscon_draw_diag: dict[int, dict],
) -> list[dict]:
    cmip_key = SITE_ID_TO_CMIP_KEY[site_id]
    runtime = SITE_RUNTIME[cmip_key]
    forcing = runtime["forcing_builder"]()
    clm_targets = load_clm_targets(cmip_key)
    fit = fit_one_site(cmip_key, runtime["config"], forcing, clm_targets)

    forcing_proj = _repeat_forcing(site_data["forcing"], max(WARMING_YEARS))
    forcing_warm = _warm_forcing(forcing_proj, 4.0)
    params_clm = site_data["params_opt"]._replace(
        log_tau=jnp.log(jnp.array(np.array(fit["tau"], dtype=np.float32) * 365.25))
    )
    state_init = ss_state_for_params(
        site_data["model"],
        forcing_proj,
        site_data["state0_obs"],
        params_clm,
    )
    out_base = run_model(site_data["model"], forcing_proj, state0=state_init, params=params_clm)
    out_warm = run_model(site_data["model"], forcing_warm, state0=state_init, params=params_clm)
    jax.block_until_ready(out_warm.C12)
    rh_base = _compute_pool_rh(site_data["model"], forcing_proj, params_clm, out_base)
    rh_warm = _compute_pool_rh(site_data["model"], forcing_warm, params_clm, out_warm)
    day100 = _year_index_map(forcing_proj, [100])[100]
    init_c = np.array(state_init.C12, dtype=np.float64)
    annual_base_resp = rh_base[:365].sum(axis=0)
    cum_base_100 = rh_base[: day100 + 1].sum(axis=0)
    cum_warm_100 = rh_warm[: day100 + 1].sum(axis=0)
    c_warm_100 = np.array(out_warm.C12[day100], dtype=np.float64)
    excess_total_100 = float(np.sum(cum_warm_100 - cum_base_100))
    slow_idx = [
        i for i, pool_name in enumerate(site_data["idx"].pool_names) if _pool_to_mode(pool_name) == "slow"
    ]
    slow_fraction = float(np.sum((cum_warm_100 - cum_base_100)[slow_idx]) / excess_total_100) if abs(excess_total_100) > 1e-12 else np.nan

    rows: list[dict] = []
    ecosystem = SITE_ID_TO_ECOSYSTEM[site_id]
    for draw_idx, diag in obscon_draw_diag.items():
        for pool_i, pool_name in enumerate(site_data["idx"].pool_names):
            mode = _pool_to_mode(pool_name)
            rows.append(
                {
                    "ecosystem": ecosystem,
                    "model_source": "observation_constrained",
                    "draw_or_member": draw_idx,
                    "mode": mode,
                    "turnover_time_years": float(diag["turnover_time_years"][pool_i]),
                    "carbon_stock": float(diag["carbon_stock"][pool_i]),
                    "baseline_respiration": float(diag["baseline_respiration"][pool_i]),
                    "warm_respiration": float(diag["baseline_respiration"][pool_i]),
                    "cumulative_carbon_loss": float(diag["cumulative_carbon_loss"][pool_i]),
                    "slow_carbon_fraction_of_excess_respiration": float(diag["slow_fraction"]) if mode == "slow" else 0.0,
                }
            )

    for pool_i, pool_name in enumerate(site_data["idx"].pool_names):
        mode = _pool_to_mode(pool_name)
        rows.append(
            {
                "ecosystem": ecosystem,
                "model_source": "CESM",
                "draw_or_member": 0,
                "mode": mode,
                "turnover_time_years": float(np.array(fit["tau"], dtype=float)[pool_i]),
                "carbon_stock": float(init_c[pool_i]),
                "baseline_respiration": float(annual_base_resp[pool_i]),
                "warm_respiration": float(rh_warm[:365, pool_i].sum()),
                "cumulative_carbon_loss": float(init_c[pool_i] - c_warm_100[pool_i]),
                "slow_carbon_fraction_of_excess_respiration": slow_fraction if mode == "slow" else 0.0,
            }
        )
    return rows


def _kernel_table(kernel_glob: str) -> pd.DataFrame:
    paths = glob.glob(kernel_glob) if any(ch in kernel_glob for ch in "*?[") else [kernel_glob]
    frames = [pd.read_csv(path) for path in sorted(paths) if os.path.isfile(path)]
    if not frames:
        raise FileNotFoundError(f"No averaging-kernel matrix CSVs matched {kernel_glob}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    fit_summary = pd.read_csv(args.fit_summary)
    output_dir = Path(args.output_dir)
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    observation_rows: list[dict] = []
    posterior_rows: list[dict] = []
    info_rows: list[dict] = []
    warming_rows: list[dict] = []
    cesm_rows: list[dict] = []
    obscon_diag_by_site: dict[str, dict[int, dict]] = {}
    site_data_by_id: dict[str, dict] = {}

    for site_id in SITE_ORDER:
        spec = SITE_SPECS[site_id]
        print(f"\nBuilding inputs for {spec.site_label} ({site_id})")
        site_data = spec.runner()
        site_data["site_id"] = site_id
        site_data["site_label"] = SITE_ID_TO_ECOSYSTEM[site_id]
        site_data_by_id[site_id] = site_data

        observation_rows.extend(_observation_rows(site_id, site_data))
        forcing_proj = _repeat_forcing(site_data["forcing"], max(WARMING_YEARS))
        forcing_warm = _warm_forcing(forcing_proj, 4.0)

        for subset_key in SUBSET_ORDER:
            samples_x = _load_chain_samples(args.chain_dir, site_data, subset_key, args.n_prior_samples, rng)
            posterior_sub, warming_sub, draw_diag = _project_subset_samples(
                site_id,
                subset_key,
                samples_x,
                site_data,
                forcing_proj,
                forcing_warm,
            )
            posterior_rows.extend(posterior_sub)
            warming_rows.extend(warming_sub)
            info_rows.extend(_info_rows(site_id, subset_key, samples_x, site_data))
            if subset_key == "all_observations":
                obscon_diag_by_site[site_id] = draw_diag

        cesm_rows.extend(_build_cesm_rows(site_id, site_data, obscon_diag_by_site[site_id]))

    observations = pd.DataFrame(observation_rows)
    posterior = pd.DataFrame(posterior_rows)
    information_metrics = pd.DataFrame(info_rows)
    warming_output = pd.DataFrame(warming_rows)
    cesm_comparison = pd.DataFrame(cesm_rows)
    averaging_kernel_matrix = _kernel_table(args.averaging_kernel_glob)

    path_map = {
        "observations": observations,
        "posterior": posterior,
        "information_metrics": information_metrics,
        "warming_output": warming_output,
        "cesm_comparison": cesm_comparison,
        "averaging_kernel_matrix": averaging_kernel_matrix,
    }
    for name, df in path_map.items():
        out_path = inputs_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"saved {out_path}")

    fig, _ = make_figure_01(output_dir=str(output_dir))
    close_or_show(fig, show=False)
    fig, _ = make_figure_02(observations, output_dir=str(output_dir))
    close_or_show(fig, show=False)
    fig, _ = make_figure_03(posterior, information_metrics, output_dir=str(output_dir))
    close_or_show(fig, show=False)
    fig, _ = make_figure_04(information_metrics, averaging_kernel_matrix=averaging_kernel_matrix, output_dir=str(output_dir))
    close_or_show(fig, show=False)
    fig, _ = make_figure_05(observations, posterior, information_metrics, warming_output, output_dir=str(output_dir))
    close_or_show(fig, show=False)
    fig, _ = make_figure_06(warming_output, output_dir=str(output_dir), horizon_year=100)
    close_or_show(fig, show=False)
    fig, _ = make_figure_07(cesm_comparison, output_dir=str(output_dir))
    close_or_show(fig, show=False)
    if os.path.isfile(args.pathway_information):
        pathway_information = pd.read_csv(args.pathway_information)
        pathway_information.to_csv(inputs_dir / "pathway_information.csv", index=False)
        fig, _ = make_figure_08(pathway_information, output_dir=str(output_dir))
        close_or_show(fig, show=False)


if __name__ == "__main__":
    main()
