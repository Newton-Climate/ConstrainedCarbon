"""
uncertainty_projections.py — Figure 4-style forecast-constraint analysis.

This script asks whether radiocarbon observations reduce uncertainty in
warming-vulnerability projections, whether they change the inferred old-carbon
contribution to warming-induced respiration, and which 14C observation type is
more valuable at each site.

Workflow
--------
For each site and observation subset:
  1. Reuse or rerun the OE inversion.
  2. Draw Gaussian parameter samples from the prior or OE posterior.
  3. Project each sample forward for 100 years under repeated baseline forcing
     and a +4 C warming perturbation applied to air and soil temperature.
  4. Compute:
       - fractional C loss after 100 years
       - absolute C loss after 100 years
       - fraction of warming-induced Rh coming from old pools
  5. Export sample-level and summary tables and render the 3-panel figure.

Run from the repository root:
    python notebooks/uncertainty_projections.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NB_ROOT = os.path.join(_SCRIPT_ROOT, "notebooks")
_SRC_ROOT = os.path.join(_SCRIPT_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_SCRIPT_ROOT)

from ecosystem_complexity.api import optimize_oe, run_model  # noqa: E402
from ecosystem_complexity._oe_helpers import (  # noqa: E402
    _analytical_c12_ss,
    _build_obs_blocks,
    _build_sa_diag,
    build_oe_prior_sigma,
)
from ecosystem_complexity.data.schemas import ForcingData  # noqa: E402
from ecosystem_complexity.oe_utils import (  # noqa: E402
    build_mean_ss_modifier,
    ss_state_for_params,
)
from ecosystem_complexity.optimizer import (  # noqa: E402
    params_to_vector,
    vector_to_params,
)
from sites.canonical import (  # noqa: E402
    run_barrow_canonical,
    run_hf_canonical,
)
from sites.eight_mile_lake import run_eml_canonical  # noqa: E402
from sites.howland_forest import run_howland_canonical  # noqa: E402


PANEL_A_TITLE = (
    "A. Posterior uncertainty in +4 C warming-induced C loss\n"
    "Median and 5-95% interval of fractional C loss after 100 years"
)
PANEL_B_TITLE = (
    "B. Posterior uncertainty in old-pool contribution to warming-induced Rh\n"
    "Old pools are defined here as slow + passive pools"
)
PANEL_C_TITLE = (
    "C. Site-specific value of soil/pool 14C versus respired 14C\n"
    "Uncertainty reduction is computed relative to the prior variance"
)

SUBSETS = [
    ("prior_only", "Prior"),
    ("stocks_only", "Stocks"),
    ("soil14c_only", "Pool 14C"),
    ("resp14c_only", "Resp 14C"),
    ("stocks_soil14c", "Stocks+\nPool 14C"),
    ("stocks_resp14c", "Stocks+\nResp 14C"),
    ("all_observations", "All"),
]
SUBSET_TO_LABEL = dict(SUBSETS)
SUBSET_COLORS = {
    "prior_only": "#B8B8B8",
    "stocks_only": "#4C78A8",
    "soil14c_only": "#4E8B5D",
    "resp14c_only": "#C76D3A",
    "stocks_soil14c": "#7A9D54",
    "stocks_resp14c": "#B6543C",
    "all_observations": "#1F3A2E",
}
SITE_ORDER = ["US-Ha1", "US-A10", "US-Ho1", "US-EML"]


@dataclass(frozen=True)
class SiteSpec:
    site_id: str
    site_label: str
    runner: Callable[[], dict]
    old_pools: tuple[str, ...] = ("soil_slow", "soil_passive")


SITE_SPECS = {
    "US-Ha1": SiteSpec("US-Ha1", "Harvard Forest", run_hf_canonical),
    "US-A10": SiteSpec("US-A10", "Barrow", run_barrow_canonical),
    "US-Ho1": SiteSpec("US-Ho1", "Howland Forest", run_howland_canonical),
    "US-EML": SiteSpec("US-EML", "Eight-mile Lake", run_eml_canonical),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sites",
        nargs="*",
        choices=SITE_ORDER,
        default=SITE_ORDER,
        help="Subset of site IDs to run.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=120,
        help="Target number of retained posterior samples per site and subset. "
        "When resuming MCMC, additional samples are drawn only until this total is reached.",
    )
    parser.add_argument(
        "--sampler",
        choices=("gaussian", "mcmc"),
        default="gaussian",
        help="Posterior sampler. 'gaussian' uses the Laplace approximation; "
        "'mcmc' runs Metropolis-Hastings on the OE posterior.",
    )
    parser.add_argument(
        "--mcmc-burn",
        type=int,
        default=40,
        help="Burn-in steps per subset when --sampler mcmc.",
    )
    parser.add_argument(
        "--mcmc-thin",
        type=int,
        default=2,
        help="Thinning interval for retained MCMC samples.",
    )
    parser.add_argument(
        "--proposal-scale",
        type=float,
        default=1.0,
        help="Additional multiplicative scale applied to the proposal covariance "
        "when --sampler mcmc.",
    )
    parser.add_argument(
        "--horizon-years",
        type=float,
        default=100.0,
        help="Projection horizon in years.",
    )
    parser.add_argument(
        "--warming-delta-c",
        type=float,
        default=4.0,
        help="Uniform warming perturbation applied to air and soil temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for posterior sampling.",
    )
    parser.add_argument(
        "--figure-path",
        default=os.path.join(_NB_ROOT, "uncertainty_projections_figure.png"),
        help="Path for the rendered figure.",
    )
    parser.add_argument(
        "--exports-dir",
        default=os.path.join(_NB_ROOT, "exports"),
        help="Directory for CSV exports.",
    )
    parser.add_argument(
        "--chain-dir",
        default=None,
        help="Directory for saved MCMC chain checkpoints. Defaults to "
        "<exports-dir>/chains.",
    )
    parser.add_argument(
        "--no-resume-mcmc",
        action="store_true",
        help="Disable loading saved MCMC checkpoints.",
    )
    return parser.parse_args()


def _count_scalar_obs(obs, extra_blocks: list) -> int:
    n = 0
    n += sum(int(np.isfinite(np.array(v)).sum()) for v in obs.delta14C_obs.values())
    if obs.delta14C_resp is not None:
        n += int(np.isfinite(np.array(obs.delta14C_resp)).sum())
    n += len(obs.C_pools_obs or {})
    n += sum(int(block.y.shape[0]) for block in extra_blocks)
    return n


def _build_subset_inputs(site_data: dict, subset_key: str):
    base_obs = site_data["obs_full"]
    use_stocks = subset_key in {
        "stocks_only",
        "stocks_soil14c",
        "stocks_resp14c",
        "all_observations",
    }
    use_soil14c = subset_key in {
        "soil14c_only",
        "stocks_soil14c",
        "all_observations",
    }
    use_resp14c = subset_key in {
        "resp14c_only",
        "stocks_resp14c",
        "all_observations",
    }
    obs_subset = base_obs._replace(
        delta14C_obs=site_data["delta14C_obs"] if use_soil14c else {},
        C_pools_obs=site_data["c_pools_obs"] if use_stocks else {},
        delta14C_resp=site_data["delta14C_resp"] if use_resp14c else None,
    )
    extra_blocks = list(site_data.get("extra_blocks", [])) if use_soil14c else []
    return obs_subset, extra_blocks


def _prior_mean_and_cov(site_data: dict) -> tuple[np.ndarray, np.ndarray]:
    opt_fields = tuple(site_data["opt_fields"])
    params_prior = site_data["params_prior"]
    x_mean = np.array(params_to_vector(params_prior, opt_fields), dtype=np.float64)
    sigma = np.array(
        build_oe_prior_sigma(site_data["model"].config, params_prior, opt_fields),
        dtype=np.float64,
    )
    cov = np.diag(np.square(sigma))
    return x_mean, cov


def _fit_subset(site_data: dict, subset_key: str) -> dict:
    opt_fields = tuple(site_data["opt_fields"])
    if subset_key == "prior_only":
        x_mean, cov = _prior_mean_and_cov(site_data)
        return {
            "subset_key": subset_key,
            "n_obs": 0,
            "converged": True,
            "cost_final": np.nan,
            "x_mean": x_mean,
            "cov": cov,
        }

    if subset_key == "all_observations":
        oe = site_data["oe_result"]
        return {
            "subset_key": subset_key,
            "n_obs": int(oe.y_obs.shape[0]),
            "converged": bool(oe.converged),
            "cost_final": float(np.array(oe.cost_history)[-1]),
            "x_mean": np.array(oe.x_opt, dtype=np.float64),
            "cov": np.array(oe.Sx, dtype=np.float64),
            "acceptance_rate": np.nan,
            "proposal_scale_final": np.nan,
        }

    obs_subset, extra_blocks = _build_subset_inputs(site_data, subset_key)
    n_obs = _count_scalar_obs(obs_subset, extra_blocks)
    if n_obs == 0:
        x_mean, cov = _prior_mean_and_cov(site_data)
        return {
            "subset_key": subset_key,
            "n_obs": 0,
            "converged": True,
            "cost_final": np.nan,
            "x_mean": x_mean,
            "cov": cov,
            "acceptance_rate": np.nan,
            "proposal_scale_final": np.nan,
        }

    print(
        f"[{site_data['site_id']}] subset={subset_key}  "
        f"n_obs={n_obs}  extras={[b.name for b in extra_blocks]}"
    )
    result = optimize_oe(
        site_data["model"],
        site_data["forcing"],
        obs_subset,
        state0=site_data["state0_obs"],
        fields=opt_fields,
        extra_obs_blocks=extra_blocks,
    )
    return {
        "subset_key": subset_key,
        "n_obs": int(result.y_obs.shape[0]),
        "converged": bool(result.converged),
        "cost_final": float(np.array(result.cost_history)[-1]),
        "x_mean": np.array(result.x_opt, dtype=np.float64),
        "cov": np.array(result.Sx, dtype=np.float64),
        "acceptance_rate": np.nan,
        "proposal_scale_final": np.nan,
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


def _chain_state_path(chain_dir: str, site_id: str, subset_key: str) -> str:
    return os.path.join(chain_dir, f"{site_id}__{subset_key}__mcmc_chain.npz")


def _save_chain_checkpoint(
    path: str,
    retained_samples: np.ndarray,
    current: np.ndarray,
    current_lp: float,
    proposal_scale: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        retained_samples=np.array(retained_samples, dtype=np.float64),
        current=np.array(current, dtype=np.float64),
        current_lp=np.array([current_lp], dtype=np.float64),
        proposal_scale=np.array([proposal_scale], dtype=np.float64),
    )


def _load_chain_checkpoint(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with np.load(path) as data:
        return {
            "retained_samples": np.array(data["retained_samples"], dtype=np.float64),
            "current": np.array(data["current"], dtype=np.float64),
            "current_lp": float(np.array(data["current_lp"], dtype=np.float64).ravel()[0]),
            "proposal_scale": float(np.array(data["proposal_scale"], dtype=np.float64).ravel()[0]),
        }


def _build_logposter_fn(site_data: dict, subset_key: str):
    if subset_key == "prior_only":
        return None

    model = site_data["model"]
    forcing = site_data["forcing"]
    state0 = site_data["state0_obs"]
    params0 = site_data["params_prior"]
    opt_fields = tuple(site_data["opt_fields"])
    obs_subset, extra_blocks = _build_subset_inputs(site_data, subset_key)

    inv_cfg = getattr(model.config, "inversion_raw", {}) or {}
    sigma_pool = float(inv_cfg.get("sigma_pool_14C", 5.0))
    sigma_resp = float(inv_cfg.get("sigma_resp_14C", 10.0))
    sigma_carbon = float(inv_cfg.get("sigma_carbon_gCm2", 1000.0))
    f_hetero = float(inv_cfg.get("f_hetero", 0.0))
    sigma_er_frac = float(inv_cfg.get("sigma_er_frac", 0.15))
    obs_blocks = _build_obs_blocks(
        obs_subset,
        model,
        sigma_pool,
        sigma_resp,
        sigma_carbon,
        f_hetero=f_hetero,
        sigma_er_frac=sigma_er_frac,
    )
    if extra_blocks:
        obs_blocks = obs_blocks + list(extra_blocks)
    if not obs_blocks:
        return None

    y_obs = jnp.concatenate([b.y for b in obs_blocks])
    se_inv = 1.0 / (jnp.concatenate([b.Se for b in obs_blocks]) + 1e-30)
    xa = params_to_vector(params0, opt_fields)
    sa_inv = 1.0 / (_build_sa_diag(model.config, params0, opt_fields) + 1e-30)

    ext_cfg = model.config.external_inputs
    cue = float(getattr(ext_cfg, "CUE", 0.47))
    mean_modifier, mean_gpp = build_mean_ss_modifier(forcing, params0)
    mean_input = mean_gpp * cue
    n_pools = len(model.pool_index)
    target_names = list(ext_cfg.partition.keys()) if ext_cfg is not None else []
    target_idx = [model.pool_index[n] for n in target_names] or None

    @jax.jit
    def _logposter(x_vec: jnp.ndarray) -> jnp.ndarray:
        params = vector_to_params(x_vec, params0, opt_fields)
        c12_ss = _analytical_c12_ss(
            params,
            n_pools,
            mean_input,
            mean_modifier,
            target_indices=target_idx,
        )
        state_ss = state0._replace(C12=c12_ss)
        out = run_model(model, forcing, state0=state_ss, params=params)
        y_hat = jnp.concatenate([b.predict(out, params) for b in obs_blocks])
        resid = y_obs - y_hat
        prior_r = xa - x_vec
        cost = jnp.sum(se_inv * resid**2) + jnp.sum(sa_inv * prior_r**2)
        return -0.5 * cost

    return _logposter


def _run_mcmc_samples(
    rng: np.random.Generator,
    logposter_fn,
    x_start: np.ndarray,
    proposal_cov: np.ndarray,
    n_keep: int,
    burn: int,
    thin: int,
    proposal_scale: float,
    current_lp_start: float | None = None,
) -> tuple[np.ndarray, float, float, np.ndarray, float]:
    cov_sym = 0.5 * (proposal_cov + proposal_cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov_sym)
    eigvals = np.clip(eigvals, 1e-12, None)
    chol = eigvecs @ np.diag(np.sqrt(eigvals))

    n_dim = x_start.size
    scale = float(proposal_scale)
    current = np.array(x_start, dtype=np.float64)
    if current_lp_start is None:
        current_lp = float(logposter_fn(jnp.array(current, dtype=jnp.float32)))
        scale = float(proposal_scale) * 2.38 / math.sqrt(max(n_dim, 1))
    else:
        current_lp = float(current_lp_start)
    total_steps = burn + n_keep * thin
    samples: list[np.ndarray] = []
    accepts_total = 0
    accepts_window = 0
    window_len = 10

    for step in range(total_steps):
        proposal = current + scale * (chol @ rng.standard_normal(n_dim))
        proposal_lp = float(logposter_fn(jnp.array(proposal, dtype=jnp.float32)))
        log_alpha = proposal_lp - current_lp
        accepted = np.log(rng.random()) < log_alpha
        if accepted:
            current = proposal
            current_lp = proposal_lp
            accepts_total += 1
            accepts_window += 1

        if step < burn and (step + 1) % window_len == 0:
            acc_rate = accepts_window / window_len
            if acc_rate < 0.15:
                scale /= 1.25
            elif acc_rate > 0.4:
                scale *= 1.25
            accepts_window = 0

        if step >= burn and ((step - burn) % thin == 0):
            samples.append(current.copy())

    acceptance_rate = accepts_total / max(total_steps, 1)
    return (
        np.array(samples, dtype=np.float64),
        float(acceptance_rate),
        float(scale),
        current.copy(),
        float(current_lp),
    )


def _draw_subset_samples(
    site_data: dict,
    fit: dict,
    subset_key: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
    chain_dir: str | None = None,
) -> tuple[np.ndarray, float, float]:
    if args.sampler == "gaussian" or subset_key == "prior_only":
        return (
            _draw_gaussian_samples(rng, fit["x_mean"], fit["cov"], args.n_samples),
            np.nan,
            np.nan,
        )

    logposter_fn = _build_logposter_fn(site_data, subset_key)
    if logposter_fn is None:
        return (
            _draw_gaussian_samples(rng, fit["x_mean"], fit["cov"], args.n_samples),
            np.nan,
            np.nan,
        )
    checkpoint = None
    checkpoint_path = None
    if chain_dir is not None and not args.no_resume_mcmc:
        checkpoint_path = _chain_state_path(chain_dir, site_data["site_id"], subset_key)
        checkpoint = _load_chain_checkpoint(checkpoint_path)

    if checkpoint is not None:
        retained = checkpoint["retained_samples"]
        if retained.shape[0] >= args.n_samples:
            return retained[: args.n_samples], np.nan, checkpoint["proposal_scale"]
        n_needed = args.n_samples - retained.shape[0]
        new_samples, acceptance_rate, proposal_scale_final, current, current_lp = _run_mcmc_samples(
            rng,
            logposter_fn,
            checkpoint["current"],
            fit["cov"],
            n_keep=n_needed,
            burn=0,
            thin=args.mcmc_thin,
            proposal_scale=checkpoint["proposal_scale"],
            current_lp_start=checkpoint["current_lp"],
        )
        all_samples = np.vstack([retained, new_samples]) if retained.size else new_samples
        if checkpoint_path is not None:
            _save_chain_checkpoint(
                checkpoint_path,
                all_samples,
                current,
                current_lp,
                proposal_scale_final,
            )
        return all_samples, acceptance_rate, proposal_scale_final

    samples, acceptance_rate, proposal_scale_final, current, current_lp = _run_mcmc_samples(
        rng,
        logposter_fn,
        fit["x_mean"],
        fit["cov"],
        n_keep=args.n_samples,
        burn=args.mcmc_burn,
        thin=args.mcmc_thin,
        proposal_scale=args.proposal_scale,
    )
    if checkpoint_path is not None:
        _save_chain_checkpoint(
            checkpoint_path,
            samples,
            current,
            current_lp,
            proposal_scale_final,
        )
    return samples, acceptance_rate, proposal_scale_final


def _repeat_forcing(forcing: ForcingData, horizon_years: float) -> ForcingData:
    target_days = max(1, int(round(horizon_years * 365.25)))
    n_in = int(forcing.time.shape[0])
    reps = int(math.ceil(target_days / n_in))

    def _tile(arr):
        arr_np = np.array(arr)
        tiled = np.concatenate([arr_np] * reps, axis=0)
        return jnp.array(tiled[:target_days], dtype=arr.dtype)

    time0 = float(np.array(forcing.time)[0])
    return ForcingData(
        time=jnp.array(time0 + np.arange(target_days), dtype=forcing.time.dtype),
        air_temp=_tile(forcing.air_temp),
        sw_radiation=_tile(forcing.sw_radiation),
        precip=_tile(forcing.precip),
        vpd=_tile(forcing.vpd),
        soil_temp=_tile(forcing.soil_temp),
        soil_moisture=_tile(forcing.soil_moisture),
        snow_depth=_tile(forcing.snow_depth),
        active_layer=_tile(forcing.active_layer),
        delta14C_atm=_tile(forcing.delta14C_atm),
        GPP_obs=_tile(forcing.GPP_obs),
        NPP_obs=_tile(forcing.NPP_obs),
    )


def _warm_forcing(forcing: ForcingData, delta_c: float) -> ForcingData:
    soil_temp = np.array(forcing.soil_temp, dtype=np.float32)
    soil_temp = np.where(np.isnan(soil_temp), soil_temp, soil_temp + delta_c)
    air_temp = np.array(forcing.air_temp, dtype=np.float32) + delta_c
    return forcing._replace(
        air_temp=jnp.array(air_temp, dtype=forcing.air_temp.dtype),
        soil_temp=jnp.array(soil_temp, dtype=forcing.soil_temp.dtype),
    )


def _softmax_last_axis(x: np.ndarray) -> np.ndarray:
    x_shift = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def _compute_pool_rh(model, forcing: ForcingData, params, output) -> np.ndarray:
    air_temp = np.nan_to_num(np.array(forcing.air_temp, dtype=np.float64), nan=5.0)
    soil_temp = np.array(forcing.soil_temp, dtype=np.float64)
    soil_temp = np.where(np.isnan(soil_temp), air_temp[:, None], soil_temp)
    soil_moisture = np.array(forcing.soil_moisture, dtype=np.float64)
    soil_moisture = np.where(np.isnan(soil_moisture), 0.3, soil_moisture)

    q10 = np.exp(np.array(params.log_Q10, dtype=np.float64))
    theta_opt = np.exp(np.array(params.log_theta_opt, dtype=np.float64))
    gamma = np.exp(np.array(params.log_gamma_moist, dtype=np.float64))
    tau = np.exp(np.array(params.log_tau, dtype=np.float64))

    ft_layers = q10[None, :] ** ((soil_temp - 15.0) / 10.0)
    fm_layers = np.exp(-gamma[None, :] * (soil_moisture - theta_opt[None, :]) ** 2)
    pool_to_layer = np.array(model._pool_to_layer, dtype=int)
    ft_vec = ft_layers[:, pool_to_layer]
    fm_vec = fm_layers[:, pool_to_layer]

    if model._pool_mid_depths is not None and model._T_annual_mean is not None:
        z_mid = np.array(model._pool_mid_depths, dtype=np.float64)
        t_surface = soil_temp[:, pool_to_layer]
        amp = np.exp(-z_mid[None, :] / float(model._damping_depth_m))
        t_depth = float(model._T_annual_mean) + (t_surface - float(model._T_annual_mean)) * amp
        ff_vec = 1.0 / (1.0 + np.exp(-10.0 * t_depth))
    else:
        ff_layers = 1.0 / (1.0 + np.exp(-10.0 * soil_temp))
        ff_vec = ff_layers[:, pool_to_layer]

    f_full = _softmax_last_axis(np.array(params.log_f_transfer, dtype=np.float64))
    resp_frac = f_full[:, -1]
    c12 = np.array(output.C12, dtype=np.float64)
    return c12 * (ft_vec * fm_vec * ff_vec) * (resp_frac[None, :] / tau[None, :])


def _project_metrics(
    site_data: dict,
    forcing_proj: ForcingData,
    forcing_warm: ForcingData,
    params,
    old_pools: tuple[str, ...],
) -> dict[str, float]:
    state_init = ss_state_for_params(
        site_data["model"],
        forcing_proj,
        site_data["state0_obs"],
        params,
    )
    out_base = run_model(site_data["model"], forcing_proj, state0=state_init, params=params)
    out_warm = run_model(site_data["model"], forcing_warm, state0=state_init, params=params)
    jax.block_until_ready(out_warm.C12)

    c0 = float(np.sum(np.array(state_init.C12, dtype=np.float64)))
    c_base_final = float(np.sum(np.array(out_base.C12[-1], dtype=np.float64)))
    c_warm_final = float(np.sum(np.array(out_warm.C12[-1], dtype=np.float64)))
    abs_loss = c_base_final - c_warm_final
    frac_loss = abs_loss / c0 if c0 > 0 else np.nan

    rh_base = _compute_pool_rh(site_data["model"], forcing_proj, params, out_base)
    rh_warm = _compute_pool_rh(site_data["model"], forcing_warm, params, out_warm)
    old_idx = [
        site_data["idx"][pool_name]
        for pool_name in old_pools
        if pool_name in site_data["idx"].pool_names
    ]
    delta_total = float(np.sum(rh_warm.sum(axis=1) - rh_base.sum(axis=1)))
    delta_old = float(
        np.sum(rh_warm[:, old_idx].sum(axis=1) - rh_base[:, old_idx].sum(axis=1))
    )
    old_fraction = delta_old / delta_total if abs(delta_total) > 1e-12 else np.nan

    return {
        "frac_loss": frac_loss,
        "abs_loss": abs_loss,
        "old_fraction": old_fraction,
    }


def _summarize_samples(samples_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (site_id, subset_key), grp in samples_df.groupby(["site_id", "subset_key"], sort=False):
        meta = grp.iloc[0]
        for metric in ("frac_loss", "abs_loss", "old_fraction"):
            vals = grp[metric].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "site_id": site_id,
                    "site_label": meta["site_label"],
                    "subset_key": subset_key,
                    "subset_label": meta["subset_label"],
                    "metric": metric,
                    "n_valid": int(vals.size),
                    "median": float(np.nanmedian(vals)) if vals.size else np.nan,
                    "p05": float(np.nanquantile(vals, 0.05)) if vals.size else np.nan,
                    "p95": float(np.nanquantile(vals, 0.95)) if vals.size else np.nan,
                    "variance": float(np.nanvar(vals, ddof=1)) if vals.size > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _compute_panel_c(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for site_id in SITE_ORDER:
        site_df = summary_df[summary_df["site_id"] == site_id]
        if site_df.empty:
            continue
        site_label = str(site_df["site_label"].iloc[0])
        for metric in ("frac_loss", "old_fraction"):
            metric_df = site_df[site_df["metric"] == metric]
            prior_var = _lookup_variance(metric_df, "prior_only")
            soil_var = _lookup_variance(metric_df, "stocks_soil14c")
            resp_var = _lookup_variance(metric_df, "stocks_resp14c")
            rows.append(
                {
                    "site_id": site_id,
                    "site_label": site_label,
                    "metric": metric,
                    "u_soil14c": _uncertainty_reduction(soil_var, prior_var),
                    "u_resp14c": _uncertainty_reduction(resp_var, prior_var),
                    "prior_variance": prior_var,
                    "stocks_soil14c_variance": soil_var,
                    "stocks_resp14c_variance": resp_var,
                }
            )
    return pd.DataFrame(rows)


def _lookup_variance(metric_df: pd.DataFrame, subset_key: str) -> float:
    sub = metric_df[metric_df["subset_key"] == subset_key]
    if sub.empty:
        return np.nan
    return float(sub["variance"].iloc[0])


def _uncertainty_reduction(var_subset: float, var_prior: float) -> float:
    if not np.isfinite(var_subset) or not np.isfinite(var_prior) or var_prior <= 0:
        return np.nan
    return 1.0 - (var_subset / var_prior)


def _metric_axis_limits(summary_df: pd.DataFrame, metric: str) -> tuple[float, float]:
    vals = summary_df.loc[summary_df["metric"] == metric, ["p05", "p95"]].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    pad = 0.08 * (vmax - vmin + 1e-9)
    return vmin - pad, vmax + pad


def _plot_interval_panel(
    ax,
    summary_df: pd.DataFrame,
    site_id: str,
    metric: str,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    site_rows = summary_df[
        (summary_df["site_id"] == site_id) & (summary_df["metric"] == metric)
    ].copy()
    if site_rows.empty:
        ax.set_visible(False)
        return
    site_rows["subset_order"] = site_rows["subset_key"].map(
        {key: i for i, (key, _) in enumerate(SUBSETS)}
    )
    site_rows = site_rows.sort_values("subset_order")

    x = np.arange(len(site_rows))
    y = site_rows["median"].to_numpy(dtype=float)
    ylo = y - site_rows["p05"].to_numpy(dtype=float)
    yhi = site_rows["p95"].to_numpy(dtype=float) - y
    colors = [SUBSET_COLORS[k] for k in site_rows["subset_key"]]
    for xi, yi, lo, hi, color in zip(x, y, ylo, yhi, colors):
        ax.errorbar(
            xi,
            yi,
            yerr=np.array([[lo], [hi]]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.8,
            capsize=3,
            markersize=6,
            markeredgecolor="#1F1F1F",
            markeredgewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SUBSET_TO_LABEL[k] for k in site_rows["subset_key"]], rotation=28, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(
        f"{site_rows['site_label'].iloc[0]} ({site_id})",
        fontsize=10,
        loc="left",
        color="#1F3A2E",
        fontweight="bold",
    )
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=8)


def _plot_panel_c(ax, panel_c_df: pd.DataFrame) -> None:
    df = panel_c_df[panel_c_df["metric"] == "frac_loss"].copy()
    finite = df[np.isfinite(df["u_soil14c"]) & np.isfinite(df["u_resp14c"])]
    if finite.empty:
        ax.set_visible(False)
        return

    xmin = min(-0.1, float(finite["u_soil14c"].min()) - 0.05)
    xmax = max(1.0, float(finite["u_soil14c"].max()) + 0.05)
    ymin = min(-0.1, float(finite["u_resp14c"].min()) - 0.05)
    ymax = max(1.0, float(finite["u_resp14c"].max()) + 0.05)
    lo = min(xmin, ymin)
    hi = max(xmax, ymax)

    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#8C8C8C", lw=1.0)
    for row in finite.itertuples(index=False):
        ax.scatter(
            row.u_soil14c,
            row.u_resp14c,
            s=85,
            color="#D4A574",
            edgecolors="#1F3A2E",
            linewidths=1.0,
            zorder=3,
        )
        ax.text(
            row.u_soil14c + 0.02,
            row.u_resp14c + 0.02,
            row.site_id,
            fontsize=9,
            color="#2A2A2A",
            ha="left",
            va="bottom",
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Uncertainty reduction from stocks + pool 14C", fontsize=10)
    ax.set_ylabel("Uncertainty reduction from stocks + respired 14C", fontsize=10)
    ax.grid(lw=0.4, alpha=0.35)
    ax.tick_params(labelsize=9)


def _make_figure(summary_df: pd.DataFrame, panel_c_df: pd.DataFrame, out_path: str) -> None:
    fig = plt.figure(figsize=(16, 18), constrained_layout=False)
    outer = gridspec.GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[1.25, 1.25, 0.9],
        hspace=0.42,
        top=0.97,
        bottom=0.06,
        left=0.06,
        right=0.98,
    )

    frac_ylim = _metric_axis_limits(summary_df, "frac_loss")
    old_ylim = _metric_axis_limits(summary_df, "old_fraction")

    gs_a = outer[0].subgridspec(2, 2, wspace=0.18, hspace=0.34)
    gs_b = outer[1].subgridspec(2, 2, wspace=0.18, hspace=0.34)

    fig.text(0.06, 0.975, PANEL_A_TITLE, fontsize=13, fontweight="bold", color="#1F3A2E", va="top")
    fig.text(0.06, 0.648, PANEL_B_TITLE, fontsize=13, fontweight="bold", color="#1F3A2E", va="top")
    fig.text(0.06, 0.320, PANEL_C_TITLE, fontsize=13, fontweight="bold", color="#1F3A2E", va="top")

    for i, site_id in enumerate(SITE_ORDER):
        ax_a = fig.add_subplot(gs_a[i // 2, i % 2])
        _plot_interval_panel(
            ax_a,
            summary_df,
            site_id,
            "frac_loss",
            "Fractional C loss",
            frac_ylim,
        )

        ax_b = fig.add_subplot(gs_b[i // 2, i % 2])
        _plot_interval_panel(
            ax_b,
            summary_df,
            site_id,
            "old_fraction",
            "Old-pool share of warming Rh",
            old_ylim,
        )

    ax_c = fig.add_subplot(outer[2])
    _plot_panel_c(ax_c, panel_c_df)

    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"saved {out_path}")


def main() -> None:
    args = _parse_args()
    os.makedirs(args.exports_dir, exist_ok=True)
    chain_dir = args.chain_dir or os.path.join(args.exports_dir, "chains")
    if args.sampler == "mcmc":
        os.makedirs(chain_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    sample_rows: list[dict] = []
    fit_rows: list[dict] = []

    for site_id in args.sites:
        spec = SITE_SPECS[site_id]
        print(f"\nRunning site: {spec.site_label} ({site_id})")
        site_data = spec.runner()
        site_data["site_id"] = site_id
        site_data["site_label"] = spec.site_label

        forcing_proj = _repeat_forcing(site_data["forcing"], args.horizon_years)
        forcing_warm = _warm_forcing(forcing_proj, args.warming_delta_c)
        fits = {subset_key: _fit_subset(site_data, subset_key) for subset_key, _ in SUBSETS}

        for subset_key, subset_label in SUBSETS:
            fit = fits[subset_key]
            fit_rows.append(
                {
                    "site_id": site_id,
                    "site_label": spec.site_label,
                    "subset_key": subset_key,
                    "subset_label": subset_label,
                    "sampler": args.sampler,
                    "n_obs": fit["n_obs"],
                    "converged": fit["converged"],
                    "cost_final": fit["cost_final"],
                    "n_retained_samples": args.n_samples,
                }
            )
            samples_x, acceptance_rate, proposal_scale_final = _draw_subset_samples(
                site_data,
                fit,
                subset_key,
                args,
                rng,
                chain_dir=chain_dir if args.sampler == "mcmc" else None,
            )
            fit_rows[-1]["n_retained_samples"] = int(samples_x.shape[0])
            fit_rows[-1]["acceptance_rate"] = acceptance_rate
            fit_rows[-1]["proposal_scale_final"] = proposal_scale_final
            for sample_idx, x_vec in enumerate(samples_x):
                params = vector_to_params(
                    jnp.array(x_vec, dtype=jnp.float32),
                    site_data["params_prior"],
                    tuple(site_data["opt_fields"]),
                )
                metrics = _project_metrics(
                    site_data,
                    forcing_proj,
                    forcing_warm,
                    params,
                    spec.old_pools,
                )
                sample_rows.append(
                    {
                        "site_id": site_id,
                        "site_label": spec.site_label,
                        "subset_key": subset_key,
                        "subset_label": subset_label,
                        "sample_idx": sample_idx,
                        **metrics,
                    }
                )

    samples_df = pd.DataFrame(sample_rows)
    fit_df = pd.DataFrame(fit_rows)
    summary_df = _summarize_samples(samples_df)
    panel_c_df = _compute_panel_c(summary_df)

    samples_path = os.path.join(args.exports_dir, "uncertainty_projection_samples.csv")
    summary_path = os.path.join(args.exports_dir, "uncertainty_projection_summary.csv")
    panel_c_path = os.path.join(args.exports_dir, "uncertainty_projection_panel_c.csv")
    fit_path = os.path.join(args.exports_dir, "uncertainty_projection_fit_summary.csv")

    samples_df.to_csv(samples_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    panel_c_df.to_csv(panel_c_path, index=False)
    fit_df.to_csv(fit_path, index=False)
    print(f"saved {samples_path}")
    print(f"saved {summary_path}")
    print(f"saved {panel_c_path}")
    print(f"saved {fit_path}")

    _make_figure(summary_df, panel_c_df, args.figure_path)


if __name__ == "__main__":
    main()
