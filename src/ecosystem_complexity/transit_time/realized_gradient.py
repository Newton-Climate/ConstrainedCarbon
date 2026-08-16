#!/usr/bin/env python3
"""Compare intrinsic and environmentally realised transit time across four sites.

Each site is re-inverted so the diagnostic uses its MAP turnover times and
transfer fractions.  Realised transit time is the expected respiratory exit
age of a unit soil-C input propagated through the site's repeating *daily*
forcing record.  Thus it retains GPP seasonality for input weighting and the
temperature, moisture, and thaw modifiers for decomposition.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecosystem_complexity.sites.driver import run_site_canonical
from ecosystem_complexity.sites.spec import load_site_spec
from ecosystem_complexity.optimizer import vector_to_params
from ecosystem_complexity.transit_time import intrinsic_mean_transit_time, realized_mean_transit_time

SITES = {
    "arctic / permafrost": "configs/multisite/eml.yaml",
    "temperate forest": "configs/multisite/harvard_forest.yaml",
    "tropical forest": "configs/multisite/baram_basin.yaml",
    "semi-arid grassland": "configs/expansion/az_mollisol.yaml",
}


def input_weights(model, params) -> np.ndarray:
    weights = np.zeros(len(model.pool_index), dtype=float)
    ext = model.config.external_inputs
    if ext is not None and ext.enabled:
        targets = [model.pool_index[name] for name in ext.partition]
        weights[targets] = np.asarray(jax.nn.softmax(params.log_external_input_partition), dtype=float)
    else:
        weights[0] = 1.0
    return weights / weights.sum()


def environment_modifiers(model, forcing, params) -> np.ndarray:
    """Return daily multiplicative decomposition modifiers by pool."""
    temp = np.asarray(forcing.soil_temp, dtype=float)
    moisture = np.asarray(forcing.soil_moisture, dtype=float)
    air = np.asarray(forcing.air_temp, dtype=float)
    temp = np.where(np.isfinite(temp), temp, air[:, None])
    moisture = np.where(np.isfinite(moisture), moisture, 0.30)
    ptl = np.asarray(model._pool_to_layer, dtype=int)
    q10 = np.exp(np.asarray(params.log_Q10, dtype=float))
    theta_opt = np.exp(np.asarray(params.log_theta_opt, dtype=float))
    gamma = np.exp(np.asarray(params.log_gamma_moist, dtype=float))
    ft = q10[None, :] ** ((temp - 15.0) / 10.0)
    fm = np.exp(-gamma[None, :] * (moisture - theta_opt[None, :]) ** 2)
    thaw = 1.0 / (1.0 + np.exp(-10.0 * temp))
    return ft[:, ptl] * fm[:, ptl] * thaw[:, ptl]


def realised_transit_time_days(tau_days, log_f_transfer, modifiers, input_phase_weights, pool_input_weights):
    """Compatibility wrapper around the package-level realized metric."""
    return realized_mean_transit_time(
        np.log(np.asarray(tau_days, dtype=float)), log_f_transfer,
        modifiers, input_phase_weights, pool_input_weights,
    )


def gaussian_draws(rng: np.random.Generator, mean: np.ndarray, covariance: np.ndarray, n_draws: int) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    transform = vectors @ np.diag(np.sqrt(np.clip(values, 0.0, None)))
    return mean[None, :] + rng.standard_normal((n_draws, mean.size)) @ transform.T


def transit_metrics(model, forcing, params) -> tuple[float, float, float]:
    weights = input_weights(model, params)
    intrinsic_days, _ = intrinsic_mean_transit_time(params.log_tau, params.log_f_transfer, weights)
    modifiers = environment_modifiers(model, forcing, params)
    gpp = np.maximum(np.asarray(forcing.GPP_obs, dtype=float), 0.0)
    if not np.isfinite(gpp).all() or gpp.sum() <= 0:
        gpp = np.ones_like(gpp)
    realised_days, _ = realised_transit_time_days(
        np.exp(np.asarray(params.log_tau, dtype=float)), params.log_f_transfer,
        modifiers, gpp, weights,
    )
    return intrinsic_days / 365.25, realised_days / 365.25, float(np.average(modifiers @ weights, weights=gpp))


def annual_climatology(modifiers: np.ndarray, gpp: np.ndarray, time_days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a multi-year forcing record to a 365-day mean seasonal cycle."""
    dates = pd.Timestamp("1970-01-01") + pd.to_timedelta(time_days, unit="D")
    phase = np.minimum(dates.dayofyear.to_numpy() - 1, 364)
    counts = np.bincount(phase, minlength=365).astype(float)
    modifier_cycle = np.vstack([
        np.bincount(phase, weights=modifiers[:, pool], minlength=365) / counts
        for pool in range(modifiers.shape[1])
    ]).T
    gpp_cycle = np.bincount(phase, weights=gpp, minlength=365) / counts
    return modifier_cycle, gpp_cycle


def transit_metrics_climatology(model, forcing, params) -> tuple[float, float]:
    """Fast posterior-draw approximation using the mean daily seasonal cycle."""
    weights = input_weights(model, params)
    intrinsic_days, _ = intrinsic_mean_transit_time(params.log_tau, params.log_f_transfer, weights)
    modifiers = environment_modifiers(model, forcing, params)
    gpp = np.maximum(np.asarray(forcing.GPP_obs, dtype=float), 0.0)
    modifiers, gpp = annual_climatology(modifiers, gpp, np.asarray(forcing.time, dtype=float))
    realised_days, _ = realised_transit_time_days(
        np.exp(np.asarray(params.log_tau, dtype=float)), params.log_f_transfer,
        modifiers, gpp, weights,
    )
    return intrinsic_days / 365.25, realised_days / 365.25


def run_site(gradient: str, config_path: str, n_draws: int, seed: int) -> tuple[dict, pd.DataFrame]:
    spec = load_site_spec(config_path)
    result = run_site_canonical(spec, observation_path=spec.observation_path, include_er_constraint=True)
    model, params, forcing = result["model"], result["params_opt"], result["forcing"]
    intrinsic, realised, modifier = transit_metrics(model, forcing, params)
    rng = np.random.default_rng(seed)
    draws = gaussian_draws(
        rng, np.asarray(result["oe_result"].x_opt, dtype=float),
        np.asarray(result["oe_result"].Sx, dtype=float), n_draws,
    )
    draw_rows = []
    for draw_id, x in enumerate(draws):
        draw_params = vector_to_params(jnp.asarray(x, dtype=jnp.float32), params, ("log_tau", "log_f_transfer"))
        draw_intrinsic, draw_realised = transit_metrics_climatology(model, forcing, draw_params)
        draw_rows.append({"gradient": gradient, "site": spec.israd_name, "draw": draw_id,
                          "intrinsic_transit_time_yr": draw_intrinsic,
                          "realised_transit_time_yr": draw_realised})
    draw_df = pd.DataFrame(draw_rows)
    qs = draw_df[["intrinsic_transit_time_yr", "realised_transit_time_yr"]].quantile([0.025, 0.5, 0.975])
    moisture = np.asarray(forcing.soil_moisture, dtype=float)
    return {
        "gradient": gradient, "site": spec.israd_name, "label": spec.label,
        "forcing_kind": spec.forcing_kind, "n_forcing_days": len(forcing.GPP_obs),
        "intrinsic_transit_time_yr": intrinsic,
        "intrinsic_q025_yr": qs.loc[0.025, "intrinsic_transit_time_yr"],
        "intrinsic_q975_yr": qs.loc[0.975, "intrinsic_transit_time_yr"],
        "realised_transit_time_yr": realised,
        "realised_q025_yr": qs.loc[0.025, "realised_transit_time_yr"],
        "realised_q975_yr": qs.loc[0.975, "realised_transit_time_yr"],
        "realised_to_intrinsic": realised / intrinsic,
        "gpp_weighted_mean_modifier": modifier,
        "mean_soil_temp_c": float(np.mean(np.asarray(forcing.soil_temp, dtype=float))),
        "mean_soil_moisture": float(np.mean(moisture)),
        "soil_moisture_sd": float(np.std(moisture)),
        "forcing_provenance": "synthetic climate + FluxCom GPP" if spec.forcing_kind == "fluxcom" else "daily tower forcing (with loader fallbacks)",
        "converged": bool(result["converged"]),
    }, draw_df


def plot(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    y = np.arange(len(df))
    ax.hlines(y, df.intrinsic_transit_time_yr, df.realised_transit_time_yr, color="0.70", lw=2)
    ax.errorbar(df.intrinsic_transit_time_yr, y, xerr=[df.intrinsic_transit_time_yr - df.intrinsic_q025_yr, df.intrinsic_q975_yr - df.intrinsic_transit_time_yr], fmt="none", color="#355C7D", alpha=.55, capsize=3)
    ax.errorbar(df.realised_transit_time_yr, y, xerr=[df.realised_transit_time_yr - df.realised_q025_yr, df.realised_q975_yr - df.realised_transit_time_yr], fmt="none", color="#C06C2B", alpha=.55, capsize=3)
    ax.scatter(df.intrinsic_transit_time_yr, y, s=70, label="Intrinsic (95% posterior interval)", color="#355C7D", zorder=3)
    ax.scatter(df.realised_transit_time_yr, y, s=70, marker="s", label="Realised (95% posterior interval)", color="#C06C2B", zorder=3)
    ax.set(yticks=y, yticklabels=df.gradient, xlabel="Input-weighted mean transit time (years)", title="Environmental forcing changes transit time across a climate gradient")
    ax.grid(axis="x", alpha=.25)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(path, dpi=250, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "notebooks/exports/realized_transit_gradient_20260731.csv"))
    parser.add_argument("--figure", default=str(ROOT / "notebooks/paper_figs/outputs/current_results/figures/figure_12_realized_transit_gradient.png"))
    parser.add_argument("--draws-out", default=str(ROOT / "notebooks/exports/realized_transit_gradient_draws_20260731.csv"))
    parser.add_argument("--posterior-draws", type=int, default=100)
    args = parser.parse_args()
    payloads = [run_site(gradient, config, args.posterior_draws, 917 + i) for i, (gradient, config) in enumerate(SITES.items())]
    rows = [payload[0] for payload in payloads]
    draws = pd.concat([payload[1] for payload in payloads], ignore_index=True)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    draws.to_csv(args.draws_out, index=False)
    plot(df, Path(args.figure))
    print(df.to_string(index=False))
    print(f"Wrote {args.out} and {args.figure}")
    return 0
