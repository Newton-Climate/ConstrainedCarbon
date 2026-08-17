"""Standardized warming-response utilities for fitted site inversions."""

from __future__ import annotations

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ecosystem_complexity.model.api import run_model
from ecosystem_complexity.data.schemas import ForcingData
from ecosystem_complexity.inference.utilities import ss_state_for_params


def repeat_forcing(forcing: ForcingData, horizon_years: float) -> ForcingData:
    """Repeat a site's forcing record until it spans ``horizon_years``."""
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


def warm_forcing(forcing: ForcingData, delta_c: float) -> ForcingData:
    """Apply a uniform warming perturbation to air and soil temperature."""
    soil_temp = np.array(forcing.soil_temp, dtype=np.float32)
    soil_temp = np.where(np.isnan(soil_temp), soil_temp, soil_temp + delta_c)
    air_temp = np.array(forcing.air_temp, dtype=np.float32) + delta_c
    return forcing._replace(
        air_temp=jnp.array(air_temp, dtype=forcing.air_temp.dtype),
        soil_temp=jnp.array(soil_temp, dtype=forcing.soil_temp.dtype),
    )


def compute_pool_rh(model, forcing: ForcingData, params, output) -> np.ndarray:
    """Return per-pool daily heterotrophic respiration (T, n_pools)."""
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
        t_depth = (
            float(model._T_annual_mean)
            + (t_surface - float(model._T_annual_mean)) * amp
        )
        ff_vec = 1.0 / (1.0 + np.exp(-10.0 * t_depth))
    else:
        ff_layers = 1.0 / (1.0 + np.exp(-10.0 * soil_temp))
        ff_vec = ff_layers[:, pool_to_layer]

    f_full = _softmax_last_axis(np.array(params.log_f_transfer, dtype=np.float64))
    resp_frac = f_full[:, -1]
    c12 = np.array(output.C12, dtype=np.float64)
    return c12 * (ft_vec * fm_vec * ff_vec) * (resp_frac[None, :] / tau[None, :])


def project_warming_response(
    model,
    forcing: ForcingData,
    state0,
    params,
    *,
    horizon_years: float = 100.0,
    warming_delta_c: float = 4.0,
    old_pools: Sequence[str] = ("soil_slow", "soil_passive"),
) -> dict[str, float]:
    """Project one fitted site under repeated baseline and warmed forcing."""
    forcing_proj = repeat_forcing(forcing, horizon_years)
    forcing_warm = warm_forcing(forcing_proj, warming_delta_c)
    state_init = ss_state_for_params(model, forcing_proj, state0, params)
    out_base = run_model(model, forcing_proj, state0=state_init, params=params)
    out_warm = run_model(model, forcing_warm, state0=state_init, params=params)
    jax.block_until_ready(out_warm.C12)

    c0 = float(np.sum(np.array(state_init.C12, dtype=np.float64)))
    c_base_final = float(np.sum(np.array(out_base.C12[-1], dtype=np.float64)))
    c_warm_final = float(np.sum(np.array(out_warm.C12[-1], dtype=np.float64)))
    abs_c_loss = c_base_final - c_warm_final
    frac_c_loss = abs_c_loss / c0 if c0 > 0.0 else float("nan")

    rh_base = compute_pool_rh(model, forcing_proj, params, out_base)
    rh_warm = compute_pool_rh(model, forcing_warm, params, out_warm)
    old_idx = [
        model.pool_index[pool_name]
        for pool_name in old_pools
        if pool_name in model.pool_index.pool_names
    ]
    delta_total = float(np.sum(rh_warm.sum(axis=1) - rh_base.sum(axis=1)))
    delta_old = float(
        np.sum(rh_warm[:, old_idx].sum(axis=1) - rh_base[:, old_idx].sum(axis=1))
    )
    old_fraction = delta_old / delta_total if abs(delta_total) > 1e-12 else float("nan")
    mean_rh_base = float(np.nanmean(np.array(out_base.Rh, dtype=np.float64)) * 365.25)
    mean_rh_warm = float(np.nanmean(np.array(out_warm.Rh, dtype=np.float64)) * 365.25)

    return {
        "horizon_years": float(horizon_years),
        "warming_delta_c": float(warming_delta_c),
        "c_initial_gCm2": c0,
        "c_base_final_gCm2": c_base_final,
        "c_warm_final_gCm2": c_warm_final,
        "abs_c_loss_gCm2": abs_c_loss,
        "frac_c_loss": frac_c_loss,
        "mean_rh_base_gCm2yr": mean_rh_base,
        "mean_rh_warm_gCm2yr": mean_rh_warm,
        "delta_rh_total_gCm2": delta_total,
        "delta_rh_annual_mean_gCm2yr": delta_total / max(horizon_years, 1e-9),
        "old_fraction_of_excess_rh": old_fraction,
    }


def _softmax_last_axis(x: np.ndarray) -> np.ndarray:
    x_shift = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
