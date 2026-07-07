"""Shared utilities for Optimal Estimation workflows and diagnostics."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .api import run_model
from .climate import f_temp as _f_temp, f_moisture as _f_moisture, thawed_frac as _ff
from .data.schemas import ObservationData
from .state import make_default_params
from ._oe_helpers import _analytical_c12_ss


def build_mean_ss_modifier(
    forcing,
    params0,
    t_nan_fill: float = 5.0,
    theta_nan_fill: float = 0.3,
) -> tuple[float, float]:
    """Compute the mean decomposition modifier and mean GPP over forcing."""
    air_t_np = np.nan_to_num(np.array(forcing.air_temp), nan=t_nan_fill)
    soil_t_raw = np.array(forcing.soil_temp[:, 0])
    soil_t_np = np.where(np.isnan(soil_t_raw), air_t_np, soil_t_raw)
    theta_raw = np.array(forcing.soil_moisture[:, 0])
    theta_np = np.where(np.isnan(theta_raw), theta_nan_fill, theta_raw)

    ft = _f_temp(jnp.array(soil_t_np, dtype=jnp.float32), params0.log_Q10[0], T_ref=15.0)
    fm = _f_moisture(
        jnp.array(theta_np, dtype=jnp.float32),
        params0.log_theta_opt[0],
        params0.log_gamma_moist[0],
    )
    ff = _ff(jnp.array(soil_t_np, dtype=jnp.float32))
    mean_modifier = float(jnp.nanmean(ft * fm * ff))
    mean_modifier = mean_modifier if (np.isfinite(mean_modifier) and mean_modifier > 0.05) else 0.05
    mean_gpp = float(jnp.nanmean(forcing.GPP_obs))
    return mean_modifier, mean_gpp


def ss_state_for_params(model, forcing, state0, params):
    """Replace ``state0.C12`` with the analytical steady-state C12 at ``params``."""
    n_pools = len(model.pool_index)
    cue = float(getattr(model.config.external_inputs, "CUE", 0.47))
    mean_mod, mean_gpp = build_mean_ss_modifier(forcing, params)
    mean_input = mean_gpp * cue
    target_names = list(model.config.external_inputs.partition.keys())
    target_idx = [model.pool_index[n] for n in target_names] or None
    c12_ss = _analytical_c12_ss(
        params, n_pools, mean_input, mean_mod, target_indices=target_idx
    )
    return state0._replace(C12=c12_ss)


def build_oe_observation_sets(forcing, delta14C_obs: dict, delta14C_resp, c_pools_obs: dict, er_sliced):
    """Build the five ObservationData variants used in OE1–OE5 workflows."""
    t_len = int(forcing.time.shape[0])
    nan_t = jnp.full(t_len, jnp.nan)

    obs_carbon_only = ObservationData(
        time=forcing.time, NEE=nan_t, GPP=nan_t, ER=nan_t, NEE_unc=nan_t,
        delta14C_obs={}, deltaD14C_obs={},
        C_pools_obs=c_pools_obs, delta14C_resp=None,
    )
    obs_carbon_pool14c = ObservationData(
        time=forcing.time, NEE=nan_t, GPP=nan_t, ER=nan_t, NEE_unc=nan_t,
        delta14C_obs=delta14C_obs, deltaD14C_obs={},
        C_pools_obs=c_pools_obs, delta14C_resp=None,
    )
    obs_carbon_resp14c = ObservationData(
        time=forcing.time, NEE=nan_t, GPP=nan_t, ER=nan_t, NEE_unc=nan_t,
        delta14C_obs={}, deltaD14C_obs={},
        C_pools_obs=c_pools_obs, delta14C_resp=delta14C_resp,
    )
    obs_all = ObservationData(
        time=forcing.time, NEE=nan_t, GPP=nan_t, ER=nan_t, NEE_unc=nan_t,
        delta14C_obs=delta14C_obs, deltaD14C_obs={},
        C_pools_obs=c_pools_obs, delta14C_resp=delta14C_resp,
    )
    obs_all_er = ObservationData(
        time=forcing.time, NEE=nan_t, GPP=nan_t, ER=er_sliced, NEE_unc=nan_t,
        delta14C_obs=delta14C_obs, deltaD14C_obs={},
        C_pools_obs=c_pools_obs, delta14C_resp=delta14C_resp,
    )
    return (
        obs_carbon_only,
        obs_carbon_pool14c,
        obs_carbon_resp14c,
        obs_all,
        obs_all_er,
    )


def run_forward_ss(model, forcing, state0, params):
    """Forward run using the analytical steady-state C12 initial condition."""
    state_ss = ss_state_for_params(model, forcing, state0, params)
    return run_model(model, forcing, state0=state_ss, params=params)


# Backward-compatible aliases for notebook callers.
mean_ss_modifier = build_mean_ss_modifier
