"""The model's own steady-state SOC prior.

This runs the forward model to steady state under the prior parameters, so it
is an inversion concern rather than a data one. The *measured* stock
constraints it is weighed against — ISRaD and SoilGrids — live in
:mod:`ecosystem_complexity.data.soc_stocks`.
"""
from __future__ import annotations

import logging

import numpy as np

from ecosystem_complexity.model.api import run_model
from ecosystem_complexity.data.schemas import ForcingData
from ecosystem_complexity.inference.utilities import ss_state_for_params
from ecosystem_complexity.sites.forcing import build_annual_mean_forcing
from ecosystem_complexity.model.state import make_default_params, make_initial_state

logger = logging.getLogger(__name__)

# Matches the σ=0.5 OE prior on log_tau. build_soc_prior sets obs = C(prior params)
# and C = I·τ/modifier at steady state with I fixed by the GPP forcing, so anything
# tighter restates the τ prior at MORE confidence than the prior itself — prior
# double-counting dressed as an observation. At 0.50 the fallback is a no-op.
_SOC_PRIOR_SIGMA_FRAC = 0.50
_SS_TOL = 1e-5
_SS_MAX_YEARS = 2000


def build_soc_prior(model, forcing: ForcingData) -> tuple:
    """Build a site-specific steady-state SOC prior from annual-mean forcing."""
    params_prior = make_default_params(model.config)
    inversion = getattr(model.config, "inversion_raw", {}) or {}
    sigma_fraction = float(inversion.get("sigma_soc_fraction", _SOC_PRIOR_SIGMA_FRAC))
    forcing_mean = build_annual_mean_forcing(forcing)
    base = make_initial_state(model.config, {})
    state = ss_state_for_params(model, forcing_mean, base, params_prior)

    prev_total = None
    n_years = 0
    for n_years in range(1, _SS_MAX_YEARS + 1):
        out = run_model(model, forcing_mean, state0=state, params=params_prior)
        state = out.final_state
        total = float(np.sum(np.array(state.C12, dtype=float)))
        if prev_total is not None:
            rel = abs(total - prev_total) / (abs(prev_total) + 1e-10)
            if rel < _SS_TOL:
                break
        prev_total = total

    c12 = np.array(state.C12, dtype=float)
    c_pools_obs = {
        name: (
            float(c12[i]),
            float(sigma_fraction * c12[i]),
        )
        for i, name in enumerate(model.pool_index.pool_names)
        if float(c12[i]) > 0.0
    }
    total = float(c12.sum())
    c_total_obs = (total, float(sigma_fraction * total)) if total > 0.0 else None
    return state, c_pools_obs, n_years, c_total_obs
