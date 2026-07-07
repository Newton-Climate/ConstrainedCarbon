"""Public runtime API for the ecosystem-complexity carbon model."""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from .config import load_config, ModelConfig, PoolIndex
from .state import EcosystemState, ModelParams, make_initial_state, make_default_params
from .model import EcosystemModel
from .climate import thawed_frac as compute_thawed_frac
from .tracer_14C import compute_delta14C
from .data.schemas import ForcingData, ObservationData
from .optimizer import (
    get_opt_fields as _get_opt_fields,
    get_oe_fields as _get_oe_fields,
    params_to_vector as _params_to_vector,
    vector_to_params as _vector_to_params,
)


# ── Output containers ─────────────────────────────────────────────────────────

class ModelOutput(NamedTuple):
    C12: jnp.ndarray        # (T, n_pools)
    C14: jnp.ndarray        # (T, n_pools)
    delta14C: jnp.ndarray   # (T, n_pools)
    NEE: jnp.ndarray        # (T,)
    GPP: jnp.ndarray        # (T,)
    ER: jnp.ndarray         # (T,)
    Rh: jnp.ndarray         # (T,)
    Ra: jnp.ndarray         # (T,)
    final_state: EcosystemState


def _build_forcing_dict(forcing: ForcingData) -> dict:
    """
    Convert ForcingData to a plain dict, NaN-filling optional fields.

    NaN values that would propagate through JAX computations are replaced
    with safe fallbacks before any model step sees them.
    """
    # ── Primary met variables: NaN → safe scalar defaults ─────────────────
    sw_rad = jnp.nan_to_num(forcing.sw_radiation, nan=0.0)
    air_t  = jnp.nan_to_num(forcing.air_temp,     nan=5.0)

    # ── Soil temperature: NaN → air temperature (already filled above) ────
    soil_temp = jnp.where(
        jnp.isnan(forcing.soil_temp), air_t[:, None], forcing.soil_temp
    )

    # ── Soil moisture: NaN → 0.3 m³ m⁻³ ──────────────────────────────────
    soil_moisture = jnp.where(
        jnp.isnan(forcing.soil_moisture),
        jnp.full_like(forcing.soil_moisture, 0.3),
        forcing.soil_moisture,
    )

    # ── Atmospheric ¹⁴C: NaN → 0.0 ‰ ─────────────────────────────────────
    delta14C_atm = jnp.where(
        jnp.isnan(forcing.delta14C_atm),
        jnp.zeros_like(forcing.delta14C_atm),
        forcing.delta14C_atm,
    )

    return dict(
        time=forcing.time,
        air_temp=air_t,
        sw_radiation=sw_rad,
        precip=forcing.precip,
        vpd=forcing.vpd,
        soil_temp=soil_temp,
        soil_moisture=soil_moisture,
        snow_depth=forcing.snow_depth,
        active_layer=forcing.active_layer,
        delta14C_atm=delta14C_atm,
        # External-inputs forcing fields (NaN = not available; model handles)
        GPP_obs=forcing.GPP_obs,
        NPP_obs=forcing.NPP_obs,
    )

def build_model(config_path: str) -> EcosystemModel:
    """Load a YAML config and return a ready-to-use EcosystemModel.

    The raw site configuration dict is attached as ``model._site_config`` so
    that ``spinup`` can call ``make_initial_state`` with the correct arguments.
    """
    config = load_config(config_path)
    pool_index = PoolIndex(config)
    params = make_default_params(config)
    model = EcosystemModel(config, params, pool_index)

    with open(config_path) as fh:
        raw_yaml = yaml.safe_load(fh)
    model._site_config = raw_yaml  # type: ignore[attr-defined]

    return model


def run_model(
    model: EcosystemModel,
    forcing: ForcingData,
    state0: Optional[EcosystemState] = None,
    params: Optional[ModelParams] = None,
) -> ModelOutput:
    """Run a forward simulation over the full forcing timeseries.

    Parameters
    ----------
    model:
        Built by :func:`build_model`.
    forcing:
        Site forcing (daily resolution).
    state0:
        Initial state; if ``None`` a default is constructed from the model config.
    params:
        Model parameters; if ``None`` the model's default parameters are used.
    """
    if params is None:
        params = make_default_params(model.config)
    if state0 is None:
        state0 = make_initial_state(
            model.config, model._site_config)  # type: ignore[attr-defined]

    forcing_dict = _build_forcing_dict(forcing)

    def _scan_body(carry, t):
        state, p = carry
        ft = jax.tree_util.tree_map(lambda x: x[t], forcing_dict)
        # Re-derive thawed_frac from the current forcing soil temperature.
        # Without this, permafrost-eligible layers stay locked at their
        # initialisation value (0 = frozen) regardless of what the forcing says.
        state = state._replace(thawed_frac=compute_thawed_frac(ft["soil_temp"]))
        state = model.step_12C(state, p, ft)
        state = model.step_14C(state, p, ft)

        C12 = state.C12
        C14 = state.C14
        delta14C = compute_delta14C(C14, C12)

        diag = model.diagnose(state, p, ft)
        return (state, p), (C12, C14, delta14C,
                            diag["NEE"], diag["GPP"], diag["ER"],
                            diag["Rh"], diag["Ra"])

    T = forcing.time.shape[0]
    (final_state, _), (C12, C14, delta14C, NEE, GPP, ER, Rh, Ra) = jax.lax.scan(
        _scan_body, (state0, params), jnp.arange(T)
    )

    return ModelOutput(
        C12=C12, C14=C14, delta14C=delta14C,
        NEE=NEE, GPP=GPP, ER=ER, Rh=Rh, Ra=Ra,
        final_state=final_state,
    )


def spinup(
    model: EcosystemModel,
    forcing: ForcingData,
    n_years: Optional[int] = None,
    convergence_tol: float = 1e-4,
    permafrost_14C_init: Optional[dict] = None,
) -> EcosystemState:
    """Spin up the model to a quasi-steady carbon state.

    Phase 1 — repeat annual forcing cycles until 12C pools converge.
    Phase 2 — run 14C spin-up over the pre-industrial atmospheric record.
    Phase 3 (optional) — initialise permafrost-layer 14C from observations.
    """
    params = make_default_params(model.config)
    state = make_initial_state(
        model.config, model._site_config)  # type: ignore[attr-defined]

    # Build annual-mean forcing for a single representative year.
    # Use the first full calendar year present in the forcing record.
    time_np = np.array(forcing.time)  # days since 1970-01-01
    years = (time_np / 365.25 + 1970).astype(int)
    unique_years = np.unique(years)

    # Prefer a full year; fall back to whatever is available.
    annual_mask = years == unique_years[len(unique_years) // 2]
    annual_forcing = jax.tree_util.tree_map(
        lambda x: x[annual_mask], forcing)

    max_years = n_years if n_years is not None else 2000
    prev_C12 = None
    for yr in range(max_years):
        out = run_model(model, annual_forcing, state0=state, params=params)
        state = out.final_state
        C12_total = float(jnp.sum(state.C12))
        if prev_C12 is not None:
            rel_change = abs(C12_total - prev_C12) / (abs(prev_C12) + 1e-10)
            if rel_change < convergence_tol:
                break
        prev_C12 = C12_total

    # Phase 3: optionally overwrite permafrost layer 14C from observations.
    if permafrost_14C_init is not None:
        for pool_name, delta14C_obs in permafrost_14C_init.items():
            idx = model.pool_index.index(pool_name)
            fm = delta14C_obs / 1000.0 + 1.0
            C14_new = state.C14.at[idx].set(fm * state.C12[idx] * params.lambda_14C)
            state = state._replace(C14=C14_new)

    return state


# ── Re-exports for backward compatibility with notebooks / site modules ──────
from .inversion import optimize, OptimizationResult  # noqa: E402
from .optimal_estimation import optimize_oe, OEResult  # noqa: E402
from ._oe_helpers import ObsBlock, _analytical_c12_ss  # noqa: E402
