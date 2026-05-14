"""Public API for the ecosystem-complexity carbon model."""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from .config import load_config, ModelConfig, PoolIndex
from .state import EcosystemState, ModelParams, make_initial_state, make_default_params
from .model import EcosystemModel
from .fluxes import thawed_frac as compute_thawed_frac
from .data.schemas import ForcingData, ObservationData


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


class OptimizationResult(NamedTuple):
    params_opt: ModelParams
    loss_history: jnp.ndarray        # (n_iter,)
    loss_flux_history: jnp.ndarray   # (n_iter,)
    loss_14C_history: jnp.ndarray    # (n_iter,)
    tau_history: jnp.ndarray         # (n_iter, n_pools)
    converged: bool
    n_iter: int


# Core fields always entered into the optimisation vector.
# lambda_14C is a fixed physical constant (never optimised).
# The three external_inputs fields are conditionally added by _get_opt_fields().
_CORE_OPTIMIZED_FIELDS = (
    "log_tau",
    "log_f_transfer",
    "log_alloc",
    "log_Q10",
    "log_theta_opt",
    "log_gamma_moist",
    "alpha_priming",
)


def _get_opt_fields(config: ModelConfig) -> tuple[str, ...]:
    """Return the list of ModelParams fields to include in the opt vector."""
    fields = list(_CORE_OPTIMIZED_FIELDS)
    ext = config.external_inputs
    if ext is not None and ext.enabled:
        if ext.optimize_CUE:
            fields.append("log_CUE")
        if ext.optimize_soil_input_fraction:
            fields.append("log_soil_input_fraction")
        if ext.optimize_partition:
            fields.append("log_external_input_partition")
    return tuple(fields)


# ── Internal helpers ──────────────────────────────────────────────────────────

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


def _params_to_vector(
    params: ModelParams, opt_fields: tuple[str, ...]
) -> jnp.ndarray:
    """Flatten optimised parameter fields into a 1-D vector."""
    parts = []
    for f in opt_fields:
        val = getattr(params, f)
        parts.append(jnp.ravel(val))
    return jnp.concatenate(parts)


def _vector_to_params(
    vec: jnp.ndarray, template: ModelParams, opt_fields: tuple[str, ...]
) -> ModelParams:
    """Unpack a 1-D optimisation vector back into ModelParams."""
    updates = {}
    offset = 0
    for f in opt_fields:
        val = getattr(template, f)
        size = int(math.prod(val.shape))
        updates[f] = vec[offset:offset + size].reshape(val.shape)
        offset += size
    return template._replace(**updates)


# ── Public API ────────────────────────────────────────────────────────────────

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

        lambda_14C = p.lambda_14C
        C12 = state.C12
        C14 = state.C14
        # Avoid division by zero for empty pools.
        fm = jnp.where(C12 > 0, C14 / (C12 * lambda_14C + 1e-30), 0.0)
        delta14C = (fm - 1.0) * 1000.0

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


def optimize(
    model: EcosystemModel,
    forcing: ForcingData,
    observations: ObservationData,
    state0: Optional[EcosystemState] = None,
) -> OptimizationResult:
    """Optimise model parameters against flux and 14C observations.

    Uses the inversion settings from the model config (``config.inversion_raw``
    or falls back to sensible defaults).
    """
    # ── Hyper-parameters ──────────────────────────────────────────────────────
    inv_cfg = getattr(model.config, "inversion_raw", {}) or {}
    optimizer_name = inv_cfg.get("optimizer", "adam")
    lr = float(inv_cfg.get("learning_rate", 1e-3))
    n_iter = int(inv_cfg.get("max_iterations", 500))
    w_flux = float(inv_cfg.get("weight_flux", 1.0))
    w_14C = float(inv_cfg.get("weight_14C", 1.0))

    params0 = make_default_params(model.config)
    if state0 is None:
        state0 = make_initial_state(
            model.config,
            model._site_config)  # type: ignore[attr-defined]

    opt_fields = _get_opt_fields(model.config)
    vec0 = _params_to_vector(params0, opt_fields)

    # Pre-build valid masks for flux observations.
    obs_NEE = jnp.array(observations.NEE)
    obs_GPP = jnp.array(observations.GPP)
    obs_ER = jnp.array(observations.ER)
    valid_NEE = ~jnp.isnan(obs_NEE)
    valid_GPP = ~jnp.isnan(obs_GPP)
    valid_ER = ~jnp.isnan(obs_ER)

    def _loss_and_components(vec: jnp.ndarray):
        p = _vector_to_params(vec, params0, opt_fields)
        out = run_model(model, forcing, state0=state0, params=p)

        # Flux loss (MSE over valid observations).
        def _mse(sim, obs, mask):
            return jnp.where(
                jnp.any(mask),
                jnp.mean(jnp.where(mask, (sim - obs) ** 2, 0.0)),
                0.0,
            )

        l_flux = (
            _mse(out.NEE, obs_NEE, valid_NEE)
            + _mse(out.GPP, obs_GPP, valid_GPP)
            + _mse(out.ER, obs_ER, valid_ER)
        ) / 3.0

        # 14C loss — mean over all pool/time pairs with observations.
        l_14C = jnp.zeros(())
        n_14C_terms = 0
        for pool_name, delta14C_obs_arr in observations.delta14C_obs.items():
            if pool_name not in model.pool_index:
                continue
            idx = model.pool_index.index(pool_name)
            obs_arr = jnp.array(delta14C_obs_arr)
            sim_arr = out.delta14C[:, idx]
            valid = ~jnp.isnan(obs_arr)
            if jnp.any(valid):
                l_14C = l_14C + jnp.mean(
                    jnp.where(valid, (sim_arr - obs_arr) ** 2, 0.0))
                n_14C_terms += 1
        if n_14C_terms > 0:
            l_14C = l_14C / n_14C_terms

        loss = w_flux * l_flux + w_14C * l_14C
        return loss, (l_flux, l_14C, jnp.exp(p.log_tau))

    grad_fn = jax.value_and_grad(_loss_and_components, has_aux=True)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    if optimizer_name.lower() == "lbfgs":
        try:
            tx = optax.lbfgs()
        except AttributeError:
            tx = optax.adam(lr)
    else:
        tx = optax.adam(lr)

    opt_state = tx.init(vec0)
    vec = vec0

    loss_hist = []
    loss_flux_hist = []
    loss_14C_hist = []
    tau_hist = []
    converged = False

    for i in range(n_iter):
        (loss_val, (l_flux, l_14C, taus)), grads = grad_fn(vec)
        updates, opt_state = tx.update(grads, opt_state, vec,
                                       value=loss_val, grad=grads,
                                       value_fn=lambda v: _loss_and_components(v)[0])
        vec = optax.apply_updates(vec, updates)

        loss_hist.append(float(loss_val))
        loss_flux_hist.append(float(l_flux))
        loss_14C_hist.append(float(l_14C))
        tau_hist.append(np.array(taus))

        if i > 10:
            recent = loss_hist[-10:]
            rel = abs(recent[0] - recent[-1]) / (abs(recent[0]) + 1e-10)
            if rel < 1e-5:
                converged = True
                break

    params_opt = _vector_to_params(vec, params0, opt_fields)

    return OptimizationResult(
        params_opt=params_opt,
        loss_history=jnp.array(loss_hist),
        loss_flux_history=jnp.array(loss_flux_hist),
        loss_14C_history=jnp.array(loss_14C_hist),
        tau_history=jnp.array(tau_hist),
        converged=converged,
        n_iter=len(loss_hist),
    )
