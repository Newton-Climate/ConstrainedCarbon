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
from .tracer_14C import compute_delta14C
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
    loss_resp_history: jnp.ndarray   # (n_iter,) — respired CO₂ Δ¹⁴C loss
    loss_carbon_history: jnp.ndarray # (n_iter,) — carbon stock constraint loss
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


def optimize(
    model: EcosystemModel,
    forcing: ForcingData,
    observations: ObservationData,
    state0: Optional[EcosystemState] = None,
    fields: Optional[tuple[str, ...]] = None,
) -> OptimizationResult:
    """Optimise model parameters against flux and 14C observations.

    Uses the inversion settings from the model config (``config.inversion_raw``
    or falls back to sensible defaults).

    Parameters
    ----------
    fields:
        Whitelist of ``ModelParams`` field names to include in the optimisation
        vector.  If ``None`` (default), all fields returned by
        ``_get_opt_fields(config)`` are used.  Pass an explicit tuple to reduce
        the problem to a tractable subset, e.g.
        ``fields=("log_tau", "log_external_input_partition")``.
    """
    # ── Hyper-parameters ──────────────────────────────────────────────────────
    inv_cfg = getattr(model.config, "inversion_raw", {}) or {}
    optimizer_name = inv_cfg.get("optimizer", "adam")
    lr = float(inv_cfg.get("learning_rate", 1e-3))
    n_iter = int(inv_cfg.get("max_iterations", 500))
    w_flux = float(inv_cfg.get("weight_flux", 1.0))
    w_14C = float(inv_cfg.get("weight_14C", 1.0))
    w_resp = float(inv_cfg.get("weight_resp_14C", 0.0))
    w_carbon = float(inv_cfg.get("weight_carbon", 0.0))
    grad_clip = float(inv_cfg.get("grad_clip", 1.0))

    params0 = make_default_params(model.config)
    if state0 is None:
        state0 = make_initial_state(
            model.config,
            model._site_config)  # type: ignore[attr-defined]

    # Allow caller to restrict which fields enter the optimisation vector.
    if fields is not None:
        opt_fields = tuple(fields)
    else:
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
        # Use the "double-where" pattern: replace obs NaN with sim so the
        # squared difference is 0 at masked timesteps in BOTH forward and
        # backward passes (avoids NaN gradients from jnp.where branch eval).
        def _mse(sim, obs, mask):
            obs_safe = jnp.where(mask, obs, sim)
            diff = sim - obs_safe
            return jnp.where(
                jnp.any(mask),
                jnp.mean(jnp.where(mask, diff ** 2, 0.0)),
                0.0,
            )

        l_flux = (
            _mse(out.NEE, obs_NEE, valid_NEE)
            + _mse(out.GPP, obs_GPP, valid_GPP)
            + _mse(out.ER, obs_ER, valid_ER)
        ) / 3.0

        # 14C loss — mean over all pool/time pairs with observations.
        # Double-where pattern applied here too to prevent NaN gradient
        # propagation through the unmasked (NaN obs) branch.
        _pool_names_set = set(model.pool_index.pool_names)
        l_14C = jnp.zeros(())
        n_14C_terms = 0
        for pool_name, delta14C_obs_arr in observations.delta14C_obs.items():
            if pool_name not in _pool_names_set:
                continue
            idx = model.pool_index[pool_name]
            obs_arr = jnp.array(delta14C_obs_arr)
            sim_arr = out.delta14C[:, idx]
            valid = ~jnp.isnan(obs_arr)
            if jnp.any(valid):
                obs_safe = jnp.where(valid, obs_arr, sim_arr)  # NaN → sim (diff=0)
                diff = sim_arr - obs_safe
                l_14C = l_14C + jnp.mean(
                    jnp.where(valid, diff ** 2, 0.0))
                n_14C_terms += 1
        if n_14C_terms > 0:
            l_14C = l_14C / n_14C_terms

        # Respired CO₂ Δ¹⁴C loss — flux-weighted mean across all pools.
        # Respiration flux from pool i ∝ C12_i / τ_i, giving the weighting.
        # Double-where pattern applied for NaN-safe gradients.
        l_resp = jnp.zeros(())
        if w_resp > 0.0 and observations.delta14C_resp is not None:
            tau_vals = jnp.exp(p.log_tau)                          # (n_pools,)
            weights = out.C12 / (tau_vals[None, :] + 1e-30)       # (T, n_pools)
            w_sum = weights.sum(-1, keepdims=False) + 1e-30        # (T,)
            d14C_resp_sim = (out.delta14C * weights).sum(-1) / w_sum  # (T,)

            obs_resp = jnp.array(observations.delta14C_resp)
            valid_resp = ~jnp.isnan(obs_resp)
            obs_resp_safe = jnp.where(valid_resp, obs_resp, d14C_resp_sim)
            diff_resp = d14C_resp_sim - obs_resp_safe
            l_resp = jnp.where(
                jnp.any(valid_resp),
                jnp.mean(jnp.where(valid_resp, diff_resp ** 2, 0.0)),
                0.0,
            )

        # Carbon stock loss — soft constraint on time-mean C12 per pool.
        # C_pools_obs: {pool_name: (mean_gC_m2, sigma_gC_m2)}
        # Uses the mean modelled C12 over the full simulation window.
        l_carbon = jnp.zeros(())
        n_carbon_terms = 0
        for pool_name, (c_obs_mean, c_obs_sigma) in (observations.C_pools_obs or {}).items():
            if pool_name not in set(model.pool_index.pool_names):
                continue
            idx = model.pool_index[pool_name]
            c_sim_mean = jnp.mean(out.C12[:, idx])
            sigma = float(c_obs_sigma) + 1.0  # avoid divide-by-zero
            l_carbon = l_carbon + ((c_sim_mean - float(c_obs_mean)) / sigma) ** 2
            n_carbon_terms += 1
        if n_carbon_terms > 0:
            l_carbon = l_carbon / n_carbon_terms

        loss = w_flux * l_flux + w_14C * l_14C + w_resp * l_resp + w_carbon * l_carbon
        return loss, (l_flux, l_14C, l_resp, l_carbon, jnp.exp(p.log_tau))

    grad_fn = jax.value_and_grad(_loss_and_components, has_aux=True)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    _use_lbfgs = False
    if optimizer_name.lower() == "lbfgs":
        try:
            tx = optax.lbfgs()
            _use_lbfgs = True
        except AttributeError:
            tx = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    else:
        tx = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))

    opt_state = tx.init(vec0)
    vec = vec0

    loss_hist = []
    loss_flux_hist = []
    loss_14C_hist = []
    loss_resp_hist = []
    loss_carbon_hist = []
    tau_hist = []
    converged = False

    best_vec = vec0
    best_loss = float("inf")

    for i in range(n_iter):
        (loss_val, (l_flux, l_14C, l_resp, l_carbon, taus)), grads = grad_fn(vec)

        # Guard against NaN/Inf divergence — stop and revert to best seen so far.
        loss_float = float(loss_val)
        if not math.isfinite(loss_float):
            vec = best_vec
            break

        if loss_float < best_loss:
            best_loss = loss_float
            best_vec = vec

        if _use_lbfgs:
            updates, opt_state = tx.update(
                grads, opt_state, vec,
                value=loss_val, grad=grads,
                value_fn=lambda v: _loss_and_components(v)[0],
            )
        else:
            updates, opt_state = tx.update(grads, opt_state, vec)
        vec = optax.apply_updates(vec, updates)

        loss_hist.append(loss_float)
        loss_flux_hist.append(float(l_flux))
        loss_14C_hist.append(float(l_14C))
        loss_resp_hist.append(float(l_resp))
        loss_carbon_hist.append(float(l_carbon))
        tau_hist.append(np.array(taus))

        if i > 10:
            recent = loss_hist[-10:]
            rel = abs(recent[0] - recent[-1]) / (abs(recent[0]) + 1e-10)
            if rel < 1e-5:
                converged = True
                break

    # Use the best-seen parameter vector (guards against overshoot at end).
    vec = best_vec

    params_opt = _vector_to_params(vec, params0, opt_fields)

    return OptimizationResult(
        params_opt=params_opt,
        loss_history=jnp.array(loss_hist),
        loss_flux_history=jnp.array(loss_flux_hist),
        loss_14C_history=jnp.array(loss_14C_hist),
        loss_resp_history=jnp.array(loss_resp_hist),
        loss_carbon_history=jnp.array(loss_carbon_hist),
        tau_history=jnp.array(tau_hist),
        converged=converged,
        n_iter=len(loss_hist),
    )


# ── Optimal Estimation ────────────────────────────────────────────────────────

class OEResult(NamedTuple):
    """Result of an Optimal Estimation inversion via Levenberg-Marquardt."""
    params_opt: ModelParams
    x_opt: jnp.ndarray              # (n_state,) optimal state vector
    x_prior: jnp.ndarray            # (n_state,) prior state vector
    Sx: jnp.ndarray                 # (n_state, n_state) posterior covariance
    averaging_kernel: jnp.ndarray   # (n_state, n_state)  A = Sₓ (KᵀSₑ⁻¹K)
    y_obs: jnp.ndarray              # (n_obs,) stacked observation vector
    y_prior: jnp.ndarray            # (n_obs,) prior model prediction
    y_opt: jnp.ndarray              # (n_obs,) posterior model prediction
    cost_history: jnp.ndarray       # (n_iter,) total OE cost per LM step
    converged: bool
    n_iter: int
    state_names: list               # length n_state — labels for diagnostics


# Default fields for OE: add log_f_transfer to the Adam-10 set.
_OE_DEFAULT_FIELDS = ("log_tau", "log_external_input_partition", "log_f_transfer")


def _build_oe_obs_vector(
    observations: ObservationData,
    model,
    sigma_pool: float,
    sigma_resp: float,
    sigma_carbon: Optional[float] = None,
    f_hetero: float = 0.0,
    sigma_er_frac: float = 0.15,
):
    """
    Flatten ObservationData into a dense (n_obs,) observation vector for OE.

    Observation blocks (in order):
      1. Pool-level Δ¹⁴C   — one entry per (time, pool) with finite obs
      2. Respired CO₂ Δ¹⁴C — one entry per time-step with finite obs
      3. Carbon stocks      — one entry per pool in C_pools_obs
                              σ taken from the (mean, sigma) tuple in
                              C_pools_obs; ``sigma_carbon`` is used as a
                              fallback when the tuple sigma is zero/None.
      4. Annual Rh flux     — annual mean Rh estimated from FluxNet ER:
                              Rh_obs = ER_obs × f_hetero  [gC m⁻² day⁻¹]
                              σ = Rh_obs × sigma_er_frac
                              Only included when f_hetero > 0 and
                              observations.ER contains finite values.
                              W_er (n_er_obs × T) is a precomputed
                              aggregation weight matrix for use in _forward.

    Returns
    -------
    y              : (n_obs,) float32  — observed values
    Se_diag        : (n_obs,) float32  — observation error variances σ²
    t_pool_idx     : (n_pool_obs,) int32
    pool_col_idx   : (n_pool_obs,) int32
    t_resp_idx     : (n_resp_obs,) int32
    carbon_col_idx : (n_carbon_obs,) int32 — pool indices for C-stock obs
    W_er           : (n_er_obs, T) float32 — annual aggregation weights
    n_pool_obs, n_resp_obs, n_carbon_obs, n_er_obs : int
    """
    pool_names_set = set(model.pool_index.pool_names)

    # ── Block 1: pool Δ¹⁴C ───────────────────────────────────────────────────
    t_p, col_p, y_p = [], [], []
    for pool_name in sorted(observations.delta14C_obs.keys()):
        if pool_name not in pool_names_set:
            continue
        obs_arr = np.array(observations.delta14C_obs[pool_name])
        valid = np.where(np.isfinite(obs_arr))[0]
        pcol = model.pool_index[pool_name]
        for t in valid:
            t_p.append(int(t)); col_p.append(pcol); y_p.append(float(obs_arr[t]))

    n_pool_obs   = len(t_p)
    t_pool_idx   = jnp.array(t_p,   dtype=jnp.int32)
    pool_col_idx = jnp.array(col_p, dtype=jnp.int32)
    y_pool       = jnp.array(y_p,   dtype=jnp.float32)

    # ── Block 2: respired Δ¹⁴C ───────────────────────────────────────────────
    t_r, y_r = [], []
    if observations.delta14C_resp is not None:
        resp_np = np.array(observations.delta14C_resp)
        for t in np.where(np.isfinite(resp_np))[0]:
            t_r.append(int(t)); y_r.append(float(resp_np[t]))

    n_resp_obs = len(t_r)
    t_resp_idx = jnp.array(t_r, dtype=jnp.int32)
    y_resp     = jnp.array(y_r, dtype=jnp.float32)

    # ── Block 3: carbon stocks ────────────────────────────────────────────────
    c_col, y_c, se_c = [], [], []
    for pool_name, (c_mean, c_sigma) in (observations.C_pools_obs or {}).items():
        if pool_name not in pool_names_set:
            continue
        sigma_c = float(c_sigma) if (c_sigma and c_sigma > 0) else (sigma_carbon or 1000.0)
        c_col.append(model.pool_index[pool_name])
        y_c.append(float(c_mean))
        se_c.append(sigma_c ** 2)

    n_carbon_obs   = len(c_col)
    carbon_col_idx = jnp.array(c_col, dtype=jnp.int32)
    y_carbon       = jnp.array(y_c,  dtype=jnp.float32)
    Se_carbon      = jnp.array(se_c, dtype=jnp.float32)

    # ── Block 4: annual Rh from FluxNet ER ───────────────────────────────────
    T = len(np.array(observations.time))
    W_er_np   = np.zeros((0, T), dtype=np.float32)
    y_er_list: list[float] = []
    se_er_list: list[float] = []

    if f_hetero > 0.0 and observations.ER is not None:
        er_np    = np.array(observations.ER, dtype=np.float64)
        time_np  = np.array(observations.time, dtype=np.float64)
        years_np = 1970.0 + time_np / 365.25

        yr_start = int(np.floor(years_np[0]))
        yr_end   = int(np.floor(years_np[-1]))
        rows: list[np.ndarray] = []

        for yr in range(yr_start, yr_end + 1):
            mask = (years_np >= yr) & (years_np < yr + 1) & np.isfinite(er_np)
            if mask.sum() < 30:          # require ≥30 days of valid ER data
                continue
            rh_est = float(np.mean(er_np[mask])) * f_hetero
            sigma  = max(abs(rh_est) * sigma_er_frac, 0.01)
            row    = np.zeros(T, dtype=np.float32)
            row[mask] = 1.0 / mask.sum()
            rows.append(row)
            y_er_list.append(rh_est)
            se_er_list.append(sigma ** 2)

        if rows:
            W_er_np = np.stack(rows, axis=0)   # (n_er_obs, T)

    n_er_obs = len(y_er_list)
    W_er     = jnp.array(W_er_np)
    y_er     = jnp.array(y_er_list,  dtype=jnp.float32)
    Se_er    = jnp.array(se_er_list, dtype=jnp.float32)

    # ── Assemble full obs vector and Se ──────────────────────────────────────
    y_parts  = [y_pool]
    se_parts = [jnp.full(n_pool_obs, sigma_pool ** 2)]

    if n_resp_obs > 0:
        y_parts.append(y_resp)
        se_parts.append(jnp.full(n_resp_obs, sigma_resp ** 2))

    if n_carbon_obs > 0:
        y_parts.append(y_carbon)
        se_parts.append(Se_carbon)

    if n_er_obs > 0:
        y_parts.append(y_er)
        se_parts.append(Se_er)

    y       = jnp.concatenate(y_parts)  if y_parts  else jnp.zeros(0)
    Se_diag = jnp.concatenate(se_parts) if se_parts else jnp.zeros(0)

    return (y, Se_diag,
            t_pool_idx, pool_col_idx, t_resp_idx,
            carbon_col_idx, W_er,
            n_pool_obs, n_resp_obs, n_carbon_obs, n_er_obs)


def _build_sa_diag(
    config: ModelConfig,
    params0: ModelParams,
    opt_fields: tuple,
) -> jnp.ndarray:
    """
    Build the diagonal of Sₐ (prior error variances) for the OE state vector.

    log_tau[i]             : σ = tau_prior_std[i] / tau_prior_days[i]  (log-space)
    log_f_transfer[i,j]    : σ = 0.5 for real transfer rules, 0.02 for structural zeros
    log_external_input_partition : σ = 1.0  (broad logit prior)
    everything else        : σ = 0.5
    """
    pool_idx = PoolIndex(config)
    n_pools  = len(pool_idx)

    # Pool name → (tau_prior_days, tau_prior_std)
    tau_info: dict[str, tuple[float, float]] = {}
    for layer in config.soil_layers:
        for pool in layer.som_pools:
            pname = f"{layer.name}_{pool.name}"
            tau_info[pname] = (pool.tau_prior_days, pool.tau_prior_std)

    # (src_i, dst_j) pairs from YAML transfer rules
    real_transfer_pairs = set()
    for src_name, dst_name, _ in config.transfer_rules:
        real_transfer_pairs.add((pool_idx[src_name], pool_idx[dst_name]))

    sa_parts = []
    for f in opt_fields:
        val = getattr(params0, f)
        n   = int(math.prod(val.shape))

        if f == "log_tau":
            sigma = np.array([
                tau_info.get(name, (1000.0, 1000.0))[1]
                / max(tau_info.get(name, (1000.0, 1000.0))[0], 1.0)
                for name in pool_idx.pool_names
            ], dtype=np.float32)
            sa_parts.append(jnp.array(sigma ** 2))

        elif f == "log_f_transfer":
            sigma = np.full(n, 0.02, dtype=np.float32)
            for (si, dj) in real_transfer_pairs:
                flat_i = si * (n_pools + 1) + dj
                sigma[flat_i] = 0.5
            sa_parts.append(jnp.array(sigma ** 2))

        elif f == "log_external_input_partition":
            sa_parts.append(jnp.full(n, 1.0 ** 2))

        else:
            sa_parts.append(jnp.full(n, 0.5 ** 2))

    return jnp.concatenate(sa_parts)


def optimize_oe(
    model,
    forcing: ForcingData,
    observations: ObservationData,
    state0: Optional[EcosystemState] = None,
    fields: Optional[tuple] = None,
) -> OEResult:
    """
    Optimal Estimation inversion via Levenberg-Marquardt.

    Minimises the OE cost function:
        J(x) = (y − F(x))ᵀ Sₑ⁻¹ (y − F(x)) + (x − xₐ)ᵀ Sₐ⁻¹ (x − xₐ)

    Both Sₐ (prior error covariance) and Sₑ (observation error covariance)
    are diagonal.  The Jacobian K = ∂F/∂x is computed via ``jax.jacobian``.

    Default state vector: log_tau (6) + log_external_input_partition (4)
    + log_f_transfer (6 × 7 = 42, but prior keeps structural zeros fixed).
    Total: 52 state variables vs ~50 observations.

    Returns an OEResult that includes the posterior covariance Sₓ and the
    averaging kernel A = Sₓ (KᵀSₑ⁻¹K), which together quantify information
    content and posterior uncertainty for each state variable.
    """
    inv_cfg      = getattr(model.config, "inversion_raw", {}) or {}
    n_iter       = int(inv_cfg.get("oe_max_iterations", 20))
    sigma_pool   = float(inv_cfg.get("sigma_pool_14C", 5.0))
    sigma_resp   = float(inv_cfg.get("sigma_resp_14C", 10.0))
    sigma_carbon = float(inv_cfg.get("sigma_carbon_gCm2", 1000.0))
    lam0         = float(inv_cfg.get("lm_lambda0", 1e-3))
    lam_factor   = float(inv_cfg.get("lm_lambda_factor", 10.0))
    eps          = float(inv_cfg.get("oe_convergence_eps", 1e-4))

    params0 = make_default_params(model.config)
    if state0 is None:
        state0 = make_initial_state(model.config, model._site_config)

    opt_fields = tuple(fields) if fields is not None else _OE_DEFAULT_FIELDS

    # ── State vector ──────────────────────────────────────────────────────────
    xa = _params_to_vector(params0, opt_fields)
    x  = xa

    Sa_diag     = _build_sa_diag(model.config, params0, opt_fields)
    Sa_inv_diag = 1.0 / (Sa_diag + 1e-30)

    state_names = []
    for f in opt_fields:
        val = getattr(params0, f)
        for i in range(int(math.prod(val.shape))):
            state_names.append(f"{f}[{i}]")

    f_hetero     = float(inv_cfg.get("f_hetero",      0.0))
    sigma_er_frac = float(inv_cfg.get("sigma_er_frac", 0.15))

    # ── Observation vector ────────────────────────────────────────────────────
    (y, Se_diag,
     t_pool_idx, pool_col_idx, t_resp_idx,
     carbon_col_idx, W_er,
     n_pool_obs, n_resp_obs, n_carbon_obs, n_er_obs) = _build_oe_obs_vector(
        observations, model, sigma_pool, sigma_resp, sigma_carbon,
        f_hetero=f_hetero, sigma_er_frac=sigma_er_frac)

    Se_inv_diag = 1.0 / (Se_diag + 1e-30)
    n_obs = int(y.shape[0])

    if n_obs == 0:
        raise ValueError("optimize_oe: no observations found in ObservationData")

    print(f"  OE obs vector: {n_pool_obs} pool Δ¹⁴C  +  {n_resp_obs} resp Δ¹⁴C"
          f"  +  {n_carbon_obs} C-stock  +  {n_er_obs} annual ER  =  {n_obs} total")

    # ── Forward function F(x) → (n_obs,) ─────────────────────────────────────
    def _forward(x_vec):
        p   = _vector_to_params(x_vec, params0, opt_fields)
        out = run_model(model, forcing, state0=state0, params=p)

        parts = [out.delta14C[t_pool_idx, pool_col_idx]]

        if n_resp_obs > 0:
            tau_v     = jnp.exp(p.log_tau)
            w         = out.C12 / (tau_v[None, :] + 1e-30)
            d14c_resp = (out.delta14C * w).sum(-1) / (w.sum(-1) + 1e-30)
            parts.append(d14c_resp[t_resp_idx])

        if n_carbon_obs > 0:
            # Predicted obs: time-mean C12 for each constrained pool
            c12_mean = jnp.mean(out.C12, axis=0)          # (n_pools,)
            parts.append(c12_mean[carbon_col_idx])

        if n_er_obs > 0:
            # Predicted Rh: sum(C12_i / tau_i) for all pools [gC m⁻² day⁻¹]
            tau_v  = jnp.exp(p.log_tau)
            rh_daily = jnp.sum(out.C12 / (tau_v[None, :] + 1e-30), axis=-1)  # (T,)
            rh_annual = W_er @ rh_daily   # (n_er_obs,)
            parts.append(rh_annual)

        return jnp.concatenate(parts)

    _jac_fn = jax.jacobian(_forward)

    y_prior = _forward(xa)

    # ── Levenberg-Marquardt loop ──────────────────────────────────────────────
    lam = lam0
    cost_hist: list[float] = []
    converged = False

    for _ in range(n_iter):
        F_x = _forward(x)
        K   = _jac_fn(x)                           # (n_obs, n_x)

        resid      = y - F_x                        # (n_obs,)
        prior_r    = xa - x                         # (n_x,)

        KtSe       = K.T * Se_inv_diag              # (n_x, n_obs)
        KtSeK      = KtSe @ K                       # (n_x, n_x)
        KtSe_r     = KtSe @ resid                   # (n_x,)

        cost = float(
            jnp.sum(Se_inv_diag * resid ** 2)
            + jnp.sum(Sa_inv_diag * prior_r ** 2)
        )
        cost_hist.append(cost)

        H  = KtSeK + jnp.diag(Sa_inv_diag) + lam * jnp.eye(int(xa.shape[0]))
        g  = KtSe_r + Sa_inv_diag * prior_r
        dx = jnp.linalg.solve(H, g)

        x_new  = x + dx
        F_new  = _forward(x_new)
        r_new  = y - F_new
        pr_new = xa - x_new
        cost_new = float(
            jnp.sum(Se_inv_diag * r_new ** 2)
            + jnp.sum(Sa_inv_diag * pr_new ** 2)
        )

        if cost_new < cost:
            x   = x_new
            lam = max(float(lam) / lam_factor, 1e-10)
        else:
            lam = min(float(lam) * lam_factor, 1e10)

        if float(jnp.max(jnp.abs(dx))) < eps:
            converged = True
            break

    # ── Posterior covariance and averaging kernel ─────────────────────────────
    K_f    = _jac_fn(x)
    KtSeK_f = (K_f.T * Se_inv_diag) @ K_f
    H_f    = KtSeK_f + jnp.diag(Sa_inv_diag)
    Sx     = jnp.linalg.inv(H_f)
    A      = Sx @ KtSeK_f

    return OEResult(
        params_opt=_vector_to_params(x, params0, opt_fields),
        x_opt=x,
        x_prior=xa,
        Sx=Sx,
        averaging_kernel=A,
        y_obs=y,
        y_prior=y_prior,
        y_opt=_forward(x),
        cost_history=jnp.array(cost_hist),
        converged=converged,
        n_iter=len(cost_hist),
        state_names=state_names,
    )
