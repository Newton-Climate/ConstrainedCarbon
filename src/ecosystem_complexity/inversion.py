"""
Gradient-based parameter inversion for the ecosystem-complexity model.

Implements ``optimize``: Adam (or L-BFGS) minimisation of a weighted loss
combining NEE/GPP/ER flux residuals, pool Δ¹⁴C residuals, respired CO₂
Δ¹⁴C residuals, and a carbon-stock soft constraint.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .api import run_model
from .tracer_14C import respired_delta14C
from .data.schemas import ForcingData, ObservationData
from .model import EcosystemModel
from .optimizer import (
    get_opt_fields as _get_opt_fields,
)
from .optimizer import (
    params_to_vector as _params_to_vector,
)
from .optimizer import (
    vector_to_params as _vector_to_params,
)
from .state import EcosystemState, ModelParams, make_default_params, make_initial_state


class OptimizationResult(NamedTuple):
    params_opt: ModelParams
    loss_history: jnp.ndarray  # (n_iter,)
    loss_flux_history: jnp.ndarray  # (n_iter,)
    loss_14C_history: jnp.ndarray  # (n_iter,)
    loss_resp_history: jnp.ndarray  # (n_iter,) — respired CO₂ Δ¹⁴C loss
    loss_carbon_history: jnp.ndarray  # (n_iter,) — carbon stock constraint loss
    tau_history: jnp.ndarray  # (n_iter, n_pools)
    converged: bool
    n_iter: int


def optimize(  # noqa: C901
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
        assert model._site_config is not None
        state0 = make_initial_state(model.config, model._site_config)

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

    def _loss_and_components(
        vec: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
        p = _vector_to_params(vec, params0, opt_fields)
        out = run_model(model, forcing, state0=state0, params=p)

        # Flux loss (MSE over valid observations).
        # Use the "double-where" pattern: replace obs NaN with sim so the
        # squared difference is 0 at masked timesteps in BOTH forward and
        # backward passes (avoids NaN gradients from jnp.where branch eval).
        def _mse(sim: jnp.ndarray, obs: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
            obs_safe = jnp.where(mask, obs, sim)
            diff = sim - obs_safe
            return jnp.where(
                jnp.any(mask),
                jnp.mean(jnp.where(mask, diff**2, 0.0)),
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
                l_14C = l_14C + jnp.mean(jnp.where(valid, diff**2, 0.0))
                n_14C_terms += 1
        if n_14C_terms > 0:
            l_14C = l_14C / n_14C_terms

        # Respired CO₂ Δ¹⁴C loss — flux-weighted mean across all pools.
        # Weights are the model's own per-pool respiration fluxes,
        #   resp_frac_i · (C12_i / τ_i) · ft · fm · ff,
        # taken straight from `Rh_by_pool` rather than recomputed here. Using
        # C12_i/τ_i alone (as this did previously) drops resp_frac and the
        # environmental scalars; resp_frac varies per pool, so it does not
        # cancel in the normalisation and biases the mixture toward pools that
        # transfer most of their outflux onward instead of respiring it.
        # Double-where pattern applied for NaN-safe gradients.
        l_resp = jnp.zeros(())
        if w_resp > 0.0 and observations.delta14C_resp is not None:
            d14C_resp_sim = respired_delta14C(
                out.delta14C, out.Rh_by_pool, out.C12,
                p.log_tau, p.log_f_transfer, out.C12.shape[-1],
            )  # (T,)

            obs_resp = jnp.array(observations.delta14C_resp)
            valid_resp = ~jnp.isnan(obs_resp)
            obs_resp_safe = jnp.where(valid_resp, obs_resp, d14C_resp_sim)
            diff_resp = d14C_resp_sim - obs_resp_safe
            l_resp = jnp.where(
                jnp.any(valid_resp),
                jnp.mean(jnp.where(valid_resp, diff_resp**2, 0.0)),
                0.0,
            )

        # Carbon stock loss — soft constraint on time-mean C12 per pool.
        # C_pools_obs: {pool_name: (mean_gC_m2, sigma_gC_m2)}
        # Uses the mean modelled C12 over the full simulation window.
        l_carbon = jnp.zeros(())
        n_carbon_terms = 0
        for pool_name, (c_obs_mean, c_obs_sigma) in (
            observations.C_pools_obs or {}
        ).items():
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
                grads,
                opt_state,
                vec,
                value=loss_val,
                grad=grads,
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
