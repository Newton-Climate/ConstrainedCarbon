"""
Optimal Estimation inversion for the ecosystem-complexity model.

Implements ``optimize_oe``: Levenberg-Marquardt minimisation of the OE cost:

    J(x) = (y − F(x))ᵀ Sₑ⁻¹ (y − F(x)) + (x − xₐ)ᵀ Sₐ⁻¹ (x − xₐ)

The Jacobian K = ∂F/∂x is computed via ``jax.jacobian`` (fully differentiable).

Public names re-exported here:
  ObsBlock       — observation block type for extra_obs_blocks kwarg
  OEResult       — inversion result
  optimize_oe    — main entry point
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ._oe_helpers import (
    ObsBlock,
    _analytical_c12_ss,
    _build_obs_blocks,
    _build_sa_diag,
    apply_ss_c12,
)
from .api import run_model
from .data.schemas import ForcingData, ObservationData
from .model import EcosystemModel
from .optimizer import (
    get_oe_fields as _get_oe_fields,
)
from .optimizer import (
    params_to_vector as _params_to_vector,
)
from .optimizer import (
    vector_to_params as _vector_to_params,
)
from .state import (
    EcosystemState,
    ModelParams,
    make_default_params,
    make_initial_state,
)

# Re-export ObsBlock so users can do: from .optimal_estimation import ObsBlock
__all__ = ["ObsBlock", "OEResult", "optimize_oe"]


class OEResult(NamedTuple):
    """Result of an Optimal Estimation inversion via Levenberg-Marquardt."""

    params_opt: ModelParams
    x_opt: jnp.ndarray  # (n_state,) optimal state vector
    x_prior: jnp.ndarray  # (n_state,) prior state vector
    Sx: jnp.ndarray  # (n_state, n_state) posterior covariance
    averaging_kernel: jnp.ndarray  # (n_state, n_state)  A = Sₓ (KᵀSₑ⁻¹K)
    y_obs: jnp.ndarray  # (n_obs,) stacked observation vector
    y_prior: jnp.ndarray  # (n_obs,) prior model prediction
    y_opt: jnp.ndarray  # (n_obs,) posterior model prediction
    cost_history: jnp.ndarray  # (n_iter,) total OE cost per LM step
    converged: bool
    n_iter: int
    state_names: list[str]  # length n_state — labels for diagnostics


def optimize_oe(  # noqa: C901
    model: EcosystemModel,
    forcing: ForcingData,
    observations: ObservationData,
    state0: Optional[EcosystemState] = None,
    fields: Optional[tuple[str, ...]] = None,
    extra_obs_blocks: Optional[list[ObsBlock]] = None,
    sa_override_diag: Optional[jnp.ndarray] = None,
) -> OEResult:
    """
    Optimal Estimation inversion via Levenberg-Marquardt.

    Minimises the OE cost function:
        J(x) = (y − F(x))ᵀ Sₑ⁻¹ (y − F(x)) + (x − xₐ)ᵀ Sₐ⁻¹ (x − xₐ)

    Both Sₐ (prior error covariance) and Sₑ (observation error covariance)
    are diagonal.  The Jacobian K = ∂F/∂x is computed via ``jax.jacobian``.

    Default state vector is derived from the config via ``_get_oe_fields``:
      always:                log_tau, log_f_transfer
      optimize_partition:    log_external_input_partition
      optimize_f_hetero:     log_f_hetero

    Pass ``fields`` explicitly to override (e.g. to add log_f_hetero for OE5
    without setting optimize_f_hetero in the config).

    Parameters
    ----------
    extra_obs_blocks : list[ObsBlock] or None
        Additional ObsBlocks appended after the standard blocks built from
        ``observations``.  Each block must supply its own ``y``, ``Se``, and
        JAX-differentiable ``predict`` callable.  Use this to inject
        externally-derived observations (e.g. ISRaD fraction Δ¹⁴C means) that
        have per-observation σ values incompatible with the uniform σ used by
        the standard pool_14C block.

    Returns an OEResult that includes the posterior covariance Sₓ and the
    averaging kernel A = Sₓ (KᵀSₑ⁻¹K), which together quantify information
    content and posterior uncertainty for each state variable.
    """
    inv_cfg = getattr(model.config, "inversion_raw", {}) or {}
    n_iter = int(inv_cfg.get("oe_max_iterations", 20))
    sigma_pool = float(inv_cfg.get("sigma_pool_14C", 5.0))
    sigma_resp = float(inv_cfg.get("sigma_resp_14C", 10.0))
    sigma_carbon = float(inv_cfg.get("sigma_carbon_gCm2", 1000.0))
    lam0 = float(inv_cfg.get("lm_lambda0", 1e-3))
    lam_factor = float(inv_cfg.get("lm_lambda_factor", 10.0))
    eps = float(inv_cfg.get("oe_convergence_eps", 1e-4))

    params0 = make_default_params(model.config)
    if state0 is None:
        assert model._site_config is not None
        state0 = make_initial_state(model.config, model._site_config)

    # Use caller-supplied fields; fall back to the config-derived default.
    # Never use a hardcoded constant here — fixed parameters (e.g. partition
    # with optimize_partition=false) must NOT enter the state vector because
    # their prior value can be -inf (from log(0)) which makes prior_r = NaN.
    opt_fields = (
        tuple(fields) if fields is not None else _get_oe_fields(model.config, inv_cfg)
    )

    # ── Steady-state mean input (for analytical C12 initialisation) ───────────
    _ext_cfg = model.config.external_inputs
    _cue = float(getattr(_ext_cfg, "CUE", 0.47))
    _mean_gpp = float(jnp.nanmean(forcing.GPP_obs))
    _mean_input = _mean_gpp * _cue  # gC m⁻² day⁻¹ entering soil
    _n_pools = len(model.pool_index)

    # Target pool indices for the external input partition (2-way softmax case).
    _target_names = list(_ext_cfg.partition.keys()) if _ext_cfg is not None else []
    _ext_target_idx = [model.pool_index[n] for n in _target_names] or None

    # Pre-compute climatological mean decomposition modifier from forcing data.
    _p0 = params0
    _air_t_np = np.nan_to_num(np.array(forcing.air_temp), nan=5.0)
    _soil_t_raw = np.array(forcing.soil_temp[:, 0])
    _T_soil_np = np.where(np.isnan(_soil_t_raw), _air_t_np, _soil_t_raw)
    _theta_raw = np.array(forcing.soil_moisture[:, 0])
    _theta_np = np.where(np.isnan(_theta_raw), 0.3, _theta_raw)
    from .climate import f_moisture as _f_moisture
    from .climate import f_temp as _f_temp
    from .climate import thawed_frac as _ff

    _ft = _f_temp(jnp.array(_T_soil_np, dtype=jnp.float32), _p0.log_Q10[0], T_ref=15.0)
    _fm = _f_moisture(
        jnp.array(_theta_np, dtype=jnp.float32),
        _p0.log_theta_opt[0],
        _p0.log_gamma_moist[0],
    )
    _fff = _ff(jnp.array(_T_soil_np, dtype=jnp.float32))
    _mod = float(jnp.nanmean(_ft * _fm * _fff))
    _mean_modifier = _mod if np.isfinite(_mod) and _mod > 0.05 else 0.05
    print(  # noqa: T201
        f"  Spinup SS: mean_input={_mean_input:.4f} gC/m²/day, "
        f"mean_modifier={_mean_modifier:.4f}, "
        f"eff_tau_active={float(jnp.exp(_p0.log_tau[0]))/_mean_modifier/365:.1f} yr"
    )

    # ── State vector ──────────────────────────────────────────────────────────
    xa = _params_to_vector(params0, opt_fields)
    x = xa

    state_names = []
    for f in opt_fields:
        val = getattr(params0, f)
        if f == "log_f_transfer":
            for i in range(int(math.prod(val[:, :-1].shape))):
                state_names.append(f"{f}[{i}]")
        else:
            for i in range(int(math.prod(val.shape))):
                state_names.append(f"{f}[{i}]")

    # Guard: -inf or NaN in the prior vector will corrupt the LM step via
    # prior_r = xa − x = −∞ − −∞ = NaN on iteration 1, poisoning the whole g.
    _bad = ~jnp.isfinite(xa)
    if bool(jnp.any(_bad)):
        bad_names = [state_names[i] for i in np.where(np.array(_bad))[0]]
        raise ValueError(
            f"optimize_oe: prior state vector has non-finite values: {bad_names}.\n"
            f"  Likely cause: a partition fraction is 0.0 so log(0)=-inf entered "
            f"the state vector.  Fix: set optimize_partition=false in the config "
            f"(so the zero-fraction logit stays out of the state vector), or use "
            f"non-zero prior fractions in the partition dict."
        )

    Sa_diag = _build_sa_diag(model.config, params0, opt_fields)
    if sa_override_diag is not None:
        Sa_diag = jnp.array(sa_override_diag, dtype=jnp.float32)
    Sa_inv_diag = 1.0 / (Sa_diag + 1e-30)

    f_hetero = float(inv_cfg.get("f_hetero", 0.0))
    sigma_er_frac = float(inv_cfg.get("sigma_er_frac", 0.15))

    # ── Observation blocks ────────────────────────────────────────────────────
    obs_blocks = _build_obs_blocks(
        observations,
        model,
        sigma_pool,
        sigma_resp,
        sigma_carbon,
        f_hetero=f_hetero,
        sigma_er_frac=sigma_er_frac,
    )

    if extra_obs_blocks:
        obs_blocks = obs_blocks + list(extra_obs_blocks)

    if not obs_blocks:
        raise ValueError("optimize_oe: no observations found in ObservationData")

    y = jnp.concatenate([b.y for b in obs_blocks])
    Se_diag = jnp.concatenate([b.Se for b in obs_blocks])

    block_summary = "  +  ".join(f"{len(b.y)} {b.name}" for b in obs_blocks)
    print(f"  OE obs vector: {block_summary}  =  {int(y.shape[0])} total")  # noqa: T201

    Se_inv_diag = 1.0 / (Se_diag + 1e-30)

    # ── Forward function F(x) → (n_obs,) ─────────────────────────────────────
    def _forward(x_vec: jnp.ndarray) -> jnp.ndarray:
        p = _vector_to_params(x_vec, params0, opt_fields)

        # Replace C12 with analytical steady-state to eliminate spinup drift,
        # rescaling C14 so the initial Δ¹⁴C stays fixed as τ (hence c12_ss) moves.
        c12_ss = _analytical_c12_ss(
            p, _n_pools, _mean_input, _mean_modifier, target_indices=_ext_target_idx
        )
        state_ss = apply_ss_c12(state0, c12_ss)
        out = run_model(model, forcing, state0=state_ss, params=p)

        return jnp.concatenate([b.predict(out, p) for b in obs_blocks])

    _jac_fn = jax.jacobian(_forward)

    y_prior = _forward(xa)

    # ── Levenberg-Marquardt loop ──────────────────────────────────────────────
    lam = lam0
    cost_hist: list[float] = []
    converged = False

    for _ in range(n_iter):
        F_x = _forward(x)
        K = _jac_fn(x)  # (n_obs, n_x)

        resid = y - F_x  # (n_obs,)
        prior_r = xa - x  # (n_x,)

        KtSe = K.T * Se_inv_diag  # (n_x, n_obs)
        KtSeK = KtSe @ K  # (n_x, n_x)
        KtSe_r = KtSe @ resid  # (n_x,)

        cost = float(
            jnp.sum(Se_inv_diag * resid**2) + jnp.sum(Sa_inv_diag * prior_r**2)
        )
        cost_hist.append(cost)

        H = KtSeK + jnp.diag(Sa_inv_diag) + lam * jnp.eye(int(xa.shape[0]))
        g = KtSe_r + Sa_inv_diag * prior_r
        dx = jnp.linalg.solve(H, g)

        x_new = x + dx
        F_new = _forward(x_new)
        r_new = y - F_new
        pr_new = xa - x_new
        cost_new = float(
            jnp.sum(Se_inv_diag * r_new**2) + jnp.sum(Sa_inv_diag * pr_new**2)
        )

        if cost_new < cost:
            x = x_new
            lam = max(float(lam) / lam_factor, 1e-10)
        else:
            lam = min(float(lam) * lam_factor, 1e10)

        if float(jnp.max(jnp.abs(dx))) < eps:
            converged = True
            break

    # ── Posterior covariance and averaging kernel ─────────────────────────────
    K_f = _jac_fn(x)
    KtSeK_f = (K_f.T * Se_inv_diag) @ K_f
    H_f = KtSeK_f + jnp.diag(Sa_inv_diag)
    Sx = jnp.linalg.inv(H_f)
    A = Sx @ KtSeK_f

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
