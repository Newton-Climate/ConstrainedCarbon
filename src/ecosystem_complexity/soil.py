"""
Soil carbon dynamics for the ecosystem-complexity model.

Covers heterotrophic decomposition fluxes and the bulk ¹²C Euler step.
All functions are pure JAX — safe for jit, lax.scan, and grad.

Functions
---------
decomp_flux         — per-pool decomposition flux
het_respiration     — total heterotrophic respiration
nee                 — net ecosystem exchange
_step_12C_pure      — one Euler step of ¹²C pool dynamics (full system)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.nn
import jax.numpy as jnp

from ecosystem_complexity.above_ground import (
    _gpp,
    compute_external_soil_inputs,
    npp_allocation,
)
from ecosystem_complexity.climate import _pool_env_vecs
from ecosystem_complexity.transfer import get_transfer_matrix

if TYPE_CHECKING:
    from ecosystem_complexity.state import EcosystemState, ModelParams


def decomp_flux(
    C12_i: jnp.ndarray,
    log_tau_i: jnp.ndarray,
    ft: jnp.ndarray,
    fm: jnp.ndarray,
    ff: jnp.ndarray,
) -> jnp.ndarray:
    """
    Decomposition flux from a single carbon pool (gC m⁻² day⁻¹).

    .. math::

        F_{decomp,i} = \\frac{C_{12,i}}{\\tau_i}
                       \\cdot f_{temp} \\cdot f_{moist} \\cdot \\text{thawed\\_frac}

    Parameters
    ----------
    C12_i :
        Pool size (gC m⁻²).  Scalar.
    log_tau_i :
        Log turnover time for pool *i* (log-days, from ``ModelParams``).
    ft :
        Temperature scalar for the layer containing pool *i*
        (output of ``f_temp``).
    fm :
        Moisture scalar for the layer containing pool *i*
        (output of ``f_moisture``).
    ff :
        Thaw fraction for the layer containing pool *i*
        (output of ``thawed_frac``).

    Returns
    -------
    jnp.ndarray
        Non-negative scalar flux in gC m⁻² day⁻¹.
    """
    tau_i = jnp.exp(log_tau_i)
    return (C12_i / tau_i) * ft * fm * ff


def het_respiration(
    C12: jnp.ndarray,
    log_tau: jnp.ndarray,
    log_f_transfer: jnp.ndarray,
    ft_vec: jnp.ndarray,
    fm_vec: jnp.ndarray,
    ff_vec: jnp.ndarray,
    n_pools: int,
) -> jnp.ndarray:
    """
    Total heterotrophic respiration (gC m⁻² day⁻¹).

    Each pool contributes ``(1 - Σ_j f_{ij}) * F_{decomp,i}`` to Rh,
    where ``Σ_j f_{ij}`` is the fraction of pool-i outflux allocated to
    other pools; the remainder is respired.

    .. math::

        F_{rh} = \\sum_i \\Bigl(1 - \\sum_j f_{ij}\\Bigr) \\cdot F_{decomp,i}

    Parameters
    ----------
    C12 :
        All pool sizes, shape ``(n_pools,)``, gC m⁻².
    log_tau :
        Log turnover times, shape ``(n_pools,)``, from ``ModelParams``.
    log_f_transfer :
        Transfer logits, shape ``(n_pools, n_pools + 1)``,
        from ``ModelParams``.
    ft_vec :
        Temperature scalars, shape ``(n_pools,)`` — one per pool
        (broadcast from per-layer values by the caller).
    fm_vec :
        Moisture scalars, shape ``(n_pools,)``.
    ff_vec :
        Thaw fractions, shape ``(n_pools,)``.
    n_pools :
        Total number of pools (static int, not a traced value).

    Returns
    -------
    jnp.ndarray
        Non-negative scalar total Rh in gC m⁻² day⁻¹.
    """
    # Transfer fractions: softmax over last axis, drop respiration column
    f_full = jax.nn.softmax(log_f_transfer, axis=-1)  # (n_pools, n_pools+1)
    f_transfer = f_full[:, :n_pools]  # (n_pools, n_pools)

    # Fraction of each pool's outflux that is respired
    resp_frac = 1.0 - f_transfer.sum(axis=-1)  # (n_pools,)

    # Per-pool decomposition fluxes (vectorised)
    tau = jnp.exp(log_tau)  # (n_pools,)
    f_decomp = (C12 / tau) * ft_vec * fm_vec * ff_vec  # (n_pools,)

    return jnp.sum(resp_frac * f_decomp)


def nee(
    GPP: jnp.ndarray,
    Ra: jnp.ndarray,
    F_rh: jnp.ndarray,
) -> jnp.ndarray:
    """
    Net ecosystem exchange (gC m⁻² day⁻¹).

    Sign convention: **positive = net source to atmosphere**.

    .. math::

        NEE = F_{rh} + R_a - GPP

    Parameters
    ----------
    GPP :
        Gross primary production (gC m⁻² day⁻¹, positive).
    Ra :
        Autotrophic respiration (gC m⁻² day⁻¹, positive).
    F_rh :
        Heterotrophic respiration (gC m⁻² day⁻¹, positive).

    Returns
    -------
    jnp.ndarray
        Scalar NEE.  Negative when ecosystem is a carbon sink.
    """
    return F_rh + Ra - GPP


def _step_12C_pure(
    state: EcosystemState,
    params: ModelParams,
    forcing_t: dict[str, jnp.ndarray],
    *,
    n_pools: int,
    n_ag_pools: int,
    pool_to_layer: jnp.ndarray,
    dt: float,
    # External-inputs static arguments (resolved at JIT trace time)
    external_inputs_active: bool = False,
    external_input_source_key: str = "GPP_obs",
    external_input_is_npp: bool = False,
    external_input_target_indices: jnp.ndarray | None = None,
    # Freeze-thaw gating at depth (optional; None = use layer-level thawed_frac)
    pool_mid_depths: jnp.ndarray | None = None,
    T_annual_mean: float | None = None,
    damping_depth_m: float = 2.0,
) -> EcosystemState:
    """
    One Euler step of ¹²C pool dynamics — pure function, no ``self``.

    Implements:

    .. math::

        \\frac{dC_{12,i}}{dt} =
            \\underbrace{\\sum_j F_{ji}\\,F_{\\text{decomp},j}}_{\\text{transfers in}}
            + F_{\\text{NPP},i}
            + F_{\\text{ext},i}
            - F_{\\text{decomp},i}

    where ``F[j, i]`` is the fraction of pool-j outflux routed to pool i
    (from ``get_transfer_matrix``), ``F_{decomp,i} = C_i / τ_i · ft · fm · ff``,
    and ``F_{ext,i}`` is the optional external soil carbon input term (zero when
    ``external_inputs_active=False``).

    Parameters
    ----------
    state :
        Current ecosystem state (only ``state.C12`` and ``state.thawed_frac``
        are read; all other fields are passed through unchanged).
    params :
        Model parameters (traced by ``jax.grad``).
    forcing_t :
        Dict with at least ``'sw_radiation'``, ``'soil_temp'``,
        ``'soil_moisture'`` for this timestep.
    n_pools, n_ag_pools :
        Static pool counts — not traced.
    pool_to_layer :
        Static pool→layer index array — not traced.
    dt :
        Timestep in days (Python float, not traced).
    external_inputs_active :
        Static Python bool — when True, prescribed GPP/NPP forcing is used
        instead of the internal LUE model.
    external_input_source_key :
        Key in ``forcing_t`` containing prescribed GPP or NPP values.
    external_input_is_npp :
        Static bool — True when the source field is already NPP (skip CUE).
    external_input_target_indices :
        Integer indices of soil pools that receive direct carbon input.

    Returns
    -------
    EcosystemState
        New state with ``C12`` updated; all other fields unchanged.
    """
    # ── GPP — use prescribed value or internal LUE model ─────────────────
    CUE = jnp.exp(params.log_CUE)
    if external_inputs_active:
        GPP_prescribed = forcing_t[external_input_source_key]
        # Fall back to LUE estimate when the prescribed value is NaN
        GPP = jnp.where(
            jnp.isnan(GPP_prescribed), _gpp(forcing_t["sw_radiation"]), GPP_prescribed
        )
        # Fraction of NPP that bypasses AG pools and enters soil directly
        soil_frac = jax.nn.sigmoid(params.log_soil_input_fraction)
        ag_frac = 1.0 - soil_frac
    else:
        GPP = _gpp(forcing_t["sw_radiation"])
        ag_frac = jnp.asarray(1.0)

    # ── Environmental scalars (per pool) ─────────────────────────────────
    ft_vec, fm_vec, ff_vec = _pool_env_vecs(
        forcing_t["soil_temp"],
        forcing_t["soil_moisture"],
        state.thawed_frac,
        params.log_Q10,
        params.log_theta_opt,
        params.log_gamma_moist,
        pool_to_layer,
        pool_mid_depths=pool_mid_depths,
        T_annual_mean=T_annual_mean,
        damping_depth_m=damping_depth_m,
    )

    # ── NPP allocation to aboveground pools ───────────────────────────────
    F_npp = npp_allocation(GPP, CUE, params.log_alloc, n_ag_pools) * ag_frac
    npp_inputs = jnp.zeros(n_pools).at[:n_ag_pools].set(F_npp)

    # ── External soil inputs (non-zero when external_inputs_active=True) ──
    if external_inputs_active:
        # When external inputs are active, target indices are always resolved.
        assert external_input_target_indices is not None
        ext_inputs = compute_external_soil_inputs(
            GPP_or_NPP=GPP if not external_input_is_npp else GPP * CUE,
            is_npp=external_input_is_npp,
            log_CUE=params.log_CUE,
            log_soil_input_fraction=params.log_soil_input_fraction,
            log_external_input_partition=params.log_external_input_partition,
            n_pools=n_pools,
            target_pool_indices=external_input_target_indices,
        )
    else:
        ext_inputs = jnp.zeros(n_pools)

    # ── Transfer matrix ───────────────────────────────────────────────────
    F_mat = get_transfer_matrix(params.log_f_transfer, n_pools)  # (n, n)

    # ── Per-pool decomposition fluxes ─────────────────────────────────────
    tau = jnp.exp(params.log_tau)  # (n_pools,)
    f_decomp = (state.C12 / tau) * ft_vec * fm_vec * ff_vec  # (n_pools,)

    # ── Carbon influx from pool-to-pool transfers ─────────────────────────
    # F_mat[i, j] = fraction of pool-i outflux going to pool j
    # influx[j]   = Σ_i  F_mat[i, j] · f_decomp[i]  =  F_mat.T @ f_decomp
    influx = F_mat.T @ f_decomp  # (n_pools,)

    # ── Euler update ──────────────────────────────────────────────────────
    dC12 = (influx + npp_inputs + ext_inputs - f_decomp) * dt
    C12_new = state.C12 + dC12

    return state._replace(C12=C12_new)
