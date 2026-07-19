"""
¹⁴C tracer dynamics for the ecosystem-complexity carbon model.

Implements TECHSPEC Section 4.4:

    step_14C                  — one Euler step of ¹⁴C pool dynamics (pure JAX)
    compute_delta14C          — Δ¹⁴C (‰) from raw C14/C12 pool masses
    spinup_14C                — advance C14 through the historical atm record
    initialize_permafrost_14C — override C14 for permafrost pools from obs

Design constraints
------------------
* **step_14C is pure JAX** — no Python loops over pools.
  All pool-level operations use matrix/vector arithmetic so the function is
  safe for ``jax.jit``, ``jax.lax.scan``, and ``jax.grad``.
* ``config`` and ``pool_index`` are treated as *static* (Python objects, not
  traced values).  ``_build_pool_to_layer`` is called once per JAX trace and
  its result is embedded as a constant.
* No circular imports: model.py imports this module; this module imports only
  config, state, fluxes, and transfer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from ecosystem_complexity.above_ground import (
    compute_external_soil_inputs,
    npp_allocation,
)
from ecosystem_complexity.climate import f_moisture, f_temp
from ecosystem_complexity.config import ModelConfig, PoolIndex
# soil.py does not import this module, so this stays acyclic.
from ecosystem_complexity.soil import respiration_fractions
from ecosystem_complexity.state import EcosystemState, ModelParams
from ecosystem_complexity.transfer import get_transfer_matrix

if TYPE_CHECKING:
    from ecosystem_complexity.model import EcosystemModel

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_R_STD: float = 1.176e-12  # IAEA modern standard ¹⁴C/¹²C ratio

# GPP placeholder constants — mirror model.py so the two step functions are
# always in sync.  When the full LUE model is replaced these should move to
# a shared constants module.
_LUE: float = 0.002  # light-use efficiency (gC MJ⁻¹)
_K_EXT: float = 0.5  # Beer–Lambert extinction coefficient
_LAI: float = 5.0  # leaf area index (m² m⁻²)
_CUE: float = 0.5  # carbon use efficiency (NPP / GPP)


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _build_pool_to_layer(
    config: ModelConfig,
    pool_index: PoolIndex,
) -> jnp.ndarray:
    """
    Build a ``(n_pools,)`` integer index mapping each pool to its soil layer.

    Aboveground pools are mapped to layer 0 (surface conditions), consistent
    with ``model._pool_to_layer``.  This function runs pure Python / NumPy;
    its result is a constant JAX array embedded once per trace.
    """
    n_pools = len(pool_index)
    ptl = np.zeros(n_pools, dtype=np.int32)
    for layer_idx, layer in enumerate(config.soil_layers):
        sl = pool_index.layer_slices[layer.name]
        ptl[sl.start : sl.stop] = layer_idx
    return jnp.array(ptl)


# ---------------------------------------------------------------------------
# step_14C
# ---------------------------------------------------------------------------


def step_14C(
    state: EcosystemState,
    params: ModelParams,
    forcing_t: dict[str, jnp.ndarray],
    config: ModelConfig,
    pool_index: PoolIndex,
    *,
    external_inputs_active: bool = False,
    external_input_source_key: str = "GPP_obs",
    external_input_is_npp: bool = False,
    external_input_target_indices: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    One Euler step of ¹⁴C tracer pool dynamics — pure JAX, no Python loops.

    Implements TECHSPEC Section 4.4:

    .. math::

        R_{atm} = R_{std} \\cdot
                  \\bigl(1 + \\Delta^{14}C_{atm} / 1000\\bigr)

        F_{14,npp,i} = F_{npp,i} \\cdot R_{atm}
                       \\quad (\\text{aboveground pools only})

        F_{14,out,i} = \\frac{C_{14,i}}{\\tau_i}
                       \\cdot f_T \\cdot f_\\theta \\cdot f_f

        F_{14,in,j}  = \\sum_i F_{ij}\\,F_{14,out,i}
                     = \\mathbf{F}^\\top \\mathbf{F}_{14,out}

        \\text{decay}_i = \\lambda_{14C} \\cdot C_{14,i}

        \\Delta C_{14,i} = F_{14,npp,i} + F_{14,in,i}
                         - F_{14,out,i} - \\text{decay}_i

    The ¹⁴C outflux uses the *same* turnover times (``log_tau``) and
    environmental scalars as the ¹²C step, so this function is fully
    differentiable w.r.t. ``params.log_tau``.

    Parameters
    ----------
    state :
        Current ecosystem state.  ``state.C14`` is evolved;
        ``state.thawed_frac`` provides the freeze/thaw mask.
        ``state.C12`` is **not** used here (the ¹⁴C dynamics depend on
        ``tau``, not on the ¹²C pool size directly).
    params :
        Model parameters — fully traced by ``jax.grad``.
    forcing_t :
        Per-timestep forcing dict.  Required keys:

        * ``'sw_radiation'``  — scalar, MJ m⁻² day⁻¹
        * ``'soil_temp'``     — shape ``(n_layers,)``, °C
        * ``'soil_moisture'`` — shape ``(n_layers,)``, m³ m⁻³
        * ``'delta14C_atm'``  — scalar, ‰
    config :
        Validated model configuration — **static**, not traced.
    pool_index :
        Pool name ↔ index mapping — **static**, not traced.

    Returns
    -------
    jnp.ndarray
        Updated C14 array, shape ``(n_total_pools,)``, gC m⁻².
    """
    n_pools = len(pool_index)
    n_ag = len(config.aboveground_pools)
    dt = float(config.dt_days)

    # ── Static pool→layer mapping (embedded as constant) ─────────────────
    pool_to_layer = _build_pool_to_layer(config, pool_index)

    # ── Environmental scalars (per layer → broadcast to per pool) ─────────
    ft_layers = f_temp(forcing_t["soil_temp"], params.log_Q10)
    fm_layers = f_moisture(
        forcing_t["soil_moisture"], params.log_theta_opt, params.log_gamma_moist
    )
    ft_vec = ft_layers[pool_to_layer]  # (n_pools,)
    fm_vec = fm_layers[pool_to_layer]
    ff_vec = state.thawed_frac[pool_to_layer]

    # ── GPP — prescribed or LUE model ────────────────────────────────────
    CUE = jnp.exp(params.log_CUE)
    if external_inputs_active:
        GPP_prescribed = forcing_t[external_input_source_key]
        GPP = jnp.where(
            jnp.isnan(GPP_prescribed),
            forcing_t["sw_radiation"] * _LUE * (1.0 - jnp.exp(-_K_EXT * _LAI)),
            GPP_prescribed,
        )
        # Fraction of NPP bypassing AG pools → scale AG ¹⁴C NPP input down
        soil_frac = jax.nn.sigmoid(params.log_soil_input_fraction)
        ag_frac = 1.0 - soil_frac
    else:
        GPP = forcing_t["sw_radiation"] * _LUE * (1.0 - jnp.exp(-_K_EXT * _LAI))
        ag_frac = jnp.asarray(1.0)

    F_npp = npp_allocation(GPP, CUE, params.log_alloc, n_ag) * ag_frac  # (n_ag,)

    # ── Atmospheric ¹⁴C ratio ─────────────────────────────────────────────
    R_atm = _R_STD * (1.0 + forcing_t["delta14C_atm"] / 1000.0)

    # ── ¹⁴C NPP inputs (aboveground pools only) ────────────────────────────
    F14_npp = jnp.zeros(n_pools).at[:n_ag].set(F_npp * R_atm)

    # ── External direct soil ¹⁴C inputs ──────────────────────────────────
    # Mirror the ¹²C external inputs, weighted by the atmospheric ¹⁴C ratio.
    if external_inputs_active:
        assert external_input_target_indices is not None
        GPP_or_NPP = GPP if not external_input_is_npp else GPP * CUE
        ext_inputs_12C = compute_external_soil_inputs(
            GPP_or_NPP=GPP_or_NPP,
            is_npp=external_input_is_npp,
            log_CUE=params.log_CUE,
            log_soil_input_fraction=params.log_soil_input_fraction,
            log_external_input_partition=params.log_external_input_partition,
            n_pools=n_pools,
            target_pool_indices=external_input_target_indices,
        )
        F14_ext = ext_inputs_12C * R_atm
    else:
        F14_ext = jnp.zeros(n_pools)

    # ── Transfer matrix (softmax logits → fractions) ──────────────────────
    F_mat = get_transfer_matrix(params.log_f_transfer, n_pools)  # (n, n)

    # ── Per-pool ¹⁴C outflux ─────────────────────────────────────────────
    tau = jnp.exp(params.log_tau)  # (n_pools,)
    F14_out = (state.C14 / tau) * ft_vec * fm_vec * ff_vec  # (n_pools,)

    # ── ¹⁴C influx from pool-to-pool transfers ────────────────────────────
    # F_mat[i, j] = fraction of pool-i outflux going to pool j
    # F14_in[j]   = Σ_i F_mat[i, j] · F14_out[i]  =  F_mat.T @ F14_out
    F14_in = F_mat.T @ F14_out  # (n_pools,)

    # ── Radioactive decay ─────────────────────────────────────────────────
    decay = params.lambda_14C * state.C14  # (n_pools,)

    # ── Euler update ──────────────────────────────────────────────────────
    dC14 = F14_npp + F14_ext + F14_in - F14_out - decay
    return jnp.asarray(state.C14 + dC14 * dt)


# ---------------------------------------------------------------------------
# compute_delta14C
# ---------------------------------------------------------------------------


def compute_delta14C(
    C14: jnp.ndarray,
    C12: jnp.ndarray,
    R_std: float = _R_STD,
) -> jnp.ndarray:
    """
    Convert raw pool masses to Δ¹⁴C values in per-mille (‰).

    .. math::

        R_{sample,i} = \\frac{C_{14,i}}{\\max(C_{12,i},\\,10^{-10})}

        \\Delta^{14}C_i =
            \\left(\\frac{R_{sample,i}}{R_{std}} - 1\\right) \\times 1000

    Parameters
    ----------
    C14 :
        ¹⁴C pool masses, shape ``(n_total_pools,)``, gC m⁻².
    C12 :
        ¹²C pool masses, shape ``(n_total_pools,)``, gC m⁻².
    R_std :
        Modern standard ¹⁴C/¹²C ratio (default ``1.176e-12``).

    Returns
    -------
    jnp.ndarray
        Δ¹⁴C in ‰, shape ``(n_total_pools,)``.
        Zero when ``R_sample == R_std``; negative for depleted / old carbon.
    """
    R_sample = C14 / jnp.maximum(C12, 1e-10)
    return (R_sample / R_std - 1.0) * 1000.0


# ---------------------------------------------------------------------------
# spinup_14C
# ---------------------------------------------------------------------------


def spinup_14C(
    model: EcosystemModel,
    C12_ss: jnp.ndarray,
    forcing_historical: dict[str, jnp.ndarray],
    params: ModelParams,
) -> jnp.ndarray:
    """
    Spin up ¹⁴C pools through the historical atmospheric ¹⁴C record.

    **Phase 1 — pre-industrial initialisation**

        Every pool is set to Δ¹⁴C ≈ 0‰ (modern standard):

        .. math::

            C_{14,i}^{(0)} = C_{12,ss,i} \\cdot R_{std}

    **Phase 2 — historical forward integration**

        Advances C14 via ``jax.lax.scan`` while holding C12 fixed at
        ``C12_ss``.  The thaw fraction is set from the layer's
        ``permafrost_eligible`` flag and held constant (no seasonal
        freeze/thaw during the spin-up).

    Parameters
    ----------
    model :
        Configured ``EcosystemModel``; supplies ``config`` and ``pool_index``.
    C12_ss :
        Steady-state ¹²C pool sizes, shape ``(n_total_pools,)``, gC m⁻².
        Held constant throughout the spin-up.
    forcing_historical :
        Dict of stacked forcing arrays, shape ``(n_steps, ...)``; scan
        slices along axis 0.  Required keys: ``'sw_radiation'``,
        ``'soil_temp'`` (n_steps, n_layers), ``'soil_moisture'``,
        ``'delta14C_atm'``.
    params :
        Model parameters.

    Returns
    -------
    jnp.ndarray
        C14 at end of historical period, shape ``(n_total_pools,)``.
    """
    cfg = model.config
    idx = model.pool_index

    # ── Phase 1: pre-industrial C14 initialisation ───────────────────────
    C14_init = C12_ss * _R_STD

    # ── Static spinup state fields (held fixed across all timesteps) ──────
    thawed = jnp.array(
        [0.0 if layer.permafrost_eligible else 1.0 for layer in cfg.soil_layers]
    )

    # ── Phase 2: scan through historical atmospheric record ───────────────
    def _scan_body(
        C14: jnp.ndarray,
        forcing_t: dict[str, jnp.ndarray],
    ) -> tuple[jnp.ndarray, None]:
        state = EcosystemState(
            C12=C12_ss,
            C14=C14,
            soil_temp=forcing_t["soil_temp"],
            soil_moisture=forcing_t["soil_moisture"],
            thawed_frac=thawed,
            active_layer_depth=jnp.array(jnp.inf),
            time=jnp.array(0.0),
        )
        new_C14 = step_14C(state, params, forcing_t, cfg, idx)
        return new_C14, None

    C14_final, _ = jax.lax.scan(_scan_body, C14_init, forcing_historical)
    return C14_final


# ---------------------------------------------------------------------------
# initialize_permafrost_14C
# ---------------------------------------------------------------------------


def initialize_permafrost_14C(
    C14: jnp.ndarray,
    C12: jnp.ndarray,
    config: ModelConfig,
    pool_index: PoolIndex,
    permafrost_obs: dict[str, float],
) -> jnp.ndarray:
    """
    Override C14 for permafrost-eligible pools using measured Δ¹⁴C values.

    Only pools listed in ``permafrost_obs`` are modified; all others are
    left unchanged.

    .. math::

        C_{14,i} = C_{12,i} \\cdot R_{std} \\cdot
                   \\Bigl(1 + \\frac{\\Delta^{14}C_{obs,i}}{1000}\\Bigr)

    Parameters
    ----------
    C14 :
        Current ¹⁴C array, shape ``(n_total_pools,)``.
    C12 :
        Current ¹²C array, shape ``(n_total_pools,)``.
    config :
        Validated model configuration (retained for API consistency and
        future validation).
    pool_index :
        Pool name ↔ index mapping.
    permafrost_obs :
        Mapping ``pool_name → Δ¹⁴C_obs`` (‰).  Pool names must be valid
        compound names (e.g. ``"permafrost_slow"``).

    Returns
    -------
    jnp.ndarray
        Updated C14 array with the specified pools set from observations.
    """
    for pool_name, delta14C_obs in permafrost_obs.items():
        i = pool_index[pool_name]
        C14 = C14.at[i].set(C12[i] * _R_STD * (1.0 + delta14C_obs / 1000.0))
    return C14


def respired_delta14C(
    delta14C: jnp.ndarray,
    Rh_by_pool: jnp.ndarray,
    C12: jnp.ndarray,
    log_tau: jnp.ndarray,
    log_f_transfer: jnp.ndarray,
    n_pools: int,
    *,
    flux_floor: float = 1e-18,
) -> jnp.ndarray:
    """Δ¹⁴C (‰) of respired CO₂: the pools' Δ¹⁴C mixed by respiration flux.

    .. math::

        \\Delta^{14}C_{resp} = \\frac{\\sum_i R_{h,i}\\,\\Delta^{14}C_i}
                                    {\\sum_i R_{h,i}}

    with ``R_h,i`` the model's own per-pool respiration
    (``resp_frac_i · C_i/τ_i · f_t · f_m · f_f``), so the operator and
    ``het_respiration`` cannot disagree about what respiration means.

    **Why the fallback exists.** Those environmental scalars are shared by all
    pools in a single-layer config, so they cancel in the ratio — but they do
    not cancel *numerically*. At a frozen timestep ``f_f → 0`` drives every
    pool's weight to zero simultaneously and the ratio becomes 0/0. This is not
    hypothetical: at Old Black Spruce 218 of 1826 daily steps have an
    identically-zero total, and 318 fall below 1e-30. Evaluated naively that
    yields a predicted 0 ‰ with a pathological gradient, which flat-lines the
    OE cost at every boreal and permafrost site.

    When the total flux underflows ``flux_floor`` the mixture therefore falls
    back to the **intrinsic** weights ``resp_frac_i · C_i/τ_i``, which are
    strictly positive. That is the right limit rather than a fudge: the
    isotopic signature of respired CO₂ is undefined when nothing is respiring,
    and the intrinsic mixture is what the soil would respire at reference
    conditions. In a single-layer config it is *exactly* the flux-weighted
    answer, since the common scalars cancel.

    Shapes: ``delta14C``, ``Rh_by_pool``, ``C12`` are ``(T, n_pools)``;
    returns ``(T,)``.
    """
    resp_frac = respiration_fractions(log_f_transfer, n_pools)  # (n_pools,)
    tau = jnp.exp(log_tau)  # (n_pools,)

    w_flux = Rh_by_pool  # (T, n_pools)
    s_flux = w_flux.sum(axis=-1)  # (T,)
    live = s_flux > flux_floor

    # Guard the denominator so the untaken branch never produces NaN/Inf and
    # poisons the gradient (the double-where pattern used elsewhere in the OE
    # loss).
    safe_flux = jnp.where(live, s_flux, 1.0)
    d14_flux = (delta14C * w_flux).sum(axis=-1) / safe_flux

    w_int = resp_frac[None, :] * (C12 / tau[None, :])  # (T, n_pools), > 0
    s_int = w_int.sum(axis=-1)
    d14_int = (delta14C * w_int).sum(axis=-1) / (s_int + 1e-30)

    return jnp.where(live, d14_flux, d14_int)
