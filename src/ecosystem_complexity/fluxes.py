"""
Pure-JAX flux functions for the ecosystem-complexity carbon model.

All functions are stateless and contain no Python control flow on traced
values — safe for ``jax.jit``, ``jax.lax.scan``, and ``jax.grad``.

Implements the equations in TECHSPEC Section 4.3:

    f_temp          — Lloyd-Taylor Q10 temperature scalar
    f_moisture      — Gaussian soil-moisture scalar
    thawed_frac     — differentiable freeze/thaw mask (sigmoid)
    decomp_flux     — per-pool decomposition flux (gC m⁻² day⁻¹)
    npp_allocation  — NPP partitioned across aboveground pools
    het_respiration — total heterotrophic respiration (scalar)
    nee             — net ecosystem exchange (scalar)
"""
from __future__ import annotations

import jax.nn
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Environmental scalars
# ---------------------------------------------------------------------------


def f_temp(
    soil_temp: jnp.ndarray,
    log_Q10: jnp.ndarray,
    T_ref: float = 15.0,
) -> jnp.ndarray:
    """
    Lloyd-Taylor Q10 temperature response scalar.

    .. math::

        f_{temp}(T) = Q_{10}^{(T - T_{ref}) / 10}

    At ``T == T_ref`` the scalar equals 1.0 exactly.

    Parameters
    ----------
    soil_temp :
        Soil temperature in °C.  Scalar or array — shape is preserved.
    log_Q10 :
        Natural log of Q10 (from ``ModelParams``).  Same shape as
        ``soil_temp`` (per-layer) or scalar (broadcast).
    T_ref :
        Reference temperature in °C (default 15 °C).

    Returns
    -------
    jnp.ndarray
        Non-negative scalar(s), same shape as ``soil_temp``.
    """
    Q10 = jnp.exp(log_Q10)
    exponent = (soil_temp - T_ref) / 10.0
    return Q10**exponent


def f_moisture(
    theta: jnp.ndarray,
    log_theta_opt: jnp.ndarray,
    log_gamma: jnp.ndarray,
) -> jnp.ndarray:
    """
    Gaussian soil-moisture response scalar.

    .. math::

        f_{moist}(\\theta) =
            \\exp\\!\\left(-\\gamma \\,(\\theta - \\theta_{opt})^2\\right)

    Equals 1.0 at ``theta == theta_opt``; falls off symmetrically.

    Parameters
    ----------
    theta :
        Volumetric soil moisture (m³ m⁻³).  Scalar or array.
    log_theta_opt :
        Natural log of optimal soil moisture (from ``ModelParams``).
    log_gamma :
        Natural log of the Gaussian width parameter (from ``ModelParams``).

    Returns
    -------
    jnp.ndarray
        Values in ``(0, 1]``, same shape as ``theta``.
    """
    theta_opt = jnp.exp(log_theta_opt)
    gamma = jnp.exp(log_gamma)
    return jnp.exp(-gamma * (theta - theta_opt) ** 2)


def thawed_frac(
    soil_temp: jnp.ndarray,
    steepness: float = 10.0,
) -> jnp.ndarray:
    """
    Differentiable freeze/thaw fraction (thaw mask).

    .. math::

        \\text{thawed\\_frac}(T) = \\sigma(k \\cdot T)

    where :math:`\\sigma` is the logistic sigmoid and *k* is ``steepness``.

    * ``T > 0`` → approaches 1.0 (fully thawed, full decomposition)
    * ``T = 0`` → exactly 0.5 (transition point)
    * ``T < 0`` → approaches 0.0 (frozen, decomposition suppressed)

    Non-permafrost layers are held at T > 0 so this always returns ≈ 1.0
    for them without any conditional logic.

    Parameters
    ----------
    soil_temp :
        Soil temperature in °C.  Shape ``(n_layers,)`` or scalar.
    steepness :
        Sigmoid steepness *k* (default 10 → ~0.1 °C transition width).

    Returns
    -------
    jnp.ndarray
        Values in ``(0, 1)``, same shape as ``soil_temp``.
    """
    return jax.nn.sigmoid(steepness * soil_temp)


# ---------------------------------------------------------------------------
# Per-pool decomposition
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# NPP allocation
# ---------------------------------------------------------------------------


def npp_allocation(
    GPP: jnp.ndarray,
    CUE: float,
    log_alloc: jnp.ndarray,
    n_ag_pools: int,
) -> jnp.ndarray:
    """
    Partition net primary production across aboveground pools.

    .. math::

        \\text{alloc} = \\mathrm{softmax}(\\log\\_alloc), \\quad
        F_{npp,i} = GPP \\cdot CUE \\cdot \\text{alloc}_i

    Parameters
    ----------
    GPP :
        Gross primary production (gC m⁻² day⁻¹).  Scalar.
    CUE :
        Carbon use efficiency (dimensionless, typically ~0.5).
        Passed as a Python float — not a traced JAX value.
    log_alloc :
        Log-ratio allocation logits, shape ``(n_ag_pools,)``
        (from ``ModelParams``).
    n_ag_pools :
        Number of aboveground pools.  Used as a static shape hint.

    Returns
    -------
    jnp.ndarray
        Shape ``(n_ag_pools,)`` NPP fluxes in gC m⁻² day⁻¹.
    """
    alloc = jax.nn.softmax(log_alloc[:n_ag_pools])
    return GPP * CUE * alloc


# ---------------------------------------------------------------------------
# Ecosystem-level fluxes
# ---------------------------------------------------------------------------


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
    f_full = jax.nn.softmax(log_f_transfer, axis=-1)   # (n_pools, n_pools+1)
    f_transfer = f_full[:, :n_pools]                    # (n_pools, n_pools)

    # Fraction of each pool's outflux that is respired
    resp_frac = 1.0 - f_transfer.sum(axis=-1)           # (n_pools,)

    # Per-pool decomposition fluxes (vectorised)
    tau = jnp.exp(log_tau)                              # (n_pools,)
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
