"""
Aboveground carbon dynamics for the ecosystem-complexity model.

Covers GPP estimation, NPP allocation, and external soil carbon inputs.
All functions are pure JAX — safe for jit, lax.scan, and grad.

Functions
---------
_gpp                        — light-use-efficiency GPP estimate
npp_allocation              — partition NPP across aboveground pools
compute_external_soil_inputs — direct soil inputs from prescribed GPP/NPP
"""
from __future__ import annotations

import jax
import jax.nn
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Model-level constants (placeholder light-use efficiency model)
# ---------------------------------------------------------------------------

_LUE: float = 0.002   # light-use efficiency (gC MJ⁻¹, placeholder)
_K_EXT: float = 0.5   # Beer–Lambert extinction coefficient (dimensionless)
_LAI: float = 5.0     # leaf area index (m² m⁻², placeholder)
_CUE: float = 0.5     # carbon use efficiency (NPP / GPP)


def _gpp(sw_radiation: jnp.ndarray) -> jnp.ndarray:
    """
    Light-use-efficiency GPP estimate (gC m⁻² day⁻¹).

    .. math::

        GPP = SW_{rad} \\cdot LUE \\cdot \\bigl(1 - e^{-k \\cdot LAI}\\bigr)

    This is a placeholder; the inversion will constrain GPP via NEE obs.
    """
    return sw_radiation * _LUE * (1.0 - jnp.exp(-_K_EXT * _LAI))


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


def compute_external_soil_inputs(
    GPP_or_NPP: jnp.ndarray,
    is_npp: bool,
    log_CUE: jnp.ndarray,
    log_soil_input_fraction: jnp.ndarray,
    log_external_input_partition: jnp.ndarray,
    n_pools: int,
    target_pool_indices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute direct soil carbon inputs from prescribed GPP or NPP.

    When ``is_npp=False`` (source is GPP), NPP is first derived as
    ``NPP = GPP × exp(log_CUE)``.

    The soil input is then:

    .. math::

        \\text{soil\\_input\\_total} = NPP \\cdot \\sigma(\\text{log\\_soil\\_input\\_fraction})

        \\text{partition} = \\mathrm{softmax}(\\text{log\\_external\\_input\\_partition})

        \\text{inputs}[\\text{target\\_pool\\_indices}] =
            \\text{soil\\_input\\_total} \\times \\text{partition}

    Parameters
    ----------
    GPP_or_NPP :
        Scalar forcing value at timestep *t* (gC m⁻² day⁻¹).
    is_npp :
        Static Python bool — if ``True`` the input is already NPP;
        skip the CUE multiplication.
    log_CUE :
        Natural log of carbon use efficiency (scalar JAX array).
    log_soil_input_fraction :
        Logit of the fraction of NPP that enters soil pools directly
        (scalar JAX array).  ``sigmoid(log_soil_input_fraction)`` gives
        the fraction in [0, 1].
    log_external_input_partition :
        Logits for partitioning among target soil pools,
        shape ``(n_target_pools,)``.  ``softmax(...)`` gives partition
        fractions that sum to 1.
    n_pools :
        Total pool count (static Python int).
    target_pool_indices :
        Integer indices of the target soil pools (static constant array).
        Shape ``(n_target_pools,)``.

    Returns
    -------
    jnp.ndarray
        Shape ``(n_pools,)``, gC m⁻² day⁻¹.
        Non-zero only at ``target_pool_indices``.
    """
    # Convert GPP → NPP if needed (is_npp is a static Python bool, not traced)
    if is_npp:
        npp = GPP_or_NPP
    else:
        npp = GPP_or_NPP * jnp.exp(log_CUE)

    # Fraction of NPP going directly to soil pools
    soil_frac = jax.nn.sigmoid(log_soil_input_fraction)
    soil_input_total = npp * soil_frac

    # Partition across target pools (softmax → sums to 1)
    partition = jax.nn.softmax(log_external_input_partition)

    # Scatter into full pool vector (no-op when target_pool_indices is empty)
    inputs = jnp.zeros(n_pools)
    inputs = inputs.at[target_pool_indices].set(soil_input_total * partition)
    return inputs
