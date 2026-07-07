"""
Abiotic environmental response functions for the ecosystem-complexity model.

All functions are pure JAX — safe for jit, lax.scan, and grad.

Functions
---------
f_temp              — Lloyd-Taylor Q10 temperature response
f_moisture          — Gaussian soil-moisture response
thawed_frac         — differentiable freeze/thaw mask
soil_temp_at_depth  — depth-attenuated soil temperature (Fourier placeholder)
_pool_env_vecs      — broadcast per-layer scalars to per-pool vectors
"""
from __future__ import annotations

import jax
import jax.nn
import jax.numpy as jnp


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


def soil_temp_at_depth(
    T_surface: jnp.ndarray,
    z_m: float,
    T_annual_mean: float,
    damping_depth_m: float = 2.0,
) -> jnp.ndarray:
    """
    Temperature at depth z_m given surface temperature T_surface.

    **Placeholder for Fourier's law heat conduction.**  A full implementation
    will solve the 1-D heat diffusion equation

        ∂T/∂t = κ(z) ∂²T/∂z²

    with depth-dependent thermal conductivity κ(z) (W m⁻¹ K⁻¹), yielding the
    classic sinusoidal solution for a periodic surface boundary condition:

        T(z, t) = T_mean + A₀ exp(−z/d) cos(ωt − z/d − φ)

    where d = √(2κ/ω) is the thermal damping depth and ω = 2π/P (P = 1 yr).

    Current approximation: exponential amplitude attenuation only (phase lag
    ignored, steady-state mean used).  This is exact for the annual-mean
    component and a reasonable first approximation for the amplitude envelope.

        T(z) ≈ T_mean + (T_surface − T_mean) × exp(−z / damping_depth_m)

    Parameters
    ----------
    T_surface :
        Current surface (shallow) soil temperature in °C.
    z_m :
        Depth in metres (positive downward).
    T_annual_mean :
        Mean annual surface temperature in °C.  Sets the asymptotic deep value.
    damping_depth_m :
        Thermal damping depth d (m).  Typical values: 1–3 m for mineral soil,
        0.5–1.5 m for organic-rich tundra.  Replace with κ-based calculation
        when Fourier's law is fully implemented.

    Returns
    -------
    jnp.ndarray
        Estimated soil temperature at depth z_m, same dtype as T_surface.
    """
    amplitude_factor = jnp.exp(jnp.array(-z_m / damping_depth_m, dtype=jnp.float32))
    return T_annual_mean + (T_surface - T_annual_mean) * amplitude_factor


def _pool_env_vecs(
    soil_temp: jnp.ndarray,
    soil_moisture: jnp.ndarray,
    ff_layers: jnp.ndarray,
    log_Q10: jnp.ndarray,
    log_theta_opt: jnp.ndarray,
    log_gamma_moist: jnp.ndarray,
    pool_to_layer: jnp.ndarray,
    *,
    pool_mid_depths: jnp.ndarray | None = None,
    T_annual_mean: float | None = None,
    damping_depth_m: float = 2.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Broadcast per-layer environmental scalars to per-pool vectors.

    When ``pool_mid_depths`` and ``T_annual_mean`` are provided, the thawed
    fraction for each pool is computed from a depth-attenuated temperature
    (via ``soil_temp_at_depth``) rather than the layer-surface temperature.
    This implements freeze-thaw gating: deep permafrost pools stay frozen
    (thawed_frac ≈ 0) even when the surface thaws in summer.

    Parameters
    ----------
    soil_temp : (n_layers,)
        Soil temperature in °C, from forcing.
    soil_moisture : (n_layers,)
        Volumetric soil moisture (m³ m⁻³), from forcing.
    ff_layers : (n_layers,)
        Layer-level thaw fraction from the current state (used when
        ``pool_mid_depths`` is None).
    log_Q10, log_theta_opt, log_gamma_moist : (n_layers,)
        Log-space environmental parameters from ``ModelParams``.
    pool_to_layer : (n_pools,) int
        Static index: ``pool_to_layer[i]`` is the layer index for pool *i*.
    pool_mid_depths : (n_pools,) float or None
        Mid-depth of each pool in metres.  When provided, enables per-pool
        depth-attenuated freeze-thaw gating.
    T_annual_mean : float or None
        Mean annual surface temperature (°C); asymptotic deep temperature.
    damping_depth_m : float
        Thermal damping depth for the ``soil_temp_at_depth`` placeholder.

    Returns
    -------
    ft_vec, fm_vec, ff_vec : each (n_pools,)
        Temperature, moisture, and thaw scalars broadcast to pool dimension.
    """
    ft_layers = f_temp(soil_temp, log_Q10)                       # (n_layers,)
    fm_layers = f_moisture(soil_moisture, log_theta_opt, log_gamma_moist)
    ft_vec = ft_layers[pool_to_layer]                            # (n_pools,)
    fm_vec = fm_layers[pool_to_layer]

    if pool_mid_depths is not None and T_annual_mean is not None:
        # Per-pool depth-attenuated temperature → per-pool thawed_frac.
        # T_surface for each pool: use the layer-mean soil temperature.
        T_surface_per_pool = soil_temp[pool_to_layer]            # (n_pools,)
        T_depth_per_pool = soil_temp_at_depth(
            T_surface_per_pool, pool_mid_depths, T_annual_mean, damping_depth_m
        )
        ff_vec = thawed_frac(T_depth_per_pool)                   # (n_pools,)
    else:
        ff_vec = ff_layers[pool_to_layer]                        # (n_pools,)

    return ft_vec, fm_vec, ff_vec
