"""
State and parameter containers for the ecosystem-complexity carbon model.

Both NamedTuples are JAX pytrees by default — no registration required.
``jax.tree.map``, ``jax.jit``, and ``jax.grad`` all work transparently.

Canonical pool order (fixed by PoolIndex)
-----------------------------------------
    [ag_pool_0, ..., ag_pool_m |
     layer_0_som_0, ..., [layer_0_mic,] |
     layer_1_som_0, ..., ]

Factory functions
-----------------
    make_initial_state(config, site_config)  -> EcosystemState
    make_default_params(config)              -> ModelParams
"""
from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax.numpy as jnp

from ecosystem_complexity.config import ModelConfig, PoolIndex

# ¹⁴C radioactive decay constant: λ = ln(2) / t½
# t½ = 5730 yr * 365.25 day/yr  →  λ ≈ 3.312e-7 day⁻¹
# (The spec quotes 3.317e-7 due to rounding; this is the exact computation.)
_LAMBDA_14C: float = math.log(2.0) / (5730.0 * 365.25)


# ---------------------------------------------------------------------------
# State and parameter NamedTuples
# ---------------------------------------------------------------------------


class EcosystemState(NamedTuple):
    """
    Full model state — a flat JAX pytree.

    All arrays use float32. Pool ordering follows ``PoolIndex`` canonical order.

    Fields
    ------
    C12 : (n_total_pools,) gC m⁻²
        Bulk ¹²C pool sizes.
    C14 : (n_total_pools,) gC m⁻²
        ¹⁴C tracer pool sizes (mass, not ratio).  Zero if ``enable_14C=False``.
    soil_temp : (n_layers,) °C
        Per-layer soil temperature.
    soil_moisture : (n_layers,) m³ m⁻³
        Per-layer volumetric soil moisture.
    thawed_frac : (n_layers,) dimensionless [0, 1]
        Thaw fraction: 1.0 = fully thawed, 0.0 = fully frozen.
        Non-permafrost layers are always 1.0.
    active_layer_depth : scalar, metres
        Depth of the thaw front.  ``jnp.inf`` for non-permafrost sites.
    time : scalar, days since 1970-01-01
    """

    C12: jnp.ndarray
    C14: jnp.ndarray
    soil_temp: jnp.ndarray
    soil_moisture: jnp.ndarray
    thawed_frac: jnp.ndarray
    active_layer_depth: jnp.ndarray
    time: jnp.ndarray


class ModelParams(NamedTuple):
    """
    Optimisable model parameters — a flat JAX pytree.

    All continuous parameters are stored in unconstrained (log / log-ratio)
    space so that ``jax.grad`` never hits a boundary.

    Fields
    ------
    log_tau : (n_total_pools,) log-days
        Log turnover time.  Exponentiate before use: ``tau = jnp.exp(log_tau)``.
    log_f_transfer : (n_total_pools, n_total_pools + 1) raw logits
        Transfer fraction logits.  Convert via softmax (last column = respiration):
        ``f = jax.nn.softmax(log_f_transfer, axis=-1)``
    log_alloc : (n_ag_pools,) log-ratios
        NPP allocation.  Convert via softmax: ``alloc = jax.nn.softmax(log_alloc)``.
    log_Q10 : (n_layers,)
        Log temperature sensitivity.  ``Q10 = jnp.exp(log_Q10)``.
    log_theta_opt : (n_layers,)
        Log optimal soil moisture.
    log_gamma_moist : (n_layers,)
        Log moisture response curve shape parameter.
    alpha_priming : (n_mic_pools,)
        Microbial priming coefficient.  Shape is ``(0,)`` when
        ``microbial_pool_per_layer=False``.
    lambda_14C : scalar, day⁻¹
        ¹⁴C radioactive decay constant.  **Fixed, never optimised.**
        Value: ``ln(2) / (5730 * 365.25) ≈ 3.312e-7 day⁻¹``.
    """

    log_tau: jnp.ndarray
    log_f_transfer: jnp.ndarray
    log_alloc: jnp.ndarray
    log_Q10: jnp.ndarray
    log_theta_opt: jnp.ndarray
    log_gamma_moist: jnp.ndarray
    alpha_priming: jnp.ndarray
    lambda_14C: jnp.ndarray


# ---------------------------------------------------------------------------
# Factory: initial state
# ---------------------------------------------------------------------------


def make_initial_state(config: ModelConfig, site_config: dict[str, Any]) -> EcosystemState:
    """
    Build a zeroed initial ``EcosystemState`` from a validated config and
    the raw YAML ``site`` section.

    Parameters
    ----------
    config :
        Validated ``ModelConfig`` (from ``load_config``).
    site_config :
        Raw ``site`` dict from the YAML, e.g.::

            {"id": "US-Ha1", "mat_c": 8.5, "permafrost": false, ...}

        Used for initial temperature and permafrost conditions.

    Returns
    -------
    EcosystemState
        All carbon pools zeroed; soil temperature set to the mean annual
        temperature (``mat_c``); soil moisture at 0.3 m³ m⁻³;
        frozen fractions and active layer depth set from ``permafrost_eligible``
        flags on each layer.
    """
    idx = PoolIndex(config)
    n_pools = len(idx)
    n_layers = len(config.soil_layers)
    is_permafrost_site = bool(site_config.get("permafrost", False))
    mat_c = float(site_config.get("mat_c", 0.0))

    # ── Carbon pools ─────────────────────────────────────────────────────
    C12 = jnp.zeros(n_pools)
    C14 = jnp.zeros(n_pools)

    # ── Soil temperature ─────────────────────────────────────────────────
    # Initialise all layers at the site's mean annual air temperature.
    soil_temp = jnp.full(n_layers, mat_c)

    # ── Soil moisture ─────────────────────────────────────────────────────
    soil_moisture = jnp.full(n_layers, 0.3)

    # ── Frozen fraction ───────────────────────────────────────────────────
    # Non-permafrost layers → fully thawed (1.0)
    # Permafrost-eligible layers → fully frozen (0.0) at initialisation
    thawed_frac = jnp.array(
        [0.0 if layer.permafrost_eligible else 1.0 for layer in config.soil_layers]
    )

    # ── Active layer depth ────────────────────────────────────────────────
    if not is_permafrost_site:
        active_layer_depth = jnp.array(jnp.inf)
    else:
        # Set to the top of the shallowest permafrost-eligible layer so that
        # the freeze/thaw front starts at the permafrost table.
        first_pf = next(
            (lay for lay in config.soil_layers if lay.permafrost_eligible), None
        )
        ald_init = first_pf.depth_top_m if first_pf is not None else 0.0
        active_layer_depth = jnp.array(float(ald_init))

    # ── Time ─────────────────────────────────────────────────────────────
    time = jnp.array(0.0)

    return EcosystemState(
        C12=C12,
        C14=C14,
        soil_temp=soil_temp,
        soil_moisture=soil_moisture,
        thawed_frac=thawed_frac,
        active_layer_depth=active_layer_depth,
        time=time,
    )


# ---------------------------------------------------------------------------
# Factory: default parameters
# ---------------------------------------------------------------------------

_DEFAULT_TAU_AG_DAYS: float = 365.0   # fallback AG pool turnover (1 year)
_DEFAULT_TAU_MIC_DAYS: float = 30.0   # default microbial turnover time
_EPS: float = 1e-6                    # floor for transfer fractions


def make_default_params(config: ModelConfig) -> ModelParams:
    """
    Build ``ModelParams`` initialised from the YAML prior values.

    Transfer-matrix logits are set so that::

        jax.nn.softmax(log_f_transfer, axis=-1)

    reproduces the fractions in ``transfer_rules`` exactly (the remainder
    goes to the respiration column at index ``n_total_pools``).  Rows for
    pools with no outgoing transfer rules are initialised to respire
    everything (respiration column = 1 − ε, others = ε).

    Allocation logits satisfy::

        jax.nn.softmax(log_alloc) == alloc_fractions_from_yaml

    Environmental parameters use per-layer arrays initialised to the prior
    values from the ``parameters`` section of the YAML (defaulting to
    Q10=2.0, θ_opt=0.3, γ=5.0 if not specified).

    Parameters
    ----------
    config :
        Validated ``ModelConfig`` (from ``load_config``).

    Returns
    -------
    ModelParams
        JAX-compatible pytree ready for ``jit`` / ``grad``.
    """
    idx = PoolIndex(config)
    n_pools = len(idx)
    n_layers = len(config.soil_layers)
    n_mic = n_layers if config.microbial_pool_per_layer else 0
    params_raw = config.parameters_raw

    # ── log_tau ───────────────────────────────────────────────────────────
    tau_overrides: dict[str, float] = {
        k: float(v)
        for k, v in params_raw.get("tau_overrides", {}).items()
    }

    tau_vals: list[float] = []
    # Aboveground pools: use tau_override if provided, else default 1 year
    for pool in config.aboveground_pools:
        tau_vals.append(tau_overrides.get(pool.name, _DEFAULT_TAU_AG_DAYS))
    # Soil SOM pools: use tau_override or tau_prior_days from SOMPoolDef
    for layer in config.soil_layers:
        for som_pool in layer.som_pools:
            compound_name = f"{layer.name}_{som_pool.name}"
            tau_vals.append(
                tau_overrides.get(compound_name, som_pool.tau_prior_days)
            )
        if config.microbial_pool_per_layer:
            tau_vals.append(_DEFAULT_TAU_MIC_DAYS)

    log_tau = jnp.log(jnp.array(tau_vals, dtype=jnp.float32))

    # ── log_f_transfer ────────────────────────────────────────────────────
    # Build a (n_pools, n_pools+1) target fraction matrix.
    # Column j (0 ≤ j < n_pools): fraction of source i routed to pool j.
    # Column n_pools (last):       fraction respired.
    # Non-specified destinations receive a tiny floor (_EPS) so that
    # log(fraction) is always finite; rows are then normalised to sum to 1
    # before taking log, giving the exact softmax inverse.

    # Use a dict to track which (row, col) pairs have explicit fractions.
    explicit: dict[tuple[int, int], float] = {}
    for source_name, dest_name, fraction in config.transfer_rules:
        src_i = idx[source_name]
        dst_j = idx[dest_name]
        # Accumulate in case the same pair appears more than once.
        explicit[(src_i, dst_j)] = explicit.get((src_i, dst_j), 0.0) + fraction

    # Build the raw (unnormalised) fraction table.
    f_rows: list[list[float]] = []
    for i in range(n_pools):
        row = [_EPS] * (n_pools + 1)
        total_explicit = 0.0
        for (si, dj), frac in explicit.items():
            if si == i:
                row[dj] = frac
                total_explicit += frac
        # Respiration gets whatever fraction is not transferred.
        row[-1] = max(1.0 - total_explicit, _EPS)
        f_rows.append(row)

    # Normalise rows to sum exactly to 1, then take log → softmax⁻¹.
    f_arr = jnp.array(f_rows, dtype=jnp.float32)
    f_norm = f_arr / f_arr.sum(axis=-1, keepdims=True)
    log_f_transfer = jnp.log(f_norm)

    # ── log_alloc ──────────────────────────────────────────────────────────
    ag_names = [p.name for p in config.aboveground_pools]
    alloc_vals = jnp.array(
        [config.alloc[name] for name in ag_names], dtype=jnp.float32
    )
    # Normalise (fractions should already sum to 1, but be defensive).
    alloc_norm = alloc_vals / alloc_vals.sum()
    log_alloc = jnp.log(alloc_norm)

    # ── Environmental parameters ───────────────────────────────────────────
    def _prior(key: str, default: float) -> float:
        entry = params_raw.get(key, {})
        if isinstance(entry, dict):
            return float(entry.get("value", default))
        return float(entry) if entry is not None else default

    log_Q10         = jnp.full(n_layers, math.log(_prior("Q10", 2.0)),           dtype=jnp.float32)
    log_theta_opt   = jnp.full(n_layers, math.log(_prior("theta_opt", 0.3)),     dtype=jnp.float32)
    log_gamma_moist = jnp.full(n_layers, math.log(_prior("gamma_moisture", 5.0)), dtype=jnp.float32)

    # ── alpha_priming ─────────────────────────────────────────────────────
    alpha_priming = jnp.zeros(n_mic, dtype=jnp.float32)

    # ── lambda_14C (fixed) ─────────────────────────────────────────────────
    lambda_14C = jnp.array(_LAMBDA_14C, dtype=jnp.float32)

    return ModelParams(
        log_tau=log_tau,
        log_f_transfer=log_f_transfer,
        log_alloc=log_alloc,
        log_Q10=log_Q10,
        log_theta_opt=log_theta_opt,
        log_gamma_moist=log_gamma_moist,
        alpha_priming=alpha_priming,
        lambda_14C=lambda_14C,
    )
