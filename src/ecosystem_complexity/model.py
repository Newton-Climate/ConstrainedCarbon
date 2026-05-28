"""
EcosystemModel: one-step forward model for the ¹²C carbon cycle.

The model is a non-frozen dataclass that holds the configuration, parameters,
pool index, and a JIT-compiled step function.  All JAX-traced computation
lives in module-level pure functions; class methods only wire in static
structural arguments (pool counts, pool-to-layer index mapping, timestep).

This separation guarantees that no ``self`` attribute is accessed inside a
traced call, making every step function safe for::

    jax.jit     — compile the step
    jax.grad    — differentiate through params
    jax.lax.scan — unroll a forward simulation

Implemented here
----------------
  step_12C   — one Euler step of ¹²C pool dynamics
  step_14C   — stub (NotImplementedError; implemented in Issue #8)
  diagnose   — diagnostic fluxes: NEE, GPP, ER, Rh, Ra, NPP

Placeholder GPP model (light-use efficiency)
---------------------------------------------
  GPP = SW_rad × LUE × (1 − exp(−k × LAI))

Constants (will be replaced by inversion-constrained parameters later):
  LUE = 0.002,  k = 0.5,  LAI = 5.0,  CUE = 0.5

Mass balance (Euler step, dt = 1 day)
--------------------------------------
  ΔC₁₂_total = NPP − Rh      (= −NEE,  positive = carbon gain)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

import ecosystem_complexity.tracer_14C as tracer_14C
from ecosystem_complexity.config import ModelConfig, PoolIndex
from ecosystem_complexity.fluxes import (
    compute_external_soil_inputs,
    f_moisture,
    f_temp,
    het_respiration,
    npp_allocation,
    soil_temp_at_depth,
    thawed_frac,
)
from ecosystem_complexity.fluxes import (
    nee as nee_flux,
)
from ecosystem_complexity.state import EcosystemState, ModelParams
from ecosystem_complexity.transfer import get_transfer_matrix

# ---------------------------------------------------------------------------
# Model-level constants (placeholder light-use efficiency model)
# ---------------------------------------------------------------------------

_LUE: float = 0.002   # light-use efficiency (gC MJ⁻¹, placeholder)
_K_EXT: float = 0.5   # Beer–Lambert extinction coefficient (dimensionless)
_LAI: float = 5.0     # leaf area index (m² m⁻², placeholder)
_CUE: float = 0.5     # carbon use efficiency (NPP / GPP)


# ---------------------------------------------------------------------------
# Module-level pure helpers — no class state
# ---------------------------------------------------------------------------


def _gpp(sw_radiation: jnp.ndarray) -> jnp.ndarray:
    """
    Light-use-efficiency GPP estimate (gC m⁻² day⁻¹).

    .. math::

        GPP = SW_{rad} \\cdot LUE \\cdot \\bigl(1 - e^{-k \\cdot LAI}\\bigr)

    This is a placeholder; the inversion will constrain GPP via NEE obs.
    """
    return sw_radiation * _LUE * (1.0 - jnp.exp(-_K_EXT * _LAI))


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


def _step_12C_pure(
    state: EcosystemState,
    params: ModelParams,
    forcing_t: dict,
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
        ag_frac = 1.0

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
    tau = jnp.exp(params.log_tau)                                # (n_pools,)
    f_decomp = (state.C12 / tau) * ft_vec * fm_vec * ff_vec     # (n_pools,)

    # ── Carbon influx from pool-to-pool transfers ─────────────────────────
    # F_mat[i, j] = fraction of pool-i outflux going to pool j
    # influx[j]   = Σ_i  F_mat[i, j] · f_decomp[i]  =  F_mat.T @ f_decomp
    influx = F_mat.T @ f_decomp                                  # (n_pools,)

    # ── Euler update ──────────────────────────────────────────────────────
    dC12 = (influx + npp_inputs + ext_inputs - f_decomp) * dt
    C12_new = state.C12 + dC12

    return state._replace(C12=C12_new)


# ---------------------------------------------------------------------------
# EcosystemModel dataclass
# ---------------------------------------------------------------------------


@dataclass
class EcosystemModel:
    """
    One-step forward model for bulk ¹²C carbon dynamics.

    **Not frozen** — ``_step_fn`` is compiled in ``__post_init__``.

    All JAX-traced computation lives in module-level pure functions;
    the methods only supply static structural arguments.

    Attributes
    ----------
    config :
        Validated model configuration.
    params :
        Initial / current model parameters (used by ``_step_fn``).
    pool_index :
        Pool name ↔ integer index mapping.
    _step_fn :
        JIT-compiled ``(state, forcing_t) → (new_state, None)`` closure.
        Closes over ``params`` — suitable for ``jax.lax.scan``.
        Re-run ``__post_init__`` (or recreate the model) after updating
        ``params`` to recompile.
    """

    config: ModelConfig
    params: ModelParams
    pool_index: PoolIndex

    # Private fields initialised in __post_init__ — not part of __init__.
    _step_fn: Callable | None = field(default=None, init=False, repr=False)
    _pool_to_layer: jnp.ndarray | None = field(default=None, init=False, repr=False)
    _pool_mid_depths: jnp.ndarray | None = field(default=None, init=False, repr=False)
    _T_annual_mean: float | None = field(default=None, init=False, repr=False)
    _damping_depth_m: float = field(default=2.0, init=False, repr=False)
    _n_pools: int = field(default=0, init=False, repr=False)
    _n_ag_pools: int = field(default=0, init=False, repr=False)
    # External-inputs static config (resolved at construction time)
    _ext_active: bool = field(default=False, init=False, repr=False)
    _ext_source_key: str = field(default="GPP_obs", init=False, repr=False)
    _ext_is_npp: bool = field(default=False, init=False, repr=False)
    _ext_target_indices: jnp.ndarray | None = field(
        default=None, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Build the pool-to-layer mapping and JIT-compile ``_step_fn``."""
        n_pools = len(self.pool_index)
        n_ag = len(self.config.aboveground_pools)

        # ── Static pool → layer index ──────────────────────────────────
        # Aboveground pools have no soil layer; map them to layer 0
        # (surface conditions) so the environment vector has no gaps.
        ptl = np.zeros(n_pools, dtype=np.int32)
        for layer_idx, layer in enumerate(self.config.soil_layers):
            sl = self.pool_index.layer_slices[layer.name]
            ptl[sl.start : sl.stop] = layer_idx
        pool_to_layer = jnp.array(ptl)

        self._pool_to_layer = pool_to_layer

        # ── Per-pool mid-depths for freeze-thaw gating ────────────────
        # Use per-pool depth_top/bot if specified in config; otherwise fall
        # back to the enclosing layer's bounds.
        T_annual_mean = getattr(self.config, "T_annual_mean_C", None)
        if T_annual_mean is not None:
            mid_depths: list[float] = []
            n_ag_pools_depth = len(self.config.aboveground_pools)
            for _ in range(n_ag_pools_depth):
                mid_depths.append(0.0)   # aboveground pools at surface
            for layer in self.config.soil_layers:
                for pool in layer.som_pools:
                    top = pool.depth_top_m if pool.depth_top_m is not None else layer.depth_top_m
                    bot = pool.depth_bot_m if pool.depth_bot_m is not None else layer.depth_bot_m
                    mid_depths.append((top + bot) / 2.0)
            self._pool_mid_depths = jnp.array(mid_depths, dtype=jnp.float32)
            self._T_annual_mean = float(T_annual_mean)
        else:
            self._pool_mid_depths = None
            self._T_annual_mean = None

        self._n_pools = n_pools
        self._n_ag_pools = n_ag

        # ── External-inputs static config ─────────────────────────────
        ext = self.config.external_inputs
        ext_active = ext is not None and ext.enabled
        if ext_active:
            ext_source_key = ext.source
            ext_is_npp = (ext.source == "NPP_obs")
            ext_target_indices = jnp.array(
                [self.pool_index[name] for name in ext.target_pool_names],
                dtype=jnp.int32,
            )
        else:
            ext_source_key = "GPP_obs"
            ext_is_npp = False
            ext_target_indices = jnp.zeros(0, dtype=jnp.int32)

        self._ext_active = ext_active
        self._ext_source_key = ext_source_key
        self._ext_is_npp = ext_is_npp
        self._ext_target_indices = ext_target_indices

        # ── JIT-compile scan-compatible step ──────────────────────────
        # Close over current params so jax.lax.scan only needs
        # (state, forcing_t) as arguments.
        params = self.params
        dt = float(self.config.dt_days)
        pool_mid_depths = self._pool_mid_depths
        T_annual_mean_val = self._T_annual_mean

        @jax.jit
        def _compiled_step(
            state: EcosystemState,
            forcing_t: dict,
        ) -> tuple[EcosystemState, None]:
            new_state = _step_12C_pure(
                state,
                params,
                forcing_t,
                n_pools=n_pools,
                n_ag_pools=n_ag,
                pool_to_layer=pool_to_layer,
                dt=dt,
                external_inputs_active=ext_active,
                external_input_source_key=ext_source_key,
                external_input_is_npp=ext_is_npp,
                external_input_target_indices=ext_target_indices,
                pool_mid_depths=pool_mid_depths,
                T_annual_mean=T_annual_mean_val,
            )
            return new_state, None

        self._step_fn = _compiled_step

    # ------------------------------------------------------------------
    # Step functions
    # ------------------------------------------------------------------

    def step_12C(
        self,
        state: EcosystemState,
        params: ModelParams,
        forcing_t: dict,
    ) -> EcosystemState:
        """
        One Euler timestep of bulk ¹²C pool dynamics.

        Only ``state.C12`` is modified; all other state fields pass through
        unchanged.  This method is the differentiable entry point — use
        ``jax.grad`` directly on it.

        Parameters
        ----------
        state :
            Current ecosystem state.
        params :
            Model parameters (may differ from ``self.params``; traced by grad).
        forcing_t :
            Scalar/array forcing for this timestep.  Required keys:
            ``'sw_radiation'``, ``'soil_temp'`` (n_layers,),
            ``'soil_moisture'`` (n_layers,).

        Returns
        -------
        EcosystemState
            Updated state with new ``C12``.
        """
        return _step_12C_pure(
            state,
            params,
            forcing_t,
            n_pools=self._n_pools,
            n_ag_pools=self._n_ag_pools,
            pool_to_layer=self._pool_to_layer,  # type: ignore[arg-type]
            dt=float(self.config.dt_days),
            external_inputs_active=self._ext_active,
            external_input_source_key=self._ext_source_key,
            external_input_is_npp=self._ext_is_npp,
            external_input_target_indices=self._ext_target_indices,
            pool_mid_depths=self._pool_mid_depths,
            T_annual_mean=self._T_annual_mean,
            damping_depth_m=self._damping_depth_m,
        )

    def step_14C(
        self,
        state: EcosystemState,
        params: ModelParams,
        forcing_t: dict,
    ) -> EcosystemState:
        """
        One Euler timestep of ¹⁴C tracer dynamics.

        Delegates to :func:`tracer_14C.step_14C` and returns a new state
        with ``C14`` updated; all other fields pass through unchanged.

        Parameters
        ----------
        state :
            Current ecosystem state.
        params :
            Model parameters (traced by ``jax.grad``).
        forcing_t :
            Per-timestep forcing dict.  Required keys: ``'sw_radiation'``,
            ``'soil_temp'``, ``'soil_moisture'``, ``'delta14C_atm'``.

        Returns
        -------
        EcosystemState
            Updated state with new ``C14``; ``C12`` and all other fields
            are unchanged.
        """
        new_C14 = tracer_14C.step_14C(
            state, params, forcing_t, self.config, self.pool_index,
            external_inputs_active=self._ext_active,
            external_input_source_key=self._ext_source_key,
            external_input_is_npp=self._ext_is_npp,
            external_input_target_indices=self._ext_target_indices,
        )
        return state._replace(C14=new_C14)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnose(
        self,
        state: EcosystemState,
        params: ModelParams,
        forcing_t: dict,
    ) -> dict:
        """
        Compute diagnostic carbon flux scalars for the current state.

        All returned values are jnp scalars (gC m⁻² day⁻¹).
        Sign convention: **NEE > 0 = net source to atmosphere**.

        Parameters
        ----------
        state :
            Current ecosystem state.
        params :
            Model parameters.
        forcing_t :
            Forcing dict (same keys as ``step_12C``).

        Returns
        -------
        dict
            Keys: ``'GPP'``, ``'Ra'``, ``'NPP'``, ``'Rh'``, ``'ER'``,
            ``'NEE'``.
        """
        CUE = jnp.exp(params.log_CUE)

        # ── GPP — prescribed or LUE estimate ──────────────────────────────
        if self._ext_active:
            GPP_prescribed = forcing_t[self._ext_source_key]
            GPP = jnp.where(
                jnp.isnan(GPP_prescribed),
                _gpp(forcing_t["sw_radiation"]),
                GPP_prescribed,
            )
        else:
            GPP = _gpp(forcing_t["sw_radiation"])

        Ra = GPP * (1.0 - CUE)
        NPP = GPP * CUE

        ft_vec, fm_vec, ff_vec = _pool_env_vecs(
            forcing_t["soil_temp"],
            forcing_t["soil_moisture"],
            state.thawed_frac,
            params.log_Q10,
            params.log_theta_opt,
            params.log_gamma_moist,
            self._pool_to_layer,  # type: ignore[arg-type]
            pool_mid_depths=self._pool_mid_depths,
            T_annual_mean=self._T_annual_mean,
            damping_depth_m=self._damping_depth_m,
        )

        Rh = het_respiration(
            state.C12,
            params.log_tau,
            params.log_f_transfer,
            ft_vec,
            fm_vec,
            ff_vec,
            n_pools=self._n_pools,
        )

        ER = Ra + Rh
        NEE = nee_flux(GPP, Ra, Rh)

        # ── External input diagnostics ────────────────────────────────────
        # Always include these keys with static shapes for jax.lax.scan.
        n_targets = len(self._ext_target_indices)  # type: ignore[arg-type]
        if self._ext_active:
            GPP_or_NPP = GPP if not self._ext_is_npp else GPP * CUE
            ext_inputs = compute_external_soil_inputs(
                GPP_or_NPP=GPP_or_NPP,
                is_npp=self._ext_is_npp,
                log_CUE=params.log_CUE,
                log_soil_input_fraction=params.log_soil_input_fraction,
                log_external_input_partition=params.log_external_input_partition,
                n_pools=self._n_pools,
                target_pool_indices=self._ext_target_indices,  # type: ignore[arg-type]
            )
            ext_total = jnp.sum(ext_inputs)
            ext_by_pool = ext_inputs[self._ext_target_indices]  # type: ignore[index]
        else:
            ext_total = jnp.array(0.0)
            ext_by_pool = jnp.zeros(n_targets)

        return {
            "GPP": GPP,
            "Ra": Ra,
            "NPP": NPP,
            "Rh": Rh,
            "ER": ER,
            "NEE": NEE,
            "GPP_forcing_used": GPP,
            "external_C_input_total": ext_total,
            "external_C_input_by_pool": ext_by_pool,
        }
