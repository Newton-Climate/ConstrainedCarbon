"""
Internal helpers for the Optimal Estimation inversion.

Contains:
  ObsBlock                 — self-contained observation block definition
  _build_obs_blocks        — build the OE observation vector from ObservationData
  build_oe_prior_sigma     — prior 1-sigma values for the OE state vector
  _build_sa_diag           — prior error variances for the OE state vector
  _analytical_c12_ss       — analytical steady-state C12 stocks
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .config import ModelConfig, PoolIndex
from .state import _LAMBDA_14C, ModelParams
from .tracer_14C import _R_STD, respired_delta14C
from .transfer import get_transfer_matrix

if TYPE_CHECKING:
    # Imported lazily to avoid circular imports (``data.israd_observations``
    # imports ``ObsBlock`` from here; ``api`` re-exports from here).  These names
    # are used only in annotations, which are strings under
    # ``from __future__ import annotations``.
    from .api import ModelOutput
    from .data.schemas import ObservationData
    from .model import EcosystemModel


@dataclass
class ObsBlock:
    """
    Self-contained observation block for Optimal Estimation.

    Each block represents one logical observation type (e.g. pool Δ¹⁴C,
    respired Δ¹⁴C, carbon stocks, annual ER flux).  Blocks are assembled
    into the full OE observation vector by simple concatenation.

    Fields
    ------
    name : str
        Human-readable label used in log messages and diagnostics.
    y : jnp.ndarray  shape (n_i,)
        Observed values.
    Se : jnp.ndarray  shape (n_i,)
        Diagonal observation-error variances σ² (same length as y).
    predict : Callable[[ModelOutput, ModelParams], jnp.ndarray]
        Pure-JAX function that takes the full model output and current
        parameters and returns the simulated counterpart of y, shape (n_i,).
        Must be differentiable via jax.jacobian.
    """

    name: str
    y: jnp.ndarray
    Se: jnp.ndarray
    predict: Callable[..., jnp.ndarray]  # (ModelOutput, ModelParams) -> (n_i,)


def _build_obs_blocks(  # noqa: C901
    observations: ObservationData,
    model: EcosystemModel,
    sigma_pool: float,
    sigma_resp: float,
    sigma_carbon: Optional[float] = None,
    f_hetero: float = 0.0,
    sigma_er_frac: float = 0.15,
) -> list[ObsBlock]:
    """
    Build the list of ObsBlock objects that together define the OE observation
    vector.  Each block is independent: it owns its observed values, error
    variances, and a JAX-differentiable ``predict`` callable.

    Adding a new observation type means appending one more ObsBlock here;
    ``optimize_oe`` and ``_forward`` need no changes.

    Current blocks (appended in order; each is skipped if it has zero obs):
      1. pool_14C    — pool-level Δ¹⁴C at sparse (time, pool) pairs
      2. resp_14C    — flux-weighted respired Δ¹⁴C at sparse time indices
      3. c_stock     — time-mean carbon stocks, one entry per constrained pool
      3b. c_sum      — time-mean TOTAL column carbon stock Σ_i C12_i; the correct
                       stock constraint when pools are co-located kinetic
                       fractions (per-pool stock is then unobservable)
      4. er_annual   — annual mean ecosystem respiration from FluxNet ER;
                       model prediction is Rh_sim / sigmoid(log_f_hetero)

    Parameters
    ----------
    observations : ObservationData
    model        : EcosystemModel
    sigma_pool   : float   Δ¹⁴C obs error [‰] for pool-level obs
    sigma_resp   : float   Δ¹⁴C obs error [‰] for respired CO₂ obs
    sigma_carbon : float   fallback C-stock obs error [gC m⁻²] (used when
                           the C_pools_obs tuple sigma is zero/None)
    f_hetero     : float   prior f_hetero > 0 enables the ER block;
                           actual value used only as on/off flag — the
                           posterior value comes from p.log_f_hetero
    sigma_er_frac: float   fractional σ on annual ER observations

    Returns
    -------
    list[ObsBlock]
        Non-empty blocks only.  Concatenate .y and .Se to get the full
        OE vectors; call b.predict(out, p) for each block to build F(x).
    """
    pool_names_set = set(model.pool_index.pool_names)
    blocks: list[ObsBlock] = []

    # ── Block 1: pool-level Δ¹⁴C ────────────────────────────────────────────
    t_p, col_p, y_p = [], [], []
    for pool_name in sorted(observations.delta14C_obs.keys()):
        if pool_name not in pool_names_set:
            continue
        obs_arr = np.array(observations.delta14C_obs[pool_name])
        valid = np.where(np.isfinite(obs_arr))[0]
        pcol = model.pool_index[pool_name]
        for t in valid:
            t_p.append(int(t))
            col_p.append(pcol)
            y_p.append(float(obs_arr[t]))

    if t_p:
        _t = jnp.array(t_p, dtype=jnp.int32)
        _col = jnp.array(col_p, dtype=jnp.int32)
        blocks.append(
            ObsBlock(
                name="pool_14C",
                y=jnp.array(y_p, dtype=jnp.float32),
                Se=jnp.full(len(y_p), sigma_pool**2),
                predict=lambda out, p, t=_t, col=_col: out.delta14C[t, col],
            )
        )

    # ── Block 2: respired CO₂ Δ¹⁴C ─────────────────────────────────────────
    t_r, y_r = [], []
    if observations.delta14C_resp is not None:
        resp_np = np.array(observations.delta14C_resp)
        for t in np.where(np.isfinite(resp_np))[0]:
            t_r.append(int(t))
            y_r.append(float(resp_np[t]))

    if t_r:
        _t_r = jnp.array(t_r, dtype=jnp.int32)

        def _predict_resp(
            out: ModelOutput, p: ModelParams, t_r: jnp.ndarray = _t_r
        ) -> jnp.ndarray:
            d14c = respired_delta14C(
                out.delta14C, out.Rh_by_pool, out.C12,
                p.log_tau, p.log_f_transfer, out.C12.shape[-1],
            )  # (T,)
            return d14c[t_r]

        blocks.append(
            ObsBlock(
                name="resp_14C",
                y=jnp.array(y_r, dtype=jnp.float32),
                Se=jnp.full(len(y_r), sigma_resp**2),
                predict=_predict_resp,
            )
        )

    # ── Block 3: carbon stocks ───────────────────────────────────────────────
    c_col, y_c, se_c = [], [], []
    for pool_name, (c_mean, c_sigma) in (observations.C_pools_obs or {}).items():
        if pool_name not in pool_names_set:
            continue
        sigma_c = (
            float(c_sigma) if (c_sigma and c_sigma > 0) else (sigma_carbon or 1000.0)
        )
        c_col.append(model.pool_index[pool_name])
        y_c.append(float(c_mean))
        se_c.append(sigma_c**2)

    if c_col:
        _c_col = jnp.array(c_col, dtype=jnp.int32)
        blocks.append(
            ObsBlock(
                name="c_stock",
                y=jnp.array(y_c, dtype=jnp.float32),
                Se=jnp.array(se_c, dtype=jnp.float32),
                predict=lambda out, p, col=_c_col: jnp.mean(out.C12, axis=0)[col],
            )
        )

    # ── Block 3b: total column carbon stock (Σ_i C12_i) ─────────────────────
    # The counterpart to Block 3 for co-located kinetic pools: when the pools are
    # density/kinetic fractions rather than depth horizons, a measured SOC stock
    # cannot be attributed to one pool by depth — only the column total is
    # observable. Given a known input I, this constrains the C-mass-weighted mean
    # turnover (Σ C_i = I·⟨τ⟩) rather than any individual τ_i.
    if observations.C_total_obs is not None:
        c_total_mean, c_total_sigma = observations.C_total_obs
        sigma_tot = (
            float(c_total_sigma)
            if (c_total_sigma and c_total_sigma > 0)
            else (sigma_carbon or 1000.0)
        )
        blocks.append(
            ObsBlock(
                name="c_sum",
                y=jnp.array([float(c_total_mean)], dtype=jnp.float32),
                Se=jnp.array([sigma_tot**2], dtype=jnp.float32),
                predict=lambda out, p: jnp.sum(
                    jnp.mean(out.C12, axis=0), keepdims=True
                ),
            )
        )

    # ── Block 4: annual ER from FluxNet (model predicts ER = Rh / f_hetero) ─
    if f_hetero > 0.0 and observations.ER is not None:
        T = len(np.array(observations.time))
        er_np = np.array(observations.ER, dtype=np.float64)
        time_np = np.array(observations.time, dtype=np.float64)
        years_np = 1970.0 + time_np / 365.25
        yr_start = int(np.floor(years_np[0]))
        yr_end = int(np.floor(years_np[-1]))

        rows: list[np.ndarray] = []
        y_er: list[float] = []
        se_er: list[float] = []

        for yr in range(yr_start, yr_end + 1):
            mask = (years_np >= yr) & (years_np < yr + 1) & np.isfinite(er_np)
            if mask.sum() < 30:
                continue
            er_est = float(np.mean(er_np[mask]))
            sigma = max(abs(er_est) * sigma_er_frac, 0.01)
            row = np.zeros(T, dtype=np.float32)
            row[mask] = 1.0 / mask.sum()
            rows.append(row)
            y_er.append(er_est)
            se_er.append(sigma**2)

        if rows:
            _W = jnp.array(np.stack(rows, axis=0))  # (n_er_obs, T)

            def _predict_er(
                out: ModelOutput, p: ModelParams, W: jnp.ndarray = _W
            ) -> jnp.ndarray:
                f_het = jax.nn.sigmoid(p.log_f_hetero)
                return (W @ out.Rh) / (f_het + 1e-6)

            blocks.append(
                ObsBlock(
                    name="er_annual",
                    y=jnp.array(y_er, dtype=jnp.float32),
                    Se=jnp.array(se_er, dtype=jnp.float32),
                    predict=_predict_er,
                )
            )

    return blocks


def _build_sa_diag(
    config: ModelConfig,
    params0: ModelParams,
    opt_fields: tuple[str, ...],
) -> jnp.ndarray:
    """
    Build the diagonal of Sₐ (prior error variances) for the OE state vector.

    log_tau[i]             : σ = tau_prior_std[i] / tau_prior_days[i]  (log-space)
    log_f_transfer[i,j]    : σ = 0.5 for real transfer rules, 0.02 for structural zeros
    log_external_input_partition : σ = 0.30  (moderate; let Δ¹⁴C inform partition)
    everything else        : σ = 0.5
    """
    sigma = build_oe_prior_sigma(config, params0, opt_fields)
    return sigma**2


def build_oe_prior_sigma(
    config: ModelConfig,
    params0: ModelParams,
    opt_fields: tuple[str, ...],
) -> jnp.ndarray:
    """Build the diagonal 1-sigma vector for the OE state vector.

    This is the sigma-space counterpart to ``_build_sa_diag`` and is the
    intended source of truth for any analysis that should match the OE prior.
    """
    pool_idx = PoolIndex(config)
    n_pools = len(pool_idx)

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

    sigma_parts = []
    for f in opt_fields:
        val = getattr(params0, f)

        if f == "log_tau":
            sigma = np.array(
                [
                    tau_info.get(name, (1000.0, 1000.0))[1]
                    / max(tau_info.get(name, (1000.0, 1000.0))[0], 1.0)
                    for name in pool_idx.pool_names
                ],
                dtype=np.float32,
            )
            sigma_parts.append(jnp.array(sigma))

        elif f == "log_f_transfer":
            n = int(math.prod(val[:, :-1].shape))
            sigma = np.full(n, 0.02, dtype=np.float32)
            for si, dj in real_transfer_pairs:
                flat_i = si * n_pools + dj
                sigma[flat_i] = 0.5
            sigma_parts.append(jnp.array(sigma))

        elif f == "log_external_input_partition":
            n = int(math.prod(val.shape))
            sigma_parts.append(jnp.full(n, 0.30))

        elif f == "log_f_hetero":
            n = int(math.prod(val.shape))
            # f_hetero prior ≈ 0.55, σ_f ≈ 0.08 (absolute).
            # In logit-space: σ_logit = σ_f / (f(1-f)) = 0.08 / (0.55×0.45) ≈ 0.323.
            sigma_parts.append(jnp.full(n, 0.323))

        else:
            n = int(math.prod(val.shape))
            sigma_parts.append(jnp.full(n, 0.5))

    return jnp.concatenate(sigma_parts)


def _analytical_c12_ss(
    params: ModelParams,
    n_pools: int,
    mean_input: float,
    mean_modifier: float = 1.0,
    target_indices: Optional[list[int]] = None,
) -> jnp.ndarray:
    """
    Compute analytical steady-state C12 stocks for a general pool system with
    both direct external inputs and cascade transfers.

    At true steady state for pool i:
        dC_i/dt = I_i_eff - (C_i / τ_i) × modifier = 0
        → C_i = I_i_eff × τ_i / modifier

    where ``I_i_eff`` is the total effective input to pool i:
        I_i_eff = f_partition[i] × mean_input      (direct external)
                + Σ_j F[j,i] × I_j_eff            (cascade from upstream)

    This forms a lower-triangular system solved by forward substitution.

    ``modifier`` is the climatological mean of (f_T × f_moisture × f_freeze).

    Parameters
    ----------
    params :
        Current ModelParams.  ``log_tau``, ``log_f_transfer``, and (if present)
        ``log_external_input_partition`` are used.
    n_pools :
        Number of carbon pools.
    mean_input :
        Long-term mean total external carbon input [gC m⁻² day⁻¹].
    mean_modifier : float, optional
        Climatological mean decomposition scalar.  Default 1.0.
    target_indices : list of int, optional
        Pool indices that receive direct external input, in the same order as
        ``log_external_input_partition``.  Required when the partition covers
        fewer pools than n_pools (e.g. 2-way softmax over active+slow only
        when n_pools=3).  If None, defaults to the first n_lep pools.

    Returns
    -------
    jnp.ndarray
        Shape ``(n_pools,)`` steady-state C12 stocks [gC m⁻²].
    """
    tau = jnp.exp(params.log_tau)  # (n_pools,)
    F = get_transfer_matrix(params.log_f_transfer, n_pools)  # (n_pools, n_pools)

    # External input partition: softmax over target pools only.
    lep = params.log_external_input_partition
    n_lep = lep.shape[0]

    if n_lep == 0:
        # No partition at all — all input to pool 0 (legacy fallback)
        f_part = jnp.zeros(n_pools).at[0].set(1.0)
    else:
        f_soft = jax.nn.softmax(lep)  # (n_lep,) summing to 1
        if n_lep == n_pools and target_indices is None:
            # Simple case: one logit per pool
            f_part = f_soft
        else:
            # Sparse case: map n_lep fractions to specific pool indices
            indices = (
                target_indices if target_indices is not None else list(range(n_lep))
            )
            f_part = jnp.zeros(n_pools)
            for k, ti in enumerate(indices):
                f_part = f_part.at[ti].set(f_soft[k])

    # Effective inputs: I_direct + cascade from upstream pools.
    # For a lower-triangular cascade (F[i,j]=0 if j<=i) this is solvable by
    # forward substitution in pool order.
    inp = f_part * float(mean_input)  # direct external to each pool (n_pools,)
    for j in range(1, n_pools):
        # Add cascade inflows from all upstream pools i < j
        inp = inp.at[j].add(jnp.dot(F[:j, j], inp[:j]))

    # Correct for decomposition modifier (true SS has longer effective τ)
    return inp * tau / float(mean_modifier)


def apply_ss_c12(state, c12_ss: jnp.ndarray):
    """Swap ``state.C12`` for the steady-state stocks, preserving initial Δ¹⁴C.

    .. warning::

       Preserving the observed ratio pins each pool's initial Δ¹⁴C independently
       of τ. If a pool-Δ¹⁴C observation is dated at the start of the forcing
       record *and* was used to seed that pool, the model reproduces it exactly
       and the observation becomes self-predicting — zero residual, zero
       sensitivity, zero information. The OE forward models therefore use
       ``apply_ss_c12_c14`` with ``analytical_c14_ss`` instead. This function
       remains for callers that genuinely want the observed initial Δ¹⁴C held
       fixed (e.g. plotting a trajectory from a known state).

    ``state`` carries an initial C12/C14 pair whose ratio C14/C12 encodes the
    intended initial Δ¹⁴C (``= R_std·(1 + Δ¹⁴C/1000)``).  When the analytical
    steady-state C12 replaces the observed C12 we must rescale C14 by the same
    factor, otherwise the *implied* initial Δ¹⁴C silently shifts by
    ``C12_old/C12_ss`` — and, because ``c12_ss`` depends on τ, that shift moves
    with the optimizer and leaks a spurious sensitivity into the Jacobian.

    Preserving the ratio keeps each pool's initial Δ¹⁴C fixed at its observed
    value independent of τ.  Pools with zero initial C12 (hence undefined ratio)
    fall back to zero C14, matching the untouched-state behaviour.
    """
    ratio = state.C14 / (state.C12 + 1e-30)  # = R_std·(1 + Δ¹⁴C/1000), constant
    c14_ss = ratio * c12_ss
    return state._replace(C12=c12_ss, C14=c14_ss)

# ── ¹⁴C initial condition from the parameters ────────────────────────────────

def analytical_c14_ss(
    params: ModelParams,
    c12_ss: jnp.ndarray,
    n_pools: int,
    mean_input: float,
    mean_modifier: float,
    atm_years: np.ndarray,
    atm_delta14C: np.ndarray,
    t0_year: float,
    target_indices: Optional[list[int]] = None,
    spinup_start_year: float = 1850.0,
    dt_years: float = 1.0,
) -> jnp.ndarray:
    """¹⁴C stocks at ``t0_year``, derived from the parameters — not from data.

    Seeding a pool's initial Δ¹⁴C from the very observation the inversion then
    fits makes that observation self-predicting: the model reproduces it exactly
    at t₀, the residual is zero by construction, and ∂prediction/∂τ vanishes. At
    UMBS that drove the three density-fraction rows of the prewhitened Jacobian
    to ~1e-3 against ~0.5 for the stock and bulk rows, so the fraction family
    contributed 0.002 DFS instead of the ~2.0 it is worth. This function removes
    that circularity by making the initial condition a *consequence* of τ, which
    is what makes radiocarbon informative about turnover in the first place.

    At fixed ``c12_ss`` the ¹⁴C system is linear in ``C14``::

        dC14/dt = b·R_atm(t) + Fᵀ·diag(k)·C14 − (diag(k) + λ)·C14
                = b·R_atm(t) − A·C14,     A = diag(k) + λI − Fᵀ·diag(k)

    where ``k_i = mean_modifier / τ_i`` and ``b_i`` is pool i's *direct*
    external input (cascades are carried by ``F``, so unlike
    ``_analytical_c12_ss`` the input here must not be pre-cascaded).

    ``A`` is time-invariant, so with the atmosphere held piecewise constant over
    each step the solution is exact::

        C14_{n+1} = C14*_n + E·(C14_n − C14*_n),
        E = exp(−A·Δt),   C14*_n = A⁻¹·b·R_atm(t_n)

    One ``expm`` is needed regardless of the number of steps, which keeps this
    cheap enough to sit inside the differentiated forward model. Integration
    starts from the pre-industrial equilibrium ``A⁻¹·b·R_atm(spinup_start_year)``
    — by 1850 a passive pool has long forgotten any earlier transient, and the
    bomb spike (the part that actually carries turnover information) is fully
    resolved.
    """
    tau = jnp.exp(params.log_tau)
    F = get_transfer_matrix(params.log_f_transfer, n_pools)

    lep = params.log_external_input_partition
    n_lep = lep.shape[0]
    if n_lep == 0:
        f_part = jnp.zeros(n_pools).at[0].set(1.0)
    else:
        w = jax.nn.softmax(lep)
        idx = target_indices if target_indices is not None else list(range(n_lep))
        f_part = jnp.zeros(n_pools).at[jnp.asarray(idx)].set(w)

    k = mean_modifier / tau                       # (n_pools,) day⁻¹
    b = f_part * mean_input                       # direct input, gC m⁻² day⁻¹
    A = jnp.diag(k + _LAMBDA_14C) - F.T * k[None, :]

    dt_days = dt_years * 365.25
    E = jax.scipy.linalg.expm(-A * dt_days)

    years = np.arange(spinup_start_year, t0_year + 1e-9, dt_years, dtype=np.float64)
    r_atm = _R_STD * (
        1.0 + np.interp(years, atm_years, atm_delta14C) / 1000.0
    )
    r_atm_j = jnp.asarray(r_atm, dtype=jnp.float32)

    c14_0 = jnp.linalg.solve(A, b * r_atm_j[0])

    def _step(c14, r):
        c_star = jnp.linalg.solve(A, b * r)
        return c_star + E @ (c14 - c_star), None

    c14_end, _ = jax.lax.scan(_step, c14_0, r_atm_j[1:])
    return jnp.maximum(c14_end, 0.0)


def apply_ss_c12_c14(state, c12_ss: jnp.ndarray, c14_ss: jnp.ndarray):
    """Set both ¹²C and ¹⁴C from the parameters (see ``analytical_c14_ss``).

    The counterpart to ``apply_ss_c12``, which preserves the *observed* initial
    Δ¹⁴C ratio and therefore holds it fixed with respect to τ. Use this when the
    ¹⁴C initial condition should follow the parameters instead.
    """
    return state._replace(C12=c12_ss, C14=c14_ss)

@functools.lru_cache(maxsize=4)
def _atm14c_record(hemisphere: str) -> tuple[np.ndarray, np.ndarray]:
    """Spliced IntCal20/Graven/Hua daily Δ¹⁴C record (cached; reads 3 CSVs)."""
    from .data.parsers_14C import load_full_14C_record
    from .data.paths import GRAVEN_PATH, HUA_PATH, INTCAL_PATH

    years, d14c = load_full_14C_record(
        HUA_PATH, GRAVEN_PATH, INTCAL_PATH, hemisphere, 1500.0, 2023.0
    )
    return np.asarray(years, dtype=np.float64), np.asarray(d14c, dtype=np.float64)


def prepare_c14_spinup(
    forcing, hemisphere: str = "NH"
) -> tuple[np.ndarray, np.ndarray, float]:
    """Constants for analytical_c14_ss: (atm_years, atm_delta14C, t0_year).

    Hoisted out of the traced forward model — these depend only on the forcing
    window, never on the parameters, so they are closure constants.
    t0_year is the first year of the forcing record, i.e. the moment the
    spinup must hand over to the simulated period.
    """
    years, d14c = _atm14c_record(hemisphere)
    t0_year = float(1970.0 + float(np.asarray(forcing.time)[0]) / 365.25)
    return years, d14c, t0_year

