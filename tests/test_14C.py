"""
Tests for src/ecosystem_complexity/tracer_14C.py.

Coverage
--------
step_14C
    - Output C14 shape equals input C14 shape.
    - All values finite after one step.
    - Zero state + zero GPP → C14 stays zero (no inputs or outputs).
    - Mass balance: sum(ΔC14) ≈ (F14_npp − F14_rh − decay) × dt  (< 1e-6).

compute_delta14C
    - Returns 0.0 ‰ when R_sample == R_std.
    - Returns negative ‰ for old carbon (R_sample < R_std).
    - Returns positive ‰ for bomb-spike carbon (R_sample > R_std) with
      the exact expected value.
    - Handles C12 = 0 without NaN (1e-10 floor).

spinup_14C
    - C14 is finite at the end of a short spinup run.
    - After a synthetic bomb-spike sequence the fast pool (organic_fast,
      τ ≈ 730 d) shows Δ¹⁴C > 50 ‰; the passive pool (mineral_B_passive,
      τ ≈ 36 500 d) shows Δ¹⁴C < 10 ‰.

initialize_permafrost_14C
    - The specified pool's Δ¹⁴C matches the input obs value (±1e-3 ‰).
    - Pools not listed in permafrost_obs are unchanged.
    - Multiple pools can be set in a single call.

Differentiability
    - jax.grad of sum(step_14C) w.r.t. params.log_tau is finite and
      non-zero for a warm state (C14 > 0, thawed_frac = 1).
    - jax.grad w.r.t. log_alloc is finite and non-zero when GPP > 0.

Integration
    - model.step_14C no longer raises NotImplementedError; returns updated
      EcosystemState with C14 changed and all other fields unchanged.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ecosystem_complexity.above_ground import npp_allocation
from ecosystem_complexity.climate import f_moisture, f_temp
from ecosystem_complexity.config import PoolIndex, load_config
from ecosystem_complexity.model import EcosystemModel
from ecosystem_complexity.state import (
    make_default_params,
    make_initial_state,
)
from ecosystem_complexity.tracer_14C import (
    _CUE,
    _K_EXT,
    _LAI,
    _LUE,
    _R_STD,
    compute_delta14C,
    spinup_14C,
    step_14C,
)
from ecosystem_complexity.transfer import get_transfer_matrix

CONFIGS_DIR = pathlib.Path(__file__).parent.parent / "configs"
_HARVARD_PATH = str(CONFIGS_DIR / "harvard_forest.yaml")
_BARROW_PATH = str(CONFIGS_DIR / "barrow_alaska.yaml")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def harvard_config():
    return load_config(_HARVARD_PATH)


@pytest.fixture(scope="module")
def harvard_params(harvard_config):
    return make_default_params(harvard_config)


@pytest.fixture(scope="module")
def harvard_index(harvard_config):
    return PoolIndex(harvard_config)


@pytest.fixture(scope="module")
def harvard_model(harvard_config, harvard_params, harvard_index):
    return EcosystemModel(
        config=harvard_config,
        params=harvard_params,
        pool_index=harvard_index,
    )


@pytest.fixture(scope="module")
def barrow_config():
    return load_config(_BARROW_PATH)


@pytest.fixture(scope="module")
def barrow_index(barrow_config):
    return PoolIndex(barrow_config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_warm_state(config, index, n_layers):
    """
    State with C14 = 1.0 gC m⁻² per pool and thawed_frac = 1 everywhere.

    C14 = 1.0 (far above the physical R_std scale) gives convenient
    values for mass-balance and gradient checks without changing the
    algebraic identities being tested.
    """
    n_pools = len(index)
    site_cfg = {"mat_c": 8.5, "permafrost": False}
    state = make_initial_state(config, site_cfg)
    return state._replace(
        C12=jnp.full(n_pools, 100.0),
        C14=jnp.full(n_pools, 1.0),
        thawed_frac=jnp.ones(n_layers),
    )


def _make_forcing(n_layers, *, sw=100.0, soil_temp=15.0, delta14C_atm=0.0):
    return {
        "sw_radiation": jnp.array(float(sw)),
        "soil_temp": jnp.full(n_layers, float(soil_temp)),
        "soil_moisture": jnp.full(n_layers, 0.3),
        "delta14C_atm": jnp.array(float(delta14C_atm)),
    }


def _pool_to_layer_array(config, index):
    """Rebuild pool→layer mapping for independent test-side computations."""
    n_pools = len(index)
    ptl = np.zeros(n_pools, dtype=np.int32)
    for li, layer in enumerate(config.soil_layers):
        sl = index.layer_slices[layer.name]
        ptl[sl.start : sl.stop] = li
    return jnp.array(ptl)


def _steady_state_c12(config, params, index, sw=100.0, soil_temp=15.0):
    """
    Compute the steady-state ¹²C pool sizes by solving the linear mass-balance.

    At steady state with environmental scalars = 1 (T = T_ref = 15 °C,
    θ = θ_opt, thawed_frac = 1):

        (I − F^T) · diag(1/τ) · C12 = NPP_inputs

    solved via numpy.linalg.solve.  This gives the only C12_ss for which
    ``C14_init = C12_ss * R_std`` produces Δ¹⁴C = 0 ‰ at all pools.

    Parameters
    ----------
    sw : float
        SW radiation used for the steady-state GPP (MJ m⁻² day⁻¹).
    soil_temp : float
        Soil temperature (°C).  At 15 °C == T_ref the temperature scalar
        is exactly 1.0 regardless of Q10.
    """
    n_pools = len(index)
    n_ag = len(config.aboveground_pools)

    # GPP and NPP (same formula as tracer_14C / model.py)
    GPP = float(sw) * _LUE * float(1.0 - np.exp(-_K_EXT * _LAI))
    NPP = GPP * _CUE

    # NPP allocation (softmax of log_alloc)
    alloc_np = np.array(jax.nn.softmax(params.log_alloc))
    npp_vec = np.zeros(n_pools, dtype=np.float64)
    npp_vec[:n_ag] = NPP * alloc_np

    # Transfer matrix and τ (numpy, float64 for stable solve)
    F_mat_np = np.array(
        get_transfer_matrix(params.log_f_transfer, n_pools), dtype=np.float64
    )
    tau_np = np.exp(np.array(params.log_tau, dtype=np.float64))

    # (I − F^T) · f_ss = npp_vec,  where f_ss = C12/τ
    A = np.eye(n_pools) - F_mat_np.T
    f_ss = np.linalg.solve(A, npp_vec)  # decomposition fluxes at SS
    C12_ss_np = np.maximum(tau_np * f_ss, 0.0)

    return jnp.array(C12_ss_np, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# step_14C — shape, finiteness, zero state
# ---------------------------------------------------------------------------


def test_step_14C_output_shape(harvard_config, harvard_params, harvard_index):
    """Output C14 has the same shape as input C14."""
    n_layers = len(harvard_config.soil_layers)
    state = _make_warm_state(harvard_config, harvard_index, n_layers)
    forcing = _make_forcing(n_layers)
    new_C14 = step_14C(state, harvard_params, forcing, harvard_config, harvard_index)
    assert new_C14.shape == state.C14.shape


def test_step_14C_finite(harvard_config, harvard_params, harvard_index):
    """C14 values are all finite after one step."""
    n_layers = len(harvard_config.soil_layers)
    state = _make_warm_state(harvard_config, harvard_index, n_layers)
    forcing = _make_forcing(n_layers)
    new_C14 = step_14C(state, harvard_params, forcing, harvard_config, harvard_index)
    assert jnp.all(jnp.isfinite(new_C14))


def test_step_14C_zero_state_zero_gpp_unchanged(
    harvard_config, harvard_params, harvard_index
):
    """
    Zero C14 + zero GPP → C14 stays zero.

    With C14 = 0: F14_out = 0, F14_in = 0, decay = 0.
    With sw_radiation = 0: GPP = 0 → F14_npp = 0.
    The Euler update leaves C14 identically zero.
    """
    n_layers = len(harvard_config.soil_layers)
    state = make_initial_state(harvard_config, {"mat_c": 8.5, "permafrost": False})
    forcing = _make_forcing(n_layers, sw=0.0, delta14C_atm=0.0)
    new_C14 = step_14C(state, harvard_params, forcing, harvard_config, harvard_index)
    assert jnp.allclose(new_C14, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# step_14C — mass balance
# ---------------------------------------------------------------------------


def test_step_14C_mass_balance(harvard_config, harvard_params, harvard_index):
    """
    Mass balance: sum(ΔC14) ≈ (F14_npp_sum − F14_rh − decay_sum) × dt.

    Algebraic derivation (identical to ¹²C):

        Σ_j F14_in_j = Σ_i F14_out_i · Σ_j F_ij
                     = Σ_i F14_out_i · (1 − resp_frac_i)
                     = Σ F14_out − F14_rh_total

        → Σ ΔC14 = Σ F14_npp + (Σ F14_out − F14_rh) − Σ F14_out − Σ decay
                 = Σ F14_npp − F14_rh − Σ decay

    Both sides of the identity are evaluated from independent code paths;
    any residual is pure float32 rounding error (≪ 1e-6 here).
    """
    n_pools = len(harvard_index)
    n_ag = len(harvard_config.aboveground_pools)
    n_layers = len(harvard_config.soil_layers)
    dt = float(harvard_config.dt_days)

    state = _make_warm_state(harvard_config, harvard_index, n_layers)
    forcing = _make_forcing(n_layers, delta14C_atm=100.0)

    new_C14 = step_14C(state, harvard_params, forcing, harvard_config, harvard_index)
    observed_delta = float(jnp.sum(new_C14 - state.C14))

    # ── Independent computation of expected Σ ΔC14 ────────────────────────
    pool_to_layer = _pool_to_layer_array(harvard_config, harvard_index)

    ft_vec = f_temp(forcing["soil_temp"], harvard_params.log_Q10)[pool_to_layer]
    fm_vec = f_moisture(
        forcing["soil_moisture"],
        harvard_params.log_theta_opt,
        harvard_params.log_gamma_moist,
    )[pool_to_layer]
    ff_vec = state.thawed_frac[pool_to_layer]

    GPP = forcing["sw_radiation"] * _LUE * (1.0 - jnp.exp(-_K_EXT * _LAI))
    F_npp = npp_allocation(GPP, _CUE, harvard_params.log_alloc, n_ag)
    R_atm = _R_STD * (1.0 + forcing["delta14C_atm"] / 1000.0)
    F14_npp_sum = float(R_atm * F_npp.sum())

    tau = jnp.exp(harvard_params.log_tau)
    F14_out = (state.C14 / tau) * ft_vec * fm_vec * ff_vec

    F_mat = get_transfer_matrix(harvard_params.log_f_transfer, n_pools)
    resp_frac = 1.0 - F_mat.sum(axis=-1)
    F14_rh = float(jnp.sum(resp_frac * F14_out))

    decay_sum = float(harvard_params.lambda_14C * state.C14.sum())

    expected = (F14_npp_sum - F14_rh - decay_sum) * dt
    assert abs(observed_delta - expected) < 1e-6, (
        f"¹⁴C mass balance violated: "
        f"observed Δ = {observed_delta:.6e}, expected = {expected:.6e}, "
        f"residual = {abs(observed_delta - expected):.2e}"
    )


# ---------------------------------------------------------------------------
# compute_delta14C
# ---------------------------------------------------------------------------


def test_compute_delta14C_at_standard_is_zero():
    """Returns 0.0 ‰ at every pool when R_sample == R_std."""
    n = 8
    C12 = jnp.ones(n)
    C14 = jnp.full(n, _R_STD)  # R_sample = _R_STD → Δ¹⁴C = 0 ‰
    delta = compute_delta14C(C14, C12)
    assert jnp.allclose(
        delta, 0.0, atol=1e-4
    ), f"Expected 0.0 ‰ everywhere, got {delta}"


def test_compute_delta14C_negative_for_old_carbon():
    """Δ¹⁴C < 0 and equals −500 ‰ when R_sample = 0.5 × R_std."""
    C12 = jnp.ones(4)
    C14 = jnp.full(4, 0.5 * _R_STD)
    delta = compute_delta14C(C14, C12)
    assert jnp.all(delta < 0.0), f"Expected negative Δ¹⁴C, got {delta}"
    assert jnp.allclose(
        delta, -500.0, atol=1.0
    ), f"Expected −500 ‰ for half-standard ratio, got {delta}"


def test_compute_delta14C_positive_for_bomb_carbon():
    """Δ¹⁴C = +1000 ‰ when R_sample = 2 × R_std."""
    C12 = jnp.ones(4)
    C14 = jnp.full(4, 2.0 * _R_STD)
    delta = compute_delta14C(C14, C12)
    assert jnp.all(delta > 0.0), f"Expected positive Δ¹⁴C, got {delta}"
    assert jnp.allclose(
        delta, 1000.0, atol=1.0
    ), f"Expected 1000 ‰ for double-standard ratio, got {delta}"


def test_compute_delta14C_zero_c12_safe():
    """Handles C12 = 0 without producing NaN (1e-10 floor on denominator)."""
    C12 = jnp.zeros(4)
    C14 = jnp.zeros(4)
    delta = compute_delta14C(C14, C12)
    assert jnp.all(jnp.isfinite(delta)), f"NaN/inf detected with zero C12: {delta}"


# ---------------------------------------------------------------------------
# spinup_14C
# ---------------------------------------------------------------------------


def test_spinup_14C_finite(
    harvard_model, harvard_params, harvard_config, harvard_index
):
    """C14 is finite at the end of a short spinup run (100 days)."""
    n_pools = len(harvard_index)
    n_layers = len(harvard_config.soil_layers)
    C12_ss = jnp.full(n_pools, 50.0)
    n_steps = 100

    forcing_hist = {
        "sw_radiation": jnp.full(n_steps, 80.0),
        "soil_temp": jnp.full((n_steps, n_layers), 10.0),
        "soil_moisture": jnp.full((n_steps, n_layers), 0.3),
        "delta14C_atm": jnp.zeros(n_steps),
    }
    C14_final = spinup_14C(harvard_model, C12_ss, forcing_hist, harvard_params)
    assert jnp.all(
        jnp.isfinite(C14_final)
    ), f"C14 contains NaN/inf after spinup: {C14_final}"


def test_spinup_14C_bomb_spike(
    harvard_model, harvard_params, harvard_config, harvard_index
):
    """
    After a synthetic bomb-spike sequence:

    * fast pool (organic_fast, τ ≈ 730 d)      → Δ¹⁴C > 50 ‰
    * passive pool (mineral_B_passive, τ = 100 yr) → Δ¹⁴C < 10 ‰

    Forcing sequence (daily timesteps, dt = 1 day):

        1 yr   Δ¹⁴C_atm =   0 ‰   — pre-industrial equilibration
        3 yr   Δ¹⁴C_atm = 900 ‰   — nuclear-bomb peak
        2 yr   Δ¹⁴C_atm = 100 ‰   — post-bomb modern decline
        ─────────────────────────────────────────────────────────
        6 yr   = 2 190 daily steps  (compiled once via jax.lax.scan)

    C12_ss is the *self-consistent* steady state computed from the mass-balance
    system (I − F^T) · (C12/τ) = NPP_inputs.  With this choice, the
    ``spinup_14C`` Phase-1 initialisation C14_init = C12_ss * R_std already
    represents Δ¹⁴C = 0 ‰ at every pool, so the 1-yr pre-industrial phase is
    only a numerical safeguard.

    Physical reasoning
    ------------------
    * organic_fast (τ ≈ 2 yr) equilibrates within the 3-yr spike; after
      the 2-yr post-spike at 100 ‰ it retains an elevated Δ¹⁴C (≫ 50 ‰).
    * mineral_B_passive (τ = 100 yr) is insulated by a long transfer chain
      (fast → mineral_A_fast → mineral_A_slow → passive at 1 %) and its
      own century-scale turnover, leaving it far below 10 ‰ in 6 yr.
    """
    n_layers = len(harvard_config.soil_layers)

    # Use the physically consistent steady-state C12 so that
    # C14_init = C12_ss * R_std gives Δ¹⁴C = 0 ‰ at all pools.
    C12_ss = _steady_state_c12(
        harvard_config, harvard_params, harvard_index, sw=100.0, soil_temp=15.0
    )

    n_pre = 1 * 365  # 365 days
    n_spike = 3 * 365  # 1095 days
    n_post = 2 * 365  # 730 days
    n_total = n_pre + n_spike + n_post  # 2190 days

    forcing_hist = {
        "sw_radiation": jnp.full(n_total, 100.0),
        "soil_temp": jnp.full((n_total, n_layers), 15.0),
        "soil_moisture": jnp.full((n_total, n_layers), 0.3),
        "delta14C_atm": jnp.concatenate(
            [
                jnp.zeros(n_pre),
                jnp.full(n_spike, 900.0),
                jnp.full(n_post, 100.0),
            ]
        ),
    }

    C14_final = spinup_14C(harvard_model, C12_ss, forcing_hist, harvard_params)
    assert jnp.all(
        jnp.isfinite(C14_final)
    ), "C14 contains NaN/inf after bomb-spike spinup"

    delta14C_pools = compute_delta14C(C14_final, C12_ss)
    fast_idx = harvard_index["organic_fast"]
    passive_idx = harvard_index["mineral_B_passive"]
    fast_d14C = float(delta14C_pools[fast_idx])
    passive_d14C = float(delta14C_pools[passive_idx])

    assert (
        fast_d14C > 50.0
    ), f"organic_fast Δ¹⁴C = {fast_d14C:.1f} ‰ — expected > 50 ‰ after bomb spike"
    assert passive_d14C < 10.0, (
        f"mineral_B_passive Δ¹⁴C = {passive_d14C:.3f} ‰ — "
        f"expected < 10 ‰ (τ = 100 yr, barely responds in 6 yr)"
    )


# ---------------------------------------------------------------------------
# initialize_permafrost_14C
# ---------------------------------------------------------------------------
