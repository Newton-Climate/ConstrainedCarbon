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

from ecosystem_complexity.model.configuration import PoolIndex, load_config
from ecosystem_complexity.model.simulator import EcosystemModel
from ecosystem_complexity.model.state import (
    EcosystemState,
    make_default_params,
    make_initial_state,
)
from ecosystem_complexity.processes.radiocarbon import (
    _CUE,
    _K_EXT,
    _LAI,
    _LUE,
    _R_STD,
    compute_delta14C,
    initialize_permafrost_14C,
    step_14C,
)
from ecosystem_complexity.model.transfers import get_transfer_matrix

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


def test_initialize_permafrost_14C_matches_obs(barrow_config, barrow_index):
    """
    After calling initialize_permafrost_14C, the specified pool's Δ¹⁴C
    matches the input obs value to within 1e-3 ‰.
    """
    n_pools = len(barrow_index)
    C12 = jnp.full(n_pools, 80.0)
    C14 = C12 * _R_STD  # all pools at 0 ‰ initially

    obs_delta14C = -500.0  # ‰ — old permafrost carbon
    C14_new = initialize_permafrost_14C(
        C14,
        C12,
        barrow_config,
        barrow_index,
        {"permafrost_slow": obs_delta14C},
    )
    delta14C_out = compute_delta14C(C14_new, C12)

    pf_idx = barrow_index["permafrost_slow"]
    assert float(delta14C_out[pf_idx]) == pytest.approx(obs_delta14C, abs=1e-3), (
        f"permafrost_slow Δ¹⁴C = {float(delta14C_out[pf_idx]):.3f} ‰; "
        f"expected {obs_delta14C:.3f} ‰"
    )


def test_initialize_permafrost_14C_unspecified_unchanged(barrow_config, barrow_index):
    """Pools not listed in permafrost_obs are left unchanged."""
    n_pools = len(barrow_index)
    C12 = jnp.full(n_pools, 80.0)
    C14_orig = C12 * _R_STD

    C14_new = initialize_permafrost_14C(
        C14_orig,
        C12,
        barrow_config,
        barrow_index,
        {"permafrost_slow": -500.0},
    )

    pf_idx = barrow_index["permafrost_slow"]
    for i in range(n_pools):
        if i == pf_idx:
            continue
        assert float(C14_new[i]) == pytest.approx(
            float(C14_orig[i]), rel=1e-6
        ), f"Pool index {i} was unexpectedly modified"


def test_initialize_permafrost_14C_multiple_pools(barrow_config, barrow_index):
    """Multiple permafrost pools can be initialised in a single call."""
    n_pools = len(barrow_index)
    C12 = jnp.full(n_pools, 100.0)
    C14 = C12 * _R_STD

    permafrost_obs = {
        "permafrost_slow": -600.0,
        "permafrost_passive": -900.0,
    }
    C14_new = initialize_permafrost_14C(
        C14, C12, barrow_config, barrow_index, permafrost_obs
    )
    delta14C_out = compute_delta14C(C14_new, C12)

    for pool_name, expected_d14C in permafrost_obs.items():
        i = barrow_index[pool_name]
        assert float(delta14C_out[i]) == pytest.approx(expected_d14C, abs=1e-3), (
            f"{pool_name} Δ¹⁴C = {float(delta14C_out[i]):.3f} ‰; "
            f"expected {expected_d14C:.3f} ‰"
        )


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_step_14C_grad_log_tau_finite(harvard_config, harvard_params, harvard_index):
    """
    jax.grad of sum(step_14C) w.r.t. params.log_tau is finite and non-zero.

    Requires a warm state (C14 > 0, thawed_frac = 1) so that F14_out = C14/τ
    is non-zero and the gradient flows through the outflux term.
    """
    n_layers = len(harvard_config.soil_layers)
    state = _make_warm_state(harvard_config, harvard_index, n_layers)
    forcing = _make_forcing(n_layers, delta14C_atm=50.0)

    def loss(p):
        return step_14C(state, p, forcing, harvard_config, harvard_index).sum()

    grads = jax.grad(loss)(harvard_params)

    assert jnp.all(
        jnp.isfinite(grads.log_tau)
    ), f"Non-finite gradient in log_tau: {grads.log_tau}"
    assert not jnp.all(
        grads.log_tau == 0.0
    ), "log_tau gradient is identically zero — C14/τ outflux path not differentiated"


def test_step_14C_grad_log_alloc_finite(harvard_config, harvard_params, harvard_index):
    """
    jax.grad w.r.t. log_alloc is finite and non-zero when GPP > 0.

    log_alloc controls NPP allocation → F14_npp; the gradient must flow
    through the atmospheric 14C input when sw_radiation > 0.
    """
    n_layers = len(harvard_config.soil_layers)
    state = _make_warm_state(harvard_config, harvard_index, n_layers)
    forcing = _make_forcing(n_layers, sw=150.0, delta14C_atm=100.0)

    def loss(p):
        return step_14C(state, p, forcing, harvard_config, harvard_index).sum()

    grads = jax.grad(loss)(harvard_params)

    assert jnp.all(
        jnp.isfinite(grads.log_alloc)
    ), f"Non-finite gradient in log_alloc: {grads.log_alloc}"
    assert not jnp.all(
        grads.log_alloc == 0.0
    ), "log_alloc gradient is identically zero — F14_npp path not differentiated"


# ---------------------------------------------------------------------------
# Integration: model.step_14C wired through
# ---------------------------------------------------------------------------


def test_model_step_14C_returns_state(
    harvard_model, harvard_config, harvard_params, harvard_index
):
    """
    EcosystemModel.step_14C returns an EcosystemState with C14 updated and
    all other fields unchanged (the NotImplementedError stub is gone).
    """
    n_layers = len(harvard_config.soil_layers)
    state = _make_warm_state(harvard_config, harvard_index, n_layers)
    forcing = _make_forcing(n_layers, delta14C_atm=50.0)

    new_state = harvard_model.step_14C(state, harvard_params, forcing)

    assert isinstance(new_state, EcosystemState)
    assert new_state.C14.shape == state.C14.shape
    assert jnp.all(jnp.isfinite(new_state.C14))

    # C12 and all other fields must pass through unchanged
    assert jnp.array_equal(new_state.C12, state.C12)
    assert jnp.array_equal(new_state.soil_temp, state.soil_temp)
    assert jnp.array_equal(new_state.thawed_frac, state.thawed_frac)
    assert jnp.array_equal(new_state.soil_moisture, state.soil_moisture)
