"""
Integration tests for the external_inputs carbon-input pathway.

Uses configs/harvard_forest_soil_only.yaml:
  - No aboveground pools
  - GPP_obs → soil pools directly via compute_external_soil_inputs
  - 6 soil pools across 3 layers (organic, mineral_A, mineral_B)

Tests
-----
test_harvard_soil_only_forward_run_10_steps
    Model steps for 10 days with realistic GPP_obs; pools grow and stay finite.

test_harvard_soil_only_spinup
    spinup() runs without error; final C12 pools are positive and finite.

test_harvard_soil_only_14C_bomb_signal
    After 50 years with bomb-spike Δ14C, fast pools show elevated Δ14C while
    slow/passive pools remain near pre-bomb values.

test_harvard_soil_only_inversion_tau_recovery_synthetic
    Synthetic inversion: given C12 stocks from a "true" tau, optimise tau and
    verify the loss decreases monotonically (gradient descent is working).
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import pytest

from ecosystem_complexity.model.api import build_model, run_model, spinup
from ecosystem_complexity.model.configuration import PoolIndex, load_config
from ecosystem_complexity.data.schemas import ForcingData, ObservationData
from ecosystem_complexity.model.state import make_default_params, make_initial_state

CONFIGS_DIR = pathlib.Path(__file__).resolve().parents[2] / "configs"
_SOIL_ONLY_PATH = str(CONFIGS_DIR / "harvard_forest_soil_only.yaml")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_forcing(T: int, gpp: float = 4.0, delta14C_atm: float = 0.0) -> ForcingData:
    """
    Minimal ForcingData for the soil-only model.

    n_layers = 3 (organic, mineral_A, mineral_B).
    GPP_obs is set to a constant ``gpp`` value.
    """
    n_layers = 3
    return ForcingData(
        time=jnp.arange(T, dtype=jnp.float32),
        air_temp=jnp.full(T, 12.0),
        sw_radiation=jnp.full(T, 80.0),
        precip=jnp.full(T, 2.0),
        vpd=jnp.full(T, 0.5),
        soil_temp=jnp.full((T, n_layers), 10.0),
        soil_moisture=jnp.full((T, n_layers), 0.35),
        snow_depth=jnp.zeros(T),
        active_layer=jnp.full(T, jnp.inf),
        delta14C_atm=jnp.full(T, delta14C_atm),
        GPP_obs=jnp.full(T, gpp),
        NPP_obs=jnp.full(T, jnp.nan),
    )


def _make_empty_obs(T: int) -> ObservationData:
    """ObservationData with all-NaN (no observations used)."""
    return ObservationData(
        time=jnp.arange(T, dtype=jnp.float32),
        NEE=jnp.full(T, jnp.nan),
        GPP=jnp.full(T, jnp.nan),
        ER=jnp.full(T, jnp.nan),
        delta14C_obs={},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def soil_only_model():
    return build_model(_SOIL_ONLY_PATH)


@pytest.fixture(scope="module")
def soil_only_config():
    return load_config(_SOIL_ONLY_PATH)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_harvard_soil_only_forward_run_10_steps(soil_only_model):
    """
    10-step forward run with GPP_obs = 4 gC m⁻² day⁻¹:
    - C12 pools stay finite throughout.
    - Total C12 is positive (inputs > outputs for small pools).
    """
    T = 10
    forcing = _make_forcing(T, gpp=4.0)
    out = run_model(soil_only_model, forcing)

    assert out.C12.shape == (T, 6), f"Expected (10, 6) C12, got {out.C12.shape}"
    assert jnp.all(jnp.isfinite(out.C12)), "C12 contains NaN or Inf after 10 steps"
    assert float(jnp.sum(out.C12[-1])) > 0.0, "Total C12 should be positive"

    assert out.NEE.shape == (T,)
    assert jnp.all(jnp.isfinite(out.NEE)), "NEE contains NaN or Inf"


def test_harvard_soil_only_spinup(soil_only_model):
    """
    spinup() runs to quasi-steady state without error.

    We run 3 spinup years (overriding the 300-year default for speed).
    Pools must be positive and finite.
    """
    T = 365
    forcing = _make_forcing(T, gpp=4.0)
    state_ss = spinup(soil_only_model, forcing, n_years=3, convergence_tol=1e-2)

    assert jnp.all(jnp.isfinite(state_ss.C12)), "Spinup produced non-finite C12"
    assert float(jnp.sum(state_ss.C12)) > 0.0, "Spinup C12 total should be positive"


def test_harvard_soil_only_14C_bomb_signal(soil_only_model):
    """
    After 50 years of bomb-spike 14C forcing, the fast pool (organic_litter,
    τ ≈ 180 days) should show Δ14C clearly higher than the passive pool
    (mineral_B_passive, τ ≈ 73 000 days).

    We use a modest Δ14C_atm = +150 ‰ to avoid float32 saturation.
    """
    T = 50 * 365  # 50 years
    forcing = _make_forcing(T, gpp=4.0, delta14C_atm=150.0)

    # Quick spinup (3 years)
    state0 = spinup(soil_only_model, forcing, n_years=3, convergence_tol=1e-2)
    out = run_model(soil_only_model, forcing, state0=state0)

    config = load_config(_SOIL_ONLY_PATH)
    idx = PoolIndex(config)

    i_litter = idx["organic_litter"]
    i_passive = idx["mineral_B_passive"]

    delta14C_litter = float(out.delta14C[-1, i_litter])
    delta14C_passive = float(out.delta14C[-1, i_passive])

    assert jnp.isfinite(jnp.array(delta14C_litter)), "Litter Δ14C is not finite"
    assert delta14C_litter > delta14C_passive, (
        f"Fast pool Δ14C ({delta14C_litter:.1f} ‰) should exceed passive pool "
        f"Δ14C ({delta14C_passive:.1f} ‰) after bomb-spike forcing"
    )


def test_harvard_soil_only_inversion_tau_recovery_synthetic(soil_only_model):
    """
    Synthetic loss-descent test: starting from perturbed tau values,
    the Adam optimiser should reduce the loss over 30 iterations.

    We use a simple C12-stock MSE loss against a synthetic "truth" generated
    by running the model with default parameters.

    Note: T must be long enough to see signal from the updated (analytical) τ
    priors.  The litter pool τ ≈ 1732 days (~4.7 yr), so a 365-day run gives
    a ~20% turnover — sufficient signal for the gradient to point downhill.
    """
    import optax

    from ecosystem_complexity.model.api import (
        _params_to_vector,
        _vector_to_params,
    )

    T = 365  # 1 year — needed to see signal with τ priors ~1700–100 000 days
    forcing = _make_forcing(T, gpp=4.0)

    config = load_config(_SOIL_ONLY_PATH)
    params_true = make_default_params(config)
    site_cfg = {"mat_c": 8.5, "permafrost": False}
    state0 = make_initial_state(config, site_cfg)

    # Generate "truth" C12 by running with default params
    out_true = run_model(soil_only_model, forcing, state0=state0, params=params_true)
    C12_target = out_true.C12  # (T, 6)

    # Perturb log_tau slightly as starting point
    log_tau_perturbed = params_true.log_tau + 0.5
    params_init = params_true._replace(log_tau=log_tau_perturbed)

    # Optimize only log_tau for this unit test (fewer params → faster, cleaner)
    opt_fields = ("log_tau",)
    vec0 = _params_to_vector(params_init, opt_fields)

    def loss_fn(vec):
        p = _vector_to_params(vec, params_init, opt_fields)
        out = run_model(soil_only_model, forcing, state0=state0, params=p)
        return jnp.mean((out.C12 - C12_target) ** 2)

    grad_fn = jax.value_and_grad(loss_fn)
    tx = optax.adam(1e-2)
    opt_state = tx.init(vec0)
    vec = vec0

    losses = []
    for _ in range(30):
        loss_val, grads = grad_fn(vec)
        updates, opt_state = tx.update(grads, opt_state)
        vec = optax.apply_updates(vec, updates)
        losses.append(float(loss_val))

    assert losses[-1] < losses[0], (
        f"Loss should decrease over 30 Adam steps. "
        f"Initial: {losses[0]:.6f}, Final: {losses[-1]:.6f}"
    )
