"""
Tests for src/ecosystem_complexity/model.py.

Coverage
--------
Instantiation
  - EcosystemModel builds from harvard_forest.yaml without error.
  - _step_fn is callable after __post_init__.
  - _pool_to_layer has shape (n_pools,) and values in [0, n_layers).

step_12C
  - Output C12 has the same shape as input C12.
  - All other state fields (C14, soil_temp, …) are unchanged.
  - C12 values are finite after one step.
  - Mass balance: sum(ΔC12) ≈ NPP − Rh within 1e-4 gC m⁻² day⁻¹.
  - jax.grad w.r.t. params is finite for all leaves.

step_14C
  - Raises NotImplementedError.

diagnose
  - Returns dict with the expected keys.
  - All values are finite jnp scalars.
  - GPP > 0 for positive SW radiation.
  - ER = Ra + Rh.
  - NEE = Rh + Ra − GPP.

_step_fn (scan compatibility)
  - _step_fn(state, forcing_t) returns (new_state, None).
  - Output state C12 shape is unchanged.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ecosystem_complexity.above_ground import _CUE
from ecosystem_complexity.config import PoolIndex, load_config
from ecosystem_complexity.model import EcosystemModel
from ecosystem_complexity.state import make_default_params, make_initial_state

CONFIGS_DIR = pathlib.Path(__file__).parent.parent / "configs"
_HARVARD_PATH = str(CONFIGS_DIR / "harvard_forest.yaml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def harvard_config():
    return load_config(_HARVARD_PATH)


@pytest.fixture(scope="module")
def harvard_params(harvard_config):
    return make_default_params(harvard_config)


@pytest.fixture(scope="module")
def harvard_model(harvard_config, harvard_params):
    idx = PoolIndex(harvard_config)
    return EcosystemModel(
        config=harvard_config,
        params=harvard_params,
        pool_index=idx,
    )


@pytest.fixture(scope="module")
def initial_state(harvard_config):
    site_cfg = {"mat_c": 8.5, "permafrost": False}
    return make_initial_state(harvard_config, site_cfg)


@pytest.fixture(scope="module")
def dummy_forcing(harvard_config):
    """Minimal forcing dict with realistic values."""
    n_layers = len(harvard_config.soil_layers)
    return {
        "air_temp": jnp.array(20.0),
        "sw_radiation": jnp.array(100.0),  # MJ m⁻² day⁻¹
        "soil_temp": jnp.full(n_layers, 15.0),
        "soil_moisture": jnp.full(n_layers, 0.3),
        "delta14C_atm": jnp.array(0.0),
    }


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_model_instantiation(harvard_model):
    """EcosystemModel builds from harvard_forest.yaml without error."""
    assert harvard_model.config.site_id == "US-Ha1"
    assert harvard_model._step_fn is not None


def test_step_fn_is_callable(harvard_model):
    """_step_fn is callable after __post_init__."""
    assert callable(harvard_model._step_fn)


def test_pool_to_layer_shape(harvard_model, harvard_config):
    """_pool_to_layer has shape (n_pools,) and values in [0, n_layers)."""
    n_pools = len(PoolIndex(harvard_config))
    n_layers = len(harvard_config.soil_layers)
    ptl = np.array(harvard_model._pool_to_layer)
    assert ptl.shape == (n_pools,)
    assert np.all(ptl >= 0) and np.all(ptl < n_layers)


# ---------------------------------------------------------------------------
# step_12C — shape and finiteness
# ---------------------------------------------------------------------------


def test_step_12C_c12_shape_unchanged(
    harvard_model, initial_state, harvard_params, dummy_forcing
):
    """Output C12 has the same shape as input C12."""
    new_state = harvard_model.step_12C(initial_state, harvard_params, dummy_forcing)
    assert new_state.C12.shape == initial_state.C12.shape


def test_step_12C_other_fields_unchanged(
    harvard_model, initial_state, harvard_params, dummy_forcing
):
    """Only C12 is modified; all other state fields pass through."""
    new_state = harvard_model.step_12C(initial_state, harvard_params, dummy_forcing)
    np.testing.assert_array_equal(np.array(new_state.C14), np.array(initial_state.C14))
    np.testing.assert_array_equal(
        np.array(new_state.soil_temp), np.array(initial_state.soil_temp)
    )
    np.testing.assert_array_equal(
        np.array(new_state.soil_moisture), np.array(initial_state.soil_moisture)
    )
    np.testing.assert_array_equal(
        np.array(new_state.thawed_frac), np.array(initial_state.thawed_frac)
    )


def test_step_12C_c12_finite(
    harvard_model, initial_state, harvard_params, dummy_forcing
):
    """C12 values are finite after one step (no NaN / inf)."""
    new_state = harvard_model.step_12C(initial_state, harvard_params, dummy_forcing)
    assert jnp.all(jnp.isfinite(new_state.C12))


# ---------------------------------------------------------------------------
# step_12C — mass balance
# ---------------------------------------------------------------------------


def test_step_12C_mass_balance(
    harvard_model, initial_state, harvard_params, dummy_forcing
):
    """
    Mass balance: sum(ΔC12) ≈ NPP − Rh within 1e-4 gC m⁻² day⁻¹.

    Derivation (Euler step, dt = 1 day):
        ΔC_total = influx_from_transfers + NPP − decomp
        influx   = F.T @ decomp  →  sum(influx) = sum(decomp) − Rh
        sum(ΔC)  = (sum(decomp) − Rh) + NPP − sum(decomp) = NPP − Rh

    Equivalently: sum(ΔC) = −NEE  (since NEE = Rh + Ra − GPP = Rh − NPP).
    """
    new_state = harvard_model.step_12C(initial_state, harvard_params, dummy_forcing)
    delta_c12 = float(jnp.sum(new_state.C12 - initial_state.C12))

    diag = harvard_model.diagnose(initial_state, harvard_params, dummy_forcing)
    npp = float(diag["NPP"])
    rh = float(diag["Rh"])
    dt = float(harvard_model.config.dt_days)

    expected = (npp - rh) * dt
    assert abs(delta_c12 - expected) < 1e-4, (
        f"Mass balance violated: sum(ΔC12)={delta_c12:.6f}, "
        f"(NPP-Rh)*dt={expected:.6f}"
    )


# ---------------------------------------------------------------------------
# step_12C — differentiability
# ---------------------------------------------------------------------------


def test_step_12C_grad_params_finite(
    harvard_model, initial_state, harvard_params, dummy_forcing
):
    """jax.grad w.r.t. params flows through step_12C without NaN or inf."""

    def loss(p):
        return harvard_model.step_12C(initial_state, p, dummy_forcing).C12.sum()

    grads = jax.grad(loss)(harvard_params)
    leaves = jax.tree.leaves(grads)
    for leaf in leaves:
        assert jnp.all(
            jnp.isfinite(leaf)
        ), f"Non-finite gradient in a ModelParams leaf: {leaf}"


# ---------------------------------------------------------------------------
# step_14C
# ---------------------------------------------------------------------------


def test_step_14C_no_longer_raises(
    harvard_model, initial_state, harvard_params, dummy_forcing
):
    """
    step_14C is now implemented (Issue #8) — must not raise NotImplementedError.

    Full correctness tests for the ¹⁴C step live in tests/test_14C.py;
    this test confirms that the stub has been replaced.
    """
    new_state = harvard_model.step_14C(initial_state, harvard_params, dummy_forcing)
    assert new_state.C14.shape == initial_state.C14.shape
    assert jnp.all(jnp.isfinite(new_state.C14))


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------


_EXPECTED_DIAG_KEYS = {
    "GPP",
    "Ra",
    "NPP",
    "Rh",
    "Rh_by_pool",
    "ER",
    "NEE",
    # external_inputs diagnostics (always present; zero when feature disabled)
    "GPP_forcing_used",
    "external_C_input_total",
    "external_C_input_by_pool",
}


def test_diagnose_keys(harvard_model, initial_state, harvard_params, dummy_forcing):
    """diagnose returns a dict with exactly the expected keys."""
    diag = harvard_model.diagnose(initial_state, harvard_params, dummy_forcing)
    assert set(diag.keys()) == _EXPECTED_DIAG_KEYS


def test_diagnose_all_finite(
    harvard_model, initial_state, harvard_params, dummy_forcing
):
    """All diagnostic values are finite jnp scalars or arrays."""
    diag = harvard_model.diagnose(initial_state, harvard_params, dummy_forcing)
    for key, val in diag.items():
        arr = jnp.asarray(val)
        if arr.size == 0:
            continue  # empty array (e.g. external_C_input_by_pool with no targets)
        assert jnp.all(jnp.isfinite(arr)), f"diagnose[{key!r}] = {val} is not finite"


def test_diagnose_gpp_positive(
    harvard_model, initial_state, harvard_params, dummy_forcing
):
    """GPP > 0 for positive SW radiation."""
    diag = harvard_model.diagnose(initial_state, harvard_params, dummy_forcing)
    assert float(diag["GPP"]) > 0.0


def test_diagnose_er_equals_ra_plus_rh(
    harvard_model, initial_state, harvard_params, dummy_forcing
):
    """ER = Ra + Rh (ecosystem respiration = autotrophic + heterotrophic)."""
    diag = harvard_model.diagnose(initial_state, harvard_params, dummy_forcing)
    assert float(diag["ER"]) == pytest.approx(
        float(diag["Ra"]) + float(diag["Rh"]), rel=1e-5
    )


def test_diagnose_nee_formula(
    harvard_model, initial_state, harvard_params, dummy_forcing
):
    """NEE = Rh + Ra − GPP (sign convention: positive = source)."""
    diag = harvard_model.diagnose(initial_state, harvard_params, dummy_forcing)
    expected = float(diag["Rh"]) + float(diag["Ra"]) - float(diag["GPP"])
    assert float(diag["NEE"]) == pytest.approx(expected, rel=1e-5)


def test_diagnose_npp_equals_gpp_times_cue(
    harvard_model, initial_state, harvard_params, dummy_forcing
):
    """NPP = GPP × CUE."""
    diag = harvard_model.diagnose(initial_state, harvard_params, dummy_forcing)
    assert float(diag["NPP"]) == pytest.approx(float(diag["GPP"]) * _CUE, rel=1e-6)


def test_diagnose_zero_radiation_zero_gpp(
    harvard_model, initial_state, harvard_params, harvard_config
):
    """GPP = 0 when SW radiation = 0 (night / polar winter)."""
    n_layers = len(harvard_config.soil_layers)
    dark_forcing = {
        "air_temp": jnp.array(10.0),
        "sw_radiation": jnp.array(0.0),
        "soil_temp": jnp.full(n_layers, 10.0),
        "soil_moisture": jnp.full(n_layers, 0.3),
        "delta14C_atm": jnp.array(0.0),
    }
    diag = harvard_model.diagnose(initial_state, harvard_params, dark_forcing)
    assert float(diag["GPP"]) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# _step_fn — scan compatibility
# ---------------------------------------------------------------------------


def test_step_fn_returns_tuple(harvard_model, initial_state, dummy_forcing):
    """_step_fn(state, forcing_t) returns (new_state, None)."""
    result = harvard_model._step_fn(initial_state, dummy_forcing)
    assert isinstance(result, tuple) and len(result) == 2
    new_state, aux = result
    assert aux is None


def test_step_fn_c12_shape(harvard_model, initial_state, dummy_forcing):
    """_step_fn produces a new state with the same C12 shape."""
    new_state, _ = harvard_model._step_fn(initial_state, dummy_forcing)
    assert new_state.C12.shape == initial_state.C12.shape


# ===========================================================================
# Multi-step correctness tests (jax.lax.scan)
# ===========================================================================
#
# Design notes
# ------------
# * All multi-step tests use jax.lax.scan — never a Python loop — so the
#   XLA compiler fuses the entire rollout into a single compiled kernel.
# * Forcing sequences are pre-built as stacked arrays of shape (n_steps, ...)
#   and passed as the `xs` argument; scan automatically slices along axis 0.
# * Mass-balance identity (dt = 1 day):
#       Σ_i ΔC12_i = (NPP − Rh) × dt
#   This holds exactly in step_12C by construction and is cross-checked
#   against the independent diagnose() code path.
# * Gradient tests use a "warm" initial state (C12 = 100 gC m⁻²) so that
#   decomposition is non-zero and gradients flow through log_tau and
#   log_f_transfer (which would be zero with the default empty pools).
# ===========================================================================

_BARROW_PATH = str(CONFIGS_DIR / "barrow_alaska.yaml")


def _sinusoidal_forcing_seq(
    n_steps: int,
    n_layers: int,
    temp_mean: float,
    temp_amp: float,
    sw_mean: float,
    sw_amp: float,
) -> dict:
    """
    Build stacked forcing arrays of shape ``(n_steps, ...)`` for use as
    ``jax.lax.scan`` ``xs``.

    Temperature and SW follow a sinusoidal annual cycle; moisture is
    constant at 0.3 m³ m⁻³.
    """
    t = jnp.arange(n_steps, dtype=jnp.float32)
    phase = 2.0 * jnp.pi * t / 365.0
    sw_seq = jnp.maximum(sw_mean + sw_amp * jnp.sin(phase), 0.0)  # (n,)
    temp_1d = temp_mean + temp_amp * jnp.sin(phase)  # (n,)
    temp_seq = jnp.outer(temp_1d, jnp.ones(n_layers))  # (n, L)
    moisture_seq = jnp.full((n_steps, n_layers), 0.3)
    return {
        "air_temp": jnp.zeros(n_steps),
        "sw_radiation": sw_seq,
        "soil_temp": temp_seq,
        "soil_moisture": moisture_seq,
        "delta14C_atm": jnp.zeros(n_steps),
    }


def _build_model_and_state(yaml_path: str, site_cfg: dict):
    """Load config, build model and initial state; return all four."""
    cfg = load_config(yaml_path)
    params = make_default_params(cfg)
    idx = PoolIndex(cfg)
    model = EcosystemModel(config=cfg, params=params, pool_index=idx)
    state = make_initial_state(cfg, site_cfg)
    return model, cfg, params, state


# ---------------------------------------------------------------------------
