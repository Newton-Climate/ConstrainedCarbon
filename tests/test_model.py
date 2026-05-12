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

from ecosystem_complexity.config import PoolIndex, load_config
from ecosystem_complexity.model import _CUE, EcosystemModel
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
        "sw_radiation": jnp.array(100.0),    # MJ m⁻² day⁻¹
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
    np.testing.assert_array_equal(
        np.array(new_state.C14), np.array(initial_state.C14)
    )
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
        assert jnp.all(jnp.isfinite(leaf)), (
            f"Non-finite gradient in a ModelParams leaf: {leaf}"
        )


# ---------------------------------------------------------------------------
# step_14C
# ---------------------------------------------------------------------------


def test_step_14C_raises(
    harvard_model, initial_state, harvard_params, dummy_forcing
):
    """step_14C raises NotImplementedError (stub until Issue #8)."""
    with pytest.raises(NotImplementedError):
        harvard_model.step_14C(initial_state, harvard_params, dummy_forcing)


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------


_EXPECTED_DIAG_KEYS = {"GPP", "Ra", "NPP", "Rh", "ER", "NEE"}


def test_diagnose_keys(harvard_model, initial_state, harvard_params, dummy_forcing):
    """diagnose returns a dict with exactly the expected keys."""
    diag = harvard_model.diagnose(initial_state, harvard_params, dummy_forcing)
    assert set(diag.keys()) == _EXPECTED_DIAG_KEYS


def test_diagnose_all_finite(
    harvard_model, initial_state, harvard_params, dummy_forcing
):
    """All diagnostic values are finite jnp scalars."""
    diag = harvard_model.diagnose(initial_state, harvard_params, dummy_forcing)
    for key, val in diag.items():
        assert jnp.isfinite(val), f"diagnose[{key!r}] = {val} is not finite"


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
    assert float(diag["NPP"]) == pytest.approx(
        float(diag["GPP"]) * _CUE, rel=1e-6
    )


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


def test_step_fn_returns_tuple(
    harvard_model, initial_state, dummy_forcing
):
    """_step_fn(state, forcing_t) returns (new_state, None)."""
    result = harvard_model._step_fn(initial_state, dummy_forcing)
    assert isinstance(result, tuple) and len(result) == 2
    new_state, aux = result
    assert aux is None


def test_step_fn_c12_shape(harvard_model, initial_state, dummy_forcing):
    """_step_fn produces a new state with the same C12 shape."""
    new_state, _ = harvard_model._step_fn(initial_state, dummy_forcing)
    assert new_state.C12.shape == initial_state.C12.shape
