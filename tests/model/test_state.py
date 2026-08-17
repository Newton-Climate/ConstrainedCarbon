"""
Tests for src/ecosystem_complexity/state.py.

Coverage
--------
* make_initial_state
    - All array shapes match pool/layer counts derived from ModelConfig.
    - thawed_frac is 1.0 for non-permafrost layers, 0.0 for permafrost layers.
    - active_layer_depth is jnp.inf for a non-permafrost site.
    - active_layer_depth is finite (and matches permafrost table) for a
      permafrost site.

* make_default_params
    - log_tau length equals n_total_pools.
    - log_f_transfer shape is (n_total_pools, n_total_pools + 1).
    - softmax(log_f_transfer) rows sum to 1 (valid probability vectors).
    - softmax(log_alloc) sums to 1.
    - lambda_14C is approximately 3.312e-7 day⁻¹ (spec quotes ~3.317e-7).

* JAX pytree compatibility
    - jax.tree.map(lambda x: x * 2, state) works without error or registration.
    - jax.tree.map(lambda x: x * 2, params) works without error.
"""

from __future__ import annotations

import math
import pathlib

import jax
import jax.numpy as jnp
import pytest

from ecosystem_complexity.model.configuration import PoolIndex, load_config
from ecosystem_complexity.model.state import (
    _LAMBDA_14C,
    EcosystemState,
    ModelParams,
    make_default_params,
    make_initial_state,
)

CONFIGS_DIR = pathlib.Path(__file__).resolve().parents[2] / "configs"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def harvard_config():
    return load_config(str(CONFIGS_DIR / "harvard_forest.yaml"))


@pytest.fixture(scope="module")
def barrow_config():
    return load_config(str(CONFIGS_DIR / "barrow_alaska.yaml"))


@pytest.fixture(scope="module")
def harvard_site():
    """Raw site dict for Harvard Forest (non-permafrost, mat_c = 8.5)."""
    return {"id": "US-Ha1", "name": "Harvard Forest", "mat_c": 8.5, "permafrost": False}


@pytest.fixture(scope="module")
def barrow_site():
    """Raw site dict for Barrow Alaska (permafrost)."""
    return {"id": "US-Brw", "name": "Barrow Alaska", "mat_c": -12.0, "permafrost": True}


@pytest.fixture(scope="module")
def harvard_state(harvard_config, harvard_site):
    return make_initial_state(harvard_config, harvard_site)


@pytest.fixture(scope="module")
def barrow_state(barrow_config, barrow_site):
    return make_initial_state(barrow_config, barrow_site)


@pytest.fixture(scope="module")
def harvard_params(harvard_config):
    return make_default_params(harvard_config)


# ---------------------------------------------------------------------------
# make_initial_state — shape checks
# ---------------------------------------------------------------------------


def test_initial_state_c12_shape(harvard_config, harvard_state):
    n_pools = len(PoolIndex(harvard_config))
    assert harvard_state.C12.shape == (
        n_pools,
    ), f"Expected C12 shape ({n_pools},), got {harvard_state.C12.shape}"


def test_initial_state_c14_shape(harvard_config, harvard_state):
    n_pools = len(PoolIndex(harvard_config))
    assert harvard_state.C14.shape == (n_pools,)


def test_initial_state_soil_temp_shape(harvard_config, harvard_state):
    n_layers = len(harvard_config.soil_layers)
    assert harvard_state.soil_temp.shape == (n_layers,)


def test_initial_state_soil_moisture_shape(harvard_config, harvard_state):
    n_layers = len(harvard_config.soil_layers)
    assert harvard_state.soil_moisture.shape == (n_layers,)


def test_initial_state_thawed_frac_shape(harvard_config, harvard_state):
    n_layers = len(harvard_config.soil_layers)
    assert harvard_state.thawed_frac.shape == (n_layers,)


def test_initial_state_c12_zeros(harvard_state):
    assert jnp.all(harvard_state.C12 == 0.0)


def test_initial_state_c14_zeros(harvard_state):
    assert jnp.all(harvard_state.C14 == 0.0)


def test_initial_state_soil_temp_from_mat(harvard_state, harvard_site):
    mat = harvard_site["mat_c"]
    assert jnp.allclose(harvard_state.soil_temp, mat)


def test_initial_state_soil_moisture_default(harvard_state):
    assert jnp.allclose(harvard_state.soil_moisture, 0.3)


# ---------------------------------------------------------------------------
# make_initial_state — thawed_frac and active_layer_depth
# ---------------------------------------------------------------------------


def test_thawed_frac_non_permafrost_layers(harvard_config, harvard_state):
    """All Harvard layers are non-permafrost → thawed_frac should be 1.0."""
    for i, layer in enumerate(harvard_config.soil_layers):
        expected = 0.0 if layer.permafrost_eligible else 1.0
        assert float(harvard_state.thawed_frac[i]) == expected, (
            f"Layer {layer.name!r}: expected thawed_frac={expected}, "
            f"got {float(harvard_state.thawed_frac[i])}"
        )


def test_thawed_frac_permafrost_layers(barrow_config, barrow_state):
    """Barrow has permafrost-eligible layers → those should start frozen (0.0)."""
    for i, layer in enumerate(barrow_config.soil_layers):
        expected = 0.0 if layer.permafrost_eligible else 1.0
        assert float(barrow_state.thawed_frac[i]) == expected, (
            f"Layer {layer.name!r}: expected thawed_frac={expected}, "
            f"got {float(barrow_state.thawed_frac[i])}"
        )


def test_active_layer_depth_non_permafrost_is_inf(harvard_state):
    """Non-permafrost site → active_layer_depth == jnp.inf."""
    assert jnp.isinf(harvard_state.active_layer_depth)


def test_active_layer_depth_permafrost_is_finite(barrow_state):
    """Permafrost site → active_layer_depth should be a finite positive value."""
    assert jnp.isfinite(barrow_state.active_layer_depth)
    assert float(barrow_state.active_layer_depth) >= 0.0


# ---------------------------------------------------------------------------
# make_default_params — shapes
# ---------------------------------------------------------------------------


def test_log_tau_length_equals_n_total_pools(harvard_config, harvard_params):
    n_pools = len(PoolIndex(harvard_config))
    assert harvard_params.log_tau.shape == (
        n_pools,
    ), f"log_tau shape {harvard_params.log_tau.shape} != ({n_pools},)"


def test_log_f_transfer_shape(harvard_config, harvard_params):
    n_pools = len(PoolIndex(harvard_config))
    expected = (n_pools, n_pools + 1)
    assert (
        harvard_params.log_f_transfer.shape == expected
    ), f"log_f_transfer shape {harvard_params.log_f_transfer.shape} != {expected}"


def test_log_f_transfer_rows_sum_to_one(harvard_params):
    """softmax(log_f_transfer) rows must be valid probability vectors."""
    f = jax.nn.softmax(harvard_params.log_f_transfer, axis=-1)
    row_sums = f.sum(axis=-1)
    assert jnp.allclose(
        row_sums, 1.0, atol=1e-5
    ), f"Transfer matrix rows do not sum to 1: {row_sums}"


def test_log_alloc_sums_to_one(harvard_params):
    """softmax(log_alloc) must sum to 1."""
    alloc = jax.nn.softmax(harvard_params.log_alloc)
    assert jnp.allclose(alloc.sum(), 1.0, atol=1e-5)


def test_log_alloc_length_equals_n_ag(harvard_config, harvard_params):
    n_ag = len(harvard_config.aboveground_pools)
    assert harvard_params.log_alloc.shape == (n_ag,)


def test_log_q10_shape(harvard_config, harvard_params):
    n_layers = len(harvard_config.soil_layers)
    assert harvard_params.log_Q10.shape == (n_layers,)


def test_alpha_priming_empty_when_no_microbial(harvard_config, harvard_params):
    """Harvard Forest has microbial_pool_per_layer=False → shape (0,)."""
    assert not harvard_config.microbial_pool_per_layer
    assert harvard_params.alpha_priming.shape == (0,)


# ---------------------------------------------------------------------------
# lambda_14C
# ---------------------------------------------------------------------------


def test_lambda14c_value(harvard_params):
    """
    lambda_14C ≈ ln(2) / (5730 yr * 365.25 d/yr) ≈ 3.312e-7 day⁻¹.

    The spec quotes 3.317e-7 (a slight rounding); we accept ±1 % tolerance.
    """
    expected = math.log(2.0) / (5730.0 * 365.25)  # 3.312e-7
    assert float(harvard_params.lambda_14C) == pytest.approx(expected, rel=1e-4)


def test_lambda14c_constant_matches_module():
    """The module-level _LAMBDA_14C constant is the exact value used."""
    assert _LAMBDA_14C == pytest.approx(math.log(2.0) / (5730.0 * 365.25), rel=1e-12)


# ---------------------------------------------------------------------------
# JAX pytree compatibility
# ---------------------------------------------------------------------------


def test_jax_tree_map_state(harvard_state):
    """EcosystemState is a JAX pytree — jax.tree.map must work without error."""
    doubled = jax.tree.map(lambda x: x * 2, harvard_state)
    assert isinstance(doubled, EcosystemState)
    # C12 is all zeros; doubling gives zeros
    assert jnp.allclose(doubled.C12, 0.0)
    # soil_temp should be doubled
    assert jnp.allclose(doubled.soil_temp, harvard_state.soil_temp * 2)


def test_jax_tree_map_params(harvard_params):
    """ModelParams is a JAX pytree — jax.tree.map must work without error."""
    doubled = jax.tree.map(lambda x: x * 2, harvard_params)
    assert isinstance(doubled, ModelParams)
    assert jnp.allclose(doubled.log_tau, harvard_params.log_tau * 2)
    assert jnp.allclose(doubled.lambda_14C, harvard_params.lambda_14C * 2)


def test_jax_tree_leaves_count(harvard_config, harvard_state, harvard_params):
    """Pytree leaf count equals the number of fields in each NamedTuple."""
    state_leaves = jax.tree.leaves(harvard_state)
    param_leaves = jax.tree.leaves(harvard_params)
    assert len(state_leaves) == len(EcosystemState._fields)
    assert len(param_leaves) == len(ModelParams._fields)


# ---------------------------------------------------------------------------
# Integration: make_default_params shapes match PoolIndex counts (both sites)
# ---------------------------------------------------------------------------
#
# These tests cross-check config.py (PoolIndex) against state.py
# (make_default_params), verifying that the parameter arrays are sized
# consistently with the pool structure declared in each YAML.
#
# Pool counts (microbial_pool_per_layer=False in both configs):
#   Harvard Forest: 4 AG + 3 layers × 2 SOM = 10 pools
#   Barrow Alaska:  2 AG + 3 layers × 2 SOM = 8 pools
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def barrow_params(barrow_config):
    return make_default_params(barrow_config)


def test_harvard_log_tau_shape_explicit(harvard_params) -> None:
    """
    log_tau for Harvard Forest has the explicit expected shape (10,).

    This cross-module test confirms that make_default_params allocates
    one turnover-time parameter per pool as counted from the YAML structure.
    """
    assert harvard_params.log_tau.shape == (
        10,
    ), f"Expected log_tau shape (10,), got {harvard_params.log_tau.shape}"


def test_barrow_log_tau_shape_explicit(barrow_params) -> None:
    """
    log_tau for Barrow Alaska has the explicit expected shape (8,).

    This cross-module test confirms that make_default_params allocates
    one turnover-time parameter per pool as counted from the YAML structure.
    """
    assert barrow_params.log_tau.shape == (
        8,
    ), f"Expected log_tau shape (8,), got {barrow_params.log_tau.shape}"


def test_log_tau_finite_and_positive(harvard_params) -> None:
    """
    All log_tau values must be finite and positive.

    Sign convention: log_tau = log(τ_days).  All τ_prior_days in the
    Harvard config are ≥ 180 days, so log(τ) > log(1) = 0 → log_tau > 0.
    """
    assert jnp.all(jnp.isfinite(harvard_params.log_tau)), "log_tau contains NaN or inf"
    assert jnp.all(harvard_params.log_tau > 0.0), (
        f"Expected all log_tau > 0 (τ > 1 day), "
        f"got min={float(harvard_params.log_tau.min()):.4f}"
    )


def test_barrow_log_tau_finite_and_positive(barrow_params) -> None:
    """All Barrow log_tau values are finite and positive (τ_prior_days ≥ 730)."""
    assert jnp.all(jnp.isfinite(barrow_params.log_tau))
    assert jnp.all(barrow_params.log_tau > 0.0), (
        f"Expected all log_tau > 0, "
        f"got min={float(barrow_params.log_tau.min()):.4f}"
    )


# ---------------------------------------------------------------------------
# Integration: make_initial_state — permafrost layer specifics (Barrow)
# ---------------------------------------------------------------------------


def test_barrow_permafrost_layer_thawed_frac_zero(barrow_config, barrow_state) -> None:
    """
    The 'permafrost' soil layer (permafrost_eligible=True) starts fully
    frozen: thawed_frac == 0.0 at initialisation.

    This test identifies the permafrost layer by name, making it explicit
    which layer is being checked (complementing the loop-based test in
    test_thawed_frac_permafrost_layers above).
    """
    layer_names = [layer.name for layer in barrow_config.soil_layers]
    pf_idx = layer_names.index("permafrost")
    assert float(barrow_state.thawed_frac[pf_idx]) == 0.0, (
        f"permafrost layer thawed_frac expected 0.0, "
        f"got {float(barrow_state.thawed_frac[pf_idx])}"
    )


def test_barrow_active_layer_depth_value(barrow_config, barrow_state) -> None:
    """
    Barrow's active_layer_depth equals the depth_top_m of the first
    permafrost-eligible layer ('active' at 0.10 m).

    This is the initial permafrost table depth — a finite, positive value
    (not jnp.inf, which would indicate a non-permafrost site).
    """
    first_pf = next(
        (lay for lay in barrow_config.soil_layers if lay.permafrost_eligible),
        None,
    )
    assert first_pf is not None, "Barrow config has no permafrost-eligible layer"
    expected_ald = first_pf.depth_top_m  # 0.10 m (top of 'active' layer)
    assert float(barrow_state.active_layer_depth) == pytest.approx(
        expected_ald, abs=1e-6
    ), (
        f"active_layer_depth expected {expected_ald} m, "
        f"got {float(barrow_state.active_layer_depth)} m"
    )
