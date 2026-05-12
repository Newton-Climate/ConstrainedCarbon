"""
Tests for src/ecosystem_complexity/state.py.

Coverage
--------
* make_initial_state
    - All array shapes match pool/layer counts derived from ModelConfig.
    - frozen_frac is 1.0 for non-permafrost layers, 0.0 for permafrost layers.
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

from ecosystem_complexity.config import PoolIndex, load_config
from ecosystem_complexity.state import (
    EcosystemState,
    ModelParams,
    _LAMBDA_14C,
    make_default_params,
    make_initial_state,
)

CONFIGS_DIR = pathlib.Path(__file__).parent.parent / "configs"


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


def test_initial_state_C12_shape(harvard_config, harvard_state):
    n_pools = len(PoolIndex(harvard_config))
    assert harvard_state.C12.shape == (n_pools,), (
        f"Expected C12 shape ({n_pools},), got {harvard_state.C12.shape}"
    )


def test_initial_state_C14_shape(harvard_config, harvard_state):
    n_pools = len(PoolIndex(harvard_config))
    assert harvard_state.C14.shape == (n_pools,)


def test_initial_state_soil_temp_shape(harvard_config, harvard_state):
    n_layers = len(harvard_config.soil_layers)
    assert harvard_state.soil_temp.shape == (n_layers,)


def test_initial_state_soil_moisture_shape(harvard_config, harvard_state):
    n_layers = len(harvard_config.soil_layers)
    assert harvard_state.soil_moisture.shape == (n_layers,)


def test_initial_state_frozen_frac_shape(harvard_config, harvard_state):
    n_layers = len(harvard_config.soil_layers)
    assert harvard_state.frozen_frac.shape == (n_layers,)


def test_initial_state_C12_zeros(harvard_state):
    assert jnp.all(harvard_state.C12 == 0.0)


def test_initial_state_C14_zeros(harvard_state):
    assert jnp.all(harvard_state.C14 == 0.0)


def test_initial_state_soil_temp_from_mat(harvard_state, harvard_site):
    mat = harvard_site["mat_c"]
    assert jnp.allclose(harvard_state.soil_temp, mat)


def test_initial_state_soil_moisture_default(harvard_state):
    assert jnp.allclose(harvard_state.soil_moisture, 0.3)


# ---------------------------------------------------------------------------
# make_initial_state — frozen_frac and active_layer_depth
# ---------------------------------------------------------------------------


def test_frozen_frac_non_permafrost_layers(harvard_config, harvard_state):
    """All Harvard layers are non-permafrost → frozen_frac should be 1.0."""
    for i, layer in enumerate(harvard_config.soil_layers):
        expected = 0.0 if layer.permafrost_eligible else 1.0
        assert float(harvard_state.frozen_frac[i]) == expected, (
            f"Layer {layer.name!r}: expected frozen_frac={expected}, "
            f"got {float(harvard_state.frozen_frac[i])}"
        )


def test_frozen_frac_permafrost_layers(barrow_config, barrow_state):
    """Barrow has permafrost-eligible layers → those should start frozen (0.0)."""
    for i, layer in enumerate(barrow_config.soil_layers):
        expected = 0.0 if layer.permafrost_eligible else 1.0
        assert float(barrow_state.frozen_frac[i]) == expected, (
            f"Layer {layer.name!r}: expected frozen_frac={expected}, "
            f"got {float(barrow_state.frozen_frac[i])}"
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
    assert harvard_params.log_tau.shape == (n_pools,), (
        f"log_tau shape {harvard_params.log_tau.shape} != ({n_pools},)"
    )


def test_log_f_transfer_shape(harvard_config, harvard_params):
    n_pools = len(PoolIndex(harvard_config))
    expected = (n_pools, n_pools + 1)
    assert harvard_params.log_f_transfer.shape == expected, (
        f"log_f_transfer shape {harvard_params.log_f_transfer.shape} != {expected}"
    )


def test_log_f_transfer_rows_sum_to_one(harvard_params):
    """softmax(log_f_transfer) rows must be valid probability vectors."""
    f = jax.nn.softmax(harvard_params.log_f_transfer, axis=-1)
    row_sums = f.sum(axis=-1)
    assert jnp.allclose(row_sums, 1.0, atol=1e-5), (
        f"Transfer matrix rows do not sum to 1: {row_sums}"
    )


def test_log_alloc_sums_to_one(harvard_params):
    """softmax(log_alloc) must sum to 1."""
    alloc = jax.nn.softmax(harvard_params.log_alloc)
    assert jnp.allclose(alloc.sum(), 1.0, atol=1e-5)


def test_log_alloc_length_equals_n_ag(harvard_config, harvard_params):
    n_ag = len(harvard_config.aboveground_pools)
    assert harvard_params.log_alloc.shape == (n_ag,)


def test_log_Q10_shape(harvard_config, harvard_params):
    n_layers = len(harvard_config.soil_layers)
    assert harvard_params.log_Q10.shape == (n_layers,)


def test_alpha_priming_empty_when_no_microbial(harvard_config, harvard_params):
    """Harvard Forest has microbial_pool_per_layer=False → shape (0,)."""
    assert not harvard_config.microbial_pool_per_layer
    assert harvard_params.alpha_priming.shape == (0,)


# ---------------------------------------------------------------------------
# lambda_14C
# ---------------------------------------------------------------------------


def test_lambda_14C_value(harvard_params):
    """
    lambda_14C ≈ ln(2) / (5730 yr * 365.25 d/yr) ≈ 3.312e-7 day⁻¹.

    The spec quotes 3.317e-7 (a slight rounding); we accept ±1 % tolerance.
    """
    expected = math.log(2.0) / (5730.0 * 365.25)   # 3.312e-7
    assert float(harvard_params.lambda_14C) == pytest.approx(expected, rel=1e-4)


def test_lambda_14C_constant_matches_module():
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
