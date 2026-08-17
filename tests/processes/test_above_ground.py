"""
Tests for src/ecosystem_complexity/above_ground.py.

Coverage: npp_allocation, compute_external_soil_inputs
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ecosystem_complexity.processes.vegetation import (
    compute_external_soil_inputs,
    npp_allocation,
)

# ---------------------------------------------------------------------------
# npp_allocation
# ---------------------------------------------------------------------------


def test_npp_allocation_shape():
    """Output shape equals n_ag_pools."""
    n_ag = 3
    log_alloc = jnp.zeros(n_ag)
    result = npp_allocation(jnp.array(10.0), 0.5, log_alloc, n_ag)
    assert result.shape == (n_ag,)


def test_npp_allocation_sum_equals_npp():
    """Sum of allocation equals GPP * CUE."""
    gpp = 8.0
    cue = 0.5
    n_ag = 4
    log_alloc = jnp.array([1.0, 2.0, 0.5, 1.5])
    result = npp_allocation(jnp.array(gpp), cue, log_alloc, n_ag)
    assert float(result.sum()) == pytest.approx(gpp * cue, rel=1e-6)


def test_npp_allocation_nonnegative():
    """All allocation values are non-negative."""
    log_alloc = jnp.array([0.5, -1.0, 2.0])
    result = npp_allocation(jnp.array(5.0), 0.6, log_alloc, 3)
    assert jnp.all(result >= 0.0)


def test_npp_allocation_uses_only_n_ag_pools():
    """Extra logits beyond n_ag_pools are ignored."""
    log_alloc_long = jnp.array([1.0, 2.0, 0.5, 999.0, -999.0])
    log_alloc_short = jnp.array([1.0, 2.0, 0.5])
    gpp = jnp.array(10.0)
    result_long = npp_allocation(gpp, 0.5, log_alloc_long, 3)
    result_short = npp_allocation(gpp, 0.5, log_alloc_short, 3)
    np.testing.assert_allclose(np.array(result_long), np.array(result_short), rtol=1e-6)


# ---------------------------------------------------------------------------
# compute_external_soil_inputs
# ---------------------------------------------------------------------------


def _make_ext_inputs(n_pools: int = 6, n_targets: int = 2):
    """Return a consistent set of inputs for compute_external_soil_inputs tests."""
    target_indices = jnp.array([2, 4], dtype=jnp.int32)[:n_targets]
    log_CUE = jnp.array(math.log(0.5))
    log_soil_frac = jnp.array(math.log(0.6 / 0.4))  # sigmoid → 0.6
    log_partition = jnp.array([math.log(0.7), math.log(0.3)])[:n_targets]
    return log_CUE, log_soil_frac, log_partition, target_indices, n_pools


def test_compute_external_soil_inputs_mass_conservation():
    """Total input equals GPP × CUE × soil_frac (mass is conserved via softmax)."""
    log_CUE, log_soil_frac, log_partition, target_indices, n_pools = _make_ext_inputs()
    GPP = jnp.array(10.0)

    result = compute_external_soil_inputs(
        GPP_or_NPP=GPP,
        is_npp=False,
        log_CUE=log_CUE,
        log_soil_input_fraction=log_soil_frac,
        log_external_input_partition=log_partition,
        n_pools=n_pools,
        target_pool_indices=target_indices,
    )

    import jax.nn as jnn

    expected_total = (
        float(GPP) * math.exp(float(log_CUE)) * float(jnn.sigmoid(log_soil_frac))
    )
    assert float(result.sum()) == pytest.approx(expected_total, rel=1e-5)


def test_compute_external_soil_inputs_disabled_returns_zeros():
    """With soil_input_fraction ≈ 0 (logit → −∞), result is essentially zero."""
    log_CUE = jnp.array(math.log(0.5))
    log_soil_frac = jnp.array(-1e6)  # sigmoid(−1e6) ≈ 0
    log_partition = jnp.array([0.0, 0.0])
    target_indices = jnp.array([1, 3], dtype=jnp.int32)

    result = compute_external_soil_inputs(
        GPP_or_NPP=jnp.array(100.0),
        is_npp=False,
        log_CUE=log_CUE,
        log_soil_input_fraction=log_soil_frac,
        log_external_input_partition=log_partition,
        n_pools=5,
        target_pool_indices=target_indices,
    )

    # Total input should be negligibly small
    assert float(result.sum()) == pytest.approx(0.0, abs=1e-20)


def test_compute_external_soil_inputs_partition_sums_to_total():
    """Individual pool inputs sum exactly to the expected soil_input_total."""
    log_CUE = jnp.array(math.log(0.4))
    log_soil_frac = jnp.array(0.0)  # sigmoid(0) = 0.5
    log_partition = jnp.log(jnp.array([0.3, 0.5, 0.2]))
    target_indices = jnp.array([0, 2, 4], dtype=jnp.int32)
    n_pools = 6
    NPP = jnp.array(8.0)

    result = compute_external_soil_inputs(
        GPP_or_NPP=NPP,
        is_npp=True,
        log_CUE=log_CUE,
        log_soil_input_fraction=log_soil_frac,
        log_external_input_partition=log_partition,
        n_pools=n_pools,
        target_pool_indices=target_indices,
    )

    # With is_npp=True: soil_input = NPP * sigmoid(0) = 8.0 * 0.5 = 4.0
    assert float(result.sum()) == pytest.approx(4.0, rel=1e-5)

    # Non-target pools must remain zero
    non_targets = [i for i in range(n_pools) if i not in [0, 2, 4]]
    for i in non_targets:
        assert float(result[i]) == pytest.approx(0.0, abs=1e-10)


def test_compute_external_soil_inputs_grad_log_CUE_finite():
    """jax.grad w.r.t. log_CUE is finite and non-zero."""
    log_CUE = jnp.array(math.log(0.5))
    log_soil_frac = jnp.array(0.0)
    log_partition = jnp.array([0.0])
    target_indices = jnp.array([1], dtype=jnp.int32)
    GPP = jnp.array(5.0)

    def loss(lc):
        return compute_external_soil_inputs(
            GPP_or_NPP=GPP,
            is_npp=False,
            log_CUE=lc,
            log_soil_input_fraction=log_soil_frac,
            log_external_input_partition=log_partition,
            n_pools=3,
            target_pool_indices=target_indices,
        ).sum()

    g = jax.grad(loss)(log_CUE)
    assert jnp.isfinite(g), f"Non-finite gradient: {g}"
    assert float(g) != 0.0, "Gradient of log_CUE is unexpectedly zero"
