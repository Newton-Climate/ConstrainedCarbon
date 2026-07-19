"""Tests for the parameter-derived ¹⁴C initial condition.

Regression coverage for the circularity these replace: seeding a pool's initial
Δ¹⁴C from the observation the inversion then fits makes that observation
self-predicting (zero residual, zero sensitivity, zero DFS).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ecosystem_complexity._oe_helpers import (
    _R_STD,
    _analytical_c12_ss,
    analytical_c14_ss,
    apply_ss_c12_c14,
)
from ecosystem_complexity.api import build_model
from ecosystem_complexity.state import make_default_params

_MEAN_INPUT = 1.79
_MEAN_MOD = 0.47
_T0 = 2007.0


@pytest.fixture(scope="module")
def setup():
    model = build_model("configs/harvard_3pool_config.yaml")
    params = make_default_params(model.config)
    n = len(model.pool_index)
    ext = model.config.external_inputs
    tgt = [model.pool_index[k] for k in ext.partition.keys()] if ext else None
    # Synthetic atmosphere: flat pre-bomb, bomb spike, decline — enough
    # structure to make pools of different τ separate.
    yrs = np.arange(1500.0, 2024.0, 1.0)
    d14 = np.where(yrs < 1955.0, 0.0, 900.0 * np.exp(-(yrs - 1963.0) / 16.0))
    d14 = np.where(yrs < 1963.0, np.maximum(d14 * 0 + (yrs - 1955) * 100.0, 0.0), d14)
    return model, params, n, tgt, yrs, d14


def _d14c(params, setup):
    model, _, n, tgt, yrs, d14 = setup
    c12 = _analytical_c12_ss(params, n, _MEAN_INPUT, _MEAN_MOD, target_indices=tgt)
    c14 = analytical_c14_ss(params, c12, n, _MEAN_INPUT, _MEAN_MOD,
                            yrs, d14, _T0, target_indices=tgt)
    return np.array((c14 / (c12 + 1e-30) / _R_STD - 1.0) * 1000.0)


def test_initial_delta14C_depends_on_tau(setup):
    """The whole point: ∂Δ¹⁴C(t₀)/∂τ must be non-negligible."""
    model, params, n, *_ = setup
    base = _d14c(params, setup)
    moved = _d14c(params._replace(log_tau=params.log_tau + 0.01), setup)
    shift = np.abs(moved - base)
    # Pre-fix this was ~1e-4 permil; against a 15 permil sigma that is nothing.
    assert shift.max() > 0.05, f"Δ¹⁴C barely responds to τ: {shift}"


def test_slower_pools_are_not_identical_to_fast_pools(setup):
    """Distinct τ must give distinct Δ¹⁴C — otherwise pools are unidentifiable."""
    _, params, *_ = setup
    d = _d14c(params, setup)
    assert np.ptp(d) > 10.0, f"pools nearly indistinguishable in Δ¹⁴C: {d}"


def test_c14_is_nonnegative_and_finite(setup):
    _, params, n, tgt, yrs, d14 = setup
    model = setup[0]
    c12 = _analytical_c12_ss(params, n, _MEAN_INPUT, _MEAN_MOD, target_indices=tgt)
    c14 = np.array(analytical_c14_ss(params, c12, n, _MEAN_INPUT, _MEAN_MOD,
                                     yrs, d14, _T0, target_indices=tgt))
    assert np.all(np.isfinite(c14))
    assert np.all(c14 >= 0.0)


def test_gradient_flows_to_tau(setup):
    """The Jacobian must see this path, or the fix buys nothing."""
    _, params, n, tgt, yrs, d14 = setup

    def _f(lt):
        p = params._replace(log_tau=lt)
        c12 = _analytical_c12_ss(p, n, _MEAN_INPUT, _MEAN_MOD, target_indices=tgt)
        c14 = analytical_c14_ss(p, c12, n, _MEAN_INPUT, _MEAN_MOD,
                                yrs, d14, _T0, target_indices=tgt)
        return jnp.sum(c14 / (c12 + 1e-30))

    g = np.array(jax.grad(_f)(params.log_tau))
    assert np.all(np.isfinite(g))
    assert np.abs(g).max() > 0.0


def test_apply_ss_c12_c14_sets_both(setup):
    model, params, n, tgt, *_ = setup
    from ecosystem_complexity.state import make_initial_state
    st = make_initial_state(model.config, {"mat_c": 8.5, "permafrost": False})
    c12 = jnp.ones(n) * 100.0
    c14 = jnp.ones(n) * 1e-10
    new = apply_ss_c12_c14(st, c12, c14)
    assert np.allclose(np.array(new.C12), 100.0)
    assert np.allclose(np.array(new.C14), 1e-10)
