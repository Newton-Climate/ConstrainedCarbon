"""
Tests for src/ecosystem_complexity/soil.py.

Coverage: decomp_flux, het_respiration, nee
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from ecosystem_complexity.soil import (
    decomp_flux,
    het_respiration,
    nee,
)

# ---------------------------------------------------------------------------
# decomp_flux
# ---------------------------------------------------------------------------


def test_decomp_flux_zero_carbon():
    """decomp_flux returns 0 when C12_i == 0."""
    result = decomp_flux(
        jnp.array(0.0),
        jnp.array(math.log(365.0)),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
    )
    assert float(result) == pytest.approx(0.0, abs=1e-10)


def test_decomp_flux_nonnegative():
    """decomp_flux is non-negative for valid inputs."""
    result = decomp_flux(
        jnp.array(500.0),
        jnp.array(math.log(365.0)),
        jnp.array(0.8),
        jnp.array(0.9),
        jnp.array(1.0),
    )
    assert float(result) >= 0.0


def test_decomp_flux_linear_in_c12():
    """Doubling C12_i should double the flux."""
    log_tau = jnp.array(math.log(500.0))
    ft = jnp.array(0.7)
    fm = jnp.array(0.8)
    ff = jnp.array(1.0)
    f1 = float(decomp_flux(jnp.array(100.0), log_tau, ft, fm, ff))
    f2 = float(decomp_flux(jnp.array(200.0), log_tau, ft, fm, ff))
    assert f2 == pytest.approx(2.0 * f1, rel=1e-6)


def test_decomp_flux_grad_wrt_c12_finite_positive():
    """jax.grad w.r.t. C12_i is finite and positive."""
    log_tau = jnp.array(math.log(365.0))
    ft = jnp.array(1.0)
    fm = jnp.array(1.0)
    ff = jnp.array(1.0)

    grad = jax.grad(decomp_flux)(jnp.array(100.0), log_tau, ft, fm, ff)
    assert jnp.isfinite(grad), f"Gradient not finite: {grad}"
    assert float(grad) > 0.0, f"Gradient not positive: {grad}"


# ---------------------------------------------------------------------------
# het_respiration
# ---------------------------------------------------------------------------


def _make_rh_inputs(n_pools: int = 4):
    """Return a consistent set of inputs for het_respiration tests."""
    C12 = jnp.ones(n_pools) * 200.0
    log_tau = jnp.log(jnp.ones(n_pools) * 365.0)
    # All flux goes to respiration: large logit on the last column
    log_f_transfer = jnp.zeros((n_pools, n_pools + 1))
    ft_vec = jnp.ones(n_pools) * 0.8
    fm_vec = jnp.ones(n_pools) * 0.9
    ff_vec = jnp.ones(n_pools)
    return C12, log_tau, log_f_transfer, ft_vec, fm_vec, ff_vec


def test_het_respiration_nonnegative():
    """het_respiration returns a non-negative scalar."""
    inputs = _make_rh_inputs()
    result = het_respiration(*inputs, n_pools=4)
    assert float(result) >= 0.0


def test_het_respiration_zero_carbon():
    """het_respiration is 0 when all pools are empty."""
    n = 4
    C12, log_tau, log_f, ft, fm, ff = _make_rh_inputs(n)
    result = het_respiration(jnp.zeros(n), log_tau, log_f, ft, fm, ff, n_pools=n)
    assert float(result) == pytest.approx(0.0, abs=1e-10)


def test_het_respiration_grad_wrt_c12_finite():
    """jax.grad w.r.t. C12 vector must be finite everywhere."""
    C12, log_tau, log_f, ft, fm, ff = _make_rh_inputs()
    n = 4

    def loss(c: jnp.ndarray) -> jnp.ndarray:
        return het_respiration(c, log_tau, log_f, ft, fm, ff, n_pools=n)

    grad = jax.grad(loss)(C12)
    assert jnp.all(jnp.isfinite(grad)), f"Gradient contains NaN or inf: {grad}"


def test_het_respiration_increases_with_carbon():
    """Doubling all pool sizes should double Rh (linear in C12)."""
    C12, log_tau, log_f, ft, fm, ff = _make_rh_inputs()
    n = 4
    rh1 = float(het_respiration(C12, log_tau, log_f, ft, fm, ff, n_pools=n))
    rh2 = float(het_respiration(2 * C12, log_tau, log_f, ft, fm, ff, n_pools=n))
    assert rh2 == pytest.approx(2.0 * rh1, rel=1e-5)


# ---------------------------------------------------------------------------
# nee
# ---------------------------------------------------------------------------


def test_nee_positive_when_source():
    """NEE > 0 when ecosystem is a net carbon source."""
    result = nee(
        GPP=jnp.array(5.0),
        Ra=jnp.array(3.0),
        F_rh=jnp.array(4.0),
    )
    # F_rh + Ra - GPP = 4 + 3 - 5 = 2 > 0
    assert float(result) > 0.0


def test_nee_negative_when_sink():
    """NEE < 0 when ecosystem is a net carbon sink."""
    result = nee(
        GPP=jnp.array(10.0),
        Ra=jnp.array(3.0),
        F_rh=jnp.array(2.0),
    )
    # F_rh + Ra - GPP = 2 + 3 - 10 = -5 < 0
    assert float(result) < 0.0


def test_nee_exact_arithmetic():
    """NEE = F_rh + Ra - GPP, verified with exact values."""
    gpp, ra, f_rh = 8.0, 2.0, 3.0
    result = float(nee(jnp.array(gpp), jnp.array(ra), jnp.array(f_rh)))
    assert result == pytest.approx(f_rh + ra - gpp, abs=1e-7)


def test_nee_zero_at_balance():
    """NEE == 0 when GPP exactly balances Ra + F_rh."""
    result = nee(
        GPP=jnp.array(7.0),
        Ra=jnp.array(4.0),
        F_rh=jnp.array(3.0),
    )
    assert float(result) == pytest.approx(0.0, abs=1e-7)
