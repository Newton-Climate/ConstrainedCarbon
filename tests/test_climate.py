"""
Tests for src/ecosystem_complexity/climate.py.

Coverage: f_temp, f_moisture, thawed_frac
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ecosystem_complexity.climate import (
    f_moisture,
    f_temp,
    thawed_frac,
)

# ---------------------------------------------------------------------------
# f_temp
# ---------------------------------------------------------------------------


def test_f_temp_at_t_ref_equals_one():
    """f_temp should return exactly 1.0 when T == T_ref."""
    result = f_temp(jnp.array(15.0), jnp.array(math.log(2.0)))
    assert float(result) == pytest.approx(1.0, abs=1e-7)


def test_f_temp_custom_t_ref():
    """f_temp with custom T_ref returns 1.0 at that temperature."""
    result = f_temp(jnp.array(10.0), jnp.array(math.log(2.0)), T_ref=10.0)
    assert float(result) == pytest.approx(1.0, abs=1e-7)


def test_f_temp_increases_with_temperature():
    """f_temp is monotonically increasing with soil_temp for Q10 > 1."""
    log_q10 = jnp.array(math.log(2.0))
    temps = jnp.array([5.0, 10.0, 15.0, 20.0, 25.0])
    values = jax.vmap(lambda t: f_temp(t, log_q10))(temps)
    diffs = jnp.diff(values)
    assert jnp.all(diffs > 0), f"f_temp not monotonically increasing: {values}"


def test_f_temp_q10_doubling():
    """With Q10=2, each 10°C increase should double the rate."""
    log_q10 = jnp.array(math.log(2.0))
    v_ref = float(f_temp(jnp.array(15.0), log_q10))
    v_plus10 = float(f_temp(jnp.array(25.0), log_q10))
    assert v_plus10 == pytest.approx(2.0 * v_ref, rel=1e-6)


def test_f_temp_grad_finite_nonzero():
    """jax.grad through f_temp w.r.t. soil_temp must be finite and non-zero."""
    log_q10 = jnp.array(math.log(2.0))

    def loss(t: jnp.ndarray) -> jnp.ndarray:
        return f_temp(t, log_q10)

    grad = jax.grad(loss)(jnp.array(15.0))
    assert jnp.isfinite(grad), f"Gradient is not finite: {grad}"
    assert float(grad) != 0.0, "Gradient is zero — no signal through f_temp"


# ---------------------------------------------------------------------------
# f_moisture
# ---------------------------------------------------------------------------


def test_f_moisture_at_theta_opt_equals_one():
    """f_moisture returns 1.0 when theta == theta_opt."""
    theta_opt = 0.3
    log_theta_opt = jnp.array(math.log(theta_opt))
    log_gamma = jnp.array(math.log(5.0))
    result = f_moisture(jnp.array(theta_opt), log_theta_opt, log_gamma)
    assert float(result) == pytest.approx(1.0, abs=1e-6)


def test_f_moisture_below_one_away_from_opt():
    """f_moisture < 1 when theta != theta_opt."""
    theta_opt = 0.3
    log_theta_opt = jnp.array(math.log(theta_opt))
    log_gamma = jnp.array(math.log(5.0))
    result = f_moisture(jnp.array(0.1), log_theta_opt, log_gamma)
    assert float(result) < 1.0


def test_f_moisture_in_unit_interval():
    """f_moisture values are in (0, 1]."""
    log_theta_opt = jnp.array(math.log(0.3))
    log_gamma = jnp.array(math.log(5.0))
    thetas = jnp.linspace(0.01, 0.8, 20)
    values = jax.vmap(lambda t: f_moisture(t, log_theta_opt, log_gamma))(thetas)
    assert jnp.all(values > 0.0) and jnp.all(values <= 1.0 + 1e-7)


def test_f_moisture_symmetric():
    """f_moisture is symmetric around theta_opt."""
    theta_opt = 0.4
    log_theta_opt = jnp.array(math.log(theta_opt))
    log_gamma = jnp.array(math.log(3.0))
    delta = 0.1
    left = float(f_moisture(jnp.array(theta_opt - delta), log_theta_opt, log_gamma))
    right = float(f_moisture(jnp.array(theta_opt + delta), log_theta_opt, log_gamma))
    assert left == pytest.approx(right, abs=1e-6)


# ---------------------------------------------------------------------------
# thawed_frac
# ---------------------------------------------------------------------------


def test_thawed_frac_at_zero_is_half():
    """thawed_frac(0) == 0.5 by sigmoid definition."""
    result = thawed_frac(jnp.array(0.0))
    assert float(result) == pytest.approx(0.5, abs=1e-7)


def test_thawed_frac_deeply_frozen_near_zero():
    """thawed_frac is near 0 for deeply frozen soil (T << 0)."""
    result = thawed_frac(jnp.array(-10.0))
    assert float(result) < 0.01


def test_thawed_frac_warm_near_one():
    """thawed_frac is near 1 for warm soil (T >> 0)."""
    result = thawed_frac(jnp.array(10.0))
    assert float(result) > 0.99


def test_thawed_frac_monotone():
    """thawed_frac is monotonically increasing with temperature.

    Use temperatures within the non-saturating region of sigmoid(10·T)
    (float32 saturates to 0/1 beyond ≈ ±3 °C at steepness=10).
    """
    temps = jnp.array([-0.5, -0.2, 0.0, 0.2, 0.5])
    values = jax.vmap(thawed_frac)(temps)
    assert jnp.all(jnp.diff(values) > 0)


def test_thawed_frac_grad_finite():
    """jax.grad through thawed_frac is finite everywhere."""
    for temp in [-5.0, 0.0, 5.0]:
        grad = jax.grad(thawed_frac)(jnp.array(temp))
        assert jnp.isfinite(grad), f"Gradient not finite at T={temp}: {grad}"
