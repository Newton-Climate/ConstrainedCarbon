"""
Tests for src/ecosystem_complexity/fluxes.py.

Coverage
--------
f_temp
  - Returns 1.0 exactly at T_ref (Q10 irrelevant).
  - Increases monotonically with temperature.
  - jax.grad w.r.t. soil_temp is finite and non-zero.

f_moisture
  - Returns 1.0 at theta_opt.
  - Decreases symmetrically away from theta_opt.
  - Values are in (0, 1].

thawed_frac
  - Returns 0.5 at T = 0.
  - Approaches 0 for deeply frozen temperatures (T << 0).
  - Approaches 1 for warm temperatures (T >> 0).
  - jax.grad is finite everywhere.

decomp_flux
  - Non-negative for non-negative C12.
  - Zero when C12 == 0.
  - jax.grad w.r.t. C12_i is finite and positive.
  - Scales linearly with C12 (doubling C12 doubles the flux).

npp_allocation
  - Output shape is (n_ag_pools,).
  - Sum equals GPP * CUE.
  - All values are non-negative.

het_respiration
  - Non-negative scalar.
  - jax.grad w.r.t. C12 vector is finite.
  - Zero when all C12 == 0.

nee
  - Positive when respiration exceeds GPP (net source).
  - Negative when GPP exceeds respiration (net sink).
  - Exact arithmetic check.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ecosystem_complexity.fluxes import (
    compute_external_soil_inputs,
    decomp_flux,
    f_moisture,
    f_temp,
    het_respiration,
    nee,
    npp_allocation,
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
    """jax.grad through thawed_frac must be finite everywhere."""
    for temp in [-5.0, 0.0, 5.0]:
        grad = jax.grad(thawed_frac)(jnp.array(temp))
        assert jnp.isfinite(grad), f"Gradient not finite at T={temp}: {grad}"


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
    np.testing.assert_allclose(
        np.array(result_long), np.array(result_short), rtol=1e-6
    )


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


# ---------------------------------------------------------------------------
# compute_external_soil_inputs
# ---------------------------------------------------------------------------


def _make_ext_inputs(n_pools: int = 6, n_targets: int = 2):
    """Return a consistent set of inputs for compute_external_soil_inputs tests."""
    target_indices = jnp.array([2, 4], dtype=jnp.int32)[:n_targets]
    log_CUE = jnp.array(math.log(0.5))
    log_soil_frac = jnp.array(math.log(0.6 / 0.4))   # sigmoid → 0.6
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
    expected_total = float(GPP) * math.exp(float(log_CUE)) * float(jnn.sigmoid(log_soil_frac))
    assert float(result.sum()) == pytest.approx(expected_total, rel=1e-5)


def test_compute_external_soil_inputs_disabled_returns_zeros():
    """With soil_input_fraction ≈ 0 (logit → −∞), result is essentially zero."""
    log_CUE = jnp.array(math.log(0.5))
    log_soil_frac = jnp.array(-1e6)   # sigmoid(−1e6) ≈ 0
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
    log_soil_frac = jnp.array(0.0)    # sigmoid(0) = 0.5
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
