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
    temp_1d = temp_mean + temp_amp * jnp.sin(phase)                # (n,)
    temp_seq = jnp.outer(temp_1d, jnp.ones(n_layers))             # (n, L)
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
# test_mass_balance_harvard
# ---------------------------------------------------------------------------


def test_mass_balance_harvard():
    """
    Mass balance holds at every step over a full annual cycle (Harvard Forest).

    Runs 365 Euler steps with sinusoidal T / SW forcing via jax.lax.scan.
    At each step the scan body cross-checks step_12C against diagnose():

        |Σ(ΔC12) − (NPP − Rh) × dt| < 1e-4 gC m⁻² day⁻¹

    The residual is O(float32 ε × flux magnitude) ≈ 1e-7, well below the
    tolerance, so this test catches any structural mismatch between the
    ODE integrator and the diagnostic code path.
    """
    model, cfg, params, init_state = _build_model_and_state(
        _HARVARD_PATH, {"mat_c": 8.5, "permafrost": False}
    )
    n_layers = len(cfg.soil_layers)
    dt = float(cfg.dt_days)

    forcing_seq = _sinusoidal_forcing_seq(
        365, n_layers,
        temp_mean=8.0, temp_amp=12.0,
        sw_mean=150.0, sw_amp=100.0,
    )

    def scan_body(state, forcing_t):
        new_state = model.step_12C(state, params, forcing_t)
        diag = model.diagnose(state, params, forcing_t)
        expected = (diag["NPP"] - diag["Rh"]) * dt
        residual = jnp.abs(jnp.sum(new_state.C12 - state.C12) - expected)
        return new_state, residual

    _, residuals = jax.lax.scan(scan_body, init_state, forcing_seq)
    max_res = float(jnp.max(residuals))
    assert max_res < 1e-4, (
        f"Harvard mass balance violated: "
        f"max |ΔC_total − (NPP−Rh)·dt| = {max_res:.3e} gC m⁻² day⁻¹"
    )


# ---------------------------------------------------------------------------
# test_mass_balance_barrow
# ---------------------------------------------------------------------------


def test_mass_balance_barrow():
    """
    Mass balance holds over a full annual cycle (Barrow Alaska, cold site).

    Forcing: T = −12 + 20·sin(2π·t/365) so soil temperatures dip well
    below 0 °C in winter.  The permafrost layers start with thawed_frac = 0,
    suppressing their decomposition; the mass balance test is valid regardless
    of freeze/thaw state because it is an algebraic identity.

    Additionally verifies that a fully frozen state (thawed_frac = 0) gives
    Rh ≈ 0: the thaw-fraction mask completely suppresses decomposition.
    """
    model, cfg, params, init_state = _build_model_and_state(
        _BARROW_PATH, {"mat_c": -12.0, "permafrost": True}
    )
    n_layers = len(cfg.soil_layers)
    n_pools = len(model.pool_index)
    dt = float(cfg.dt_days)

    forcing_seq = _sinusoidal_forcing_seq(
        365, n_layers,
        temp_mean=-12.0, temp_amp=20.0,
        sw_mean=150.0, sw_amp=100.0,
    )

    # ── Mass balance scan ─────────────────────────────────────────────────
    def scan_body(state, forcing_t):
        new_state = model.step_12C(state, params, forcing_t)
        diag = model.diagnose(state, params, forcing_t)
        expected = (diag["NPP"] - diag["Rh"]) * dt
        residual = jnp.abs(jnp.sum(new_state.C12 - state.C12) - expected)
        return new_state, residual

    _, residuals = jax.lax.scan(scan_body, init_state, forcing_seq)
    max_res = float(jnp.max(residuals))
    assert max_res < 1e-4, (
        f"Barrow mass balance violated: "
        f"max |ΔC_total − (NPP−Rh)·dt| = {max_res:.3e} gC m⁻² day⁻¹"
    )

    # ── Frozen-state suppresses decomposition ─────────────────────────────
    # Fill all pools with 50 gC m⁻² then freeze all layers.
    # With thawed_frac = 0 everywhere, f_decomp = 0 for every pool → Rh = 0.
    filled_state = init_state._replace(C12=jnp.ones(n_pools) * 50.0)
    frozen_state = filled_state._replace(thawed_frac=jnp.zeros(n_layers))

    winter_forcing = {
        "air_temp": jnp.array(-20.0),
        "sw_radiation": jnp.array(5.0),
        "soil_temp": jnp.full(n_layers, -15.0),
        "soil_moisture": jnp.full(n_layers, 0.3),
        "delta14C_atm": jnp.array(0.0),
    }
    diag_frozen = model.diagnose(frozen_state, params, winter_forcing)
    assert float(diag_frozen["Rh"]) == pytest.approx(0.0, abs=1e-6), (
        f"Expected Rh = 0 with thawed_frac = 0, "
        f"got Rh = {float(diag_frozen['Rh']):.3e} gC m⁻² day⁻¹"
    )

    # Sanity: fully thawed state gives Rh > 0 (decomposition is active)
    thawed_state = filled_state._replace(thawed_frac=jnp.ones(n_layers))
    diag_thawed = model.diagnose(thawed_state, params, winter_forcing)
    assert float(diag_thawed["Rh"]) > 0.0, (
        "Expected Rh > 0 with thawed_frac = 1 and non-zero C12"
    )


# ---------------------------------------------------------------------------
# test_no_nan_harvard
# ---------------------------------------------------------------------------


def test_no_nan_harvard():
    """
    C12 remains finite at every step over 10 years (3 650 steps, Harvard).

    Uses a repeating annual sinusoidal forcing cycle via jax.lax.scan.
    A NaN or inf anywhere would indicate a numerical instability (e.g.
    division by zero in log_tau or overflow in the softmax).
    """
    model, cfg, params, init_state = _build_model_and_state(
        _HARVARD_PATH, {"mat_c": 8.5, "permafrost": False}
    )
    n_layers = len(cfg.soil_layers)

    # Repeat the annual cycle 10 times
    forcing_seq = _sinusoidal_forcing_seq(
        3650, n_layers,
        temp_mean=8.0, temp_amp=12.0,
        sw_mean=150.0, sw_amp=100.0,
    )

    def scan_body(state, forcing_t):
        new_state = model.step_12C(state, params, forcing_t)
        all_finite = jnp.all(jnp.isfinite(new_state.C12))
        return new_state, all_finite

    _, finite_flags = jax.lax.scan(scan_body, init_state, forcing_seq)
    n_bad = int(jnp.sum(~finite_flags))
    assert n_bad == 0, (
        f"C12 became non-finite at {n_bad} / 3650 timesteps"
    )


# ---------------------------------------------------------------------------
# test_gradient_flow
# ---------------------------------------------------------------------------


def test_gradient_flow():
    """
    Gradients of the one-step C12 total w.r.t. every ModelParams field are
    finite and non-zero for the key differentiable parameters.

    Uses a "warm" initial state (C12 = 100 gC m⁻² per pool) so that
    decomposition is non-zero and gradients flow through log_tau and
    log_f_transfer.  With the default empty state (C12 = 0), those
    gradients would be identically zero (no carbon → no decomp flux).
    """
    model, cfg, params, init_state = _build_model_and_state(
        _HARVARD_PATH, {"mat_c": 8.5, "permafrost": False}
    )
    n_pools = len(model.pool_index)
    n_layers = len(cfg.soil_layers)

    # Warm state: 100 gC m⁻² per pool so decomp flux is active
    warm_state = init_state._replace(C12=jnp.full(n_pools, 100.0))

    forcing = {
        "air_temp": jnp.array(20.0),
        "sw_radiation": jnp.array(150.0),
        "soil_temp": jnp.full(n_layers, 15.0),
        "soil_moisture": jnp.full(n_layers, 0.3),
        "delta14C_atm": jnp.array(0.0),
    }

    def loss(p):
        return model.step_12C(warm_state, p, forcing).C12.sum()

    grads = jax.grad(loss)(params)

    # All key fields must have finite gradients
    assert jnp.all(jnp.isfinite(grads.log_tau)), (
        f"Non-finite gradient in log_tau: {grads.log_tau}"
    )
    assert jnp.all(jnp.isfinite(grads.log_alloc)), (
        f"Non-finite gradient in log_alloc: {grads.log_alloc}"
    )
    assert jnp.all(jnp.isfinite(grads.log_f_transfer)), (
        f"Non-finite gradient in log_f_transfer (shape "
        f"{grads.log_f_transfer.shape})"
    )

    # Gradients must actually flow (not all zero)
    assert not jnp.all(grads.log_tau == 0.0), (
        "log_tau gradient is identically zero — "
        "no gradient signal flowing through decomposition"
    )
    assert not jnp.all(grads.log_alloc == 0.0), (
        "log_alloc gradient is identically zero — "
        "no gradient signal flowing through NPP allocation"
    )
    assert not jnp.all(grads.log_f_transfer == 0.0), (
        "log_f_transfer gradient is identically zero — "
        "no gradient signal flowing through carbon transfers"
    )


# ---------------------------------------------------------------------------
# test_pool_positivity
# ---------------------------------------------------------------------------


def test_pool_positivity():
    """
    C12 never goes negative over 3 650 steps (Harvard Forest, 10 years).

    Proof sketch (Euler step, dt = 1 day, τ ≥ 1 day):
        C12_new[i] = C12[i] + dt·(influx[i] + npp[i] − C12[i]/τ[i]·scalars)
                   ≥ C12[i]·(1 − dt/τ[i]) ≥ 0
    because influx ≥ 0, npp ≥ 0, and dt/τ ≤ 1 for all pools (τ ≥ 180 days
    in the Harvard config).  This test catches mis-initialised tau values
    or softmax outputs that could violate the τ ≥ 1 assumption.
    """
    model, cfg, params, init_state = _build_model_and_state(
        _HARVARD_PATH, {"mat_c": 8.5, "permafrost": False}
    )
    n_layers = len(cfg.soil_layers)

    forcing_seq = _sinusoidal_forcing_seq(
        3650, n_layers,
        temp_mean=8.0, temp_amp=12.0,
        sw_mean=150.0, sw_amp=100.0,
    )

    def scan_body(state, forcing_t):
        new_state = model.step_12C(state, params, forcing_t)
        all_nonneg = jnp.all(new_state.C12 >= 0.0)
        return new_state, all_nonneg

    _, nonneg_flags = jax.lax.scan(scan_body, init_state, forcing_seq)
    n_bad = int(jnp.sum(~nonneg_flags))
    assert n_bad == 0, (
        f"C12 went negative at {n_bad} / 3650 timesteps — "
        f"Euler step is unstable for the current tau values"
    )
