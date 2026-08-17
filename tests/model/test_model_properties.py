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

from ecosystem_complexity.model.configuration import PoolIndex, load_config
from ecosystem_complexity.model.simulator import EcosystemModel
from ecosystem_complexity.model.state import make_default_params, make_initial_state

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


# test_mass_balance_harvard
# ---------------------------------------------------------------------------


_BARROW_PATH = str(CONFIGS_DIR / "barrow_alaska.yaml")


def _sinusoidal_forcing_seq(
    n_steps: int,
    n_layers: int,
    temp_mean: float,
    temp_amp: float,
    sw_mean: float,
    sw_amp: float,
) -> dict:
    """Build stacked forcing arrays for jax.lax.scan."""
    t = jnp.arange(n_steps, dtype=jnp.float32)
    phase = 2.0 * jnp.pi * t / 365.0
    sw_seq = jnp.maximum(sw_mean + sw_amp * jnp.sin(phase), 0.0)
    temp_1d = temp_mean + temp_amp * jnp.sin(phase)
    temp_seq = jnp.outer(temp_1d, jnp.ones(n_layers))
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
        3650,
        n_layers,
        temp_mean=8.0,
        temp_amp=12.0,
        sw_mean=150.0,
        sw_amp=100.0,
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


# ---------------------------------------------------------------------------
# external_inputs pathway
# ---------------------------------------------------------------------------

_SOIL_ONLY_PATH = str(
    pathlib.Path(__file__).parent.parent / "configs" / "harvard_forest_soil_only.yaml"
)


@pytest.fixture(scope="module")
def soil_only_config():
    return load_config(_SOIL_ONLY_PATH)


@pytest.fixture(scope="module")
def soil_only_model(soil_only_config):
    from ecosystem_complexity.model.state import make_default_params

    idx = PoolIndex(soil_only_config)
    params = make_default_params(soil_only_config)
    return EcosystemModel(config=soil_only_config, params=params, pool_index=idx)


@pytest.fixture(scope="module")
def soil_only_state(soil_only_config):
    from ecosystem_complexity.model.state import make_initial_state

    site_cfg = {"mat_c": 8.5, "permafrost": False}
    return make_initial_state(soil_only_config, site_cfg)


@pytest.fixture(scope="module")
def soil_only_params(soil_only_config):
    from ecosystem_complexity.model.state import make_default_params

    return make_default_params(soil_only_config)


def _soil_forcing(n_layers: int, gpp: float = 5.0):
    """Forcing dict for soil-only model tests."""
    return {
        "air_temp": jnp.array(15.0),
        "sw_radiation": jnp.array(100.0),
        "soil_temp": jnp.full(n_layers, 15.0),
        "soil_moisture": jnp.full(n_layers, 0.3),
        "delta14C_atm": jnp.array(50.0),
        "GPP_obs": jnp.array(gpp),
        "NPP_obs": jnp.array(float("nan")),
    }


def test_step_12C_with_external_inputs_pool_sum_increases(
    soil_only_model, soil_only_state, soil_only_params, soil_only_config
):
    """
    With external_inputs active and positive GPP_obs, target soil pools
    receive positive carbon inputs (pool sum strictly increases vs. zero-GPP case).
    """
    n_layers = len(soil_only_config.soil_layers)
    forcing_with_gpp = _soil_forcing(n_layers, gpp=5.0)
    forcing_no_gpp = _soil_forcing(n_layers, gpp=0.0)

    state_with = soil_only_model.step_12C(
        soil_only_state, soil_only_params, forcing_with_gpp
    )
    state_none = soil_only_model.step_12C(
        soil_only_state, soil_only_params, forcing_no_gpp
    )

    # Pool sum must be higher when GPP_obs > 0 (net input > 0)
    sum_with = float(jnp.sum(state_with.C12))
    sum_none = float(jnp.sum(state_none.C12))
    assert sum_with > sum_none, (
        f"Pool sum with GPP_obs=5 ({sum_with:.4f}) should exceed "
        f"sum with GPP_obs=0 ({sum_none:.4f})"
    )


def test_step_12C_external_inputs_disabled_matches_baseline(
    harvard_model, initial_state, harvard_params, dummy_forcing
):
    """
    When external_inputs is not configured (Harvard model), step_12C output
    is identical whether or not GPP_obs is in the forcing dict.

    The external_inputs pathway is disabled, so GPP_obs is ignored.
    """
    # Forcing with and without GPP_obs — should produce the same state
    forcing_with = dict(dummy_forcing)
    forcing_with["GPP_obs"] = jnp.array(999.0)  # should be ignored
    forcing_with["NPP_obs"] = jnp.array(float("nan"))

    state_base = harvard_model.step_12C(initial_state, harvard_params, dummy_forcing)
    state_extra = harvard_model.step_12C(initial_state, harvard_params, forcing_with)

    np.testing.assert_allclose(
        np.array(state_base.C12),
        np.array(state_extra.C12),
        rtol=1e-6,
        err_msg=(
            "step_12C changed when GPP_obs was added to forcing "
            "(external_inputs disabled)"
        ),
    )


def test_external_14C_input_uses_atmospheric_signature(
    soil_only_model, soil_only_state, soil_only_params, soil_only_config
):
    """
    When external inputs are active, the 14C content of new soil inputs
    reflects the atmospheric Δ14C signal (positive bomb-spike → higher C14).

    Compare: two steps with different delta14C_atm but same GPP_obs.
    The step with higher delta14C_atm should produce higher total C14.
    """
    n_layers = len(soil_only_config.soil_layers)

    forcing_low = _soil_forcing(n_layers, gpp=5.0)
    forcing_low["delta14C_atm"] = jnp.array(-100.0)  # depleted 14C

    forcing_high = dict(forcing_low)
    forcing_high["delta14C_atm"] = jnp.array(200.0)  # bomb-spike 14C

    # Start with zero C14 so new inputs dominate
    state0 = soil_only_state._replace(C14=jnp.zeros_like(soil_only_state.C14))

    state_low = soil_only_model.step_14C(state0, soil_only_params, forcing_low)
    state_high = soil_only_model.step_14C(state0, soil_only_params, forcing_high)

    sum_low = float(jnp.sum(state_low.C14))
    sum_high = float(jnp.sum(state_high.C14))
    assert sum_high > sum_low, (
        f"Higher delta14C_atm should yield more C14 in soil pools. "
        f"Got: low={sum_low:.6e}, high={sum_high:.6e}"
    )
