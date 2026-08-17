from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ecosystem_complexity.data.schemas import ForcingData
from ecosystem_complexity.synthesis.warming import repeat_forcing, warm_forcing


def _forcing() -> ForcingData:
    return ForcingData(
        time=jnp.array([0.0, 1.0], dtype=jnp.float32),
        air_temp=jnp.array([1.0, 2.0], dtype=jnp.float32),
        sw_radiation=jnp.array([10.0, 20.0], dtype=jnp.float32),
        precip=jnp.array([0.5, 0.0], dtype=jnp.float32),
        vpd=jnp.array([0.2, 0.3], dtype=jnp.float32),
        soil_temp=jnp.array([[0.0], [1.0]], dtype=jnp.float32),
        soil_moisture=jnp.array([[0.3], [0.4]], dtype=jnp.float32),
        snow_depth=jnp.array([0.0, 0.0], dtype=jnp.float32),
        active_layer=jnp.array([jnp.inf, jnp.inf], dtype=jnp.float32),
        delta14C_atm=jnp.array([0.0, 0.0], dtype=jnp.float32),
        GPP_obs=jnp.array([2.0, 3.0], dtype=jnp.float32),
        NPP_obs=jnp.array([1.0, 1.5], dtype=jnp.float32),
    )


def test_repeat_forcing_tiles_to_requested_horizon() -> None:
    repeated = repeat_forcing(_forcing(), horizon_years=5.0 / 365.25)
    assert repeated.time.shape == (5,)
    np.testing.assert_allclose(np.asarray(repeated.air_temp), [1.0, 2.0, 1.0, 2.0, 1.0])


def test_warm_forcing_offsets_air_and_soil_temperature() -> None:
    warmed = warm_forcing(_forcing(), delta_c=4.0)
    np.testing.assert_allclose(np.asarray(warmed.air_temp), [5.0, 6.0])
    np.testing.assert_allclose(np.asarray(warmed.soil_temp[:, 0]), [4.0, 5.0])
