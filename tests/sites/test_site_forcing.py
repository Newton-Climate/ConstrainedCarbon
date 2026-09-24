from __future__ import annotations

import pathlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import jax.numpy as jnp

from ecosystem_complexity.model.configuration import load_config
from ecosystem_complexity.data.forcing import (
    _prepend_era5_history,
    align_daily_observations,
    load_fluxcom_forcing,
    load_fluxcom_observations,
)
from ecosystem_complexity.data.schemas import ForcingData, ObservationData
from ecosystem_complexity.sites.spec import SiteSpec, load_site_spec

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_load_site_spec_reads_fluxcom_metadata() -> None:
    spec = load_site_spec(
        str(REPO_ROOT / "configs" / "expansion" / "luquillo_experimental_forest.yaml")
    )
    assert spec.forcing_kind == "fluxcom"
    assert spec.er_observation_glob.endswith("_fluxcom_er.csv")
    assert spec.mat_c == 24.0
    assert spec.map_mm == 3800.0
    assert spec.observation_path == "fraction"


def test_load_fluxcom_forcing_builds_finite_daily_series(tmp_path: pathlib.Path) -> None:
    fluxcom_csv = tmp_path / "fluxcom.csv"
    pd.DataFrame(
        {
            "date": ["2001-01-01", "2001-01-02", "2001-01-03", "2001-01-04"],
            "gpp_gCm2day": [2.0, 2.5, 3.0, 2.8],
        }
    ).to_csv(fluxcom_csv, index=False)

    cfg = load_config(str(REPO_ROOT / "configs" / "israd_multisite_3pool_config.yaml"))
    model = SimpleNamespace(config=cfg)
    spec = SiteSpec(
        config_path="",
        config_stem="test_fluxcom",
        israd_name="Test",
        forcing_glob="",
        lat=18.3157,
        lon=-65.7487,
        label="Test FluxCom Site",
        biome="tropical wet forest",
        forcing_kind="fluxcom",
        observation_path="fraction",
        mat_c=24.0,
        map_mm=3800.0,
        elevation_m=250.0,
    )

    forcing = load_fluxcom_forcing(str(fluxcom_csv), model, spec)
    assert forcing.GPP_obs.shape[0] > 4
    assert float(np.asarray(forcing.time)[0]) == 4018.0  # 1981-01-01
    np.testing.assert_allclose(
        np.asarray(forcing.GPP_obs)[-4:], [2.0, 2.5, 3.0, 2.8]
    )
    assert np.isfinite(np.asarray(forcing.air_temp)).all()
    assert np.isfinite(np.asarray(forcing.soil_temp)).all()
    assert np.isfinite(np.asarray(forcing.soil_moisture)).all()
    assert np.isfinite(np.asarray(forcing.vpd)).all()


def test_load_fluxcom_forcing_sets_finite_permafrost_active_layer(tmp_path: pathlib.Path) -> None:
    fluxcom_csv = tmp_path / "fluxcom_permafrost.csv"
    pd.DataFrame(
        {
            "date": ["2005-01-01", "2005-06-01", "2005-12-01"],
            "GPP_obs": [0.05, 1.20, 0.03],
        }
    ).to_csv(fluxcom_csv, index=False)

    cfg = load_config(str(REPO_ROOT / "configs" / "israd_multisite_3pool_config.yaml"))
    model = SimpleNamespace(config=cfg)
    spec = SiteSpec(
        config_path="",
        config_stem="test_permafrost_fluxcom",
        israd_name="Permafrost Test",
        forcing_glob="",
        lat=68.0,
        lon=-149.0,
        label="Permafrost Test",
        biome="high-arctic tundra/permafrost",
        forcing_kind="fluxcom",
        observation_path="bulk_resp",
        mat_c=-7.0,
        map_mm=250.0,
        elevation_m=50.0,
    )

    forcing = load_fluxcom_forcing(str(fluxcom_csv), model, spec)
    active_layer = np.asarray(forcing.active_layer)
    assert np.isfinite(active_layer).all()
    assert (active_layer > 0.0).all()
    assert active_layer.max() > active_layer.min()


def test_load_fluxcom_observations_aligns_daily_er_to_reference_time(
    tmp_path: pathlib.Path,
) -> None:
    er_csv = tmp_path / "fluxcom_er.csv"
    pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-03"],
            "ER": [1.0, 3.0],
        }
    ).to_csv(er_csv, index=False)

    reference_time = np.array([18628.0, 18629.0, 18630.0], dtype=np.float32)
    observations = load_fluxcom_observations(str(er_csv), reference_time)

    np.testing.assert_allclose(np.asarray(observations.ER), [1.0, 2.0, 3.0])
    assert np.isnan(np.asarray(observations.NEE)).all()


def test_load_fluxcom_observations_does_not_extrapolate_outside_record(
    tmp_path: pathlib.Path,
) -> None:
    er_csv = tmp_path / "fluxcom_er.csv"
    pd.DataFrame({
        "date": ["2021-01-02", "2021-01-03"],
        "ER": [2.0, 3.0],
    }).to_csv(er_csv, index=False)
    reference_time = np.array([18628.0, 18629.0, 18630.0, 18631.0])
    observations = load_fluxcom_observations(str(er_csv), reference_time)
    assert np.isnan(np.asarray(observations.ER)[[0, 3]]).all()
    np.testing.assert_allclose(np.asarray(observations.ER)[1:3], [2.0, 3.0])


def test_era5_history_is_prepended_and_tower_observations_are_aligned(
    tmp_path: pathlib.Path,
) -> None:
    tower_path = tmp_path / "SITE_FLUXMET_DD_2001-2001.csv"
    tower_path.write_text("placeholder")
    pd.DataFrame({
        "TIMESTAMP": [20000101, 20000102],
        "TA_ERA": [1.0, 2.0],
        "SW_IN_ERA": [10.0, 20.0],
        "VPD_ERA": [0.5, 0.6],
        "P_ERA": [3.0, 4.0],
    }).to_csv(tmp_path / "SITE_ERA5_DD_1981-2001.csv", index=False)

    tower_days = np.array([11323.0, 11324.0], dtype=np.float32)  # 2001-01-01/02
    forcing = ForcingData(
        time=jnp.asarray(tower_days), air_temp=jnp.array([3.0, 4.0]),
        sw_radiation=jnp.array([30.0, 40.0]), precip=jnp.array([5.0, 6.0]),
        vpd=jnp.array([0.7, 0.8]), soil_temp=jnp.ones((2, 1)),
        soil_moisture=jnp.full((2, 1), 0.25), snow_depth=jnp.zeros(2),
        active_layer=jnp.full(2, jnp.inf), delta14C_atm=jnp.full(2, jnp.nan),
        GPP_obs=jnp.array([2.0, 4.0]), NPP_obs=jnp.full(2, jnp.nan),
    )
    extended = _prepend_era5_history(forcing, str(tower_path), 1)
    assert extended.time.shape == (4,)
    np.testing.assert_allclose(np.asarray(extended.air_temp[:2]), [1.0, 2.0])
    np.testing.assert_allclose(np.asarray(extended.GPP_obs[:2]), [2.0, 4.0])

    nan = jnp.full(2, jnp.nan)
    observations = ObservationData(
        time=jnp.asarray(tower_days), NEE=nan, GPP=nan, ER=jnp.array([7.0, 8.0]),
        NEE_unc=nan, delta14C_obs={}, deltaD14C_obs={}, C_pools_obs={},
    )
    aligned = align_daily_observations(observations, extended.time)
    assert np.isnan(np.asarray(aligned.ER[:2])).all()
    np.testing.assert_allclose(np.asarray(aligned.ER[2:]), [7.0, 8.0])
