from __future__ import annotations

import pathlib
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ecosystem_complexity.model.configuration import load_config
from ecosystem_complexity.data.forcing import (
    load_fluxcom_forcing,
    load_fluxcom_observations,
)
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
    assert forcing.GPP_obs.shape == (4,)
    np.testing.assert_allclose(np.asarray(forcing.GPP_obs), [2.0, 2.5, 3.0, 2.8])
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
