from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from ecosystem_complexity.model.api import build_model
from ecosystem_complexity.data.israd_observations import (
    FractionMappingRule,
    add_layer_midpoint,
    build_fraction_obs_blocks,
    obs_blocks_from_single_year_summary,
    obs_dict_from_single_year_summary,
    summarize_by_depth,
)

CONFIGS_DIR = pathlib.Path(__file__).parent.parent / "configs"
_HF_3POOL_PATH = str(CONFIGS_DIR / "harvard_3pool_config.yaml")


@pytest.fixture(scope="module")
def hf_3pool_model():
    return build_model(_HF_3POOL_PATH)


def test_summarize_by_depth():
    df = pd.DataFrame(
        {
            "lyr_top": [-10, 0, 20, 25],
            "lyr_bot": [0, 20, 40, 45],
            "lyr_14c": [50.0, -10.0, -200.0, -240.0],
        }
    )
    df = add_layer_midpoint(df)
    summary = summarize_by_depth(
        df,
        "lyr_14c",
        [
            ("soil_active", (-15.0, 0.0)),
            ("soil_slow", (0.0, 20.0)),
            ("soil_passive", (20.0, 50.0)),
        ],
        min_n=1,
    )
    assert summary["soil_active"][0] == pytest.approx(50.0)
    assert summary["soil_slow"][0] == pytest.approx(-10.0)
    assert summary["soil_passive"][0] == pytest.approx(-220.0)
    assert summary["soil_active"][1] == pytest.approx(25.0)
    assert summary["soil_slow"][1] == pytest.approx(25.0)
    assert summary["soil_passive"][1] == pytest.approx(28.284271247461902)


def test_obs_dict_and_blocks_from_single_year_summary(hf_3pool_model):
    forcing_time = np.arange(20, dtype=float)
    summary = {
        "soil_active": (25.0, 15.0, 3),
        "soil_slow": (-80.0, 20.0, 4),
    }
    obs_dict = obs_dict_from_single_year_summary(summary, forcing_time, 1970.0)
    assert set(obs_dict) == {"soil_active", "soil_slow"}
    assert np.isfinite(np.array(obs_dict["soil_active"])).sum() == 1

    blocks = obs_blocks_from_single_year_summary(
        summary, hf_3pool_model.pool_index, forcing_time, 1970.0
    )
    assert [block.name for block in blocks] == [
        "israd_layer_soil_active",
        "israd_layer_soil_slow",
    ]


def test_build_fraction_obs_blocks_weighted_and_depth_filtered(hf_3pool_model):
    df = pd.DataFrame(
        {
            "entry_name": ["A", "A", "B", "B", "C"],
            "frc_property": [
                "free light",
                "free light",
                "heavy",
                "heavy",
                "free light",
            ],
            "frc_14c": [100.0, 0.0, -200.0, -100.0, 999.0],
            "frc_mass_perc": [3.0, 1.0, 1.0, 1.0, 1.0],
            "lyr_top": [0.0, 0.0, 0.0, 0.0, 20.0],
            "lyr_bot": [5.0, 5.0, 40.0, 40.0, 25.0],
        }
    )
    df = add_layer_midpoint(df)
    rows = build_fraction_obs_blocks(
        df,
        forcing_time=np.arange(365, dtype=float),
        pool_index=hf_3pool_model.pool_index,
        rules=[
            FractionMappingRule("soil_active", "free light", (0.0, 10.0)),
            FractionMappingRule("soil_passive", "heavy", (0.0, 60.0)),
        ],
        entry_to_year={"A": 1996, "B": 2007, "C": 1996},
        weight_col="frc_mass_perc",
        min_sigma=15.0,
        singleton_sigma=50.0,
        name_prefix="israd_density",
    )
    assert len(rows) == 2

    active = next(row for row in rows if row["pool_name"] == "soil_active")
    passive = next(row for row in rows if row["pool_name"] == "soil_passive")

    assert active["obs_year"] == 1996
    assert active["mean"] == pytest.approx(75.0)
    assert passive["obs_year"] == 2007
    assert passive["mean"] == pytest.approx(-150.0)
    assert active["block"].name == "israd_density_soil_active_1996"
