from __future__ import annotations

from pathlib import Path

import numpy as np

from ecosystem_complexity.model.api import build_model
from ecosystem_complexity.data.custom_14c import (
    build_custom_14c_observations,
    load_custom_14c_manifest,
)
from ecosystem_complexity.sites.spec import load_site_spec


def test_example_custom_lab_data_builds_observations():
    root = Path(__file__).resolve().parents[2]
    data = load_custom_14c_manifest(root / "examples/custom_14c/example_lab_14c.yaml")
    model = build_model(str(root / "configs/harvard_3pool_config.yaml"))
    forcing_time = np.arange(365 * 60, dtype=float)
    blocks, respiration = build_custom_14c_observations(
        data, forcing_time, model.pool_index
    )

    assert [block.name for block in blocks] == [
        "custom_bulk_2020_07_15",
        "custom_fraction_soil_active_2020_07_15",
        "custom_fraction_soil_passive_2020_07_15",
    ]
    assert float(blocks[1].y[0]) == 110.0
    assert np.isfinite(np.asarray(respiration)).sum() == 1


def test_custom_manifest_site_does_not_require_israd_name(tmp_path):
    config = tmp_path / "custom_site.yaml"
    config.write_text(
        "site:\n  name: My lab site\ndatasource:\n"
        "  forcing_glob: my_forcing\n"
        "  radiocarbon_manifest: ../../data/custom/my_site_14c.yaml\n"
    )
    spec = load_site_spec(str(config))
    assert spec.israd_name == "My lab site"
