"""
Tests for the data pipeline.

All tests use synthetic data that mirrors the real file structures exactly.
No real data files are read.
"""

from __future__ import annotations

import math
import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from ecosystem_complexity.config import ModelConfig, load_config
from ecosystem_complexity.data.alignment import align_to_layers
from ecosystem_complexity.data.loaders import (
    load_harvard_forest,
)
from ecosystem_complexity.data.parsers import (
    validate_forcing,
)
from ecosystem_complexity.data.parsers_14C import (
    fm_to_delta14C,
    load_full_14C_record,
)

# ---------------------------------------------------------------------------
# Shared fixture: minimal Harvard Forest config (3 soil layers)
# ---------------------------------------------------------------------------

_HARVARD_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "harvard_forest.yaml"
)
_BARROW_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "barrow_alaska.yaml"
)


@pytest.fixture(scope="module")
def hf_config() -> ModelConfig:
    return load_config(_HARVARD_CONFIG_PATH)


@pytest.fixture(scope="module")
def barrow_config() -> ModelConfig:
    return load_config(_BARROW_CONFIG_PATH)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

_SENTINEL = -9999


def _make_harvard_hr_csv(
    n_days: int = 2,
    nee_umol: float = 1.0,
    qc_above_threshold: int = 0,
) -> str:
    """
    Build a minimal Harvard Forest HR CSV string with exact column structure.

    nee_umol          : constant NEE value in μmol CO₂ m⁻² s⁻¹
    qc_above_threshold: number of half-hours in day 0 with QC > 0.5
    """
    rows = []
    base = pd.Timestamp("2000-01-01")
    hh_per_day = 48

    for day in range(n_days):
        for hh in range(hh_per_day):
            ts_start = base + pd.Timedelta(days=day) + pd.Timedelta(minutes=30 * hh)
            ts_end = ts_start + pd.Timedelta(minutes=30)

            qc_val = 0.0
            if day == 0 and hh < qc_above_threshold:
                qc_val = 0.8  # > 0.5 threshold → will be masked

            rows.append(
                {
                    "TIMESTAMP_START": int(ts_start.strftime("%Y%m%d%H%M")),
                    "TIMESTAMP_END": int(ts_end.strftime("%Y%m%d%H%M")),
                    "NEE_VUT_REF": nee_umol,
                    "NEE_VUT_REF_QC": qc_val,
                    "NEE_VUT_REF_RANDUNC": 0.1,
                    "GPP_NT_VUT_REF": nee_umol * 2,
                    "RECO_NT_VUT_REF": nee_umol,
                    "TA_F": 15.0,
                    "SW_IN_F": 200.0,
                    "VPD_F": 10.0,
                    "P_F": 0.1,
                    "WS_F": 2.0,
                    "TS_F_MDS_1": 12.0,
                    "TS_F_MDS_2": 11.0,
                    "TS_F_MDS_3": 10.0,
                    "TS_F_MDS_4": 9.0,
                    "TS_F_MDS_1_QC": 0,
                    "TS_F_MDS_2_QC": 0,
                    "TS_F_MDS_3_QC": 0,
                    "TS_F_MDS_4_QC": 0,
                }
            )

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def _make_barrow_era5_csv(n_days: int = 5, start_year: int = 1981) -> str:
    rows = []
    base = pd.Timestamp(f"{start_year}-01-01")
    for d in range(n_days):
        dt = base + pd.Timedelta(days=d)
        rows.append(
            {
                "TIMESTAMP": int(dt.strftime("%Y%m%d")),
                "TA_ERA": -10.0,
                "SW_IN_ERA": 50.0,
                "LW_IN_ERA": 200.0,
                "VPD_ERA": 2.0,
                "PA_ERA": 101.0,
                "P_ERA": 1.0,
                "WS_ERA": 3.0,
                "TA_ERA_NIGHT": -12.0,
                "TA_ERA_NIGHT_SD": 1.0,
                "TA_ERA_DAY": -8.0,
                "TA_ERA_DAY_SD": 1.0,
            }
        )
    return pd.DataFrame(rows).to_csv(index=False)


def _make_barrow_fluxmet_csv(
    n_days: int = 3,
    start_year: int = 2011,
    nee_cut: float = 2.0,
) -> str:
    rows = []
    base = pd.Timestamp(f"{start_year}-06-01")
    for d in range(n_days):
        dt = base + pd.Timedelta(days=d)
        rows.append(
            {
                "TIMESTAMP": int(dt.strftime("%Y%m%d")),
                "NEE_CUT_REF": nee_cut,
                "NEE_CUT_REF_QC": 0.1,
                "NEE_CUT_REF_RANDUNC": 0.2,
                "GPP_NT_VUT_REF": nee_cut * 2,
                "RECO_NT_VUT_REF": nee_cut,
                "TS_F_MDS_1": 5.0,
                "TS_F_MDS_1_QC": 0.0,
                "TS_F_MDS_2": 3.0,
                "TS_F_MDS_2_QC": 0.0,
                "TS_F_MDS_3": 1.0,
                "TS_F_MDS_3_QC": 0.0,
                "SWC_F_MDS_1": 0.35,
                "SWC_F_MDS_1_QC": 0.0,
                "SWC_F_MDS_2": 0.40,
                "SWC_F_MDS_2_QC": 0.0,
            }
        )
    return pd.DataFrame(rows).to_csv(index=False)


def _make_hua_csv() -> str:
    """Synthetic Hua 2021: flat pre-bomb, bomb spike around 1963, decay after."""
    years = list(range(1941, 2020))
    vals = []
    for y in years:
        if 1955 <= y <= 1970:
            # Crude bomb spike centred at 1963
            vals.append(200.0 + 500.0 * math.exp(-0.5 * ((y - 1963) / 4.0) ** 2))
        else:
            vals.append(0.0)
    df = pd.DataFrame(
        {"Year.AD": years, "NH14C": vals, "SH14C": [v * 0.9 for v in vals]}
    )
    return df.to_csv(index=False)


def _make_graven_csv() -> str:
    """Synthetic Graven 2017: pre-bomb baseline, near 0‰."""
    years = [1850.5 + i for i in range(91)]  # 1850.5 to 1940.5
    vals = [-5.0 + 0.05 * (y - 1850) for y in years]  # slight Suess effect
    df = pd.DataFrame(
        {
            "Unnamed: 0": range(len(years)),
            "Date": years,
            "NHc14": vals,
            "Tropicsc14": vals,
            "SHc14": [v - 2 for v in vals],
            "Globalc13": [-7.0] * len(years),
        }
    )
    return df.to_csv(index=False)


def _make_intcal20_content() -> str:
    """Synthetic IntCal20: annual data from 500 BP (AD 1450) to 0 BP (AD 1950)."""
    lines = ["# IntCal20 synthetic test data\n"]
    for cal_bp in range(500, -1, -1):
        delta14c = -5.0 + 0.005 * cal_bp  # small pre-industrial variation
        lines.append(f"{cal_bp},{400 + cal_bp},30,{delta14c:.2f},5.0\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_atm14C_splice():
    """Bomb spike at 1963.5 > 100‰; splice continuity at 1940 < 20‰."""
    hua_content = _make_hua_csv()
    graven_content = _make_graven_csv()
    intcal_content = _make_intcal20_content()

    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as fh,
        tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as fg,
        tempfile.NamedTemporaryFile(mode="w", suffix=".14c", delete=False) as fi,
    ):
        fh.write(hua_content)
        fg.write(graven_content)
        fi.write(intcal_content)
        hua_path, graven_path, intcal_path = fh.name, fg.name, fi.name

    try:
        years, d14c = load_full_14C_record(
            hua_path,
            graven_path,
            intcal_path,
            hemisphere="NH",
            start_year=1500.0,
            end_year=2019.0,
        )
        # Bomb spike check
        idx_1963 = np.argmin(np.abs(years - 1963.5))
        assert (
            d14c[idx_1963] > 100.0
        ), f"Expected bomb spike > 100‰ at 1963.5, got {d14c[idx_1963]:.1f}‰"
        # Splice continuity at 1940
        idx_1940 = np.argmin(np.abs(years - 1940.0))
        jump = abs(d14c[idx_1940] - d14c[max(0, idx_1940 - 1)])
        assert jump < 20.0, f"Splice jump at 1940 = {jump:.1f}‰ > 20‰"
    finally:
        for p in [hua_path, graven_path, intcal_path]:
            os.unlink(p)


def test_intcal20_parsing():
    """cal_BP=0 → year_AD=1950; value near pre-bomb baseline (−50 to +50 ‰)."""
    hua_content = _make_hua_csv()
    graven_content = _make_graven_csv()
    intcal_content = _make_intcal20_content()

    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as fh,
        tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as fg,
        tempfile.NamedTemporaryFile(mode="w", suffix=".14c", delete=False) as fi,
    ):
        fh.write(hua_content)
        fg.write(graven_content)
        fi.write(intcal_content)
        hua_path, graven_path, intcal_path = fh.name, fg.name, fi.name

    try:
        years, d14c = load_full_14C_record(
            hua_path,
            graven_path,
            intcal_path,
            start_year=1500.0,
            end_year=2019.0,
        )
        idx_1950 = np.argmin(np.abs(years - 1950.0))
        assert (
            -50 < d14c[idx_1950] < 50
        ), f"Pre-bomb 1950 value should be near 0‰, got {d14c[idx_1950]:.1f}‰"
    finally:
        for p in [hua_path, graven_path, intcal_path]:
            os.unlink(p)


def test_align_to_layers_depth_weighted(hf_config):
    """Two measurements spanning Harvard's organic layer → weighted average."""
    # Harvard organic layer: 0.00–0.05 m
    # Measurement 0: 0.00–0.05 m, value=10.0 (full overlap)
    # Measurement 1: 0.05–0.10 m, value=20.0 (no overlap with organic layer)
    vals, uncs = align_to_layers(
        [0.00, 0.05],
        [0.05, 0.10],
        [10.0, 20.0],
        [1.0, 1.0],
        hf_config,
        method="depth_weighted",
    )
    # organic layer (0–0.05 m): measurement 0 has full overlap → 10.0
    assert (
        abs(float(vals[0]) - 10.0) < 0.01
    ), f"organic layer should be 10.0, got {float(vals[0]):.4f}"
    # mineral_A layer (0.05–0.20 m): measurement 1 covers 0.05–0.10 (partial overlap)
    assert not jnp.isnan(vals[1]), "mineral_A should have a value from meas 1"


def test_align_no_overlap_returns_nan(hf_config):
    """Measurement at 0.5–1.0 m has no overlap with organic layer (0–0.05 m)."""
    vals, _ = align_to_layers(
        [0.5], [1.0], [100.0], [5.0], hf_config, method="depth_weighted"
    )
    # organic layer (0–0.05 m) has no overlap with [0.5, 1.0]
    assert jnp.isnan(vals[0]), f"Expected NaN for organic layer, got {float(vals[0])}"


def test_israd_depth_convention(hf_config):
    """Organic horizon at lyr_top=-5 cm → depth_top_m=0.0 after conversion."""
    # Depth convention: negative lyr_top → clamp to 0.0 m
    tops_cm = np.array([-5.0, 10.0])
    bots_cm = np.array([0.0, 20.0])
    tops_m = np.where(tops_cm < 0, 0.0, tops_cm / 100.0)
    bots_m = np.maximum(bots_cm, 0.0) / 100.0

    assert tops_m[0] == 0.0, "Negative lyr_top should map to 0.0 m"
    assert abs(tops_m[1] - 0.10) < 1e-9, "10 cm should map to 0.10 m"

    # Also verify it can be passed to align_to_layers without error
    vals, _ = align_to_layers(tops_m, bots_m, [50.0, 30.0], [5.0, 5.0], hf_config)
    # organic layer (0–0.05 m): measurement 0 covers 0–0.0 m → but lyr_bot=0 → 0 width
    # After clamp: tops_m[0]=0.0, bots_m[0]=0.0 → zero thickness measurement, no overlap
    # This should not raise an error


def test_israd_uncertainty_priority():
    """lyr_14c_sigma preferred over lyr_14c_sd; lyr_14c_sd used if no sigma."""
    from ecosystem_complexity.data.parsers_14C import _get_uncertainty

    row_both = {"lyr_14c_sigma": 3.0, "lyr_14c_sd": 8.0}
    assert _get_uncertainty(row_both, "lyr_14c_sigma", "lyr_14c_sd") == 3.0

    row_sd_only = {"lyr_14c_sigma": float("nan"), "lyr_14c_sd": 8.0}
    assert _get_uncertainty(row_sd_only, "lyr_14c_sigma", "lyr_14c_sd") == 8.0

    row_neither = {"lyr_14c_sigma": float("nan"), "lyr_14c_sd": float("nan")}
    assert _get_uncertainty(row_neither, "lyr_14c_sigma", "lyr_14c_sd") == 5.0


def test_fm_to_delta14C():
    """Fm=1.0 at obs_year=1950 → Δ¹⁴C = 0.0‰."""
    result = fm_to_delta14C(1.0, 1950.0)
    assert abs(result) < 1e-9, f"Expected 0.0, got {result}"


def test_attach_atm14C(hf_config):
    """attach_atm14C fills delta14C_atm from interpolated record."""
    csv_content = _make_harvard_hr_csv(n_days=3)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    try:
        forcing, _ = load_harvard_forest(path, hf_config)
        assert jnp.all(
            jnp.isnan(forcing.delta14C_atm)
        ), "Before attach_atm14C, delta14C_atm should be NaN"

        # Simple synthetic record
        years_daily = np.linspace(1990.0, 2005.0, 5479)
        d14c_daily = np.full(5479, 50.0)
        from ecosystem_complexity.data.parsers import attach_atm14C

        forcing2 = attach_atm14C(forcing, d14c_daily, years_daily)
        assert not jnp.any(
            jnp.isnan(forcing2.delta14C_atm)
        ), "After attach_atm14C, delta14C_atm should not be NaN"
    finally:
        os.unlink(path)


def test_validate_forcing_warns_nan(hf_config):
    """validate_forcing returns a warning when > 5% of air_temp is NaN."""
    csv_content = _make_harvard_hr_csv(n_days=3)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    try:
        forcing, _ = load_harvard_forest(path, hf_config)
        # Inject NaN into air_temp
        at = np.array(forcing.air_temp)
        at[:] = np.nan  # 100% NaN
        forcing2 = forcing._replace(air_temp=jnp.array(at, dtype=jnp.float32))
        warnings = validate_forcing(forcing2, hf_config)
        assert any(
            "air_temp" in w for w in warnings
        ), f"Expected air_temp NaN warning, got: {warnings}"
    finally:
        os.unlink(path)
