"""Tests for ecosystem_complexity.data.loaders: site-specific data loaders.

Tests for the data pipeline.

All tests use synthetic data that mirrors the real file structures exactly.
No real data files are read.
"""

from __future__ import annotations

import io
import math
import os
import tempfile

import jax.numpy as jnp
import pandas as pd
import pytest

from ecosystem_complexity.config import ModelConfig, load_config
from ecosystem_complexity.data.loaders import (
    load_barrow_alaska,
    load_harvard_forest,
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

    nee_umol          : constant NEE value in umol CO2 m-2 s-1
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
                qc_val = 0.8  # > 0.5 threshold -> will be masked

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
    """Synthetic Graven 2017: pre-bomb baseline, near 0 permil."""
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
        lines.append(f"{cal_bp} {400 + cal_bp} 30 {delta14c:.2f} 5.0\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_harvard_hr_unit_conversion(hf_config):
    """1 umol m-2 s-1 x 48 HH x 1800 s x 1e-6 x 12 = 1.0368 gC m-2 day-1."""
    csv_content = _make_harvard_hr_csv(n_days=2, nee_umol=1.0)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    try:
        _, obs = load_harvard_forest(path, hf_config, qc_threshold=0.5)
        expected = 1.0 * 0.02160 * 48  # = 1.0368
        assert (
            abs(float(obs.NEE[0]) - expected) < 0.001
        ), f"Expected {expected:.4f}, got {float(obs.NEE[0]):.4f}"
    finally:
        os.unlink(path)


def test_harvard_qc_filter(hf_config):
    """Days where < 24 HH are valid -> daily NEE is NaN."""
    # 30 half-hours out of 48 have QC > 0.5 -> only 18 valid -> < 24 -> NaN
    csv_content = _make_harvard_hr_csv(n_days=2, nee_umol=1.0, qc_above_threshold=30)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    try:
        _, obs = load_harvard_forest(path, hf_config, qc_threshold=0.5)
        assert jnp.isnan(obs.NEE[0]), f"Expected NaN for day 0, got {obs.NEE[0]}"
        # Day 1 has all 48 valid -> should not be NaN
        assert not jnp.isnan(obs.NEE[1]), f"Day 1 should be valid, got {obs.NEE[1]}"
    finally:
        os.unlink(path)


def test_harvard_no_soil_moisture(hf_config):
    """Harvard CSV has no SWC columns -> soil_moisture must be all NaN."""
    csv_content = _make_harvard_hr_csv(n_days=2)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    try:
        forcing, _ = load_harvard_forest(path, hf_config)
        assert jnp.all(
            jnp.isnan(forcing.soil_moisture)
        ), "soil_moisture should be all NaN for Harvard (no SWC columns)"
    finally:
        os.unlink(path)


def test_harvard_active_layer_inf(hf_config):
    """Harvard is non-permafrost -> active_layer must be all inf."""
    csv_content = _make_harvard_hr_csv(n_days=2)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    try:
        forcing, _ = load_harvard_forest(path, hf_config)
        assert jnp.all(
            jnp.isinf(forcing.active_layer)
        ), "active_layer should be all inf for Harvard (non-permafrost)"
    finally:
        os.unlink(path)


def test_barrow_nee_uses_cut(barrow_config):
    """Barrow parser reads NEE_CUT_REF, not NEE_VUT_REF."""
    era5_csv = _make_barrow_era5_csv(n_days=5, start_year=2011)
    fluxmet_csv = _make_barrow_fluxmet_csv(n_days=3, start_year=2011, nee_cut=2.0)

    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f1,
        tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f2,
    ):
        f1.write(era5_csv)
        f2.write(fluxmet_csv)
        era5_path, fluxmet_path = f1.name, f2.name

    try:
        _, obs = load_barrow_alaska(era5_path, fluxmet_path, barrow_config)
        # Find a day that overlaps with FLUXMET (2011-06-01 onwards)
        # ERA5 starts 2011-01-01, FLUXMET starts 2011-06-01
        # With ERA5 n_days=5 starting 2011-01-01, days are Jan 1-5
        # FLUXMET starts 2011-06-01 -> no overlap in this fixture
        # Use aligned fixture: both start 2011-06-01
    finally:
        os.unlink(era5_path)
        os.unlink(fluxmet_path)

    # Re-run with matching dates
    era5_csv2 = _make_barrow_era5_csv(n_days=5, start_year=2011)
    # Patch ERA5 to start 2011-06-01 to overlap with FLUXMET
    era5_df = pd.read_csv(io.StringIO(era5_csv2))
    era5_df["TIMESTAMP"] = [
        int((pd.Timestamp("2011-06-01") + pd.Timedelta(days=i)).strftime("%Y%m%d"))
        for i in range(len(era5_df))
    ]
    era5_csv3 = era5_df.to_csv(index=False)

    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f1,
        tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f2,
    ):
        f1.write(era5_csv3)
        f2.write(fluxmet_csv)
        era5_path, fluxmet_path = f1.name, f2.name

    try:
        _, obs = load_barrow_alaska(era5_path, fluxmet_path, barrow_config)
        # First 3 days should have NEE from NEE_CUT_REF=2.0
        for i in range(3):
            assert not jnp.isnan(obs.NEE[i]), f"NEE[{i}] should not be NaN"
            assert (
                abs(float(obs.NEE[i]) - 2.0) < 0.01
            ), f"Expected NEE=2.0, got {float(obs.NEE[i]):.4f}"
    finally:
        os.unlink(era5_path)
        os.unlink(fluxmet_path)


def test_barrow_merge_dates(barrow_config):
    """ERA5 dates outside FLUXMET range -> NaN NEE but valid met."""
    # ERA5: 5 days starting 1981-01-01 (before FLUXMET range 2011-2022)
    # FLUXMET: 3 days starting 2011-06-01 (no overlap)
    era5_csv = _make_barrow_era5_csv(n_days=5, start_year=1981)
    fluxmet_csv = _make_barrow_fluxmet_csv(n_days=3, start_year=2011, nee_cut=2.0)

    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f1,
        tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f2,
    ):
        f1.write(era5_csv)
        f2.write(fluxmet_csv)
        era5_path, fluxmet_path = f1.name, f2.name

    try:
        forcing, obs = load_barrow_alaska(era5_path, fluxmet_path, barrow_config)
        # All ERA5 days are pre-2011 -> NEE should be NaN
        assert jnp.all(jnp.isnan(obs.NEE)), "NEE should be NaN for pre-FLUXMET dates"
        # Met forcing should be valid (from ERA5)
        assert not jnp.any(
            jnp.isnan(forcing.air_temp)
        ), "air_temp should be valid from ERA5"
    finally:
        os.unlink(era5_path)
        os.unlink(fluxmet_path)
