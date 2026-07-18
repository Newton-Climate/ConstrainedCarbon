"""
Site-specific data loaders for the ecosystem-complexity carbon model.

load_harvard_forest()    — AmeriFlux HR FULLSET → ForcingData + ObservationData
load_barrow_alaska()     — ERA5_DD + FLUXMET_DD → ForcingData + ObservationData
load_eight_mile_lake()   — AmeriFlux BASE HH → ForcingData + ObservationData
load_howland_forest()    — AmeriFlux BASE HH → ForcingData + ObservationData
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import jax.numpy as jnp

from ecosystem_complexity.config import ModelConfig
from ecosystem_complexity.data.alignment import align_to_layers
from ecosystem_complexity.data.schemas import ForcingData, ObservationData

logger = logging.getLogger(__name__)

# Reference epoch
_EPOCH = pd.Timestamp("1970-01-01")

# Half-hourly μmol CO₂ m⁻² s⁻¹ → gC m⁻² per half-hour
_HH_TO_GC = 1e-6 * 12.0 * 1800.0  # = 0.02160

def _days_since_epoch(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(
        ((dates - _EPOCH) / pd.Timedelta("1D")).values, dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Harvard Forest
# ---------------------------------------------------------------------------


def load_harvard_forest(
    hr_path: str,
    config: ModelConfig,
    qc_threshold: float = 0.5,
    include_gpp_forcing: bool = False,
) -> tuple[ForcingData, ObservationData]:
    """
    Load Harvard Forest from the single half-hourly FULLSET CSV.

    Parameters
    ----------
    hr_path :
        Path to AMF_US-Ha1_FLUXNET_FULLSET_HR_*.csv
    config :
        ModelConfig (used for soil layer alignment).
    qc_threshold :
        Maximum allowed gap-fill fraction (0–1).  Half-hours with
        NEE_VUT_REF_QC > qc_threshold are set to NaN before aggregation.
    include_gpp_forcing :
        When True, populate ``ForcingData.GPP_obs`` with the QC-filtered
        daily GPP series (``GPP_NT_VUT_REF``) so it can be used as an
        external forcing input.  When False (default), ``GPP_obs`` is NaN.
    """
    df = pd.read_csv(hr_path, na_values=[-9999], low_memory=False)

    # Parse timestamp manually (12-digit integer YYYYMMDDhhmm).
    # Kept as a standalone grouping key rather than assigned into `df`: the
    # FULLSET frame is ~227 columns, and inserting into a frame that wide
    # fragments its block manager (pandas raises PerformanceWarning, and every
    # later column access pays for the fragmentation). `datetime` was only ever
    # used to derive `date`, and `date` only to group, so neither needs to be a
    # column. The name is what makes `reset_index()` below emit a "date" column.
    date_key = pd.to_datetime(
        df["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M"
    ).dt.date.rename("date")

    # ── QC filter ────────────────────────────────────────────────────────────
    if "NEE_VUT_REF_QC" in df.columns:
        bad_qc = df["NEE_VUT_REF_QC"] > qc_threshold
        for col in ["NEE_VUT_REF", "GPP_NT_VUT_REF", "RECO_NT_VUT_REF"]:
            if col in df.columns:
                df.loc[bad_qc, col] = np.nan

    # ── Unit conversion: μmol CO₂ m⁻² s⁻¹ → gC m⁻² per half-hour ──────────
    for col in ["NEE_VUT_REF", "GPP_NT_VUT_REF", "RECO_NT_VUT_REF"]:
        if col in df.columns:
            df[col] = df[col] * _HH_TO_GC

    # ── Daily aggregation ────────────────────────────────────────────────────
    def _daily_flux_sum(series: pd.Series) -> float:
        """Sum if ≥ 24 valid half-hours, else NaN."""
        valid = series.dropna()
        return float(valid.sum()) if len(valid) >= 24 else np.nan

    agg_dict: dict[str, Any] = {}

    # Fluxes: sum with ≥ 24 valid HH
    for raw, out in [
        ("NEE_VUT_REF", "NEE"),
        ("GPP_NT_VUT_REF", "GPP"),
        ("RECO_NT_VUT_REF", "ER"),
    ]:
        if raw in df.columns:
            agg_dict[out] = pd.NamedAgg(column=raw, aggfunc=_daily_flux_sum)

    # Uncertainty: daily mean of available values.
    # Guarded like the met aggregators below: a day whose half-hours are all NaN
    # (gap-filled uncertainty is absent for whole days early in the record) makes
    # np.nanmean warn "Mean of empty slice" and return NaN anyway. Returning NaN
    # explicitly keeps the same value without the warning — and, unlike silencing
    # it, still leaves a genuinely empty day as NaN rather than a fabricated 0.
    if "NEE_VUT_REF_RANDUNC" in df.columns:
        agg_dict["NEE_unc_hh"] = pd.NamedAgg(
            column="NEE_VUT_REF_RANDUNC",
            aggfunc=lambda s: (
                float(np.nanmean(s.dropna())) * _HH_TO_GC * 48
                if s.notna().any()
                else np.nan
            ),
        )

    # Met: daily mean / sum
    for met_col, out_name, method in [
        ("TA_F", "air_temp", "mean"),
        ("SW_IN_F", "sw_radiation", "mean"),
        ("VPD_F", "vpd", "mean"),
        ("P_F", "precip", "sum"),
    ]:
        if met_col in df.columns:
            aggfunc = np.nanmean if method == "mean" else np.nansum
            agg_dict[out_name] = pd.NamedAgg(column=met_col, aggfunc=lambda s, f=aggfunc: float(f(s.dropna())) if s.notna().any() else np.nan)

    # Soil temperature: daily mean
    ts_cols = ["TS_F_MDS_1", "TS_F_MDS_2", "TS_F_MDS_3", "TS_F_MDS_4"]
    ts_present = [c for c in ts_cols if c in df.columns]
    for ts_col in ts_present:
        agg_dict[ts_col] = pd.NamedAgg(
            column=ts_col,
            aggfunc=lambda s: float(np.nanmean(s.dropna())) if s.notna().any() else np.nan,
        )

    daily = df.groupby(date_key).agg(**agg_dict).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    T = len(daily)
    dates_idx = pd.DatetimeIndex(daily["date"])
    time_arr = _days_since_epoch(dates_idx)

    # ── Soil temperature alignment ───────────────────────────────────────────
    # Harvard TS sensor approximate depths (AmeriFlux BADM)
    ts_depths_top = np.array([0.00, 0.075, 0.15, 0.35])
    ts_depths_bot = np.array([0.075, 0.15, 0.35, 0.65])
    # Sensor midpoints: 0.05, 0.10, 0.20, 0.50 m
    ts_sensor_mid = np.array([0.05, 0.10, 0.20, 0.50])
    ts_depths_top = ts_sensor_mid - 0.025
    ts_depths_top = np.clip(ts_depths_top, 0, None)
    ts_depths_bot = ts_sensor_mid + 0.025

    n_layers = len(config.soil_layers)
    soil_temp_arr = np.full((T, n_layers), np.nan)
    soil_moist_arr = np.full((T, n_layers), np.nan)  # Harvard has no SWC

    ts_data = np.column_stack([
        daily[c].values if c in daily.columns else np.full(T, np.nan)
        for c in ts_cols
    ])  # (T, 4)

    for t in range(T):
        row_vals = ts_data[t]
        valid_mask = np.isfinite(row_vals)
        if valid_mask.any():
            layer_vals, _ = align_to_layers(
                ts_depths_top[valid_mask],
                ts_depths_bot[valid_mask],
                row_vals[valid_mask],
                np.ones(valid_mask.sum()),
                config,
            )
            soil_temp_arr[t] = np.array(layer_vals, dtype=np.float64)

    # ── Assemble arrays ──────────────────────────────────────────────────────
    def _col(name: str) -> np.ndarray:
        if name in daily.columns:
            return np.asarray(daily[name].values, dtype=np.float64)
        return np.full(T, np.nan)

    GPP_obs_arr = _col("GPP") if include_gpp_forcing else np.full(T, np.nan)

    forcing = ForcingData(
        time=jnp.array(time_arr, dtype=jnp.float32),
        air_temp=jnp.array(_col("air_temp"), dtype=jnp.float32),
        sw_radiation=jnp.array(_col("sw_radiation"), dtype=jnp.float32),
        precip=jnp.array(_col("precip"), dtype=jnp.float32),
        vpd=jnp.array(_col("vpd"), dtype=jnp.float32),
        soil_temp=jnp.array(soil_temp_arr, dtype=jnp.float32),
        soil_moisture=jnp.array(soil_moist_arr, dtype=jnp.float32),
        snow_depth=jnp.full(T, jnp.nan, dtype=jnp.float32),
        active_layer=jnp.full(T, jnp.inf, dtype=jnp.float32),
        delta14C_atm=jnp.full(T, jnp.nan, dtype=jnp.float32),
        GPP_obs=jnp.array(GPP_obs_arr, dtype=jnp.float32),
        NPP_obs=jnp.full(T, jnp.nan, dtype=jnp.float32),
    )

    nee_unc = _col("NEE_unc_hh")

    obs = ObservationData(
        time=jnp.array(time_arr, dtype=jnp.float32),
        NEE=jnp.array(_col("NEE"), dtype=jnp.float32),
        GPP=jnp.array(_col("GPP"), dtype=jnp.float32),
        ER=jnp.array(_col("ER"), dtype=jnp.float32),
        NEE_unc=jnp.array(nee_unc, dtype=jnp.float32),
        delta14C_obs={},
        deltaD14C_obs={},
        C_pools_obs={},
    )

    return forcing, obs


# ---------------------------------------------------------------------------
# Barrow, Alaska
# ---------------------------------------------------------------------------


def load_barrow_alaska(
    era5_path: str,
    fluxmet_path: str,
    config: ModelConfig,
    qc_threshold: float = 0.5,
    include_gpp_forcing: bool = False,
) -> tuple[ForcingData, ObservationData]:
    """
    Load Barrow, Alaska from ERA5_DD (met backbone) + FLUXMET_DD (fluxes + soil).

    Merges on date; flux/soil columns are NaN outside the FLUXMET range.

    Parameters
    ----------
    include_gpp_forcing :
        When True, populate ``ForcingData.GPP_obs`` with the QC-filtered
        daily GPP series (``GPP_NT_VUT_REF``) so it can be used as an
        external forcing input.  When False (default), ``GPP_obs`` is NaN.
    """
    # ── ERA5 DD (met backbone, 1981–2025) ────────────────────────────────────
    era5 = pd.read_csv(era5_path, na_values=[-9999], low_memory=False)
    era5["date"] = pd.to_datetime(era5["TIMESTAMP"].astype(str), format="%Y%m%d")

    # ── FLUXMET DD (fluxes + soil, 2011–2022) ────────────────────────────────
    fluxmet = pd.read_csv(fluxmet_path, na_values=[-9999], low_memory=False)
    fluxmet["date"] = pd.to_datetime(fluxmet["TIMESTAMP"].astype(str), format="%Y%m%d")

    # Apply QC to fluxes.
    # US-A10 FLUXNET DD uses the standard fraction-gap-filled convention
    # (0=all measured, 1=all gap-filled); mask rows where the gap-filled
    # fraction EXCEEDS the threshold (bad quality).
    if "NEE_CUT_REF_QC" in fluxmet.columns:
        bad_qc = fluxmet["NEE_CUT_REF_QC"] > qc_threshold
        for col in ["NEE_CUT_REF", "GPP_DT_CUT_REF", "RECO_NT_CUT_REF"]:
            if col in fluxmet.columns:
                fluxmet.loc[bad_qc, col] = np.nan

    # Apply QC to soil columns
    ts_cols = ["TS_F_MDS_1", "TS_F_MDS_2", "TS_F_MDS_3"]
    swc_cols = ["SWC_F_MDS_1", "SWC_F_MDS_2"]
    for ts_col, qc_col in [
        ("TS_F_MDS_1", "TS_F_MDS_1_QC"),
        ("TS_F_MDS_2", "TS_F_MDS_2_QC"),
        ("TS_F_MDS_3", "TS_F_MDS_3_QC"),
        ("SWC_F_MDS_1", "SWC_F_MDS_1_QC"),
        ("SWC_F_MDS_2", "SWC_F_MDS_2_QC"),
    ]:
        if ts_col in fluxmet.columns and qc_col in fluxmet.columns:
            fluxmet.loc[fluxmet[qc_col] > qc_threshold, ts_col] = np.nan

    # ── Merge (left join on ERA5 dates) ──────────────────────────────────────
    merged = era5.merge(fluxmet, on="date", how="left", suffixes=("_era5", "_fluxmet"))
    merged = merged.sort_values("date").reset_index(drop=True)

    T = len(merged)
    dates_idx = pd.DatetimeIndex(merged["date"])
    time_arr = _days_since_epoch(dates_idx)

    # ── Soil temperature alignment ───────────────────────────────────────────
    # Barrow TS sensor depths (NGEE-Arctic metadata)
    ts_sensor_mid = np.array([0.05, 0.20, 0.40])
    ts_depths_top = ts_sensor_mid - 0.025
    ts_depths_top = np.clip(ts_depths_top, 0, None)
    ts_depths_bot = ts_sensor_mid + 0.025

    n_layers = len(config.soil_layers)
    soil_temp_arr = np.full((T, n_layers), np.nan)
    soil_moist_arr = np.full((T, n_layers), np.nan)

    ts_col_names = [c + "_fluxmet" if c + "_fluxmet" in merged.columns else c for c in ts_cols]
    swc_col_names = [c + "_fluxmet" if c + "_fluxmet" in merged.columns else c for c in swc_cols]

    # Fallback: check plain names
    ts_col_names = [c if c in merged.columns else c.replace("_fluxmet", "") for c in ts_col_names]
    swc_col_names = [c if c in merged.columns else c.replace("_fluxmet", "") for c in swc_col_names]

    ts_data = np.column_stack([
        merged[c].values.astype(np.float64) if c in merged.columns else np.full(T, np.nan)
        for c in ts_cols
    ])

    swc_data = np.column_stack([
        merged[c].values.astype(np.float64) if c in merged.columns else np.full(T, np.nan)
        for c in swc_cols
    ])

    # SWC sensor depths: 0.05 m and 0.20 m
    swc_sensor_mid = np.array([0.05, 0.20])
    swc_depths_top = np.clip(swc_sensor_mid - 0.025, 0, None)
    swc_depths_bot = swc_sensor_mid + 0.025

    for t in range(T):
        # Soil temperature
        ts_row = ts_data[t]
        valid_ts = np.isfinite(ts_row)
        if valid_ts.any():
            layer_vals, _ = align_to_layers(
                ts_depths_top[valid_ts],
                ts_depths_bot[valid_ts],
                ts_row[valid_ts],
                np.ones(valid_ts.sum()),
                config,
            )
            soil_temp_arr[t] = np.array(layer_vals, dtype=np.float64)

        # Soil moisture (AmeriFlux SWC is in volumetric %; convert to m³ m⁻³)
        swc_row = swc_data[t] / 100.0
        valid_swc = np.isfinite(swc_row)
        if valid_swc.any():
            layer_vals_m, _ = align_to_layers(
                swc_depths_top[valid_swc],
                swc_depths_bot[valid_swc],
                swc_row[valid_swc],
                np.ones(valid_swc.sum()),
                config,
            )
            soil_moist_arr[t] = np.array(layer_vals_m, dtype=np.float64)

    # ── Active layer depth proxy from TS_F_MDS_3 (40 cm sensor) ─────────────
    # Rough proxy only — replace with dedicated ALT dataset in production.
    ts3_col = "TS_F_MDS_3"
    if ts3_col in merged.columns:
        ts3 = merged[ts3_col].fillna(0.0).values.astype(np.float64)
    else:
        ts3 = np.zeros(T)
    active_layer_arr = 0.20 + 0.20 / (1.0 + np.exp(-5.0 * ts3))  # sigmoid proxy

    # ── Helper to get column (handles _fluxmet suffix or plain) ─────────────
    def _get_col(name: str) -> np.ndarray:
        if name in merged.columns:
            return np.asarray(merged[name].values, dtype=np.float64)
        alt = name + "_fluxmet"
        if alt in merged.columns:
            return np.asarray(merged[alt].values, dtype=np.float64)
        return np.full(T, np.nan)

    # GPP: daytime-partitioned CUT variant (NT variant is absent for US-A10)
    GPP_obs_arr = _get_col("GPP_DT_CUT_REF") if include_gpp_forcing else np.full(T, np.nan)

    forcing = ForcingData(
        time=jnp.array(time_arr, dtype=jnp.float32),
        air_temp=jnp.array(_get_col("TA_ERA"), dtype=jnp.float32),
        sw_radiation=jnp.array(_get_col("SW_IN_ERA"), dtype=jnp.float32),
        precip=jnp.array(_get_col("P_ERA"), dtype=jnp.float32),
        vpd=jnp.array(_get_col("VPD_ERA"), dtype=jnp.float32),
        soil_temp=jnp.array(soil_temp_arr, dtype=jnp.float32),
        soil_moisture=jnp.array(soil_moist_arr, dtype=jnp.float32),
        snow_depth=jnp.full(T, jnp.nan, dtype=jnp.float32),
        active_layer=jnp.array(active_layer_arr, dtype=jnp.float32),
        delta14C_atm=jnp.full(T, jnp.nan, dtype=jnp.float32),
        GPP_obs=jnp.array(GPP_obs_arr, dtype=jnp.float32),
        NPP_obs=jnp.full(T, jnp.nan, dtype=jnp.float32),
    )

    # Uncertainty: scale from RANDUNC if available (daily file, already daily units)
    nee_unc = _get_col("NEE_CUT_REF_RANDUNC")

    obs = ObservationData(
        time=jnp.array(time_arr, dtype=jnp.float32),
        NEE=jnp.array(_get_col("NEE_CUT_REF"), dtype=jnp.float32),
        GPP=jnp.array(_get_col("GPP_DT_CUT_REF"), dtype=jnp.float32),
        ER=jnp.array(_get_col("RECO_NT_CUT_REF"), dtype=jnp.float32),
        NEE_unc=jnp.array(nee_unc, dtype=jnp.float32),
        delta14C_obs={},
        deltaD14C_obs={},
        C_pools_obs={},
    )

    return forcing, obs


# ---------------------------------------------------------------------------
# Eight-mile Lake, Alaska
# ---------------------------------------------------------------------------


def load_eight_mile_lake(
    hh_path: str,
    config: ModelConfig,
    include_gpp_forcing: bool = False,
) -> tuple[ForcingData, ObservationData]:
    """
    Load Eight-mile Lake (US-EML) from the AmeriFlux BASE half-hourly CSV.

    The EML BASE file has a commented metadata header and a compact
    single-depth soil schema (`TS`, `SWC`). The loader aggregates to daily
    forcing, converts half-hourly carbon fluxes to gC m⁻² day⁻¹, and
    broadcasts the single soil measurement to all model layers.
    """
    df = pd.read_csv(hh_path, comment="#", na_values=[-9999], low_memory=False)

    # Standalone grouping key rather than two inserts into the wide BASE frame
    # — see load_harvard_forest for why.
    date_key = pd.to_datetime(
        df["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M"
    ).dt.date.rename("date")

    for col in ["NEE_PI_F", "GPP_PI_F", "RECO_PI_F"]:
        if col in df.columns:
            df[col] = df[col] * _HH_TO_GC

    def _daily_flux_sum(series: pd.Series) -> float:
        valid = series.dropna()
        return float(valid.sum()) if len(valid) >= 24 else np.nan

    agg_dict: dict[str, Any] = {}
    for raw, out in [
        ("NEE_PI_F", "NEE"),
        ("GPP_PI_F", "GPP"),
        ("RECO_PI_F", "ER"),
    ]:
        if raw in df.columns:
            agg_dict[out] = pd.NamedAgg(column=raw, aggfunc=_daily_flux_sum)

    for met_col, out_name, method in [
        ("TA", "air_temp", "mean"),
        ("SW_IN", "sw_radiation", "mean"),
        ("P", "precip", "sum"),
        ("D_SNOW", "snow_depth", "mean"),
        ("TS", "soil_temp", "mean"),
        ("SWC", "soil_moisture", "mean"),
    ]:
        if met_col in df.columns:
            aggfunc = np.nanmean if method == "mean" else np.nansum
            agg_dict[out_name] = pd.NamedAgg(
                column=met_col,
                aggfunc=lambda s, f=aggfunc: float(f(s.dropna())) if s.notna().any() else np.nan,
            )

    daily = df.groupby(date_key).agg(**agg_dict).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    T = len(daily)
    dates_idx = pd.DatetimeIndex(daily["date"])
    time_arr = _days_since_epoch(dates_idx)
    n_layers = len(config.soil_layers)

    def _col(name: str) -> np.ndarray:
        if name in daily.columns:
            return np.asarray(daily[name].values, dtype=np.float64)
        return np.full(T, np.nan)

    soil_temp_1d = _col("soil_temp")
    soil_moist_1d = _col("soil_moisture")
    soil_moist_1d = np.where(np.isfinite(soil_moist_1d), soil_moist_1d / 100.0, np.nan)
    soil_temp_arr = np.repeat(soil_temp_1d[:, None], n_layers, axis=1)
    soil_moist_arr = np.repeat(soil_moist_1d[:, None], n_layers, axis=1)

    snow_depth = _col("snow_depth")
    snow_depth = np.where(np.isfinite(snow_depth), snow_depth / 100.0, np.nan)
    GPP_obs_arr = _col("GPP") if include_gpp_forcing else np.full(T, np.nan)

    forcing = ForcingData(
        time=jnp.array(time_arr, dtype=jnp.float32),
        air_temp=jnp.array(_col("air_temp"), dtype=jnp.float32),
        sw_radiation=jnp.array(_col("sw_radiation"), dtype=jnp.float32),
        precip=jnp.array(_col("precip"), dtype=jnp.float32),
        vpd=jnp.full(T, jnp.nan, dtype=jnp.float32),
        soil_temp=jnp.array(soil_temp_arr, dtype=jnp.float32),
        soil_moisture=jnp.array(soil_moist_arr, dtype=jnp.float32),
        snow_depth=jnp.array(snow_depth, dtype=jnp.float32),
        active_layer=jnp.full(T, jnp.inf, dtype=jnp.float32),
        delta14C_atm=jnp.full(T, jnp.nan, dtype=jnp.float32),
        GPP_obs=jnp.array(GPP_obs_arr, dtype=jnp.float32),
        NPP_obs=jnp.full(T, jnp.nan, dtype=jnp.float32),
    )

    obs = ObservationData(
        time=jnp.array(time_arr, dtype=jnp.float32),
        NEE=jnp.array(_col("NEE"), dtype=jnp.float32),
        GPP=jnp.array(_col("GPP"), dtype=jnp.float32),
        ER=jnp.array(_col("ER"), dtype=jnp.float32),
        NEE_unc=jnp.full(T, jnp.nan, dtype=jnp.float32),
        delta14C_obs={},
        deltaD14C_obs={},
        C_pools_obs={},
    )

    return forcing, obs


# ---------------------------------------------------------------------------
# Howland Forest, Maine
# ---------------------------------------------------------------------------


def load_howland_forest(
    dd_path: str,
    config: ModelConfig,
    qc_threshold: float = 0.5,
    include_gpp_forcing: bool = False,
) -> tuple[ForcingData, ObservationData]:
    """
    Load Howland Forest (US-Ho1) from the FLUXNET FULLSET daily CSV.

    Uses the gap-filled daily meteorology and carbon products already provided
    by the FULLSET package, including `GPP_NT_VUT_REF` for external soil-input
    forcing when `include_gpp_forcing=True`.
    """
    df = pd.read_csv(dd_path, na_values=[-9999], low_memory=False)
    dates = pd.to_datetime(df["TIMESTAMP"].astype(str), format="%Y%m%d")

    if "NEE_VUT_REF_QC" in df.columns:
        bad_qc = df["NEE_VUT_REF_QC"] > qc_threshold
        for col in ["NEE_VUT_REF", "GPP_NT_VUT_REF", "RECO_NT_VUT_REF"]:
            if col in df.columns:
                df.loc[bad_qc, col] = np.nan

    T = len(df)
    time_arr = _days_since_epoch(pd.DatetimeIndex(dates))
    n_layers = len(config.soil_layers)

    def _col(name: str) -> np.ndarray:
        if name in df.columns:
            return np.asarray(df[name].values, dtype=np.float64)
        return np.full(T, np.nan)

    ts_cols = [c for c in df.columns if c.startswith("TS_F_MDS_") and not c.endswith("_QC")]
    swc_cols = [c for c in df.columns if c.startswith("SWC_F_MDS_") and not c.endswith("_QC")]
    if ts_cols:
        ts_vals = df[ts_cols].values.astype(np.float64)
        ts_mask = np.isfinite(ts_vals)
        ts_count = ts_mask.sum(axis=1)
        ts_sum = np.where(ts_mask, ts_vals, 0.0).sum(axis=1)
        soil_temp_1d = np.full(T, np.nan)
        np.divide(ts_sum, ts_count, out=soil_temp_1d, where=ts_count > 0)
    else:
        soil_temp_1d = np.full(T, np.nan)
    if swc_cols:
        swc_vals = df[swc_cols].values.astype(np.float64)
        swc_mask = np.isfinite(swc_vals)
        swc_count = swc_mask.sum(axis=1)
        swc_sum = np.where(swc_mask, swc_vals, 0.0).sum(axis=1)
        soil_moist_1d = np.full(T, np.nan)
        np.divide(swc_sum, swc_count, out=soil_moist_1d, where=swc_count > 0)
    else:
        soil_moist_1d = np.full(T, np.nan)
    soil_moist_1d = np.where(np.isfinite(soil_moist_1d), soil_moist_1d / 100.0, np.nan)
    soil_temp_arr = np.repeat(soil_temp_1d[:, None], n_layers, axis=1)
    soil_moist_arr = np.repeat(soil_moist_1d[:, None], n_layers, axis=1)
    gpp_daily = _col("GPP_NT_VUT_REF") if include_gpp_forcing else np.full(T, np.nan)

    forcing = ForcingData(
        time=jnp.array(time_arr, dtype=jnp.float32),
        air_temp=jnp.array(_col("TA_F_MDS"), dtype=jnp.float32),
        sw_radiation=jnp.array(_col("SW_IN_F_MDS"), dtype=jnp.float32),
        precip=jnp.array(_col("P_F"), dtype=jnp.float32),
        vpd=jnp.array(_col("VPD_F_MDS"), dtype=jnp.float32),
        soil_temp=jnp.array(soil_temp_arr, dtype=jnp.float32),
        soil_moisture=jnp.array(soil_moist_arr, dtype=jnp.float32),
        snow_depth=jnp.full(T, jnp.nan, dtype=jnp.float32),
        active_layer=jnp.full(T, jnp.inf, dtype=jnp.float32),
        delta14C_atm=jnp.full(T, jnp.nan, dtype=jnp.float32),
        GPP_obs=jnp.array(gpp_daily, dtype=jnp.float32),
        NPP_obs=jnp.full(T, jnp.nan, dtype=jnp.float32),
    )

    obs = ObservationData(
        time=jnp.array(time_arr, dtype=jnp.float32),
        NEE=jnp.array(_col("NEE_VUT_REF"), dtype=jnp.float32),
        GPP=jnp.full(T, jnp.nan, dtype=jnp.float32),
        ER=jnp.array(_col("RECO_NT_VUT_REF"), dtype=jnp.float32),
        NEE_unc=jnp.array(_col("NEE_VUT_REF_RANDUNC"), dtype=jnp.float32),
        delta14C_obs={},
        deltaD14C_obs={},
        C_pools_obs={},
    )

    return forcing, obs


# ---------------------------------------------------------------------------
# attach_atm14C
