"""
Site-specific data loaders for the ecosystem-complexity carbon model.

slice_forcing()         — slice all fields of a ForcingData to a time window
load_harvard_forest()   — AmeriFlux HR FULLSET → ForcingData + ObservationData
load_barrow_alaska()    — ERA5_DD + FLUXMET_DD → ForcingData + ObservationData
attach_atm14C()         — attach interpolated atmospheric Δ¹⁴C to ForcingData
validate_forcing()      — sanity-check a ForcingData, return warning strings
"""
from __future__ import annotations

import logging
from datetime import date

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
# = 1e-6 mol × 12 g/mol × 1800 s
_HH_TO_GC = 1e-6 * 12.0 * 1800.0  # = 0.02160


def slice_forcing(forcing: ForcingData, start: int, end: int) -> ForcingData:
    """
    Slice all time-axis fields of a ``ForcingData`` to ``[start:end]``.

    Parameters
    ----------
    forcing :
        Source ``ForcingData`` (all fields shape ``(T, ...)``).
    start, end :
        Integer indices into the time axis.  Follows standard Python slice
        semantics: ``end`` is exclusive, negative indices are supported.

    Returns
    -------
    ForcingData
        New object with every field sliced; same dtype as input.
    """
    return ForcingData(
        time=forcing.time[start:end],
        air_temp=forcing.air_temp[start:end],
        sw_radiation=forcing.sw_radiation[start:end],
        precip=forcing.precip[start:end],
        vpd=forcing.vpd[start:end],
        soil_temp=forcing.soil_temp[start:end],
        soil_moisture=forcing.soil_moisture[start:end],
        snow_depth=forcing.snow_depth[start:end],
        active_layer=forcing.active_layer[start:end],
        delta14C_atm=forcing.delta14C_atm[start:end],
        GPP_obs=forcing.GPP_obs[start:end],
        NPP_obs=forcing.NPP_obs[start:end],
    )


def _days_since_epoch(dates: pd.DatetimeIndex) -> np.ndarray:
    return ((dates - _EPOCH) / pd.Timedelta("1D")).values.astype(np.float64)


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

    # Parse timestamp manually (12-digit integer YYYYMMDDhhmm)
    df["datetime"] = pd.to_datetime(
        df["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M"
    )
    df["date"] = df["datetime"].dt.date

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

    agg_dict: dict = {}

    # Fluxes: sum with ≥ 24 valid HH
    for raw, out in [
        ("NEE_VUT_REF", "NEE"),
        ("GPP_NT_VUT_REF", "GPP"),
        ("RECO_NT_VUT_REF", "ER"),
    ]:
        if raw in df.columns:
            agg_dict[out] = pd.NamedAgg(column=raw, aggfunc=_daily_flux_sum)

    # Uncertainty: daily mean of available values
    if "NEE_VUT_REF_RANDUNC" in df.columns:
        agg_dict["NEE_unc_hh"] = pd.NamedAgg(
            column="NEE_VUT_REF_RANDUNC", aggfunc=lambda s: float(np.nanmean(s)) * _HH_TO_GC * 48
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

    daily = df.groupby("date").agg(**agg_dict).reset_index()
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
            return daily[name].values.astype(np.float64)
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
            return merged[name].values.astype(np.float64)
        alt = name + "_fluxmet"
        if alt in merged.columns:
            return merged[alt].values.astype(np.float64)
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
# attach_atm14C
# ---------------------------------------------------------------------------


def attach_atm14C(
    forcing: ForcingData,
    atm14C_record: np.ndarray,
    years_daily: np.ndarray,
) -> ForcingData:
    """
    Attach the atmospheric ¹⁴C record to a ForcingData object.

    Interpolates atm14C_record (on years_daily grid) to the exact dates
    stored in forcing.time (days since 1970-01-01).
    """
    # Convert forcing.time (days since epoch) to decimal years
    forcing_time_np = np.array(forcing.time, dtype=np.float64)
    forcing_years = 1970.0 + forcing_time_np / 365.25

    delta14C_interp = np.interp(forcing_years, years_daily, atm14C_record)

    return forcing._replace(
        delta14C_atm=jnp.array(delta14C_interp, dtype=jnp.float32)
    )


# ---------------------------------------------------------------------------
# validate_forcing
# ---------------------------------------------------------------------------


def validate_forcing(
    forcing: ForcingData,
    config: ModelConfig,
) -> list[str]:
    """
    Sanity-check a ForcingData. Returns warning strings (does not raise).
    """
    warnings_out: list[str] = []

    def _nan_frac(arr: jnp.ndarray) -> float:
        a = np.array(arr, dtype=np.float64).ravel()
        return float(np.isnan(a).mean())

    scalar_fields = {
        "air_temp": forcing.air_temp,
        "sw_radiation": forcing.sw_radiation,
        "precip": forcing.precip,
        "vpd": forcing.vpd,
        "snow_depth": forcing.snow_depth,
    }
    for name, arr in scalar_fields.items():
        frac = _nan_frac(arr)
        if frac > 0.05:
            warnings_out.append(f"{name}: {frac*100:.1f}% NaN (threshold 5%)")

    # Physical range checks
    air_temp_np = np.array(forcing.air_temp, dtype=np.float64)
    valid_at = air_temp_np[~np.isnan(air_temp_np)]
    if len(valid_at) > 0:
        if valid_at.min() < -70 or valid_at.max() > 50:
            warnings_out.append(
                f"air_temp out of range [-70, 50]°C: "
                f"min={valid_at.min():.1f}, max={valid_at.max():.1f}"
            )

    sw_np = np.array(forcing.sw_radiation, dtype=np.float64)
    valid_sw = sw_np[~np.isnan(sw_np)]
    if len(valid_sw) > 0:
        if valid_sw.min() < 0 or valid_sw.max() > 1400:
            warnings_out.append(
                f"sw_radiation out of range [0, 1400] W m⁻²: "
                f"min={valid_sw.min():.1f}, max={valid_sw.max():.1f}"
            )

    sm_np = np.array(forcing.soil_moisture, dtype=np.float64).ravel()
    valid_sm = sm_np[~np.isnan(sm_np)]
    if len(valid_sm) > 0:
        if valid_sm.min() < 0 or valid_sm.max() > 0.7:
            warnings_out.append(
                f"soil_moisture out of range [0, 0.7] m³ m⁻³: "
                f"min={valid_sm.min():.3f}, max={valid_sm.max():.3f}"
            )

    precip_np = np.array(forcing.precip, dtype=np.float64)
    valid_pr = precip_np[~np.isnan(precip_np)]
    if len(valid_pr) > 0 and valid_pr.min() < 0:
        warnings_out.append(f"precip has negative values: min={valid_pr.min():.3f}")

    d14c_np = np.array(forcing.delta14C_atm, dtype=np.float64)
    if np.all(np.isnan(d14c_np)):
        warnings_out.append("delta14C_atm is all NaN — call attach_atm14C() to populate")

    # Check for gaps > 14 consecutive NaN days in NEE — requires ObservationData,
    # but we only have ForcingData here; skip if not passed.

    return warnings_out


def validate_obs_nee_gaps(obs: ObservationData, max_gap: int = 14) -> list[str]:
    """Check for runs of > max_gap consecutive NaN days in NEE."""
    warnings_out: list[str] = []
    nee = np.array(obs.NEE, dtype=np.float64)
    is_nan = np.isnan(nee)
    run = 0
    max_run = 0
    for v in is_nan:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    if max_run > max_gap:
        warnings_out.append(
            f"NEE has a run of {max_run} consecutive NaN days (threshold {max_gap})"
        )
    return warnings_out
