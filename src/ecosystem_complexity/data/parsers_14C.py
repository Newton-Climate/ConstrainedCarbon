"""
Atmospheric and soil radiocarbon data loaders.

load_full_14C_record()  — splice IntCal20 / Graven 2017 / Hua 2021 into a
                           daily continuous record
load_israd_14C()        — load ISRaD layer/fraction observations, map to pools
fm_to_delta14C()        — helper: fraction-modern → Δ¹⁴C ‰
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from ecosystem_complexity.config import ModelConfig
from ecosystem_complexity.data.alignment import align_to_layers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atmospheric ¹⁴C record
# ---------------------------------------------------------------------------


def load_full_14C_record(
    hua_path: str,
    graven_path: str,
    intcal_path: str,
    hemisphere: str = "NH",
    start_year: float = 1500.0,
    end_year: float = 2023.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Splice IntCal20 / Graven 2017 / Hua 2021 into a daily Δ¹⁴C ‰ record.

    Returns
    -------
    years_daily : np.ndarray float64  (decimal years)
    delta14C_daily : np.ndarray float64  (Δ¹⁴C ‰)
    """
    # ── Hua 2021 ────────────────────────────────────────────────────────────
    hua = pd.read_csv(hua_path)
    hua_col = "NH14C" if hemisphere == "NH" else "SH14C"
    hua = hua.dropna(subset=["Year.AD", hua_col])
    hua_years = hua["Year.AD"].values.astype(np.float64)
    hua_vals = hua[hua_col].values.astype(np.float64)

    # ── Graven 2017 ─────────────────────────────────────────────────────────
    graven = pd.read_csv(graven_path)
    graven_col = "NHc14" if hemisphere == "NH" else "SHc14"
    graven = graven.dropna(subset=["Date", graven_col])
    graven_years = graven["Date"].values.astype(np.float64)
    graven_vals = graven[graven_col].values.astype(np.float64)

    # ── IntCal20 ─────────────────────────────────────────────────────────────
    intcal = pd.read_csv(
        intcal_path,
        comment="#",
        header=None,
        names=["cal_BP", "age14C", "sigma_age", "Delta14C", "sigma_D14C"],
        sep=r"\s+",
        engine="python",
    )
    intcal = intcal.dropna(subset=["cal_BP", "Delta14C"])
    intcal["year_AD"] = 1950.0 - intcal["cal_BP"].values
    # File is oldest-first; reverse so year_AD is ascending
    intcal = intcal.sort_values("year_AD")
    intcal_years = intcal["year_AD"].values.astype(np.float64)
    intcal_vals = intcal["Delta14C"].values.astype(np.float64)

    # ── Splice ───────────────────────────────────────────────────────────────
    # IntCal20: [start_year, 1850)
    mask_ic = (intcal_years >= start_year) & (intcal_years < 1850.0)
    # Graven:   [1850, 1941)
    mask_gr = (graven_years >= 1850.0) & (graven_years < 1941.0)
    # Hua:      [1941, end_year]
    mask_hu = (hua_years >= 1941.0) & (hua_years <= end_year)

    splice_years = np.concatenate([
        intcal_years[mask_ic],
        graven_years[mask_gr],
        hua_years[mask_hu],
    ])
    splice_vals = np.concatenate([
        intcal_vals[mask_ic],
        graven_vals[mask_gr],
        hua_vals[mask_hu],
    ])

    # Check continuity at splice points
    _check_splice_continuity(intcal_years[mask_ic], intcal_vals[mask_ic],
                             graven_years[mask_gr], graven_vals[mask_gr],
                             label="IntCal20/Graven at 1850")
    _check_splice_continuity(graven_years[mask_gr], graven_vals[mask_gr],
                             hua_years[mask_hu], hua_vals[mask_hu],
                             label="Graven/Hua at 1941")

    # Sort by year (should already be sorted, but defensive)
    order = np.argsort(splice_years)
    splice_years = splice_years[order]
    splice_vals = splice_vals[order]

    # ── Interpolate to daily ─────────────────────────────────────────────────
    cs = CubicSpline(splice_years, splice_vals, extrapolate=False)
    n_days = int(round((end_year - start_year) * 365.25))
    years_daily = np.linspace(start_year, end_year, n_days)
    delta14C_daily = cs(years_daily)

    # Fill any NaN at edges with nearest valid
    valid = np.isfinite(delta14C_daily)
    if not valid.all():
        first_valid = np.argmax(valid)
        last_valid = len(valid) - 1 - np.argmax(valid[::-1])
        delta14C_daily[:first_valid] = delta14C_daily[first_valid]
        delta14C_daily[last_valid + 1:] = delta14C_daily[last_valid]

    # Validation: NH bomb spike should peak > 100‰ around 1963–1965
    if hemisphere == "NH":
        idx_1963 = np.argmin(np.abs(years_daily - 1963.5))
        if delta14C_daily[idx_1963] <= 100.0:
            warnings.warn(
                f"NH Δ¹⁴C at 1963.5 = {delta14C_daily[idx_1963]:.1f}‰; "
                "expected > 100‰ for bomb spike — check Hua 2021 data."
            )

    return years_daily, delta14C_daily


def _check_splice_continuity(
    years_a: np.ndarray, vals_a: np.ndarray,
    years_b: np.ndarray, vals_b: np.ndarray,
    label: str,
    threshold: float = 20.0,
) -> None:
    if len(years_a) == 0 or len(years_b) == 0:
        return
    boundary = years_b[0]
    # Evaluate both datasets near the boundary
    idx_a = np.argmin(np.abs(years_a - boundary))
    idx_b = 0
    jump = abs(float(vals_a[idx_a]) - float(vals_b[idx_b]))
    if jump > threshold:
        logger.warning(
            "Splice discontinuity at %s: %.1f‰ jump (threshold %.1f‰)",
            label, jump, threshold,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def fm_to_delta14C(fm: float, obs_year: float) -> float:
    """Convert fraction modern to Δ¹⁴C ‰."""
    return (fm * np.exp((1.0 / 8267.0) * (1950.0 - obs_year)) - 1.0) * 1000.0


def _get_uncertainty(row: dict | pd.Series, sigma_col: str, sd_col: str,
                     default: float = 5.0) -> float:
    sigma = row.get(sigma_col, np.nan)
    if pd.notna(sigma) and sigma > 0:
        return float(sigma)
    sd = row.get(sd_col, np.nan)
    if pd.notna(sd) and sd > 0:
        return float(sd)
    return default


# ---------------------------------------------------------------------------
# ISRaD radiocarbon observations
# ---------------------------------------------------------------------------


def load_israd_14C(
    layer_path: str,
    fraction_path: str,
    config: ModelConfig,
    site_name_filter: str | None = None,
    lat_filter: float | None = None,
    lon_filter: float | None = None,
    radius_deg: float = 2.0,
    pool_name_map: dict[str, str] | None = None,
    prefer_fractions: bool = True,
    use_dd14c: bool = True,
) -> tuple[dict, dict]:
    """
    Load ISRaD radiocarbon observations and map to model pool names.

    Returns
    -------
    delta14C_obs  : {pool_name: (Δ14C ‰, uncertainty ‰, year)}
    deltaD14C_obs : {pool_name: (Δ-Δ14C ‰, uncertainty ‰, year)}
    """
    if pool_name_map is None:
        pool_name_map = {}

    # ── Load layer file ──────────────────────────────────────────────────────
    lyr = pd.read_csv(layer_path, low_memory=False)
    lyr = _apply_site_filter(lyr, site_name_filter, lat_filter, lon_filter, radius_deg)
    lyr = lyr.dropna(subset=["lyr_14c"])

    # ── Load fraction file ───────────────────────────────────────────────────
    frc = pd.read_csv(fraction_path, low_memory=False)
    frc = _apply_site_filter(frc, site_name_filter, lat_filter, lon_filter, radius_deg)
    frc = frc.dropna(subset=["frc_14c"])

    delta14C_obs: dict[str, tuple] = {}
    deltaD14C_obs: dict[str, tuple] = {}

    # ── Layer-level observations → model layers via depth alignment ──────────
    if not lyr.empty:
        tops_cm = lyr["lyr_top"].values
        bots_cm = lyr["lyr_bot"].values
        d14c_vals = lyr["lyr_14c"].values.astype(np.float64)
        sigmas = np.array([
            _get_uncertainty(row, "lyr_14c_sigma", "lyr_14c_sd")
            for _, row in lyr.iterrows()
        ])

        # Convert depth convention: cm with 0=mineral-organic interface,
        # negative = organic. Convert to positive-downward metres.
        tops_m = np.where(tops_cm < 0, 0.0, tops_cm / 100.0)
        bots_m = np.maximum(bots_cm, 0.0) / 100.0

        layer_vals, layer_uncs = align_to_layers(
            tops_m, bots_m, d14c_vals, sigmas, config, method="depth_weighted"
        )

        for li, layer in enumerate(config.soil_layers):
            v = float(layer_vals[li])
            u = float(layer_uncs[li])
            if not np.isnan(v):
                # Use midpoint observation year
                obs_years = lyr["lyr_obs_date_y"].dropna().values
                year = float(np.nanmean(obs_years)) if len(obs_years) > 0 else np.nan
                # Name: use first SOM pool name for this layer
                pool_name = f"{layer.name}_{layer.som_pools[0].name}"
                delta14C_obs[pool_name] = (v, u if not np.isnan(u) else 5.0, year)

        # Δ-Δ¹⁴C from lyr_dd14c if available and use_dd14c
        if use_dd14c and "lyr_dd14c" in lyr.columns:
            dd_vals = lyr["lyr_dd14c"].dropna().values.astype(np.float64)
            if len(dd_vals) > 0:
                dd_layer_vals, dd_layer_uncs = align_to_layers(
                    tops_m, bots_m,
                    lyr["lyr_dd14c"].fillna(np.nan).values.astype(np.float64),
                    sigmas, config, method="depth_weighted"
                )
                for li, layer in enumerate(config.soil_layers):
                    v = float(dd_layer_vals[li])
                    u = float(dd_layer_uncs[li])
                    if not np.isnan(v):
                        obs_years = lyr["lyr_obs_date_y"].dropna().values
                        year = float(np.nanmean(obs_years)) if len(obs_years) > 0 else np.nan
                        pool_name = f"{layer.name}_{layer.som_pools[0].name}"
                        deltaD14C_obs[pool_name] = (v, u if not np.isnan(u) else 5.0, year)

    # ── Fraction-level observations → named pools via pool_name_map ──────────
    if prefer_fractions and not frc.empty and pool_name_map:
        # Only density fractions
        density_frc = frc[frc.get("frc_scheme", pd.Series(dtype=str)).str.lower() == "density"] \
            if "frc_scheme" in frc.columns else frc

        # Accumulate per mapped pool
        pool_measurements: dict[str, list[tuple[float, float, float]]] = {}

        for _, row in density_frc.iterrows():
            frc_name = str(row.get("frc_name", "")).lower().strip()
            # Find matching pool name
            mapped = None
            for key, pool in pool_name_map.items():
                if key.lower() in frc_name:
                    mapped = pool
                    break
            if mapped is None:
                continue

            d14c = row.get("frc_14c", np.nan)
            if pd.isna(d14c):
                # Try fm conversion
                fm = row.get("frc_fm", np.nan)
                obs_yr = row.get("frc_obs_date_y", np.nan)
                if pd.notna(fm) and pd.notna(obs_yr):
                    d14c = fm_to_delta14C(float(fm), float(obs_yr))
                else:
                    continue

            unc = _get_uncertainty(row, "frc_14c_sigma", "frc_14c_sd")
            obs_year_val = row.get("frc_obs_date_y", np.nan)
            if pd.isna(obs_year_val):
                obs_year_val = row.get("lyr_obs_date_y", np.nan)

            if pd.notna(d14c):
                pool_measurements.setdefault(mapped, []).append(
                    (float(d14c), float(unc), float(obs_year_val) if pd.notna(obs_year_val) else np.nan)
                )

            # Δ-Δ¹⁴C from frc_dd14c if present
            if use_dd14c and "frc_dd14c" in frc.columns:
                dd = row.get("frc_dd14c", np.nan)
                if pd.notna(dd):
                    pass  # handled below

        # Weighted mean per pool
        for pool, meas in pool_measurements.items():
            vals_arr = np.array([m[0] for m in meas])
            uncs_arr = np.array([m[1] for m in meas])
            years_arr = np.array([m[2] for m in meas])
            inv_var = 1.0 / np.maximum(uncs_arr, 1e-6) ** 2
            wmean = np.sum(inv_var * vals_arr) / np.sum(inv_var)
            wcomb = 1.0 / np.sqrt(np.sum(inv_var))
            year_mean = float(np.nanmean(years_arr))
            delta14C_obs[pool] = (float(wmean), float(wcomb), year_mean)

        # Δ-Δ¹⁴C from frc_dd14c
        if use_dd14c and "frc_dd14c" in density_frc.columns:
            dd_pool_meas: dict[str, list[tuple[float, float, float]]] = {}
            for _, row in density_frc.iterrows():
                frc_name = str(row.get("frc_name", "")).lower().strip()
                mapped = None
                for key, pool in pool_name_map.items():
                    if key.lower() in frc_name:
                        mapped = pool
                        break
                if mapped is None:
                    continue
                dd = row.get("frc_dd14c", np.nan)
                if pd.isna(dd):
                    continue
                unc = _get_uncertainty(row, "frc_14c_sigma", "frc_14c_sd")
                obs_year_val = row.get("frc_obs_date_y", np.nan)
                if pd.isna(obs_year_val):
                    obs_year_val = row.get("lyr_obs_date_y", np.nan)
                dd_pool_meas.setdefault(mapped, []).append(
                    (float(dd), float(unc), float(obs_year_val) if pd.notna(obs_year_val) else np.nan)
                )

            for pool, meas in dd_pool_meas.items():
                vals_arr = np.array([m[0] for m in meas])
                uncs_arr = np.array([m[1] for m in meas])
                years_arr = np.array([m[2] for m in meas])
                inv_var = 1.0 / np.maximum(uncs_arr, 1e-6) ** 2
                wmean = np.sum(inv_var * vals_arr) / np.sum(inv_var)
                wcomb = 1.0 / np.sqrt(np.sum(inv_var))
                deltaD14C_obs[pool] = (float(wmean), float(wcomb), float(np.nanmean(years_arr)))

    return delta14C_obs, deltaD14C_obs


def _apply_site_filter(
    df: pd.DataFrame,
    site_name_filter: str | None,
    lat_filter: float | None,
    lon_filter: float | None,
    radius_deg: float,
) -> pd.DataFrame:
    if site_name_filter is not None and "site_name" in df.columns:
        df = df[df["site_name"].str.contains(site_name_filter, case=False, na=False)]
    if lat_filter is not None and "site_lat" in df.columns:
        df = df[np.abs(df["site_lat"] - lat_filter) < radius_deg]
    if lon_filter is not None and "site_long" in df.columns:
        df = df[np.abs(df["site_long"] - lon_filter) < radius_deg]
    return df
