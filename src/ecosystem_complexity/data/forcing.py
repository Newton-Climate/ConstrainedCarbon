"""Flux-tower forcing: reading, sanitising, annual-mean collapse.

Wraps :mod:`ecosystem_complexity.data.loaders` with the gap-filling and
GPP-column selection the inversion drivers need, so a tower record missing soil
temperature, or with an unpopulated GPP partition, still yields a finite series.

Everything here takes a path — never a ``SiteSpec``. Deciding *which* file a
configured site uses stays in :mod:`ecosystem_complexity.sites.forcing`, which
keeps ``data`` free of any dependency on ``sites``.
"""
from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ecosystem_complexity.data.loaders import load_howland_forest
from ecosystem_complexity.data.paths import REPO_ROOT as _REPO_ROOT
from ecosystem_complexity.data.schemas import ForcingData, ObservationData

if TYPE_CHECKING:
    from ecosystem_complexity.sites.spec import SiteSpec

# GPP columns in preference order (tropical FLUXNET only fills the DT variants).
_GPP_COLS = ("GPP_NT_VUT_REF", "GPP_DT_VUT_REF", "GPP_NT_CUT_REF", "GPP_DT_CUT_REF")
_EPOCH = pd.Timestamp("1970-01-01")


def resolve_dd_file(forcing_glob: str) -> str:
    """Find the flux DD csv (with GPP) inside a tower directory."""
    root = os.path.join(_REPO_ROOT, "data", forcing_glob)
    dd = [
        f for f in glob.glob(os.path.join(root, "**", "*_DD_*.csv"), recursive=True)
        if "ERA5" not in f and "BIFVARINFO" not in f
    ]
    for f in dd:
        with open(f, encoding="latin-1") as fh:
            if "GPP_NT_VUT_REF" in fh.readline():
                return f
    if not dd:
        raise FileNotFoundError(f"No flux DD file under data/{forcing_glob}")
    return dd[0]


def load_daily_forcing(path: str, model):
    """Read one FLUXNET daily product into a sanitised ForcingData.

    All FLUXNET DD files (AmeriFlux FULLSET and ICOS/EUF/JPF FLUXMET) share a
    column convention, so one reader covers every configured site.
    """
    forcing, _ = load_howland_forest(path, config=model.config, include_gpp_forcing=True)
    return _sanitize_forcing(forcing, _load_gpp_series(path))


def load_daily_observations(path: str, model) -> ObservationData:
    """Read the tower daily product into an ``ObservationData`` record."""
    _, observations = load_howland_forest(
        path, config=model.config, include_gpp_forcing=True
    )
    return observations


def load_fluxcom_forcing(path: str, model, spec: SiteSpec) -> ForcingData:
    """Build a synthetic daily forcing using FluxCom GPP and ISRaD climatology.

    This path is for radiocarbon sites without a colocated flux tower. The site
    still needs a full ``ForcingData`` record, so the loader combines a local
    FluxCom GPP time series with simple seasonal climate fields derived from the
    site's mean annual temperature/precipitation metadata from ISRaD.
    """
    gpp_df = _load_fluxcom_gpp(path)
    return _build_synthetic_site_forcing(
        model=model,
        spec=spec,
        dates=pd.DatetimeIndex(gpp_df["date"]),
        gpp=np.asarray(gpp_df["GPP_obs"], dtype=np.float64),
    )


def load_fluxcom_observations(
    path: str,
    reference_time: jnp.ndarray,
) -> ObservationData:
    """Read a site-level FluxCom ER series and align it to ``reference_time``."""
    er_df = _load_fluxcom_daily_series(
        path,
        value_cols=("ER", "er", "Reco", "reco", "TER", "ter"),
        output_col="ER",
    )
    dates = _reference_time_to_dates(reference_time)
    aligned = (
        er_df.set_index("date")
        .reindex(dates)
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )
    er = np.asarray(aligned["ER"], dtype=np.float32)
    time = jnp.array(np.asarray(reference_time), dtype=jnp.float32)
    nan = jnp.full(time.shape[0], jnp.nan, dtype=jnp.float32)
    return ObservationData(
        time=time,
        NEE=nan,
        GPP=nan,
        ER=jnp.array(er, dtype=jnp.float32),
        NEE_unc=nan,
        delta14C_obs={},
        deltaD14C_obs={},
        C_pools_obs={},
    )


def _load_gpp_series(dd_path: str) -> np.ndarray:
    """Robust, fully-finite daily GPP (gC m⁻² day⁻¹), aligned to the DD rows.

    Picks the best-populated GPP partition column (tropical towers only fill the
    daytime DT variants) and gap-fills the remaining days so the forcing has no
    NaNs — a single NaN GPP day propagates through the ODE and NaNs the cost.
    """
    df = pd.read_csv(dd_path, na_values=[-9999], low_memory=False)
    series = None
    for col in _GPP_COLS:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().mean() > 0.2:
            series = pd.to_numeric(df[col], errors="coerce")
            break
    if series is None:
        raise ValueError(f"No usable GPP column in {os.path.basename(dd_path)}")
    series = series.interpolate(limit_direction="both").ffill().bfill()
    return np.asarray(series.fillna(series.mean()).values, dtype=np.float64)


def _load_fluxcom_gpp(path: str) -> pd.DataFrame:
    """Return a daily ``date`` / ``GPP_obs`` frame from a FluxCom csv or netcdf."""
    return _load_fluxcom_daily_series(
        path,
        value_cols=("GPP_obs", "gpp_gCm2day", "gpp", "GPP"),
        output_col="GPP_obs",
    )


def _load_fluxcom_daily_series(
    path: str,
    *,
    value_cols: tuple[str, ...],
    output_col: str,
) -> pd.DataFrame:
    """Return a daily ``date`` / value frame from a FluxCom csv or netcdf."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        frame = pd.read_csv(path)
        date_col = next(
            (c for c in ("date", "time", "timestamp") if c in frame.columns), None
        )
        value_col = next((c for c in value_cols if c in frame.columns), None)
        if date_col is None or value_col is None:
            value_list = "/".join(value_cols)
            raise ValueError(
                f"FluxCom csv {os.path.basename(path)!r} must contain a date/time "
                f"column and one of {value_list}."
            )
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(frame[date_col]),
                output_col: pd.to_numeric(frame[value_col], errors="coerce"),
            }
        )
    elif ext in {".nc", ".nc4", ".netcdf"}:
        import xarray as xr

        ds = xr.open_dataset(path)
        data_var = next(
            (name for name in value_cols if name in ds.data_vars),
            None,
        )
        if data_var is None or "time" not in ds.coords:
            value_list = "/".join(value_cols)
            raise ValueError(
                f"FluxCom netcdf {os.path.basename(path)!r} must have a time "
                f"coordinate and one of {value_list}."
            )
        out = ds[[data_var]].to_dataframe().reset_index()[["time", data_var]]
        out = out.rename(columns={"time": "date", data_var: output_col})
    else:
        raise ValueError(
            f"Unsupported FluxCom file type {ext!r} for {os.path.basename(path)!r}."
        )

    out = out.dropna(subset=["date"]).sort_values("date")
    if out.empty:
        raise ValueError(f"FluxCom input {os.path.basename(path)!r} had no dated rows.")
    out["date"] = pd.to_datetime(out["date"])
    out[output_col] = pd.to_numeric(out[output_col], errors="coerce")
    daily = (
        out.set_index("date")
        .resample("D")
        .mean()
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
        .reset_index()
    )
    if daily[output_col].isna().all():
        raise ValueError(
            f"FluxCom input {os.path.basename(path)!r} had no finite {output_col}."
        )
    daily[output_col] = daily[output_col].fillna(float(daily[output_col].mean()))
    return daily


def _reference_time_to_dates(reference_time: jnp.ndarray) -> pd.DatetimeIndex:
    days = np.asarray(reference_time, dtype=np.float64)
    return pd.DatetimeIndex(_EPOCH + pd.to_timedelta(days, unit="D"))


def _build_synthetic_site_forcing(
    *,
    model,
    spec: SiteSpec,
    dates: pd.DatetimeIndex,
    gpp: np.ndarray,
) -> ForcingData:
    """Construct a finite daily forcing from site MAT/MAP plus FluxCom GPP."""
    n_layers = max(len(model.config.soil_layers), 1)
    lat = float(spec.lat)
    mat_c = float(spec.mat_c if spec.mat_c is not None else 10.0)
    map_mm = float(spec.map_mm if spec.map_mm is not None else 1000.0)
    time = ((dates - _EPOCH) / pd.Timedelta("1D")).to_numpy(dtype=np.float32)
    doy = dates.dayofyear.to_numpy(dtype=np.float64)
    lat_rad = np.deg2rad(lat)
    seasonal = np.sin(2.0 * np.pi * (doy - 80.0) / 365.25)
    temp_amp = float(np.clip(4.0 + abs(lat) / 4.0, 4.0, 18.0))
    air_temp = mat_c + temp_amp * seasonal
    soil_amp = temp_amp * np.linspace(0.65, 0.25, n_layers, dtype=np.float64)
    soil_phase = np.linspace(0.0, 35.0, n_layers, dtype=np.float64)
    soil_temp = np.stack(
        [
            mat_c + amp * np.sin(2.0 * np.pi * (doy - 80.0 - lag) / 365.25)
            for amp, lag in zip(soil_amp, soil_phase)
        ],
        axis=1,
    )
    # Mild seasonal radiation and moisture structure; these are placeholders
    # for non-tower sites and are intentionally conservative rather than tuned.
    sw_radiation = np.clip(
        180.0 + 140.0 * np.cos(lat_rad) * np.sin(2.0 * np.pi * (doy - 172.0) / 365.25),
        20.0,
        None,
    )
    precip = np.clip((map_mm / 365.25) * (1.0 + 0.35 * np.cos(2.0 * np.pi * doy / 365.25)), 0.0, None)
    moisture_base = float(np.clip(0.18 + map_mm / 4000.0, 0.18, 0.48))
    soil_moisture = np.clip(
        moisture_base - 0.03 * seasonal[:, None] + np.linspace(0.01, -0.02, n_layers),
        0.08,
        0.60,
    )
    vpd = np.clip(0.35 + np.maximum(air_temp - 2.0, 0.0) * 0.08, 0.2, 3.5)
    snow_depth = np.where(air_temp < 0.0, np.clip(0.02 * precip, 0.0, 0.25), 0.0)
    is_permafrost = "permafrost" in spec.biome.lower() or mat_c < -1.0
    if is_permafrost:
        thaw = np.clip((air_temp + 8.0) / 18.0, 0.05, 1.0)
        active_layer = 0.15 + 0.65 * thaw
    else:
        active_layer = np.full_like(air_temp, np.inf)

    return ForcingData(
        time=jnp.array(time, dtype=jnp.float32),
        air_temp=jnp.array(air_temp, dtype=jnp.float32),
        sw_radiation=jnp.array(sw_radiation, dtype=jnp.float32),
        precip=jnp.array(precip, dtype=jnp.float32),
        vpd=jnp.array(vpd, dtype=jnp.float32),
        soil_temp=jnp.array(soil_temp, dtype=jnp.float32),
        soil_moisture=jnp.array(soil_moisture, dtype=jnp.float32),
        snow_depth=jnp.array(snow_depth, dtype=jnp.float32),
        active_layer=jnp.array(active_layer, dtype=jnp.float32),
        delta14C_atm=jnp.full(len(dates), jnp.nan, dtype=jnp.float32),
        GPP_obs=jnp.array(gpp, dtype=jnp.float32),
        NPP_obs=jnp.full(len(dates), jnp.nan, dtype=jnp.float32),
    )


def _sanitize_forcing(forcing, gpp: np.ndarray):
    """Override GPP with the robust series and make climate drivers finite.

    Fills missing soil temperature with air temperature (then 10 °C) and missing
    soil moisture with 0.30 v/v, so towers lacking TS/SWC (e.g. SJ-Adv) still run.
    """
    air = np.array(forcing.air_temp, dtype=np.float64)
    air = np.where(np.isfinite(air), air, np.nanmean(air) if np.isfinite(np.nanmean(air)) else 10.0)
    st = np.array(forcing.soil_temp, dtype=np.float64)
    st = np.where(np.isfinite(st), st, air[:, None])
    st = np.where(np.isfinite(st), st, 10.0)
    sm = np.array(forcing.soil_moisture, dtype=np.float64)
    sm = np.where(np.isfinite(sm), sm, 0.30)
    return forcing._replace(
        air_temp=jnp.array(air, dtype=jnp.float32),
        soil_temp=jnp.array(st, dtype=jnp.float32),
        soil_moisture=jnp.array(sm, dtype=jnp.float32),
        GPP_obs=jnp.array(gpp, dtype=jnp.float32),
    )


def _repeat_mean_field(arr: np.ndarray, n_days: int) -> np.ndarray:
    """Repeat the finite time mean of a forcing field into a synthetic year."""
    if arr.ndim == 1:
        if np.isposinf(arr).all():
            return np.full((n_days,), np.inf, dtype=np.float32)
        if np.isnan(arr).all():
            return np.full((n_days,), 0.0, dtype=np.float32)
        mean_val = np.nanmean(arr)
        if not np.isfinite(mean_val):
            mean_val = 0.0
        return np.full((n_days,), mean_val, dtype=np.float32)

    if np.isnan(arr).all():
        return np.zeros((n_days,) + arr.shape[1:], dtype=np.float32)
    mean_val = np.nanmean(arr, axis=0)
    mean_val = np.where(np.isfinite(mean_val), mean_val, 0.0).astype(np.float32)
    return np.repeat(mean_val[None, :], n_days, axis=0)


def build_annual_mean_forcing(forcing: ForcingData, n_days: int = 365) -> ForcingData:
    """Collapse a site forcing record to a synthetic constant annual-mean year."""
    return ForcingData(
        time=jnp.arange(n_days, dtype=jnp.float32),
        air_temp=jnp.array(_repeat_mean_field(np.array(forcing.air_temp), n_days)),
        sw_radiation=jnp.array(_repeat_mean_field(np.array(forcing.sw_radiation), n_days)),
        precip=jnp.array(_repeat_mean_field(np.array(forcing.precip), n_days)),
        vpd=jnp.array(_repeat_mean_field(np.array(forcing.vpd), n_days)),
        soil_temp=jnp.array(_repeat_mean_field(np.array(forcing.soil_temp), n_days)),
        soil_moisture=jnp.array(_repeat_mean_field(np.array(forcing.soil_moisture), n_days)),
        snow_depth=jnp.array(_repeat_mean_field(np.array(forcing.snow_depth), n_days)),
        active_layer=jnp.array(_repeat_mean_field(np.array(forcing.active_layer), n_days)),
        delta14C_atm=jnp.array(_repeat_mean_field(np.array(forcing.delta14C_atm), n_days)),
        GPP_obs=jnp.array(_repeat_mean_field(np.array(forcing.GPP_obs), n_days)),
        NPP_obs=jnp.array(_repeat_mean_field(np.array(forcing.NPP_obs), n_days)),
    )
