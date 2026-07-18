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

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ecosystem_complexity.data.loaders import load_howland_forest
from ecosystem_complexity.data.paths import REPO_ROOT as _REPO_ROOT
from ecosystem_complexity.data.schemas import ForcingData

# GPP columns in preference order (tropical FLUXNET only fills the DT variants).
_GPP_COLS = ("GPP_NT_VUT_REF", "GPP_DT_VUT_REF", "GPP_NT_CUT_REF", "GPP_DT_CUT_REF")


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
