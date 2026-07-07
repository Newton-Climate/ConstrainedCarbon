"""
Helpers for reading global CESM2 CMIP output from ``data/cmip``.

This supersedes the older ``download_clm.py`` per-grid-cell workflow. The
global NetCDFs are opened directly and the nearest land-model grid cell is
selected for each analysis site.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import glob
import math
import os

import numpy as np
import xarray as xr

_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CMIP_DIR = os.path.join(_SCRIPT_ROOT, "data", "cmip")
SEC_PER_YR = 365.25 * 86400.0


@dataclass(frozen=True)
class SiteSpec:
    key: str
    code: str
    label: str
    short_label: str
    lat: float
    lon: float


SITE_SPECS: tuple[SiteSpec, ...] = (
    SiteSpec(
        key="harvard_forest",
        code="US-Ha1",
        label="Harvard Forest",
        short_label="HF",
        lat=42.5377,
        lon=-72.1715,
    ),
    SiteSpec(
        key="barrow",
        code="US-A10",
        label="Barrow, Alaska",
        short_label="Barrow",
        lat=71.2750,
        lon=-156.5500,
    ),
    SiteSpec(
        key="howland_forest",
        code="US-Ho1",
        label="Howland Forest",
        short_label="Howland",
        lat=45.1670,
        lon=-68.6670,
    ),
    SiteSpec(
        key="eight_mile_lake",
        code="US-EML",
        label="Eight-mile Lake",
        short_label="EML",
        lat=63.878361,
        lon=-149.253583,
    ),
)

SITE_INDEX = {site.key: site for site in SITE_SPECS}

_VAR_PATTERNS = {
    "cSoilFast": "cSoilFast_Lmon_CESM2_historical_r8i1p1f1_gn_*.nc",
    "cSoilMedium": "cSoilMedium_Lmon_CESM2_historical_r8i1p1f1_gn_*.nc",
    "cSoilSlow": "cSoilSlow_Lmon_CESM2_historical_r8i1p1f1_gn_*.nc",
    "cSoil": "cSoil_Emon_CESM2_historical_r8i1p1f1_gn_*.nc",
    "rhSoil": "rhSoil_Emon_CESM2_historical_r8i1p1f1_gn_*.nc",
    "npp": "npp_Lmon_CESM2_historical_r8i1p1f1_gn_*.nc",
}


def get_site_spec(site_key: str) -> SiteSpec:
    try:
        return SITE_INDEX[site_key]
    except KeyError as exc:
        raise KeyError(f"unknown CMIP site key: {site_key}") from exc


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_earth = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * r_earth * math.asin(math.sqrt(a))


@lru_cache(maxsize=None)
def _open_global_var(var_name: str) -> xr.DataArray:
    pattern = _VAR_PATTERNS[var_name]
    paths = sorted(glob.glob(os.path.join(CMIP_DIR, pattern)))
    if not paths:
        raise FileNotFoundError(f"no files matched {pattern} in {CMIP_DIR}")

    arrays = [xr.open_dataset(path)[var_name] for path in paths]
    return xr.concat(arrays, dim="time").sortby("time")


def _select_site_cell(da: xr.DataArray, site: SiteSpec) -> tuple[xr.DataArray, float, float, float]:
    ref = _reference_cell(site)
    lon_vals = da["lon"].values
    target_lon = ref["cell_lon"] % 360.0 if float(np.nanmin(lon_vals)) >= 0.0 else ref["cell_lon"]
    pt = da.sel(lat=ref["cell_lat"], lon=target_lon, method="nearest")
    cell_lat = float(pt["lat"].values)
    cell_lon = float(pt["lon"].values)
    if cell_lon > 180.0:
        cell_lon -= 360.0
    dist_km = float(ref["dist_km"])
    return pt, cell_lat, cell_lon, dist_km


@lru_cache(maxsize=None)
def _reference_cell(site: SiteSpec) -> dict[str, float]:
    """
    Pick a stable reference cell for a site.

    The nearest cell is used when it has viable recent-decade soil respiration.
    If not, fall back to the nearest cell with positive 2005–2014 ``rhSoil``.
    This preserves the old Barrow land-cell workaround without hand-coded site
    aliases.
    """
    rh = _open_global_var("rhSoil")
    recent = rh.sel(time=slice("2005-01-01", "2014-12-31")).mean(dim="time") * SEC_PER_YR * 1000.0
    lat_vals = recent["lat"].values.astype(float)
    lon_native = recent["lon"].values.astype(float)
    lon_signed = np.where(lon_native > 180.0, lon_native - 360.0, lon_native)
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_signed, indexing="ij")

    dist = np.vectorize(_haversine_km)(site.lat, site.lon, lat_grid, lon_grid)
    recent_vals = recent.values.astype(float)
    viable = np.isfinite(recent_vals) & (recent_vals > 1.0)
    if not np.any(viable):
        raise RuntimeError(f"no viable rhSoil cells found for {site.key}")

    choice = np.unravel_index(np.argmin(np.where(viable, dist, np.inf)), dist.shape)
    cell_lat = float(lat_vals[choice[0]])
    cell_lon = float(lon_signed[choice[1]])
    dist_km = float(dist[choice])
    return {"cell_lat": cell_lat, "cell_lon": cell_lon, "dist_km": dist_km}


def load_site_cesm(site_key: str) -> dict:
    """
    Return annual-mean CESM2 time series at the nearest grid cell.

    Notes
    -----
    The local global archive does not include litter-carbon output, so the
    returned structure sets ``cLitter`` to ``None`` and bulk turnover uses the
    soil-only totals.
    """
    site = get_site_spec(site_key)
    out: dict[str, object] = {
        "site": site.code,
        "site_key": site.key,
        "site_label": site.label,
        "site_lat": site.lat,
        "site_lon": site.lon,
        "cLitter": None,
    }

    series: dict[str, np.ndarray | None] = {}
    meta = None
    for var_name in ("cSoilFast", "cSoilMedium", "cSoilSlow", "cSoil", "rhSoil", "npp"):
        da = _open_global_var(var_name)
        pt, cell_lat, cell_lon, dist_km = _select_site_cell(da, site)
        annual = pt.groupby("time.year").mean(dim="time")
        vals = annual.values.astype(float)
        if var_name == "rhSoil" or var_name == "npp":
            vals = vals * SEC_PER_YR * 1000.0
        else:
            vals = vals * 1000.0
        series[var_name] = vals
        if meta is None:
            meta = {
                "years": annual["year"].values.astype(int),
                "cell_lat": cell_lat,
                "cell_lon": cell_lon,
                "dist_km": dist_km,
            }

    out.update(meta or {})
    out.update(
        cSoilFast=series["cSoilFast"],
        cSoilMedium=series["cSoilMedium"],
        cSoilSlow=series["cSoilSlow"],
        cSoil=series["cSoil"],
        rh=series["rhSoil"],
        npp=series["npp"],
    )
    return out


def load_clm_targets(site_key: str, years: tuple[int, int] = (2005, 2014)) -> dict:
    """Recent-decade means used as CLM-emulator inversion targets."""
    data = load_site_cesm(site_key)
    yr0, yr1 = years
    mask = (data["years"] >= yr0) & (data["years"] <= yr1)
    if not np.any(mask):
        raise RuntimeError(f"no CESM2 data in {yr0}–{yr1} for {site_key}")
    return {
        "site": data["site"],
        "site_key": site_key,
        "site_label": data["site_label"],
        "cLitter": 0.0,
        "cFast": float(np.nanmean(data["cSoilFast"][mask])),
        "cMed": float(np.nanmean(data["cSoilMedium"][mask])),
        "cSlow": float(np.nanmean(data["cSoilSlow"][mask])),
        "rh": float(np.nanmean(data["rh"][mask])),
        "cell_lat": float(data["cell_lat"]),
        "cell_lon": float(data["cell_lon"]),
        "dist_km": float(data["dist_km"]),
    }
