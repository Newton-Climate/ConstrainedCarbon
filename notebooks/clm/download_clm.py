"""
download_clm.py — Deprecated per-grid-cell CESM2 downloader.

This older workflow pulled CESM2 (CLM5) variables from the CMIP6 archive for
Harvard Forest (US-Ha1) and Barrow, Alaska (US-A10), then saved one extracted
grid-cell NetCDF per site.

The active workflow now reads the global historical files already stored in
``data/cmip`` and selects the nearest cell on demand for the four canonical
analysis sites. See:

  - notebooks/clm/cmip_global.py
  - notebooks/clm/analyze_clm.py
  - notebooks/clm/fit_clm.py
  - notebooks/clm/clm_emulator_14c.py

Two source paths are attempted in order:

  1. Pangeo CMIP6 catalog (Google Cloud Storage, zarr).  Fast, no auth.
  2. LLNL ESGF (canonical CMIP6 archive).  Requires ``pyesgf``.

Output layout
-------------
  data/cmip6/<model>/<variable>_<experiment>_<site>.nc

Each file is a 1-D time series (a single grid cell extracted from the
gridded output) with attributes documenting source, member, table, and
the great-circle distance from the requested site coordinate to the
grid-cell centroid.

Run
---
    python notebooks/clm/download_clm.py                  # all defaults
    python notebooks/clm/download_clm.py --dry-run        # show plan, don't fetch
    python notebooks/clm/download_clm.py --experiment historical --variables cSoilFast rh
    python notebooks/clm/download_clm.py --source esgf    # force LLNL ESGF path

Dependencies
------------
    pip install intake-esm xarray gcsfs zarr netCDF4
    # ESGF fallback also needs:
    pip install pyesgf requests

Author: Newton H. Nguyen
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np

# ─── Repo paths ──────────────────────────────────────────────────────────────
_SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
_NB_ROOT     = os.path.dirname(_SCRIPT_ROOT)
_REPO_ROOT   = os.path.dirname(_NB_ROOT)
OUT_DIR      = os.path.join(_REPO_ROOT, "data", "cmip6")


# ─── Sites + variable plan ──────────────────────────────────────────────────
@dataclass
class Site:
    short: str
    name:  str
    lat:   float
    lon:   float   # signed (−180 … 180)


SITES: list[Site] = [
    Site("us-ha1",       "Harvard Forest",      42.5378,  -72.1715),
    # CESM2's grid cell at Barrow's exact coords (71.28°N) lies in the
    # Beaufort Sea — gpp = rh = 0.  Use a coordinate ~100 km inland that
    # falls on the nearest land cell (tundra biome) for the comparison.
    Site("us-a10",       "Barrow, Alaska",      71.28,   -156.61),
    Site("us-a10-land",  "Barrow inland (CESM2 land cell)",
                                                70.50,   -156.00),
    # Coastal CESM2 cells near Barrow have rh ≈ 0 (sea-ice / mixed cells).
    # Use a North-Slope tundra cell ~250 km inland that's solidly land.
    Site("us-a10-tundra","North Slope tundra (NEON Toolik-area)",
                                                68.50,   -149.50),
]


# Pool / flux variables — CMIP6 short names, Lmon/Emon tables.
# Notes:
#   cSoilFast, cSoilMedium, cSoilSlow are reported by CESM2 in the Emon table.
#   cSoil, cLitter, rh, gpp, tas are Amon/Lmon.
#   tsl (soil temperature on depth levels) is Lmon.
VARIABLES_DEFAULT = [
    "cSoilFast", "cSoilMedium", "cSoilSlow",
    "cSoil", "cLitter",
    "rh", "gpp",
    "tas", "tsl",
]

# CMIP6 table associations (Pangeo catalog uses "table_id").
# Note: CESM2 publishes cSoilFast/Medium/Slow under Lmon (not Emon as in
# the CMIP6 standard request), and cSoil under Emon.  Verified against the
# Pangeo CMIP6 catalog in June 2026.
TABLES = {
    "cSoilFast":   "Lmon",
    "cSoilMedium": "Lmon",
    "cSoilSlow":   "Lmon",
    "cSoil":       "Emon",
    "cLitter":     "Lmon",
    "rh":          "Lmon",
    "gpp":         "Lmon",
    "tas":         "Amon",
    "tsl":         "Lmon",
}

EXPERIMENTS_DEFAULT = ["historical", "ssp585"]

# Sites for which to fetch GPP across all experiments (historical + future).
# GPP is what we'll use to FORCE our model in a planned warming-scenario run.
GPP_FORCING_SITES = ["us-ha1", "us-a10-tundra"]

# Pangeo CMIP6 catalog (Google Cloud)
PANGEO_CATALOG_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"


# ─── Geographic helper ──────────────────────────────────────────────────────
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi   = np.radians(lat2 - lat1)
    dlam   = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlam/2)**2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def _nearest_cell(ds, lat: float, lon: float):
    """
    Select the nearest grid cell to (lat, lon) from a CMIP6 lat/lon dataset.

    Handles both 0..360 and −180..180 longitude conventions and returns the
    sliced dataset plus the centroid distance.
    """
    # Detect lon convention
    lon_vals = ds["lon"].values
    target_lon = lon if lon_vals.min() >= -180 and lon_vals.max() <= 180 else (lon % 360)
    ds_pt = ds.sel(lat=lat, lon=target_lon, method="nearest")
    cell_lat = float(ds_pt["lat"].values)
    cell_lon = float(ds_pt["lon"].values)
    if cell_lon > 180:
        cell_lon -= 360
    dist_km = _haversine_km(lat, lon, cell_lat, cell_lon)
    return ds_pt, cell_lat, cell_lon, dist_km


# ─── Pangeo CMIP6 catalog path ──────────────────────────────────────────────
def _open_pangeo_catalog():
    try:
        import intake
    except ImportError as e:
        raise RuntimeError(
            "intake-esm not installed. Run:\n"
            "    pip install intake-esm xarray gcsfs zarr"
        ) from e

    try:
        col = intake.open_esm_datastore(PANGEO_CATALOG_URL)
    except Exception as e:
        raise RuntimeError(
            f"Could not open Pangeo CMIP6 catalog at {PANGEO_CATALOG_URL}.\n"
            f"  Error: {e}\n"
            "Falling back to ESGF requires --source esgf."
        ) from e
    return col


def _pangeo_query(
    col, model: str, experiment: str, variable: str,
    member: str = "r1i1p1f1",
):
    table = TABLES.get(variable, "Lmon")
    return col.search(
        source_id=model,
        experiment_id=experiment,
        variable_id=variable,
        table_id=table,
        member_id=member,
    )


def _download_pangeo(
    model: str, experiment: str, variables: list, sites: list,
    member: str, dry_run: bool,
) -> None:
    print(f"\n── Pangeo CMIP6 catalog ─ {model} / {experiment} / member={member}")
    col = _open_pangeo_catalog()

    import xarray as xr  # delayed import — heavy

    out_root = os.path.join(OUT_DIR, model)
    os.makedirs(out_root, exist_ok=True)

    for var in variables:
        cat = _pangeo_query(col, model, experiment, var, member)
        n_hits = len(cat.df)
        print(f"  {var:<12s} ({TABLES.get(var, '?')}):  {n_hits} dataset(s)")
        if n_hits == 0:
            continue
        if dry_run:
            # Show a sample row
            print(f"    sample → {cat.df.iloc[0]['zstore']}")
            continue

        try:
            zstore = cat.df.iloc[0]["zstore"]
            t0 = time.perf_counter()
            ds = xr.open_zarr(zstore, consolidated=True, decode_times=True)
            print(f"    opened zarr [{time.perf_counter()-t0:.1f}s]  "
                  f"vars={list(ds.data_vars)}  dims={dict(ds.sizes)}")
        except Exception as e:
            print(f"    ERROR opening {zstore}: {e}")
            continue

        # Drop ensemble/member dim if present
        if "member_id" in ds.dims:
            ds = ds.isel(member_id=0)

        for site in sites:
            try:
                ds_pt, clat, clon, dist = _nearest_cell(ds, site.lat, site.lon)
            except Exception as e:
                print(f"    {site.short}: nearest-cell failed: {e}")
                continue

            # Compute & save: load into memory first (Pangeo zarr is lazy)
            t0 = time.perf_counter()
            ds_pt = ds_pt.load()
            ds_pt.attrs.update({
                "source": "Pangeo CMIP6 catalog (GCS zarr)",
                "model": model,
                "experiment": experiment,
                "variable": var,
                "table_id": TABLES.get(var, "?"),
                "member": member,
                "site_short": site.short,
                "site_name": site.name,
                "site_lat_requested": site.lat,
                "site_lon_requested": site.lon,
                "cell_lat": clat,
                "cell_lon": clon,
                "dist_km": dist,
            })

            fname = f"{var}_{experiment}_{site.short}.nc"
            fpath = os.path.join(out_root, fname)
            ds_pt.to_netcdf(fpath)
            sz = os.path.getsize(fpath) / 1024
            print(f"    {site.short}: cell ({clat:.2f},{clon:.2f}), dist {dist:.0f} km  "
                  f"→ {fname} [{sz:.0f} KB, {time.perf_counter()-t0:.1f}s]")


# ─── ESGF fallback path ─────────────────────────────────────────────────────
def _download_esgf(
    model: str, experiment: str, variables: list, sites: list,
    member: str, dry_run: bool,
) -> None:
    try:
        from pyesgf.search import SearchConnection
    except ImportError as e:
        raise RuntimeError(
            "pyesgf not installed. Run:\n"
            "    pip install pyesgf requests"
        ) from e

    print(f"\n── LLNL ESGF ─ {model} / {experiment} / member={member}")
    conn = SearchConnection("https://esgf-node.llnl.gov/esg-search", distrib=True)

    out_root = os.path.join(OUT_DIR, model)
    os.makedirs(out_root, exist_ok=True)

    for var in variables:
        table = TABLES.get(var, "Lmon")
        ctx = conn.new_context(
            project="CMIP6",
            source_id=model,
            experiment_id=experiment,
            variable_id=var,
            table_id=table,
            variant_label=member,
        )
        nhits = ctx.hit_count
        print(f"  {var:<12s} ({table}):  {nhits} hit(s)")
        if nhits == 0:
            continue
        if dry_run:
            r = next(iter(ctx.search()))
            print(f"    sample → {r.dataset_id}")
            continue

        ds_search = ctx.search()
        try:
            r = next(iter(ds_search))
            file_ctx = r.file_context()
            files = list(file_ctx.search())
            print(f"    {len(files)} netCDF file(s) under {r.dataset_id}")
        except Exception as e:
            print(f"    ERROR enumerating files: {e}")
            continue

        # Use xarray OPeNDAP to slice without downloading whole gridded file
        import xarray as xr
        url0 = files[0].opendap_url
        try:
            ds = xr.open_dataset(url0, decode_times=True)
        except Exception as e:
            print(f"    ERROR opening OPeNDAP {url0}: {e}")
            continue
        for site in sites:
            ds_pt, clat, clon, dist = _nearest_cell(ds, site.lat, site.lon)
            ds_pt = ds_pt.load()
            ds_pt.attrs.update({
                "source": "LLNL ESGF (OPeNDAP)",
                "model": model, "experiment": experiment, "variable": var,
                "table_id": table, "member": member,
                "site_short": site.short, "site_name": site.name,
                "site_lat_requested": site.lat, "site_lon_requested": site.lon,
                "cell_lat": clat, "cell_lon": clon, "dist_km": dist,
            })
            fname = f"{var}_{experiment}_{site.short}.nc"
            ds_pt.to_netcdf(os.path.join(out_root, fname))
            print(f"    {site.short}: cell ({clat:.2f},{clon:.2f}), dist {dist:.0f} km → {fname}")


# ─── Entry point ────────────────────────────────────────────────────────────
def main():
    print("DEPRECATED: download_clm.py is retained for reference only.")
    print("Use the global CESM2 files in data/cmip via notebooks/clm/cmip_global.py.\n")
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",      default="CESM2",
        help="CMIP6 source_id (default: CESM2 — uses CLM5)")
    p.add_argument("--experiment", nargs="+", default=EXPERIMENTS_DEFAULT,
        help=f"CMIP6 experiment_id(s) (default: {EXPERIMENTS_DEFAULT})")
    p.add_argument("--variables",  nargs="+", default=VARIABLES_DEFAULT,
        help="CMIP6 variable_id(s)")
    p.add_argument("--member",     default="r1i1p1f1",
        help="CMIP6 variant_label (default: r1i1p1f1)")
    p.add_argument("--source",     choices=("pangeo", "esgf", "auto"), default="auto",
        help="Data source (auto = try Pangeo first, fall back to ESGF)")
    p.add_argument("--dry-run",    action="store_true",
        help="List what would be downloaded without fetching")
    p.add_argument("--gpp-forcing", action="store_true",
        help="Shortcut: pull GPP (historical + ssp585 + ssp245) for the "
             "GPP_FORCING_SITES, used as forcing for warming-scenario runs.")
    args = p.parse_args()

    if args.gpp_forcing:
        args.variables  = ["gpp"]
        args.experiment = ["historical", "ssp585", "ssp245"]
        sites = [s for s in SITES if s.short in GPP_FORCING_SITES]
        print(f"GPP-forcing mode: sites = {[s.short for s in sites]}, "
              f"experiments = {args.experiment}")
    else:
        sites = SITES

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")
    print(f"Sites:")
    for site in sites:
        print(f"  {site.short}: {site.name:<18s} ({site.lat:6.2f}°N, {site.lon:7.2f}°E)")

    for experiment in args.experiment:
        if args.source in ("pangeo", "auto"):
            try:
                _download_pangeo(args.model, experiment, args.variables,
                                 sites, args.member, args.dry_run)
                continue   # success
            except Exception as e:
                if args.source == "pangeo":
                    raise
                print(f"\nPangeo path failed: {e}\n→ falling back to ESGF…")

        # ESGF (either explicit or fallback)
        _download_esgf(args.model, experiment, args.variables,
                       sites, args.member, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
