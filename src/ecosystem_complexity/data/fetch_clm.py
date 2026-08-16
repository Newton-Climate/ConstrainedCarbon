"""Extract site-level CLM (Community Land Model) forcing and diagnostics.

CLM output — whether from CTSM single-point runs, NCAR NEON evaluation
simulations, or CMIP6 CLM history files — is delivered as NetCDF grids.
The task here is to pull the nearest-grid daily/monthly series for each
site the model tracks and write per-site CSVs into ``data/shared/clm/``
so the OE driver can load them via a config's ``forcing_kind: clm``
data source.

The default variables extracted are GPP, NEE, HR (heterotrophic
respiration), TSOI (soil temperature), and H2OSOI (soil moisture). CLM
history file names vary between conventions — the extractor searches a
small list of common variable-name aliases before failing loudly.

Live download is optional: pass ``source_dir`` to point at a local
directory of CLM NetCDFs that were staged out-of-band (from NCAR's
Casper archive, ESGF, or an internal HPC scratch mount). A future
extension can add a URL-templated downloader once we settle on a
canonical CLM output source.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import xarray as xr

from ecosystem_complexity.fetch.external import download_file

if TYPE_CHECKING:
    from ecosystem_complexity.sites.spec import SiteSpec

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLM_ROOT = _REPO_ROOT / "data" / "shared" / "clm"


# CLM variable-name aliases (the physical field is stable but naming
# differs across CLM4/CLM5/CTSM/NEON conventions). Order matters: the
# first present in the dataset wins.
_ALIASES = {
    "GPP":    ("GPP", "gpp", "FPSN_TO_GPP"),
    "NEE":    ("NEE", "nee", "NEP", "nep"),
    "HR":     ("HR", "hr", "HETEROTROPH_RESP", "heterotroph_resp"),
    "ER":     ("ER", "er", "TOTECOSYSRESP", "totecosysresp"),
    "TSOI":   ("TSOI", "tsoi", "soil_t"),
    "H2OSOI": ("H2OSOI", "h2osoi", "soil_moisture"),
}

# Default output columns and their aliases
_DEFAULT_VARS: tuple[str, ...] = ("GPP", "NEE", "HR")

PANGEO_CMIP6_CATALOG = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
_PANGEO_VARIABLES = {
    # CMIP6 uses lowercase controlled-vocabulary names; CESM2 exposes CLM5
    # GPP and soil heterotrophic respiration in Lmon.
    "GPP": ("gpp", "Lmon"),
    "HR": ("rh", "Lmon"),
    # NEE is not a standard CMIP6 land-output variable.  NBP is the closest
    # available carbon-balance diagnostic when a caller explicitly requests it.
    "NEE": ("nbp", "Lmon"),
    "ER": ("rh", "Lmon"),
    "TSOI": ("tsl", "Lmon"),
    "H2OSOI": ("mrsos", "Lmon"),
    "CSOILFAST": ("cSoilFast", "Lmon"),
    "CSOILMEDIUM": ("cSoilMedium", "Lmon"),
    "CSOILSLOW": ("cSoilSlow", "Lmon"),
    "CSOIL": ("cSoil", "Emon"),
    "CLITTER": ("cLitter", "Lmon"),
    "NPP": ("npp", "Lmon"),
    # Not part of the standard CESM2 CMIP6 publication; retained as an
    # opt-in discovery request for archives that expose the field.
    "C14SOIL": ("c14Soil", "Lmon"),
}
_PANGEO_SOIL_CARBON_VARS = (
    "CSOILFAST", "CSOILMEDIUM", "CSOILSLOW", "CSOIL", "CLITTER",
    "GPP", "NPP", "HR",
)


def load_clm_site_specs(config_paths: Iterable[str | Path]) -> list[SiteSpec]:
    """Load only site configs that declare ``forcing_kind: clm``."""
    from ecosystem_complexity.sites.spec import load_site_spec
    specs = [load_site_spec(str(p)) for p in config_paths]
    return [s for s in specs if getattr(s, "forcing_kind", None) == "clm"]


def _match_longitude(site_lon: float, grid_lons: np.ndarray) -> float:
    lon_min = float(np.nanmin(grid_lons))
    if lon_min >= 0.0 and site_lon < 0.0:
        return site_lon % 360.0
    return site_lon


def _restore_longitude(grid_lon: float, grid_lons: np.ndarray) -> float:
    lon_max = float(np.nanmax(grid_lons))
    if lon_max > 180.0 and grid_lon > 180.0:
        return grid_lon - 360.0
    return grid_lon


def _pick_var(ds: xr.Dataset, wanted: str) -> str:
    """Return the first alias of ``wanted`` present in ``ds``."""
    aliases = _ALIASES.get(wanted, (wanted,))
    for name in aliases:
        if name in ds.data_vars:
            return name
    raise ValueError(
        f"CLM dataset lacks any alias for {wanted!r}. "
        f"Tried: {aliases}. Present: {sorted(ds.data_vars)[:20]}…"
    )


def _pangeo_values(point: xr.DataArray, wanted: str) -> np.ndarray:
    """Convert CMIP6 carbon fluxes to the forcing CSV's gC m-2 day-1."""
    values = np.asarray(point.values, dtype=np.float64)
    if wanted in {"GPP", "HR", "NEE", "ER"}:
        units = str(point.attrs.get("units", "")).replace(" ", "").lower()
        if "kg" in units and ("s-1" in units or "/s" in units):
            return values * 1000.0 * 86400.0
    return values


def extract_site_series_from_clm_netcdf(
    nc_path: str | Path,
    *,
    lat: float,
    lon: float,
    variables: Sequence[str] = _DEFAULT_VARS,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Extract a nearest-grid daily/monthly series for one site.

    Returns ``(dataframe, metadata)`` where the dataframe has a ``date``
    column plus one column per requested variable. Metadata records the
    grid cell actually sampled and per-variable means so callers can
    audit the pull without re-opening the NetCDF.
    """
    with xr.open_dataset(nc_path) as ds:
        lat_name = "lat" if "lat" in ds.coords else "latitude"
        lon_name = "lon" if "lon" in ds.coords else "longitude"
        if lat_name not in ds.coords or lon_name not in ds.coords:
            raise ValueError(f"{Path(nc_path).name}: missing lat/lon coordinates.")

        lon_q = _match_longitude(lon, np.asarray(ds[lon_name].values, dtype=np.float64))
        selectors = {lat_name: lat, lon_name: lon_q}

        cols: dict[str, np.ndarray] = {}
        picked: dict[str, str] = {}
        for wanted in variables:
            var = _pick_var(ds, wanted)
            picked[wanted] = var
            point = ds[var].sel(selectors, method="nearest")
            cols[wanted] = np.asarray(point.values, dtype=np.float64).squeeze()

        # Use the first variable's time axis as the frame index; CLM
        # history files share a single time dimension across variables.
        first = variables[0]
        first_point = ds[picked[first]].sel(selectors, method="nearest")
        dates = pd.to_datetime(first_point["time"].values)
        frame = pd.DataFrame({"date": dates, **cols})

        grid_lat = float(first_point[lat_name].item())
        grid_lon = _restore_longitude(
            float(first_point[lon_name].item()),
            np.asarray(ds[lon_name].values, dtype=np.float64),
        )
        meta: dict[str, float] = {
            "grid_lat": grid_lat,
            "grid_lon": grid_lon,
            "n_days": float(len(frame)),
        }
        for v in variables:
            arr = frame[v].to_numpy(dtype=np.float64)
            meta[f"mean_{v.lower()}"] = float(np.nanmean(arr))
    return frame, meta


def _discover_source_netcdf(source_dir: Path, spec: SiteSpec) -> Path:
    """Find a CLM NetCDF in ``source_dir`` for ``spec``.

    Preference order:
      1. File whose name contains the tower id (e.g. ``US-Ha1``)
      2. Single .nc file in the directory (unambiguous)
      3. Raise — the caller must disambiguate
    """
    ncs = sorted(source_dir.glob("*.nc"))
    if not ncs:
        raise FileNotFoundError(f"No .nc files in {source_dir}")
    tower = (spec.tower_id or "").strip()
    if tower:
        matches = [p for p in ncs if tower in p.name]
        if matches:
            return matches[0]
    if len(ncs) == 1:
        return ncs[0]
    raise ValueError(
        f"{source_dir}: multiple .nc files and none matched tower {tower!r}. "
        f"Point --source-dir at a subdirectory with one file, or add the tower "
        f"id to the filename."
    )


def write_clm_site_csvs(
    spec: SiteSpec,
    nc_path: str | Path,
    *,
    variables: Sequence[str] = _DEFAULT_VARS,
    overwrite: bool = False,
    out_root: str | Path = _CLM_ROOT,
) -> dict[str, float | str]:
    """Write one site's CLM series as ``{out_root}/{stem}.csv``.

    A companion ``{stem}.er.csv`` is written when the extracted variables
    include enough terms to derive ER (either directly, or as
    ``ER = GPP + NEE`` when only those two are present).
    """
    frame, meta = extract_site_series_from_clm_netcdf(
        nc_path, lat=spec.lat, lon=spec.lon, variables=variables,
    )
    return write_clm_site_frame(
        spec, frame, metadata=meta, overwrite=overwrite, out_root=out_root,
    )


def write_clm_site_frame(
    spec: SiteSpec,
    frame: pd.DataFrame,
    *,
    metadata: dict[str, float | str],
    overwrite: bool = False,
    out_root: str | Path = _CLM_ROOT,
) -> dict[str, float | str]:
    """Write an extracted CLM site frame in the canonical forcing layout."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    forcing_path = out_root / f"{spec.config_stem}.csv"
    er_path = out_root / f"{spec.config_stem}.er.csv"
    if forcing_path.exists() and er_path.exists() and not overwrite:
        return {"site_id": spec.config_stem, "site_name": spec.label,
                "forcing_output_path": str(forcing_path), "er_output_path": str(er_path),
                "status": "skipped", "n_days": float(len(pd.read_csv(forcing_path)))}
    if frame.empty:
        raise ValueError(f"{spec.config_stem}: CLM extraction produced no rows.")

    # Rename to the canonical column names the driver reads
    rename = {"GPP": "gpp_gCm2day", "NEE": "nee_gCm2day", "HR": "hr_gCm2day",
              "TSOI": "soil_t", "H2OSOI": "soil_moisture"}
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
    frame.to_csv(forcing_path, index=False)

    if {"gpp_gCm2day", "nee_gCm2day"}.issubset(frame.columns):
        er = pd.DataFrame({"date": frame["date"],
                           "ER": frame["gpp_gCm2day"] + frame["nee_gCm2day"]})
        er.to_csv(er_path, index=False)
        er_out = str(er_path)
    elif "hr_gCm2day" in frame.columns:
        pd.DataFrame({"date": frame["date"], "ER": frame["hr_gCm2day"]}).to_csv(er_path, index=False)
        er_out = str(er_path)
    elif "ER" in frame.columns:
        pd.DataFrame({"date": frame["date"], "ER": frame["ER"]}).to_csv(er_path, index=False)
        er_out = str(er_path)
    else:
        er_out = ""

    return {
        "site_id": spec.config_stem,
        "site_name": spec.label,
        "forcing_output_path": str(forcing_path),
        "er_output_path": er_out,
        "status": "written",
        **metadata,
    }


def fetch_clm_for_configs(
    config_paths: Iterable[str | Path],
    *,
    source_dir: str | Path,
    variables: Sequence[str] = _DEFAULT_VARS,
    overwrite: bool = False,
    out_root: str | Path = _CLM_ROOT,
) -> list[dict[str, float | str]]:
    """Extract CLM site series for every ``forcing_kind: clm`` config.

    ``source_dir`` must point at a directory containing the CLM
    NetCDF(s). See :func:`_discover_source_netcdf` for the per-site
    matching rules.
    """
    source = Path(source_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"CLM source_dir does not exist: {source}")
    specs = load_clm_site_specs(config_paths)
    if not specs:
        return []
    rows: list[dict[str, float | str]] = []
    for spec in specs:
        nc = _discover_source_netcdf(source, spec)
        rows.append(write_clm_site_csvs(
            spec, nc, variables=variables, overwrite=overwrite, out_root=out_root,
        ))
    return rows


def download_clm_sources(
    urls: Iterable[str],
    *,
    source_dir: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    """Stage CLM NetCDF files supplied as direct HTTP(S) URLs.

    CLM/CTSM and CMIP products are not one dataset: their experiment, member,
    variables, and temporal resolution are analysis choices.  Callers therefore
    provide the exact file URLs selected from ESGF, NCAR, or another archive.
    """
    destination = Path(source_dir)
    outputs: list[Path] = []
    for url in urls:
        name = Path(urlparse(url).path).name
        if not name.lower().endswith((".nc", ".nc4")):
            raise ValueError(f"CLM source URL must name a NetCDF (.nc/.nc4): {url}")
        outputs.append(download_file(url, destination / name, overwrite=overwrite))
    return outputs


def fetch_pangeo_clm_for_configs(
    config_paths: Iterable[str | Path],
    *,
    source_id: str = "CESM2",
    experiment_ids: Sequence[str] = ("historical",),
    member_id: str = "r1i1p1f1",
    variables: Sequence[str] = _PANGEO_SOIL_CARBON_VARS,
    overwrite: bool = False,
    out_root: str | Path = _CLM_ROOT,
    catalog_url: str = PANGEO_CMIP6_CATALOG,
) -> list[dict[str, float | str]]:
    """Extract configured CLM sites directly from Pangeo CMIP6 Zarr stores."""
    try:
        import intake
    except ImportError as exc:  # pragma: no cover - optional deployment dependency
        raise RuntimeError("Pangeo fetch requires intake-esm, gcsfs, and zarr.") from exc
    specs = load_clm_site_specs(config_paths)
    if not specs:
        return []
    unknown = [name for name in variables if name not in _PANGEO_VARIABLES]
    if unknown:
        raise ValueError(f"Pangeo does not map CLM variables: {', '.join(unknown)}")
    catalog = intake.open_esm_datastore(catalog_url)
    rows: list[dict[str, float | str]] = []
    out_root = Path(out_root)
    for experiment_id in experiment_ids:
        for spec in specs:
            archive_path = out_root / f"{spec.config_stem}_{experiment_id}_pangeo.nc"
            if archive_path.exists() and not overwrite:
                rows.append({"site_id": spec.config_stem, "experiment_id": experiment_id,
                             "archive_output_path": str(archive_path), "status": "skipped"})
                continue
            columns: dict[str, np.ndarray] = {}
            series: dict[str, xr.DataArray] = {}
            dates = None
            metadata: dict[str, float | str] = {
                "source": "Pangeo CMIP6 Zarr", "source_id": source_id,
                "experiment_id": experiment_id, "member_id": member_id,
            }
            for wanted in variables:
                variable_id, table_id = _PANGEO_VARIABLES[wanted]
                found = catalog.search(
                    source_id=source_id, experiment_id=experiment_id,
                    member_id=member_id, variable_id=variable_id, table_id=table_id,
                )
                if found.df.empty:
                    raise FileNotFoundError(
                        "Pangeo has no "
                        f"{source_id}/{experiment_id}/{member_id} {table_id}/{variable_id} store."
                    )
                zstore = str(found.df.iloc[0]["zstore"])
                ds = xr.open_zarr(
                    zstore, consolidated=True, storage_options={"token": "anon"},
                )
                lons = np.asarray(ds["lon"].values, dtype=np.float64)
                point = ds[variable_id].sel(
                    lat=spec.lat, lon=_match_longitude(spec.lon, lons), method="nearest",
                ).squeeze(drop=True).load()
                series[variable_id] = point
                columns[wanted] = _pangeo_values(point, wanted)
                if dates is None:
                    dates = pd.to_datetime(point["time"].values)
                    metadata.update({
                        "grid_lat": float(point["lat"].item()),
                        "grid_lon": _restore_longitude(float(point["lon"].item()), lons),
                        "pangeo_zstore": zstore,
                    })
            out_root.mkdir(parents=True, exist_ok=True)
            archive = xr.Dataset(series, attrs=metadata)
            archive.to_netcdf(archive_path)
            record: dict[str, float | str] = {
                "site_id": spec.config_stem, "experiment_id": experiment_id,
                "archive_output_path": str(archive_path), "status": "written",
            }
            # The operational forcing files retain their historical default.
            if experiment_id == "historical":
                frame = pd.DataFrame({"date": dates, **columns})
                metadata["n_days"] = float(len(frame))
                record.update(write_clm_site_frame(
                    spec, frame, metadata=metadata, overwrite=overwrite, out_root=out_root,
                ))
            rows.append(record)
    return rows
