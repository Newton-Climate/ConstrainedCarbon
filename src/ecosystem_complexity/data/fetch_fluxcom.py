"""Download and extract site-level FluxCom forcing and ER series."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import requests
import xarray as xr

if TYPE_CHECKING:
    from ecosystem_complexity.sites.spec import SiteSpec

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_ROOT = _REPO_ROOT / "data"
_ICOS_DOWNLOAD_URL = "https://data.icos-cp.eu/licence_accept"


FLUXCOM_X_2021 = {
    "year": 2021,
    "file_stem": "GPP_2021_025_daily",
    "object_ids": (
        "68unxlD2Z6jxwEe6ql0XxK_1",
        "KTbxZpEAqivQVKyNf9Ec8fCv",
    ),
    "landing_page": "https://meta.icos-cp.eu/objects/KTbxZpEAqivQVKyNf9Ec8fCv",
}

FLUXCOM_X_2021_NEE = {
    "year": 2021,
    "file_stem": "NEE_2021_025_daily",
    "object_ids": (
        "DQozv4y9DR61Ks8F5R6KrLIT",
        "uLYdGGSwHV8U57JYRICwQlLV",
    ),
    "landing_page": "https://meta.icos-cp.eu/objects/DQozv4y9DR61Ks8F5R6KrLIT",
}


def load_fluxcom_site_specs(config_paths: Iterable[str | Path]) -> list[SiteSpec]:
    """Load only the site configs that declare ``forcing_kind: fluxcom``."""
    from ecosystem_complexity.sites.spec import load_site_spec

    specs = [load_site_spec(str(path)) for path in config_paths]
    return [spec for spec in specs if spec.forcing_kind == "fluxcom"]


def download_icos_object(
    object_ids: Sequence[str],
    file_stem: str,
    download_dir: str | Path,
) -> Path:
    """Download one ICOS object zip, trying the supplied object ids in order."""
    target_dir = Path(download_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for object_id in object_ids:
        zip_path = target_dir / f"{file_stem}-{object_id}.zip"
        try:
            with requests.Session() as session:
                response = session.get(
                    _ICOS_DOWNLOAD_URL,
                    params={
                        "ids": json.dumps([object_id], separators=(",", ":")),
                        "fileName": file_stem,
                    },
                    stream=True,
                    timeout=120,
                )
                response.raise_for_status()
                with zip_path.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            if zipfile.is_zipfile(zip_path):
                return zip_path
            raise ValueError(f"{zip_path.name} was not a valid zip archive")
        except Exception as exc:  # pragma: no cover - network failure path
            last_error = exc
            if zip_path.exists():
                zip_path.unlink()

    msg = f"Failed to download ICOS object for {file_stem}"
    if last_error is not None:
        raise RuntimeError(msg) from last_error
    raise RuntimeError(msg)


def extract_single_netcdf(zip_path: str | Path, dest_dir: str | Path) -> Path:
    """Extract the sole NetCDF member from a downloaded ICOS zip archive."""
    out_dir = Path(dest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".nc")]
        if len(members) != 1:
            raise ValueError(
                f"{Path(zip_path).name} contained {len(members)} netcdf files, expected 1"
            )
        member = members[0]
        nc_path = out_dir / Path(member).name
        with zf.open(member) as src, nc_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return nc_path


def extract_site_series_from_netcdf(
    nc_path: str | Path,
    *,
    lat: float,
    lon: float,
    variable_names: Sequence[str],
    output_col: str,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Extract the nearest-grid daily series for one site."""
    with xr.open_dataset(nc_path) as ds:
        var_name = next((name for name in variable_names if name in ds.data_vars), None)
        if var_name is None:
            value_list = "/".join(variable_names)
            raise ValueError(
                f"{Path(nc_path).name} did not contain one of {value_list}."
            )

        lat_name = "lat" if "lat" in ds.coords else "latitude"
        lon_name = "lon" if "lon" in ds.coords else "longitude"
        if lat_name not in ds.coords or lon_name not in ds.coords:
            raise ValueError(f"{Path(nc_path).name} must contain lat/lon coordinates.")

        lon_query = _match_longitude(lon, np.asarray(ds[lon_name].values, dtype=np.float64))
        point = ds[var_name].sel({lat_name: lat, lon_name: lon_query}, method="nearest")
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(point["time"].values),
                output_col: np.asarray(point.values, dtype=np.float64),
            }
        )
        frame = frame.dropna(subset=["date"]).reset_index(drop=True)
        mean_value = float(np.nanmean(frame[output_col].to_numpy(dtype=np.float64)))
        meta = {
            "grid_lat": float(point[lat_name].item()),
            "grid_lon": _restore_longitude(
                float(point[lon_name].item()),
                np.asarray(ds[lon_name].values, dtype=np.float64),
            ),
            "n_days": float(len(frame)),
            f"mean_{output_col}": mean_value,
        }
    return frame, meta


def extract_site_gpp_from_netcdf(
    nc_path: str | Path,
    *,
    lat: float,
    lon: float,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Extract the nearest-grid daily GPP series for one site."""
    frame, meta = extract_site_series_from_netcdf(
        nc_path,
        lat=lat,
        lon=lon,
        variable_names=("GPP_obs", "gpp_gCm2day", "gpp", "GPP"),
        output_col="gpp_gCm2day",
    )
    meta["mean_gpp_gCm2day"] = meta.pop("mean_gpp_gCm2day")
    return frame, meta


def write_fluxcom_site_csvs(
    spec: SiteSpec,
    gpp_netcdf: str | Path,
    nee_netcdf: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, float | str]:
    """Write site-level FluxCom GPP forcing and derived ER CSVs."""
    forcing_path = _DATA_ROOT / spec.forcing_glob
    forcing_path.parent.mkdir(parents=True, exist_ok=True)
    er_path = (
        (_DATA_ROOT / spec.er_observation_glob)
        if spec.er_observation_glob
        else None
    )
    if er_path is not None:
        er_path.parent.mkdir(parents=True, exist_ok=True)

    forcing_exists = forcing_path.exists()
    er_exists = er_path is None or er_path.exists()
    if forcing_exists and er_exists and not overwrite:
        forcing_frame = pd.read_csv(forcing_path)
        er_frame = pd.read_csv(er_path) if er_path is not None else None
        return {
            "site_id": spec.config_stem,
            "site_name": spec.label,
            "forcing_output_path": str(forcing_path),
            "er_output_path": str(er_path) if er_path is not None else "",
            "status": "skipped",
            "n_days": float(len(forcing_frame)),
            "mean_gpp_gCm2day": float(
                pd.to_numeric(forcing_frame.iloc[:, 1], errors="coerce").mean()
            ),
            "mean_ER": (
                float(pd.to_numeric(er_frame.iloc[:, 1], errors="coerce").mean())
                if er_frame is not None
                else float("nan")
            ),
        }

    gpp_frame, gpp_meta = extract_site_series_from_netcdf(
        gpp_netcdf,
        lat=spec.lat,
        lon=spec.lon,
        variable_names=("GPP_obs", "gpp_gCm2day", "gpp", "GPP"),
        output_col="gpp_gCm2day",
    )
    nee_frame, _ = extract_site_series_from_netcdf(
        nee_netcdf,
        lat=spec.lat,
        lon=spec.lon,
        variable_names=("NEE", "nee", "NEE_obs"),
        output_col="nee_gCm2day",
    )
    merged = gpp_frame.merge(nee_frame, on="date", how="inner")
    if merged.empty:
        raise ValueError(f"{spec.config_stem}: FluxCom GPP/NEE merge produced no rows")
    forcing_frame = merged[["date", "gpp_gCm2day"]].copy()
    er_frame = pd.DataFrame(
        {
            "date": merged["date"],
            "ER": merged["gpp_gCm2day"] + merged["nee_gCm2day"],
        }
    )
    forcing_frame.to_csv(forcing_path, index=False)
    if er_path is not None:
        er_frame.to_csv(er_path, index=False)
    return {
        "site_id": spec.config_stem,
        "site_name": spec.label,
        "forcing_output_path": str(forcing_path),
        "er_output_path": str(er_path) if er_path is not None else "",
        "status": "written",
        **gpp_meta,
        "mean_gpp_gCm2day": float(
            np.nanmean(forcing_frame["gpp_gCm2day"].to_numpy(dtype=np.float64))
        ),
        "mean_ER": float(np.nanmean(er_frame["ER"].to_numpy(dtype=np.float64))),
    }


def fetch_fluxcom_x_for_configs(
    config_paths: Iterable[str | Path],
    *,
    overwrite: bool = False,
) -> list[dict[str, float | str]]:
    """Download FLUXCOM-X 2021 GPP and NEE grids and write site CSVs."""
    specs = load_fluxcom_site_specs(config_paths)
    if not specs:
        return []

    with tempfile.TemporaryDirectory(prefix="fluxcom-x-") as tmp:
        gpp_zip_path = download_icos_object(
            FLUXCOM_X_2021["object_ids"],
            FLUXCOM_X_2021["file_stem"],
            tmp,
        )
        nee_zip_path = download_icos_object(
            FLUXCOM_X_2021_NEE["object_ids"],
            FLUXCOM_X_2021_NEE["file_stem"],
            tmp,
        )
        gpp_nc_path = extract_single_netcdf(gpp_zip_path, tmp)
        nee_nc_path = extract_single_netcdf(nee_zip_path, tmp)
        return [
            write_fluxcom_site_csvs(
                spec,
                gpp_nc_path,
                nee_nc_path,
                overwrite=overwrite,
            )
            for spec in specs
        ]


def _match_longitude(site_lon: float, grid_lons: np.ndarray) -> float:
    lon_min = float(np.nanmin(grid_lons))
    lon_max = float(np.nanmax(grid_lons))
    if lon_min >= 0.0 and site_lon < 0.0:
        return site_lon % 360.0
    if lon_max <= 180.0 and site_lon > 180.0:
        return ((site_lon + 180.0) % 360.0) - 180.0
    return site_lon


def _restore_longitude(grid_lon: float, grid_lons: np.ndarray) -> float:
    lon_min = float(np.nanmin(grid_lons))
    if lon_min >= 0.0 and grid_lon > 180.0:
        return grid_lon - 360.0
    return grid_lon
