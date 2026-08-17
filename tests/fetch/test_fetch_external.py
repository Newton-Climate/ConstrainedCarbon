"""Tests for archive validation used by external dataset fetchers."""
from __future__ import annotations

import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest
import xarray as xr

import ecosystem_complexity.data.fetch_clm as fetch_clm
from ecosystem_complexity.fetch.external import DatasetDownloadError, extract_named_zip_members


def test_extract_named_zip_members_validates_all_members_before_replacing(tmp_path) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("layer.csv", "new-layer")

    destination = tmp_path / "data"
    destination.mkdir()
    existing = destination / "layer.csv"
    existing.write_text("old-layer")

    with pytest.raises(DatasetDownloadError, match="flux.csv"):
        extract_named_zip_members(
            archive, destination, ["layer.csv", "flux.csv"], overwrite=True,
        )

    assert existing.read_text() == "old-layer"


def test_extract_named_zip_members_extracts_exact_basenames(tmp_path) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("compiled/layer.csv", "layer")
        zf.writestr("compiled/flux.csv", "flux")

    outputs = extract_named_zip_members(archive, tmp_path / "data", ["layer.csv", "flux.csv"])

    assert [path.name for path in outputs] == ["layer.csv", "flux.csv"]
    assert outputs[1].read_text() == "flux"


def test_pangeo_flux_values_are_converted_to_forcing_units() -> None:
    point = xr.DataArray([1e-6], attrs={"units": "kg m-2 s-1"})

    assert fetch_clm._pangeo_values(point, "GPP").tolist() == [86.4]


def test_pangeo_fetch_writes_site_archive_and_historical_forcing(
    tmp_path, monkeypatch
) -> None:
    spec = SimpleNamespace(config_stem="test_site", label="Test site", lat=42.0, lon=-72.0)
    monkeypatch.setattr(fetch_clm, "load_clm_site_specs", lambda _: [spec])

    class Catalog:
        def search(self, **query):
            return SimpleNamespace(df=pd.DataFrame({"zstore": [query["variable_id"]]}))

    monkeypatch.setitem(
        __import__("sys").modules,
        "intake",
        SimpleNamespace(open_esm_datastore=lambda _: Catalog()),
    )

    def open_zarr(zstore, **_):
        values = [1e-6, 2e-6] if zstore == "gpp" else [3e-6, 4e-6]
        dataset = xr.Dataset(
            {zstore: (("time", "lat", "lon"), [[[values[0]]], [[values[1]]]])},
            coords={"time": pd.date_range("2000-01-01", periods=2, freq="MS"),
                    "lat": [42.0], "lon": [288.0]},
        )
        dataset[zstore].attrs["units"] = "kg m-2 s-1"
        return dataset

    monkeypatch.setattr(fetch_clm.xr, "open_zarr", open_zarr)
    rows = fetch_clm.fetch_pangeo_clm_for_configs(
        ["unused.yaml"], variables=("GPP", "HR"), out_root=tmp_path,
    )

    assert rows[0]["status"] == "written"
    assert (tmp_path / "test_site_historical_pangeo.nc").is_file()
    forcing = pd.read_csv(tmp_path / "test_site.csv")
    assert forcing["gpp_gCm2day"].tolist() == [86.4, 172.8]
    assert pd.read_csv(tmp_path / "test_site.er.csv")["ER"].tolist() == [259.2, 345.6]
