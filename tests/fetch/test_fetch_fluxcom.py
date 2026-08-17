import pandas as pd
import xarray as xr

from ecosystem_complexity.data.fetch_fluxcom import extract_site_gpp_from_netcdf


def test_extract_site_gpp_from_netcdf_wraps_negative_longitude(tmp_path) -> None:
    ds = xr.Dataset(
        {
            "GPP": (
                ("time", "lat", "lon"),
                [[[1.0, 2.0]], [[3.0, 4.0]]],
            )
        },
        coords={
            "time": pd.date_range("2021-01-01", periods=2, freq="D"),
            "lat": [18.25],
            "lon": [294.0, 294.25],
        },
    )
    path = tmp_path / "fluxcom.nc"
    ds.to_netcdf(path)

    frame, meta = extract_site_gpp_from_netcdf(path, lat=18.3157, lon=-65.7487)

    assert frame["gpp_gCm2day"].tolist() == [2.0, 4.0]
    assert str(frame["date"].iloc[0].date()) == "2021-01-01"
    assert meta["grid_lat"] == 18.25
    assert meta["grid_lon"] == -65.75
