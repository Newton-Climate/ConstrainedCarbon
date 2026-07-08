"""
export_flux_israd_colocations.py — Colocate flux towers with ISRaD radiocarbon data.

Builds two tower catalogs from current workspace data:
  - local AmeriFlux sites with coordinates from the model configs
  - NEON tower sites represented in ISRaD (tower-location NEON entries)

For each tower, finds the nearest ISRaD site with:
  - bulk soil layer 14C
  - fraction 14C
  - flux / respired 14C

Outputs
-------
  notebooks/exports/flux_tower_israd_colocations.csv
  notebooks/exports/flux_tower_israd_priority_sites.csv
  notebooks/exports/flux_tower_israd_best_sites.csv
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_DIR = os.path.join(ROOT, "notebooks", "exports")
AMERIFLUX_SITES = os.path.join(ROOT, "data", "shared", "ameriflux_field_sites.csv")
NEON_SITES = os.path.join(ROOT, "data", "shared", "neon_field_sites.csv")

ISRAD_LAYER = os.path.join(
    ROOT, "data", "shared", "israd", "ISRaD_data_flat_layer_v 2.6.6.2024-01-25.csv"
)
ISRAD_FRACTION = os.path.join(
    ROOT, "data", "shared", "israd", "ISRaD_data_flat_fraction_v 2.6.6.2024-01-25.csv"
)
ISRAD_FLUX = os.path.join(
    ROOT, "data", "shared", "israd", "ISRaD_data_flat_flux_v 2.6.6.2024-01-25.csv"
)

@dataclass
class TowerSite:
    network: str
    tower_id: str
    tower_name: str
    lat: float
    lon: float


def _load_ameriflux_sites() -> list[TowerSite]:
    df = pd.read_csv(AMERIFLUX_SITES, sep="\t", low_memory=False)
    df["field_latitude"] = pd.to_numeric(df["field_latitude"], errors="coerce")
    df["field_longitude"] = pd.to_numeric(df["field_longitude"], errors="coerce")
    df = df.dropna(subset=["field_site_id", "field_latitude", "field_longitude"]).copy()
    return [
        TowerSite(
            network="AmeriFlux",
            tower_id=str(row.field_site_id),
            tower_name=str(row.Name),
            lat=float(row.field_latitude),
            lon=float(row.field_longitude),
        )
        for row in df.itertuples(index=False)
    ]


def _is_neon_row(df: pd.DataFrame) -> pd.Series:
    return (
        df["entry_name"].astype(str).isin(["Beem-Miller_2022", "Nave_2021"])
        | df["bibliographical_reference"].astype(str).str.contains(
            "NEON|National Ecological Observatory|SOMMOS", case=False, na=False
        )
        | df["site_note"].astype(str).str.contains("tower location", case=False, na=False)
    )


def _load_neon_sites_from_israd() -> list[TowerSite]:
    df = pd.read_csv(NEON_SITES, low_memory=False)
    df["field_latitude"] = pd.to_numeric(df["field_latitude"], errors="coerce")
    df["field_longitude"] = pd.to_numeric(df["field_longitude"], errors="coerce")
    df["field_tower_height_m"] = pd.to_numeric(df["field_tower_height_m"], errors="coerce")
    df["field_number_tower_levels"] = pd.to_numeric(df["field_number_tower_levels"], errors="coerce")
    terrestrial = df["field_site_type"].astype(str).str.contains("Terrestrial", case=False, na=False)
    has_tower = df["field_tower_height_m"].notna() | df["field_number_tower_levels"].notna()
    sub = df.loc[terrestrial & has_tower, ["field_site_id", "field_site_name", "field_latitude", "field_longitude"]].dropna()
    return [
        TowerSite(
            network="NEON",
            tower_id=str(row.field_site_id),
            tower_name=str(row.field_site_name),
            lat=float(row.field_latitude),
            lon=float(row.field_longitude),
        )
        for row in sub.itertuples(index=False)
    ]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _aggregate_israd_sites(path: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    cols = ["site_name", "site_lat", "site_long", "entry_name", value_col]
    sub = df.loc[df[value_col].notna(), cols].dropna(subset=["site_name", "site_lat", "site_long"]).copy()
    out = (
        sub.groupby(["site_name", "site_lat", "site_long"], as_index=False)
        .agg(
            n_obs=(value_col, "count"),
            n_entries=("entry_name", "nunique"),
            entry_names=("entry_name", lambda s: "|".join(sorted(set(map(str, s))))),
        )
        .rename(columns={"site_lat": "lat", "site_long": "lon"})
    )
    return out


def _nearest_match(tower: TowerSite, candidates: pd.DataFrame) -> dict[str, object]:
    if candidates.empty:
        return {
            "site_name": None,
            "distance_km": math.nan,
            "n_obs": 0,
            "n_entries": 0,
            "entry_names": "",
        }
    dists = candidates.apply(
        lambda r: _haversine_km(tower.lat, tower.lon, float(r["lat"]), float(r["lon"])),
        axis=1,
    )
    idx = int(dists.idxmin())
    row = candidates.loc[idx]
    return {
        "site_name": row["site_name"],
        "distance_km": float(dists.loc[idx]),
        "n_obs": int(row["n_obs"]),
        "n_entries": int(row["n_entries"]),
        "entry_names": row["entry_names"],
    }


def _colocation_row(tower: TowerSite, bulk_sites: pd.DataFrame, frac_sites: pd.DataFrame, flux_sites: pd.DataFrame) -> dict[str, object]:
    bulk = _nearest_match(tower, bulk_sites)
    frac = _nearest_match(tower, frac_sites)
    flux = _nearest_match(tower, flux_sites)

    exact_types = 0
    for match in (bulk, frac, flux):
        if math.isfinite(match["distance_km"]) and match["distance_km"] <= 1.0:
            exact_types += 1

    near_types = 0
    for match in (bulk, frac, flux):
        if math.isfinite(match["distance_km"]) and match["distance_km"] <= 25.0:
            near_types += 1

    return {
        "network": tower.network,
        "tower_id": tower.tower_id,
        "tower_name": tower.tower_name,
        "tower_lat": tower.lat,
        "tower_lon": tower.lon,
        "nearest_bulk_site": bulk["site_name"],
        "nearest_bulk_km": bulk["distance_km"],
        "nearest_bulk_obs": bulk["n_obs"],
        "nearest_bulk_entries": bulk["n_entries"],
        "nearest_bulk_entry_names": bulk["entry_names"],
        "nearest_fraction_site": frac["site_name"],
        "nearest_fraction_km": frac["distance_km"],
        "nearest_fraction_obs": frac["n_obs"],
        "nearest_fraction_entries": frac["n_entries"],
        "nearest_fraction_entry_names": frac["entry_names"],
        "nearest_respired_site": flux["site_name"],
        "nearest_respired_km": flux["distance_km"],
        "nearest_respired_obs": flux["n_obs"],
        "nearest_respired_entries": flux["n_entries"],
        "nearest_respired_entry_names": flux["entry_names"],
        "exact_data_type_count_le_1km": exact_types,
        "near_data_type_count_le_25km": near_types,
    }


def main() -> None:
    os.makedirs(EXPORT_DIR, exist_ok=True)

    ameriflux_sites = _load_ameriflux_sites()
    neon_sites = _load_neon_sites_from_israd()
    towers = ameriflux_sites + neon_sites

    bulk_sites = _aggregate_israd_sites(ISRAD_LAYER, "lyr_14c")
    fraction_sites = _aggregate_israd_sites(ISRAD_FRACTION, "frc_14c")
    flux_sites = _aggregate_israd_sites(ISRAD_FLUX, "flx_14c")

    rows = [_colocation_row(tower, bulk_sites, fraction_sites, flux_sites) for tower in towers]
    coloc_df = pd.DataFrame(rows).sort_values(
        ["network", "exact_data_type_count_le_1km", "near_data_type_count_le_25km", "tower_id"],
        ascending=[True, False, False, True],
    )

    priority_df = coloc_df.loc[
        (coloc_df["exact_data_type_count_le_1km"] > 0)
        | (coloc_df["near_data_type_count_le_25km"] >= 2)
    ].copy()
    priority_df = priority_df.sort_values(
        ["exact_data_type_count_le_1km", "near_data_type_count_le_25km", "network", "tower_id"],
        ascending=[False, False, True, True],
    )
    best_df = priority_df.loc[
        (priority_df["exact_data_type_count_le_1km"] >= 2)
        | (
            (priority_df["nearest_fraction_km"] <= 1.0)
            & (priority_df["nearest_respired_km"] <= 1.0)
        )
    ].copy()

    coloc_path = os.path.join(EXPORT_DIR, "flux_tower_israd_colocations.csv")
    priority_path = os.path.join(EXPORT_DIR, "flux_tower_israd_priority_sites.csv")
    best_path = os.path.join(EXPORT_DIR, "flux_tower_israd_best_sites.csv")
    coloc_df.to_csv(coloc_path, index=False)
    priority_df.to_csv(priority_path, index=False)
    best_df.to_csv(best_path, index=False)

    print(f"Saved colocation table: {coloc_path}")
    print(f"Saved priority table:   {priority_path}")
    print(f"Saved best table:       {best_path}")
    print(f"AmeriFlux towers: {len(ameriflux_sites)}")
    print(f"NEON towers:      {len(neon_sites)}")
    print(f"Priority rows:    {len(priority_df)}")
    print(f"Best rows:        {len(best_df)}")


if __name__ == "__main__":
    main()
