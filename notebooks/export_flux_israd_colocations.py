"""
export_flux_israd_colocations.py — Colocate flux towers with ISRaD radiocarbon data.

Builds two tower catalogs from current workspace data:
  - local AmeriFlux sites with coordinates from the model configs
  - NEON tower sites represented in ISRaD (tower-location NEON entries)

Covers every ISRaD flat datatype that carries a Δ¹⁴C measurement:
  - layer        (lyr_14c)  bulk soil layer 14C
  - fraction     (frc_14c)  fractionated soil 14C
  - flux         (flx_14c)  field CO₂ efflux (respired) 14C
  - incubation   (inc_14c)  lab-incubated respired CO₂ 14C
  - interstitial (ist_14c)  soil pore-space gas 14C

An ISRaD site is treated as *eligible* for a soil-carbon inversion when it has
either
  - fraction 14C, or
  - bulk/layer 14C together with respiration 14C (flux and/or incubation).
Interstitial 14C is reported for reference but does not count as respiration.

Outputs
-------
  notebooks/exports/israd_eligible_sites.csv
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

ISRAD_DIR = os.path.join(ROOT, "data", "shared", "israd")
ISRAD_VERSION = "2.6.6.2024-01-25"

# datatype -> (flat-file basename, Δ¹⁴C value column)
DATATYPES: dict[str, tuple[str, str]] = {
    "layer": (f"ISRaD_data_flat_layer_v {ISRAD_VERSION}.csv", "lyr_14c"),
    "fraction": (f"ISRaD_data_flat_fraction_v {ISRAD_VERSION}.csv", "frc_14c"),
    "flux": (f"ISRaD_data_flat_flux_v {ISRAD_VERSION}.csv", "flx_14c"),
    "incubation": (f"ISRaD_data_flat_incubation_v {ISRAD_VERSION}.csv", "inc_14c"),
    "interstitial": (f"ISRaD_data_flat_interstitial_v {ISRAD_VERSION}.csv", "ist_14c"),
}

# datatypes whose 14C measures respired CO₂ (field efflux or lab incubation)
RESPIRATION_TYPES = ("flux", "incubation")

# colocation distance bands (km)
EXACT_KM = 1.0
NEAR_KM = 25.0


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


def _aggregate_israd_sites(datatype: str) -> pd.DataFrame:
    """Per-site Δ¹⁴C observation counts for one ISRaD datatype."""
    basename, value_col = DATATYPES[datatype]
    path = os.path.join(ISRAD_DIR, basename)
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


def _build_site_catalog(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge per-datatype aggregates into one site table with eligibility flags."""
    master: pd.DataFrame | None = None
    for name, df in frames.items():
        renamed = df.rename(
            columns={
                "n_obs": f"{name}_obs",
                "n_entries": f"{name}_entries",
                "entry_names": f"{name}_entry_names",
            }
        )
        master = renamed if master is None else master.merge(renamed, on=["site_name", "lat", "lon"], how="outer")
    assert master is not None

    for name in frames:
        master[f"{name}_obs"] = master[f"{name}_obs"].fillna(0).astype(int)
        master[f"{name}_entries"] = master[f"{name}_entries"].fillna(0).astype(int)
        master[f"{name}_entry_names"] = master[f"{name}_entry_names"].fillna("")

    master["has_layer"] = master["layer_obs"] > 0
    master["has_fraction"] = master["fraction_obs"] > 0
    master["has_respiration"] = master[[f"{t}_obs" for t in RESPIRATION_TYPES]].gt(0).any(axis=1)
    master["has_interstitial"] = master["interstitial_obs"] > 0

    master["eligible_fraction"] = master["has_fraction"]
    master["eligible_pool_resp"] = master["has_layer"] & master["has_respiration"]
    master["eligible"] = master["eligible_fraction"] | master["eligible_pool_resp"]

    def _path(row: pd.Series) -> str:
        paths = []
        if row["eligible_fraction"]:
            paths.append("fraction")
        if row["eligible_pool_resp"]:
            paths.append("pool+respiration")
        return "|".join(paths)

    master["eligibility_path"] = master.apply(_path, axis=1)
    return master.sort_values(["eligible", "site_name"], ascending=[False, True]).reset_index(drop=True)


def _nearest_match(tower: TowerSite, candidates: pd.DataFrame) -> dict[str, object]:
    """Nearest candidate site to a tower; candidates need site_name/lat/lon columns."""
    if candidates.empty:
        return {"site_name": None, "distance_km": math.nan, "n_obs": 0}
    dists = candidates.apply(
        lambda r: _haversine_km(tower.lat, tower.lon, float(r["lat"]), float(r["lon"])),
        axis=1,
    )
    idx = int(dists.idxmin())
    row = candidates.loc[idx]
    n_obs = int(row["n_obs"]) if "n_obs" in row and pd.notna(row["n_obs"]) else 0
    return {"site_name": row["site_name"], "distance_km": float(dists.loc[idx]), "n_obs": n_obs}


def _colocation_row(
    tower: TowerSite,
    frames: dict[str, pd.DataFrame],
    catalog: pd.DataFrame,
) -> dict[str, object]:
    row: dict[str, object] = {
        "network": tower.network,
        "tower_id": tower.tower_id,
        "tower_name": tower.tower_name,
        "tower_lat": tower.lat,
        "tower_lon": tower.lon,
    }

    # nearest raw site per datatype (diagnostic detail)
    for name, df in frames.items():
        m = _nearest_match(tower, df)
        row[f"nearest_{name}_site"] = m["site_name"]
        row[f"nearest_{name}_km"] = m["distance_km"]
        row[f"nearest_{name}_obs"] = m["n_obs"]

    # nearest eligible site (any path) and per-path
    eligible = catalog.loc[catalog["eligible"]].reset_index(drop=True)
    frac_path = catalog.loc[catalog["eligible_fraction"]].reset_index(drop=True)
    poolresp_path = catalog.loc[catalog["eligible_pool_resp"]].reset_index(drop=True)

    m_elig = _nearest_match(tower, eligible)
    m_frac = _nearest_match(tower, frac_path)
    m_pr = _nearest_match(tower, poolresp_path)

    if m_elig["site_name"] is not None:
        path = catalog.loc[catalog["site_name"] == m_elig["site_name"], "eligibility_path"]
        row["nearest_eligible_path"] = path.iloc[0] if not path.empty else ""
    else:
        row["nearest_eligible_path"] = ""
    row["nearest_eligible_site"] = m_elig["site_name"]
    row["nearest_eligible_km"] = m_elig["distance_km"]
    row["nearest_fraction_path_site"] = m_frac["site_name"]
    row["nearest_fraction_path_km"] = m_frac["distance_km"]
    row["nearest_pool_resp_path_site"] = m_pr["site_name"]
    row["nearest_pool_resp_path_km"] = m_pr["distance_km"]

    d = m_elig["distance_km"]
    row["eligible_within_1km"] = bool(math.isfinite(d) and d <= EXACT_KM)
    row["eligible_within_25km"] = bool(math.isfinite(d) and d <= NEAR_KM)
    return row


def main() -> None:
    os.makedirs(EXPORT_DIR, exist_ok=True)

    frames = {name: _aggregate_israd_sites(name) for name in DATATYPES}
    catalog = _build_site_catalog(frames)

    eligible_catalog = catalog.loc[catalog["eligible"]].copy()
    eligible_path = os.path.join(EXPORT_DIR, "israd_eligible_sites.csv")
    eligible_catalog.to_csv(eligible_path, index=False)

    ameriflux_sites = _load_ameriflux_sites()
    neon_sites = _load_neon_sites_from_israd()
    towers = ameriflux_sites + neon_sites

    rows = [_colocation_row(tower, frames, catalog) for tower in towers]
    coloc_df = pd.DataFrame(rows).sort_values(
        ["network", "eligible_within_1km", "eligible_within_25km", "nearest_eligible_km", "tower_id"],
        ascending=[True, False, False, True, True],
    )

    priority_df = coloc_df.loc[coloc_df["eligible_within_25km"]].sort_values(
        ["eligible_within_1km", "nearest_eligible_km", "network", "tower_id"],
        ascending=[False, True, True, True],
    )
    best_df = coloc_df.loc[coloc_df["eligible_within_1km"]].sort_values(
        ["nearest_eligible_km", "network", "tower_id"],
        ascending=[True, True, True],
    )

    coloc_path = os.path.join(EXPORT_DIR, "flux_tower_israd_colocations.csv")
    priority_path = os.path.join(EXPORT_DIR, "flux_tower_israd_priority_sites.csv")
    best_path = os.path.join(EXPORT_DIR, "flux_tower_israd_best_sites.csv")
    coloc_df.to_csv(coloc_path, index=False)
    priority_df.to_csv(priority_path, index=False)
    best_df.to_csv(best_path, index=False)

    print(f"Saved eligible ISRaD sites: {eligible_path}")
    print(f"Saved colocation table:     {coloc_path}")
    print(f"Saved priority table:       {priority_path}")
    print(f"Saved best table:           {best_path}")
    for name, df in frames.items():
        print(f"  {name:13s} sites with 14C: {len(df)}")
    print(f"Eligible ISRaD sites: {len(eligible_catalog)} "
          f"(fraction={int(catalog['eligible_fraction'].sum())}, "
          f"pool+resp={int(catalog['eligible_pool_resp'].sum())})")
    print(f"AmeriFlux towers:     {len(ameriflux_sites)}")
    print(f"NEON towers:          {len(neon_sites)}")
    print(f"Priority rows (≤25km): {len(priority_df)}")
    print(f"Best rows (≤1km):      {len(best_df)}")


if __name__ == "__main__":
    main()
