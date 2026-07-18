"""Locate colocated ISRaD and flux-tower sites from workspace data."""

from __future__ import annotations

import math
import os
from typing import Any

import pandas as pd

from ecosystem_complexity.data.paths import ISRAD_DIR, ISRAD_VERSION, REPO_ROOT
from ecosystem_complexity.sites.spec import discover_site_specs

AMERIFLUX_SITES = os.path.join(REPO_ROOT, "data", "shared", "ameriflux_field_sites.csv")
NEON_SITES = os.path.join(REPO_ROOT, "data", "shared", "neon_field_sites.csv")

DATATYPES: dict[str, tuple[str, str]] = {
    "layer": (f"ISRaD_data_flat_layer_v {ISRAD_VERSION}.csv", "lyr_14c"),
    "fraction": (f"ISRaD_data_flat_fraction_v {ISRAD_VERSION}.csv", "frc_14c"),
    "flux": (f"ISRaD_data_flat_flux_v {ISRAD_VERSION}.csv", "flx_14c"),
    "incubation": (f"ISRaD_data_flat_incubation_v {ISRAD_VERSION}.csv", "inc_14c"),
    "interstitial": (f"ISRaD_data_flat_interstitial_v {ISRAD_VERSION}.csv", "ist_14c"),
}
RESPIRATION_TYPES = ("flux", "incubation")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def _load_ameriflux_sites() -> pd.DataFrame:
    df = pd.read_csv(AMERIFLUX_SITES, sep="\t", low_memory=False)
    df["field_latitude"] = pd.to_numeric(df["field_latitude"], errors="coerce")
    df["field_longitude"] = pd.to_numeric(df["field_longitude"], errors="coerce")
    sub = df.dropna(subset=["field_site_id", "field_latitude", "field_longitude"]).copy()
    out = pd.DataFrame(
        {
            "network": "AmeriFlux",
            "tower_id": sub["field_site_id"].astype(str),
            "tower_name": sub["Name"].astype(str),
            "tower_lat": sub["field_latitude"].astype(float),
            "tower_lon": sub["field_longitude"].astype(float),
        }
    )
    return out


def _load_neon_sites() -> pd.DataFrame:
    df = pd.read_csv(NEON_SITES, low_memory=False)
    df["field_latitude"] = pd.to_numeric(df["field_latitude"], errors="coerce")
    df["field_longitude"] = pd.to_numeric(df["field_longitude"], errors="coerce")
    df["field_tower_height_m"] = pd.to_numeric(df["field_tower_height_m"], errors="coerce")
    df["field_number_tower_levels"] = pd.to_numeric(
        df["field_number_tower_levels"], errors="coerce"
    )
    terrestrial = df["field_site_type"].astype(str).str.contains(
        "Terrestrial", case=False, na=False
    )
    has_tower = df["field_tower_height_m"].notna() | df["field_number_tower_levels"].notna()
    sub = df.loc[
        terrestrial & has_tower,
        ["field_site_id", "field_site_name", "field_latitude", "field_longitude"],
    ].dropna()
    out = pd.DataFrame(
        {
            "network": "NEON",
            "tower_id": sub["field_site_id"].astype(str),
            "tower_name": sub["field_site_name"].astype(str),
            "tower_lat": sub["field_latitude"].astype(float),
            "tower_lon": sub["field_longitude"].astype(float),
        }
    )
    return out


def load_flux_tower_catalog() -> pd.DataFrame:
    """Return a normalized catalog of known tower sites."""
    towers = pd.concat([_load_ameriflux_sites(), _load_neon_sites()], ignore_index=True)
    towers["biome"] = "unclassified"

    specs = discover_site_specs()
    biome_by_tower: dict[str, str] = {}
    for spec in specs.values():
        if spec.tower_id:
            biome_by_tower[spec.tower_id] = spec.biome
    towers["biome"] = towers["tower_id"].map(biome_by_tower).fillna(towers["biome"])
    return towers.sort_values(["network", "tower_id"]).reset_index(drop=True)


def _aggregate_israd_sites(datatype: str) -> pd.DataFrame:
    basename, value_col = DATATYPES[datatype]
    path = os.path.join(ISRAD_DIR, basename)
    df = pd.read_csv(path, low_memory=False)
    cols = ["site_name", "site_lat", "site_long", "entry_name", value_col]
    sub = df.loc[df[value_col].notna(), cols].dropna(
        subset=["site_name", "site_lat", "site_long"]
    )
    return (
        sub.groupby(["site_name", "site_lat", "site_long"], as_index=False)
        .agg(
            n_obs=(value_col, "count"),
            n_entries=("entry_name", "nunique"),
            entry_names=("entry_name", lambda s: "|".join(sorted(set(map(str, s))))),
        )
        .rename(columns={"site_lat": "site_lat", "site_long": "site_lon"})
    )


def build_israd_site_catalog() -> pd.DataFrame:
    """Return one row per ISRaD site with radiocarbon availability metadata."""
    frames = {name: _aggregate_israd_sites(name) for name in DATATYPES}
    master: pd.DataFrame | None = None
    for name, frame in frames.items():
        renamed = frame.rename(
            columns={
                "n_obs": f"{name}_obs",
                "n_entries": f"{name}_entries",
                "entry_names": f"{name}_entry_names",
            }
        )
        master = renamed if master is None else master.merge(
            renamed, on=["site_name", "site_lat", "site_lon"], how="outer"
        )
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

    def _eligibility_path(row: pd.Series) -> str:
        paths: list[str] = []
        if bool(row["eligible_fraction"]):
            paths.append("fraction")
        if bool(row["eligible_pool_resp"]):
            paths.append("pool+respiration")
        return "|".join(paths)

    master["eligibility_path"] = master.apply(_eligibility_path, axis=1)

    biome_by_site: dict[str, str] = {}
    for spec in discover_site_specs().values():
        biome_by_site[spec.israd_name] = spec.biome
    master["biome"] = master["site_name"].map(biome_by_site).fillna("unclassified")

    return master.sort_values(["eligible", "site_name"], ascending=[False, True]).reset_index(
        drop=True
    )


def build_colocation_table(max_distance_km: float | None = 50.0) -> pd.DataFrame:
    """Return all tower/ISRaD pairs, optionally filtered by distance."""
    towers = load_flux_tower_catalog()
    israd = build_israd_site_catalog()

    rows: list[dict[str, Any]] = []
    for tower in towers.itertuples(index=False):
        for site in israd.itertuples(index=False):
            distance = _haversine_km(
                float(tower.tower_lat),
                float(tower.tower_lon),
                float(site.site_lat),
                float(site.site_lon),
            )
            if max_distance_km is not None and distance > max_distance_km:
                continue
            rows.append(
                {
                    "network": tower.network,
                    "tower_id": tower.tower_id,
                    "tower_name": tower.tower_name,
                    "tower_lat": float(tower.tower_lat),
                    "tower_lon": float(tower.tower_lon),
                    "site_name": site.site_name,
                    "site_lat": float(site.site_lat),
                    "site_lon": float(site.site_lon),
                    "biome": site.biome if site.biome != "unclassified" else tower.biome,
                    "distance_km": distance,
                    "eligibility_path": site.eligibility_path,
                    "eligible": bool(site.eligible),
                    "eligible_fraction": bool(site.eligible_fraction),
                    "eligible_pool_resp": bool(site.eligible_pool_resp),
                    "layer_obs": int(site.layer_obs),
                    "fraction_obs": int(site.fraction_obs),
                    "flux_obs": int(site.flux_obs),
                    "incubation_obs": int(site.incubation_obs),
                    "interstitial_obs": int(site.interstitial_obs),
                    "layer_entries": int(site.layer_entries),
                    "fraction_entries": int(site.fraction_entries),
                    "flux_entries": int(site.flux_entries),
                    "incubation_entries": int(site.incubation_entries),
                    "interstitial_entries": int(site.interstitial_entries),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["distance_km", "tower_id", "site_name"]
    ).reset_index(drop=True)


def locate_site(
    *,
    flux_tower: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    biome: str | None = None,
    max_distance_km: float | None = 50.0,
) -> pd.DataFrame:
    """Locate colocated tower / ISRaD sites by tower id, coordinates, or biome."""
    table = build_colocation_table(max_distance_km=max_distance_km)
    if table.empty:
        return table

    if flux_tower:
        needle = flux_tower.casefold()
        mask = (
            table["tower_id"].astype(str).str.casefold().eq(needle)
            | table["tower_name"].astype(str).str.casefold().str.contains(needle, na=False)
            | table["site_name"].astype(str).str.casefold().str.contains(needle, na=False)
        )
        table = table.loc[mask].copy()

    if biome:
        biome_needle = biome.casefold()
        mask = table["biome"].astype(str).str.casefold().str.contains(biome_needle, na=False)
        table = table.loc[mask].copy()

    if lat is not None and lon is not None:
        table["query_distance_km"] = table.apply(
            lambda row: _haversine_km(lat, lon, float(row["tower_lat"]), float(row["tower_lon"])),
            axis=1,
        )
        table = table.sort_values(["query_distance_km", "distance_km", "tower_id", "site_name"])

    return table.reset_index(drop=True)

