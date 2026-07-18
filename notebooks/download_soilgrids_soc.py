"""
download_soilgrids_soc.py — Fetch ¹²C soil organic carbon (SOC) stocks per site
from SoilGrids (ISRIC) and bin them onto the 3 model pools by depth.

ISRaD carries almost no `lyr_soc` for the field-flux + respiration sites, so the
multi-site inversion had no C-stock constraint and the deep (passive) pool was
weakly constrained.  SoilGrids v2.0 provides global, depth-resolved SOC content,
bulk density and coarse-fragment fractions, from which a SOC **stock** (gC m⁻²)
is computed per standard depth interval and prorated onto the pool depth bins.

    SOC_stock(gC m⁻²) = 10 · SOC(g/kg) · BD(g/cm³) · thickness(cm) · (1 − coarse)

Pool depth bins (from configs/israd_multisite_3pool_config.yaml):
    soil_active   0–10 cm
    soil_slow     10–30 cm
    soil_passive  30–130 cm

Output
------
    notebooks/exports/soilgrids_soc_pools.csv
        site, pool, soc_gCm2, soc_sigma_gCm2, depth_top_cm, depth_bot_cm

No account or licence acknowledgement is required (SoilGrids is CC-BY-4.0).

Run:
    python notebooks/download_soilgrids_soc.py
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd
import requests


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_DIR = os.path.join(ROOT, "notebooks", "exports")
OUT_CSV = os.path.join(EXPORT_DIR, "soilgrids_soc_pools.csv")

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SG_DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

# Pool depth bins (cm) — mirror the shared config's som_pool depths.
POOL_BINS: list[tuple[str, tuple[float, float]]] = [
    ("soil_active", (0.0, 10.0)),
    ("soil_slow", (10.0, 30.0)),
    ("soil_passive", (30.0, 130.0)),
]

# ISRaD field-flux + respiration sites → (lat, lon).  Mirrors the configs read by
# ecosystem_complexity.sites (kept local so this script needs no jax).
SITES: dict[str, tuple[float, float]] = {
    "Howland Forest": (45.20, -68.74),
    "Harvard Forest": (42.54, -72.17),
    "FLONA": (-3.02, -54.97),
    "ZF2": (-2.55, -60.11),
    "CZ_Old_Black_Spruce": (55.88, -98.48),
    "CZ_1964burn_NSA": (55.91, -98.38),
    "Auchencorth Moss": (55.79, -3.24),
    "Adventdalen Valley": (78.17, 16.10),
    "Solling": (51.52, 9.57),
    "Appi forest": (40.00, 140.56),
    "Baram Basin": (4.48, 114.31),
}

# stock uncertainty: at least this fraction of the stock (SoilGrids point values
# at 250 m are uncertain relative to a specific profile).
SIGMA_FRAC = 0.5


def _depth_bounds(label: str) -> tuple[float, float]:
    a, b = label.replace("cm", "").split("-")
    return float(a), float(b)


def _query_site(lat: float, lon: float, retries: int = 4) -> dict[str, dict[str, float]]:
    """Return {property: {depth_label: mean}} for soc/bdod/cfvo at a point."""
    params = [("lon", lon), ("lat", lat), ("value", "mean")]
    for prop in ("soc", "bdod", "cfvo"):
        params.append(("property", prop))
    for d in SG_DEPTHS:
        params.append(("depth", d))
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(SOILGRIDS_URL, params=params, timeout=90)
            if r.status_code == 200:
                out: dict[str, dict[str, float]] = {}
                for lyr in r.json()["properties"]["layers"]:
                    vals = {}
                    for d in lyr["depths"]:
                        m = d["values"].get("mean")
                        if m is not None:
                            vals[d["label"]] = float(m)
                    out[lyr["name"]] = vals
                return out
            last = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(5 * (attempt + 1))  # SoilGrids throttles; back off
    raise RuntimeError(f"SoilGrids query failed after {retries} attempts: {last}")


def _layer_stocks(props: dict[str, dict[str, float]]) -> list[tuple[float, float, float]]:
    """Per SoilGrids layer → (top_cm, bot_cm, SOC_stock gC m⁻²)."""
    soc, bd, cf = props.get("soc", {}), props.get("bdod", {}), props.get("cfvo", {})
    rows = []
    for label in SG_DEPTHS:
        if label not in soc or label not in bd:
            continue
        top, bot = _depth_bounds(label)
        soc_gkg = soc[label] / 10.0        # dg/kg → g/kg
        bd_gcm3 = bd[label] / 100.0        # cg/cm³ → g/cm³
        cf_frac = cf.get(label, 0.0) / 1000.0  # cm³/dm³ → volume fraction
        stock = 10.0 * soc_gkg * bd_gcm3 * (bot - top) * (1.0 - cf_frac)
        rows.append((top, bot, stock))
    return rows


def _bin_to_pools(layers: list[tuple[float, float, float]]) -> dict[str, float]:
    """Prorate SoilGrids-layer stocks onto pool depth bins by depth overlap."""
    pools = {name: 0.0 for name, _ in POOL_BINS}
    for top, bot, stock in layers:
        thick = bot - top
        if thick <= 0:
            continue
        for name, (pt, pb) in POOL_BINS:
            overlap = max(0.0, min(bot, pb) - max(top, pt))
            if overlap > 0:
                pools[name] += stock * (overlap / thick)
    return pools


def main() -> int:
    os.makedirs(EXPORT_DIR, exist_ok=True)
    rows = []
    print(f"Querying SoilGrids for {len(SITES)} sites…")
    for i, (site, (lat, lon)) in enumerate(SITES.items()):
        try:
            props = _query_site(lat, lon)
            pools = _bin_to_pools(_layer_stocks(props))
        except Exception as exc:  # noqa: BLE001
            print(f"  {site:22s}: FAILED ({exc})")
            continue
        for name, (top, bot) in POOL_BINS:
            stock = pools[name]
            rows.append({
                "site": site, "pool": name,
                "soc_gCm2": round(stock, 1),
                "soc_sigma_gCm2": round(max(SIGMA_FRAC * stock, 500.0), 1),
                "depth_top_cm": top, "depth_bot_cm": bot,
            })
        total = sum(pools.values())
        print(f"  {site:22s}: {total:6.0f} gC m⁻²  "
              f"(A={pools['soil_active']:.0f}, S={pools['soil_slow']:.0f}, P={pools['soil_passive']:.0f})")
        if i < len(SITES) - 1:
            time.sleep(12)  # stay under the SoilGrids rate limit
    if rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"\nSaved → {os.path.relpath(OUT_CSV, ROOT)}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
