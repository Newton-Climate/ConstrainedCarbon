"""
download_icos_sites.py — Fetch international (non-AmeriFlux) flux-tower data for
the field-flux + respiration ISRaD sites, via the ICOS Carbon Portal.

The ICOS Carbon Portal is a multi-network hub: it hosts ONEFlux FLUXNET products
(same FLUXNET2015 archive format as AmeriFlux — with GPP_DT/NT partitioning) from
several regional networks, distinguished by filename prefix:

    ICOS_...   ICOS (Europe)
    EUF_...    European Fluxes Database
    JPF_...    JapanFlux / AsiaFlux
    FLX_...    FLUXNET2015

So a single ICOS-portal downloader covers our international ISRaD sites across
continents.  Colocating the international field-flux + bulk-layer ISRaD sites
against the ICOS station catalogue (474 ecosystem stations) gives:

    Auchencorth Moss (UK)      → UK-AMo  0.1 km   ICOS_ FLUXNET  (Europe)
    Adventdalen (Svalbard)     → SJ-Adv  4.4 km   EUF_  FLUXNET  (Arctic)
    Baram Basin (Malaysia)     → MY-LHP  43 km    JPF_  FLUXNET  (Asia)

Sites with no downloadable FLUXNET product at the nearest station (Solling,
Fichtelgebirge, Heidelberg in DE; Appi in JP; Kivu in CD) are reported and
skipped.  Australia has no field-flux ISRaD site in our set.

ICOS data is CC-BY-4.0 and needs no account, but downloading requires
acknowledging the licence.  This script therefore requires an explicit
``--accept-license`` flag (analogous to the AmeriFlux ``--accept-data-policy``):
without it, it prints the licence URL and the resolved site→station mapping, and
downloads nothing.

Examples
--------
    # Dry run — colocate + list products, download nothing:
    python notebooks/download_icos_sites.py

    # Download the FLUXNET zips into data/ and unzip:
    python notebooks/download_icos_sites.py --accept-license
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import zipfile

import requests


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

SPARQL_ENDPOINT = "https://meta.icos-cp.eu/sparql"
OBJECT_BASE = "https://data.icos-cp.eu/objects"
LICENSE_URL = "https://data.icos-cp.eu/licence"

# International field-flux + bulk-layer ISRaD sites (flux_obs>0 & layer_obs>0,
# outside the AmeriFlux domain).  Coords from notebooks/exports/israd_eligible_sites.csv.
INTL_ISRAD_SITES: dict[str, tuple[float, float]] = {
    "Auchencorth Moss": (55.793, -3.243),   # UK
    "Adventdalen Valley": (78.170, 16.100),  # Svalbard
    "Baram Basin": (4.478, 114.305),         # Malaysia
    "Solling": (51.517, 9.567),              # Germany
    "Fichtelgeberge": (50.133, 11.867),      # Germany
    "Heidelberg": (49.400, 8.670),           # Germany
    "Appi forest": (40.000, 140.560),        # Japan
    "Kivu": (2.313, 28.755),                 # DR Congo
}

DEFAULT_MAX_KM = 50.0


def _sparql(query: str, retries: int = 4) -> list[dict]:
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(SPARQL_ENDPOINT, data={"query": query},
                              headers={"Accept": "application/json"}, timeout=90)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                return r.json()["results"]["bindings"]
            last = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(2 * (attempt + 1))  # linear backoff for transient 503s
    raise RuntimeError(f"ICOS SPARQL failed after {retries} attempts: {last}")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _icos_stations() -> list[tuple[str, str, float, float]]:
    """All ICOS ecosystem (ES) stations: (name, station_id, lat, lon)."""
    rows = _sparql(
        "prefix cpmeta: <http://meta.icos-cp.eu/ontologies/cpmeta/>\n"
        "select ?station ?id ?lat ?lon where{\n"
        " ?s a cpmeta:ES ; cpmeta:hasName ?station ; cpmeta:hasStationId ?id ;\n"
        "    cpmeta:hasLatitude ?lat ; cpmeta:hasLongitude ?lon .}"
    )
    out = []
    for r in rows:
        try:
            out.append((r["station"]["value"], r["id"]["value"],
                        float(r["lat"]["value"]), float(r["lon"]["value"])))
        except (KeyError, ValueError):
            continue
    return out


def _fluxnet_product(station_id: str) -> dict | None:
    """Best downloadable FLUXNET product for a station, or None.

    Prefers the full FLUXNET archive (with GPP partitioning) over the half-hourly
    FLUXMET file, and the latest version (no newer version exists).
    """
    rows = _sparql(
        "prefix cpmeta: <http://meta.icos-cp.eu/ontologies/cpmeta/>\n"
        "prefix prov: <http://www.w3.org/ns/prov#>\n"
        "select ?dobj ?fileName ?size where{\n"
        " ?dobj cpmeta:hasName ?fileName ; cpmeta:hasSizeInBytes ?size .\n"
        f" ?dobj cpmeta:wasAcquiredBy/prov:wasAssociatedWith/cpmeta:hasStationId \"{station_id}\" .\n"
        " FILTER(CONTAINS(?fileName,\"FLUXNET\") && STRENDS(?fileName,\".zip\"))\n"
        " FILTER NOT EXISTS {[] cpmeta:isNextVersionOf ?dobj} }"
    )
    products = [
        {"name": r["fileName"]["value"],
         "hash": r["dobj"]["value"].rstrip("/").split("/")[-1],
         "size": int(r["size"]["value"])}
        for r in rows
    ]
    # Drop the FLUXMET half-hourly companion; keep the full FLUXNET product.
    full = [p for p in products if "FLUXMET" not in p["name"]]
    pool = full or products
    if not pool:
        return None
    return max(pool, key=lambda p: p["size"])  # full archive is the larger file


def _resolve_targets(sites: dict[str, tuple[float, float]], max_km: float) -> list[dict]:
    """Colocate ISRaD sites to nearest ICOS station and attach FLUXNET product."""
    stations = _icos_stations()
    targets = []
    for name, (lat, lon) in sites.items():
        near = min(stations, key=lambda s: _haversine_km(lat, lon, s[2], s[3]))
        dist = _haversine_km(lat, lon, near[2], near[3])
        prod = _fluxnet_product(near[1]) if dist <= max_km else None
        targets.append({
            "israd_site": name, "station": near[0], "station_id": near[1],
            "distance_km": dist, "product": prod,
        })
    return targets


def _already_present(name_stem: str, out_dir: str) -> bool:
    return os.path.isdir(os.path.join(out_dir, name_stem))


def _download(product: dict, out_dir: str) -> str:
    """Download one ICOS object (CC-BY licence cookie) to out_dir; return path."""
    url = f"{OBJECT_BASE}/{product['hash']}"
    dest = os.path.join(out_dir, product["name"])
    # The CpLicenseAcceptedFor cookie signals CC-BY acceptance (see --accept-license).
    with requests.get(url, stream=True, timeout=900,
                      cookies={"CpLicenseAcceptedFor": product["hash"]}) as r:
        r.raise_for_status()
        total = 0
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                total += len(chunk)
    print(f"    downloaded {product['name']}  ({total / 1e6:.1f} MB)")
    return dest


def _unzip(zip_path: str, out_dir: str, remove_zip: bool = True) -> None:
    stem = os.path.splitext(os.path.basename(zip_path))[0]
    target = os.path.join(out_dir, stem)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    print(f"    extracted → {os.path.relpath(target, ROOT)}")
    if remove_zip:
        os.remove(zip_path)
        print("    removed archive to save space")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", default="",
                    help="Comma-separated ICOS station ids to fetch directly "
                         "(e.g. UK-AMo,SJ-Adv,MY-LHP), bypassing colocation.")
    ap.add_argument("--max-km", type=float, default=DEFAULT_MAX_KM,
                    help=f"Max colocation distance (default: {DEFAULT_MAX_KM:.0f} km).")
    ap.add_argument("--out-dir", default=DATA_DIR, help="Destination directory (default: data/).")
    ap.add_argument("--keep-zips", action="store_true",
                    help="Keep downloaded zips after extraction (default: delete to save disk).")
    ap.add_argument("--accept-license", action="store_true",
                    help="Confirm you accept the ICOS CC-BY-4.0 data licence. "
                         "Required before anything is downloaded.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Resolving ICOS FLUXNET products…")
    if args.stations.strip():
        targets = []
        for sid in [s.strip() for s in args.stations.split(",") if s.strip()]:
            prod = _fluxnet_product(sid)
            targets.append({"israd_site": "(manual)", "station": sid, "station_id": sid,
                            "distance_km": 0.0, "product": prod})
    else:
        targets = _resolve_targets(INTL_ISRAD_SITES, args.max_km)

    print(f"\n  {'ISRaD site':22s} {'station':10s} {'dist':>7s}  product")
    print("  " + "─" * 78)
    downloadable = []
    for t in sorted(targets, key=lambda x: x["distance_km"]):
        prod = t["product"]
        if prod is None:
            reason = "no FLUXNET product" if t["distance_km"] <= args.max_km else f">{args.max_km:.0f} km"
            print(f"  {t['israd_site']:22s} {t['station_id']:10s} {t['distance_km']:6.1f}k  — {reason}")
            continue
        stem = os.path.splitext(prod["name"])[0]
        tag = "already present" if _already_present(stem, args.out_dir) else f"{prod['size']/1e6:.0f} MB"
        print(f"  {t['israd_site']:22s} {t['station_id']:10s} {t['distance_km']:6.1f}k  {prod['name']}  [{tag}]")
        if not _already_present(stem, args.out_dir):
            downloadable.append(prod)

    if not downloadable:
        print("\nNothing to download (no new products resolved).")
        return 0

    if not args.accept_license:
        print(f"\nICOS data is CC-BY-4.0. Review the licence: {LICENSE_URL}")
        print("Re-run with --accept-license to download. Nothing was downloaded.")
        return 1

    total_mb = sum(p["size"] for p in downloadable) / 1e6
    print(f"\nDownloading {len(downloadable)} product(s) (~{total_mb:.0f} MB)…")
    for prod in downloadable:
        try:
            zip_path = _download(prod, args.out_dir)
            _unzip(zip_path, args.out_dir, remove_zip=not args.keep_zips)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED {prod['name']}: {exc}")

    print("\nDone. Downloaded ICOS data into", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
