"""
download_ameriflux_sites.py — Fetch AmeriFlux tower data for the field-flux +
respiration ISRaD sites.

These towers supply the annual-mean GPP that drives the canonical soil-carbon
¹⁴C inversion at each co-located ISRaD site (bulk-layer ¹⁴C + field CO₂-efflux
¹⁴C).  Only towers within ~25 km of such an ISRaD site have AmeriFlux coverage;
the default site list below is exactly that set (minus US-EML / US-Ho1, which are
already in ``data/``).

Data is pulled through the AmeriFlux data-download REST API — the same endpoint
the ``amerifluxr`` R package uses.  Using AmeriFlux data requires a **registered
user account** and **acceptance of the AmeriFlux Data Use Policy**:

    https://ameriflux.lbl.gov/data/data-policy/

This script therefore does NOT hardcode credentials or silently accept the
policy.  Supply your AmeriFlux user id + email via a gitignored ``.env`` file at
the repo root (``AMERIFLUX_USER_ID`` / ``AMERIFLUX_EMAIL``) — or via real env
vars / ``--user-id`` / ``--email`` — and pass ``--accept-data-policy`` to confirm
you personally accept the policy.  Without that flag the script prints the policy
URL and exits before any network write.

``.env`` format (repo root)::

    AMERIFLUX_USER_ID=your_id
    AMERIFLUX_EMAIL=you@example.com

Examples
--------
    # Dry run — shows the resolved payload and target sites, downloads nothing:
    python notebooks/download_ameriflux_sites.py --sites US-Ha1

    # Real download (reads .env for credentials; fetches FLUXNET zips into data/):
    python notebooks/download_ameriflux_sites.py --accept-data-policy
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile

import requests


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
ENV_FILE = os.path.join(ROOT, ".env")


def _load_dotenv(path: str) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no overwrite).

    Zero-dependency: existing real environment variables take precedence, so
    ``AMERIFLUX_USER_ID=... python download_ameriflux_sites.py`` still wins over
    the file.  Ignores blanks, comments, and optional surrounding quotes.
    """
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

# AmeriFlux data-download API (as used by the amerifluxr R package).
DOWNLOAD_API = "https://amfcdn.lbl.gov/api/v1/data_download"
DATA_POLICY_URL = "https://ameriflux.lbl.gov/data/data-policy/"

# Co-located field-flux + bulk-layer ISRaD towers (see plan / colocation export).
# US-EML (BASE) and US-Ho1 (FLUXNET) are already in data/, so they are omitted.
DEFAULT_SITES = [
    "BR-Sa3",   # FLONA / Santarém Km83  (0.14 km from ISRaD "FLONA")
    "CA-NS4",   # UCI 1964 burn          (0.14 km from "CZ_1964burn_NSA")
    "US-Ha1",   # Harvard Forest EMS     (canonical long-record tower)
    "US-Ha2",   # Harvard Forest hemlock (0.19 km from ISRaD "Harvard Forest")
    "CA-NS1",   # UCI 1850 / Old Black Spruce (0.24 km from "CZ_Old_Black_Spruce")
    "US-Bes",   # Barrow-Bes             (1.78 km from ISRaD "Barrow")
    "BR-Ma2",   # Manaus ZF2 K34         (12.9 km from ISRaD "ZF2")
]

# AmeriFlux data products: FLUXNET is gap-filled with GPP_DT/NT partitioning
# (best for annual-mean GPP); BASE-BADM is the base half-hourly product.
VALID_PRODUCTS = {"FLUXNET", "BASE-BADM"}

# FLUXNET variants: FULLSET (all variables incl. GPP partitioning) or SUBSET.
VALID_VARIANTS = {"FULLSET", "SUBSET"}

# Policy string the API expects for the modern CC-BY-4.0 licence.
DATA_POLICY = "CCBY4.0"

# intended_use must come from the AmeriFlux controlled vocabulary.
INTENDED_USE_CHOICES = (
    "Research - Multi-site synthesis",
    "Research - Remote sensing",
    "Research - Land model/Earth system model",
    "Research - Other",
    "Education (Teacher or Student)",
    "Other",
)
DEFAULT_INTENDED_USE = "Research - Land model/Earth system model"


def _already_present(site: str, out_dir: str) -> bool:
    """True if a directory for this AmeriFlux site already exists under out_dir."""
    prefix = f"AMF_{site}_"
    return any(
        name.startswith(prefix) and os.path.isdir(os.path.join(out_dir, name))
        for name in os.listdir(out_dir)
    )


def _request_urls(payload: dict) -> list[str]:
    """POST to the AmeriFlux API and return the list of download URLs.

    The API returns ``data_urls`` as a list of objects each carrying a ``url``
    field (see the amerifluxr client).  Returns an empty list if the API had no
    matching files for the requested sites/product (not an error — the caller
    may fall back to another product).
    """
    resp = requests.post(DOWNLOAD_API, json=payload, timeout=120)
    resp.raise_for_status()
    body = resp.json()
    urls: list[str] = []
    for item in body.get("data_urls", []) or []:
        if isinstance(item, dict):
            url = item.get("url")
            if isinstance(url, str) and url.startswith("http"):
                urls.append(url)
        elif isinstance(item, str) and item.startswith("http"):
            urls.append(item)
    return sorted(set(urls))


def _build_payload(product: str, variant: str, sites: list[str], args) -> dict:
    """Assemble the AmeriFlux download JSON body for a product + site list."""
    payload = {
        "user_id": args.user_id,
        "user_email": args.email,
        "data_product": product,
        "data_policy": DATA_POLICY,
        "site_ids": sites,
        "intended_use": args.intended_use,
        "description": f"{args.description} [ecosystem-complexity]",
        "is_test": "",  # API expects "true" or empty string, not a bool
    }
    if product == "FLUXNET":
        payload["data_variant"] = variant
    return payload


def _sites_covered(urls: list[str], sites: list[str]) -> set[str]:
    """Which of `sites` appear in the returned download filenames."""
    return {s for s in sites if any(f"_{s}_" in u or f"/{s}_" in u for u in urls)}


def _download(url: str, out_dir: str) -> str:
    """Stream a zip to out_dir; return the local path."""
    fname = url.split("?")[0].rstrip("/").split("/")[-1] or "ameriflux_download.zip"
    dest = os.path.join(out_dir, fname)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = 0
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                total += len(chunk)
    print(f"    downloaded {fname}  ({total / 1e6:.1f} MB)")
    return dest


def _unzip(zip_path: str, out_dir: str, remove_zip: bool = True) -> None:
    """Extract a downloaded AmeriFlux zip into out_dir/<zip-stem>/.

    Removes the zip after a successful extraction (``remove_zip=True``) so the
    archive and its extracted copy don't both sit on disk — AmeriFlux zips are
    large and this halves the peak footprint.
    """
    stem = os.path.splitext(os.path.basename(zip_path))[0]
    target = os.path.join(out_dir, stem)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    print(f"    extracted → {os.path.relpath(target, ROOT)}")
    if remove_zip:
        os.remove(zip_path)
        print("    removed archive to save space")


def main() -> int:
    # Pre-parse only --env-file so the .env can populate argparse env-var defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=ENV_FILE)
    pre_args, _ = pre.parse_known_args()
    _load_dotenv(pre_args.env_file)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-file", default=ENV_FILE,
                    help="Path to a .env file with AMERIFLUX_USER_ID / AMERIFLUX_EMAIL "
                         "(default: repo-root/.env).")
    ap.add_argument("--sites", default=",".join(DEFAULT_SITES),
                    help="Comma-separated AmeriFlux site ids (default: co-located set).")
    ap.add_argument("--data-product", default="FLUXNET", choices=sorted(VALID_PRODUCTS),
                    help="AmeriFlux data product (default: FLUXNET).")
    ap.add_argument("--data-variant", default="FULLSET", choices=sorted(VALID_VARIANTS),
                    help="FLUXNET variant (default: FULLSET). Ignored for BASE-BADM.")
    ap.add_argument("--no-base-fallback", action="store_true",
                    help="Do not fall back to BASE-BADM for sites with no FLUXNET release.")
    ap.add_argument("--keep-zips", action="store_true",
                    help="Keep downloaded zips after extraction (default: delete to save disk).")
    ap.add_argument("--out-dir", default=DATA_DIR, help="Destination directory (default: data/).")
    ap.add_argument("--user-id", default=os.environ.get("AMERIFLUX_USER_ID"),
                    help="AmeriFlux user id (or env AMERIFLUX_USER_ID).")
    ap.add_argument("--email", default=os.environ.get("AMERIFLUX_EMAIL"),
                    help="AmeriFlux account email (or env AMERIFLUX_EMAIL).")
    ap.add_argument("--intended-use", default=DEFAULT_INTENDED_USE,
                    choices=INTENDED_USE_CHOICES,
                    help="Intended-use category reported to AmeriFlux "
                         f"(default: {DEFAULT_INTENDED_USE!r}).")
    ap.add_argument("--description",
                    default="Soil-carbon radiocarbon inversion: annual-mean GPP "
                            "forcing at ISRaD field-flux + respiration sites.",
                    help="Free-text description reported to AmeriFlux.")
    ap.add_argument("--accept-data-policy", action="store_true",
                    help="Confirm you personally accept the AmeriFlux Data Use "
                         "Policy. Required before anything is downloaded.")
    args = ap.parse_args()

    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    pending = [s for s in sites if not _already_present(s, args.out_dir)]
    skipped = [s for s in sites if s not in pending]

    print(f"AmeriFlux data product : {args.data_product}")
    print(f"Requested sites        : {', '.join(sites)}")
    if skipped:
        print(f"Already present (skip) : {', '.join(skipped)}")
    print(f"To download            : {', '.join(pending) or '(none)'}")
    print(f"Destination            : {args.out_dir}")

    if not pending:
        print("\nNothing to download — all requested sites already present.")
        return 0

    # Gate: credentials + explicit policy acceptance before any network write.
    if not args.accept_data_policy:
        print("\nAmeriFlux data requires accepting the Data Use Policy:")
        print(f"  {DATA_POLICY_URL}")
        print("Re-run with --accept-data-policy (and your AmeriFlux user id/email) "
              "to download. Nothing was downloaded.")
        return 1
    if not args.user_id or not args.email:
        print("\nMissing AmeriFlux credentials. Provide --user-id and --email "
              "(or set AMERIFLUX_USER_ID / AMERIFLUX_EMAIL). Nothing was downloaded.")
        return 1

    print(f"\nRequesting {args.data_product} manifest from AmeriFlux…")
    try:
        urls = _request_urls(_build_payload(args.data_product, args.data_variant, pending, args))
    except Exception as exc:  # noqa: BLE001 — surface the API error to the user
        print(f"API request failed: {exc}")
        return 2

    # Sites the primary product covered; fall back to BASE-BADM for the rest.
    covered = _sites_covered(urls, pending)
    missing = [s for s in pending if s not in covered]
    if missing and args.data_product == "FLUXNET" and not args.no_base_fallback:
        print(f"  No FLUXNET release for: {', '.join(missing)} — retrying as BASE-BADM…")
        try:
            base_urls = _request_urls(_build_payload("BASE-BADM", args.data_variant, missing, args))
        except Exception as exc:  # noqa: BLE001
            print(f"  BASE-BADM request failed: {exc}")
            base_urls = []
        urls = sorted(set(urls) | set(base_urls))
        covered |= _sites_covered(base_urls, missing)

    still_missing = [s for s in pending if s not in covered]
    if still_missing:
        print(f"  WARNING: no data returned for: {', '.join(still_missing)}")
    if not urls:
        print("\nNo download URLs returned for any requested site. Nothing downloaded.")
        return 2

    print(f"  {len(urls)} file(s) to fetch.")
    for url in urls:
        try:
            local = _download(url, args.out_dir)
            if local.endswith(".zip"):
                _unzip(local, args.out_dir, remove_zip=not args.keep_zips)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED {url}: {exc}")

    print("\nDone. Downloaded AmeriFlux data into", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
