"""Flux-tower download planning and fetch helpers for AmeriFlux and ICOS."""

from __future__ import annotations

import json
import os
import shutil
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Any

from ecosystem_complexity.data.paths import REPO_ROOT
from ecosystem_complexity.sites.spec import SiteSpec, discover_site_specs

AMERIFLUX_API = "https://amfcdn.lbl.gov/api/v1/data_download"
AMERIFLUX_POLICY = "https://ameriflux.lbl.gov/data/data-policy/"
ICOS_SPARQL = "https://meta.icos-cp.eu/sparql"
ICOS_OBJECT_BASE = "https://data.icos-cp.eu/objects"
ICOS_LICENSE = "https://data.icos-cp.eu/licence"
DEFAULT_AMERIFLUX_USE = "Research - Land model/Earth system model"


@dataclass(frozen=True)
class FluxDownloadPlan:
    """Resolved download target for one flux tower."""

    selector: str
    tower_id: str
    source: str
    forcing_glob: str
    local_path: str
    remote_label: str


def _load_dotenv(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key and key not in os.environ:
                os.environ[key.strip()] = value.strip().strip('"').strip("'")


def _local_data_path(forcing_glob: str) -> str:
    return os.path.join(REPO_ROOT, "data", forcing_glob)


def _resolve_spec(selector: str) -> SiteSpec:
    specs = discover_site_specs()
    by_key: dict[str, SiteSpec] = {}
    for spec in specs.values():
        for key in {spec.config_stem, spec.israd_name, spec.label, spec.tower_id}:
            if key:
                by_key[key] = spec
    if selector not in by_key:
        raise KeyError(f"Unknown site selector {selector!r}.")
    return by_key[selector]


def resolve_flux_download_plan(selector: str) -> FluxDownloadPlan:
    """Resolve a user-facing selector to a concrete download plan."""
    spec = _resolve_spec(selector)
    forcing_glob = spec.forcing_glob
    local_path = _local_data_path(forcing_glob)
    source_prefix = os.path.basename(forcing_glob).split("_", 1)[0]
    if source_prefix == "AMF":
        source = "AmeriFlux"
    elif source_prefix in {"ICOS", "EUF", "JPF", "FLX"}:
        source = "ICOS"
    else:
        raise ValueError(
            f"Could not infer data source for forcing_glob={forcing_glob!r}."
        )
    tower_id = spec.tower_id or os.path.basename(forcing_glob).split("_")[1]
    return FluxDownloadPlan(
        selector=selector,
        tower_id=tower_id,
        source=source,
        forcing_glob=forcing_glob,
        local_path=local_path,
        remote_label=os.path.basename(forcing_glob),
    )


def _extract_zip(zip_path: str, out_dir: str, keep_archive: bool) -> str:
    stem = os.path.splitext(os.path.basename(zip_path))[0]
    target = os.path.join(out_dir, stem)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    if not keep_archive:
        os.remove(zip_path)
    return target


def _http_json(url: str, *, data: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
    encoded = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_form_json(url: str, data: dict[str, str]) -> Any:
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        headers={"Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_file(url: str, dest: str, *, headers: dict[str, str] | None = None) -> str:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=900) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    return dest


def _ameriflux_urls(
    *,
    tower_id: str,
    user_id: str,
    email: str,
    product: str = "FLUXNET",
    variant: str = "FULLSET",
    intended_use: str = DEFAULT_AMERIFLUX_USE,
    description: str = "ecosystem-complexity flux download",
) -> list[str]:
    payload: dict[str, Any] = {
        "user_id": user_id,
        "user_email": email,
        "data_product": product,
        "data_policy": "CCBY4.0",
        "site_ids": [tower_id],
        "intended_use": intended_use,
        "description": description,
        "is_test": "",
    }
    if product == "FLUXNET":
        payload["data_variant"] = variant
    body = _http_json(AMERIFLUX_API, data=payload)
    urls: list[str] = []
    for item in body.get("data_urls", []) or []:
        if isinstance(item, dict) and isinstance(item.get("url"), str):
            urls.append(item["url"])
        elif isinstance(item, str):
            urls.append(item)
    return sorted(set(urls))


def _icos_query(query: str) -> list[dict[str, Any]]:
    body = _http_form_json(ICOS_SPARQL, {"query": query})
    return body["results"]["bindings"]


def _icos_product(station_id: str) -> dict[str, Any] | None:
    rows = _icos_query(
        "prefix cpmeta: <http://meta.icos-cp.eu/ontologies/cpmeta/>\n"
        "prefix prov: <http://www.w3.org/ns/prov#>\n"
        "select ?dobj ?fileName ?size where{\n"
        " ?dobj cpmeta:hasName ?fileName ; cpmeta:hasSizeInBytes ?size .\n"
        f" ?dobj cpmeta:wasAcquiredBy/prov:wasAssociatedWith/cpmeta:hasStationId \"{station_id}\" .\n"
        " FILTER(CONTAINS(?fileName,\"FLUXNET\") && STRENDS(?fileName,\".zip\"))\n"
        " FILTER NOT EXISTS {[] cpmeta:isNextVersionOf ?dobj} }"
    )
    products = [
        {
            "name": row["fileName"]["value"],
            "hash": row["dobj"]["value"].rstrip("/").split("/")[-1],
            "size": int(row["size"]["value"]),
        }
        for row in rows
    ]
    non_fluxmet = [p for p in products if "FLUXMET" not in p["name"]]
    pool = non_fluxmet or products
    if not pool:
        return None
    return max(pool, key=lambda product: product["size"])


def download_flux_data(
    selector: str,
    *,
    out_dir: str | None = None,
    accept_policy: bool = False,
    accept_license: bool = False,
    user_id: str | None = None,
    email: str | None = None,
    env_file: str | None = None,
    keep_archive: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """Download flux data for a configured site into the repo data directory."""
    plan = resolve_flux_download_plan(selector)
    if os.path.isdir(plan.local_path):
        return [plan.local_path]

    target_dir = os.path.abspath(out_dir or os.path.join(REPO_ROOT, "data"))
    os.makedirs(target_dir, exist_ok=True)
    if env_file is not None:
        _load_dotenv(env_file)

    if plan.source == "AmeriFlux":
        if not accept_policy:
            raise PermissionError(
                f"AmeriFlux download requires explicit policy acceptance: {AMERIFLUX_POLICY}"
            )
        user_id = user_id or os.environ.get("AMERIFLUX_USER_ID")
        email = email or os.environ.get("AMERIFLUX_EMAIL")
        if not user_id or not email:
            raise ValueError("Missing AmeriFlux credentials (user_id and email).")
        urls = _ameriflux_urls(tower_id=plan.tower_id, user_id=user_id, email=email)
        if dry_run:
            return urls
        outputs: list[str] = []
        for url in urls:
            name = url.split("?")[0].rstrip("/").split("/")[-1] or "ameriflux_download.zip"
            archive = os.path.join(target_dir, name)
            _download_file(url, archive)
            outputs.append(_extract_zip(archive, target_dir, keep_archive))
        return outputs

    if not accept_license:
        raise PermissionError(
            f"ICOS download requires explicit licence acceptance: {ICOS_LICENSE}"
        )
    product = _icos_product(plan.tower_id)
    if product is None:
        raise FileNotFoundError(f"No ICOS FLUXNET product found for station {plan.tower_id}.")
    url = f"{ICOS_OBJECT_BASE}/{product['hash']}"
    if dry_run:
        return [url]
    archive = os.path.join(target_dir, product["name"])
    _download_file(url, archive, headers={"Cookie": f"CpLicenseAcceptedFor={product['hash']}"})
    return [_extract_zip(archive, target_dir, keep_archive)]

