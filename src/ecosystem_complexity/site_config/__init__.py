"""Helpers for building and normalizing site-level config YAML content."""

from __future__ import annotations

import copy
import glob
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import yaml

from ecosystem_complexity.data.paths import REPO_ROOT

if TYPE_CHECKING:
    from ecosystem_complexity.sites.spec import SiteSpec

DEFAULT_ANALYSIS_CONFIG: dict[str, Any] = {
    "metrics": {
        "reduced_chi2": True,
        "rmse": True,
        "degrees_of_freedom": True,
        "averaging_kernel": True,
        "gain_matrix": True,
        "jacobian": True,
        "constraint_ladder": True,
        "shapley": False,
        "ablation": False,
        "orthogonality": False,
    },
    "plots": {
        "model_results": True,
        "information_content": True,
        "observation_fit": True,
    },
}

DEFAULT_OUTPUT_CONFIG: dict[str, Any] = {
    "artifact_dir": "results/{config_stem}",
    "summary_json": True,
    "matrices_npz": True,
    "observation_csv": True,
    "diagnostics_csv": True,
    "figure_png": True,
}

_MULTISITE_TEMPLATE = os.path.join(REPO_ROOT, "configs", "israd_multisite_3pool_config.yaml")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings into a new dict."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def normalize_analysis_config(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fill missing analysis options with project defaults."""
    return _deep_merge(DEFAULT_ANALYSIS_CONFIG, dict(raw or {}))


def normalize_output_config(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fill missing output options with project defaults."""
    return _deep_merge(DEFAULT_OUTPUT_CONFIG, dict(raw or {}))


def render_artifact_dir(template: str, *, config_stem: str, site_id: str) -> str:
    """Render an output template relative to the repo root."""
    rendered = template.format(config_stem=config_stem, site_id=site_id)
    if os.path.isabs(rendered):
        return rendered
    return os.path.join(REPO_ROOT, rendered)


def load_template_config(template_path: str = _MULTISITE_TEMPLATE) -> dict[str, Any]:
    """Load the canonical multisite config template."""
    with open(template_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _sanitize_stem(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "site"


def _guess_forcing_glob(tower_id: str) -> str:
    pattern = os.path.join(REPO_ROOT, "data", f"*{tower_id}*")
    matches = sorted(glob.glob(pattern))
    if matches:
        return os.path.basename(matches[0])
    raise FileNotFoundError(
        f"No local forcing directory or file matched tower id {tower_id!r} under data/."
    )


def _select_existing_spec(selector: str) -> SiteSpec | None:
    from ecosystem_complexity.sites.spec import discover_site_specs

    specs = discover_site_specs()
    for spec in specs.values():
        if selector in {spec.config_stem, spec.israd_name, spec.label, spec.tower_id}:
            return spec
    return None


def build_site_config_dict(
    *,
    selector: str | None = None,
    tower_id: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    biome: str | None = None,
    observation_path: str | None = None,
    template_path: str = _MULTISITE_TEMPLATE,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a multisite-style config dict for one site.

    The builder is intentionally permissive because the exact CLI inputs are
    still evolving. It can clone an existing config, or derive one from the
    colocation catalog and whatever local forcing data is already present.
    """
    existing = _select_existing_spec(selector) if selector else None
    if existing is not None:
        with open(existing.config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        cfg["analysis"] = normalize_analysis_config(cfg.get("analysis"))
        cfg["output"] = normalize_output_config(cfg.get("output"))
        return existing.config_stem, _deep_merge(cfg, dict(overrides or {}))

    cfg = load_template_config(template_path)

    from ecosystem_complexity.fetch.colocation import locate_site

    matches = locate_site(
        flux_tower=selector or tower_id,
        lat=lat,
        lon=lon,
        biome=biome,
        max_distance_km=None,
    )
    if matches.empty:
        raise ValueError("No colocated ISRaD / flux-tower match found for the inputs.")

    row = matches.iloc[0]
    chosen_tower_id = tower_id or str(row["tower_id"])
    chosen_site = str(row["site_name"])
    chosen_label = f"{chosen_site} ({chosen_tower_id})"
    chosen_stem = _sanitize_stem(chosen_site)
    forcing_glob = _guess_forcing_glob(chosen_tower_id)

    ds = cfg.setdefault("datasource", {})
    ds["israd_name"] = chosen_site
    ds["forcing_glob"] = forcing_glob
    ds["forcing_kind"] = "daily"
    if observation_path is not None:
        ds["observation_path"] = observation_path
    else:
        eligibility_path = str(row.get("eligibility_path", ""))
        ds["observation_path"] = "fraction" if "fraction" in eligibility_path else "bulk_resp"

    site = cfg.setdefault("site", {})
    site["id"] = f"israd-multisite-{chosen_stem}"
    site["name"] = chosen_label
    site["lat"] = float(row["tower_lat"])
    site["lon"] = float(row["tower_lon"])
    site["tower_id"] = chosen_tower_id
    site["biome"] = biome or str(row.get("biome", "unclassified"))

    cfg["analysis"] = normalize_analysis_config(cfg.get("analysis"))
    cfg["output"] = normalize_output_config(cfg.get("output"))
    cfg = _deep_merge(cfg, dict(overrides or {}))
    return chosen_stem, cfg


def write_site_config(
    output_path: str,
    *,
    selector: str | None = None,
    tower_id: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    biome: str | None = None,
    observation_path: str | None = None,
    template_path: str = _MULTISITE_TEMPLATE,
    overrides: Mapping[str, Any] | None = None,
) -> str:
    """Build and write a site config YAML, returning the absolute path."""
    _, cfg = build_site_config_dict(
        selector=selector,
        tower_id=tower_id,
        lat=lat,
        lon=lon,
        biome=biome,
        observation_path=observation_path,
        template_path=template_path,
        overrides=overrides,
    )
    dest = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return dest
