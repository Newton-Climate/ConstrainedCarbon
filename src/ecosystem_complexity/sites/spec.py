"""Per-site configuration: the SiteSpec record and config discovery.

Each site is defined entirely by a config YAML under ``configs/multisite/``.
Adding a site is a new YAML, not new code — see
:mod:`ecosystem_complexity.data.fraction_mapping` for the one piece that used
to require a code change and no longer does.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import yaml

from ecosystem_complexity.sites.paths import CONFIG_DIR


@dataclass(frozen=True)
class SiteSpec:
    config_path: str         # configs/multisite/<stem>.yaml this spec came from
    config_stem: str         # filename stem (the CLI selector for the site)
    israd_name: str          # ISRaD site_name (filters the layer/flux tables)
    forcing_glob: str        # data/<glob> for the tower directory
    lat: float
    lon: float
    label: str
    tower_id: str = ""
    biome: str = "unclassified"
    forcing_kind: str = "daily"
    observation_path: str = "bulk_resp"   # bulk_resp | fraction | combined
    # Per-site overrides of the ISRaD frc_property → pool mapping. Empty for
    # every current site: the shared vocabulary in sites.fraction_mapping
    # covers them, and this exists so an unusual protocol stays a config
    # change rather than the per-site branch this replaced.
    fraction_rules: dict[str, str | None] = field(default_factory=dict)



def load_site_spec(config_path: str) -> SiteSpec:
    """Build a SiteSpec from a per-site config YAML's site + datasource blocks."""
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    site = cfg.get("site") or {}
    ds = cfg.get("datasource") or {}
    missing = [k for k in ("israd_name", "forcing_glob") if not ds.get(k)]
    if missing:
        raise ValueError(
            f"{os.path.basename(config_path)}: datasource missing {missing}"
        )
    stem = os.path.splitext(os.path.basename(config_path))[0]
    return SiteSpec(
        config_path=config_path,
        config_stem=stem,
        israd_name=str(ds["israd_name"]),
        forcing_glob=str(ds["forcing_glob"]),
        lat=float(site.get("lat", 0.0)),
        lon=float(site.get("lon", 0.0)),
        label=str(site.get("name", stem)),
        tower_id=str(site.get("tower_id", "")),
        biome=str(site.get("biome", "unclassified")),
        forcing_kind=str(ds.get("forcing_kind", "daily")),
        observation_path=str(ds.get("observation_path", "bulk_resp")),
        fraction_rules=dict(ds.get("fraction_rules") or {}),
    )


def discover_site_specs(config_dir: str = CONFIG_DIR) -> dict[str, SiteSpec]:
    """Load every configs/multisite/*.yaml into a {config_stem: SiteSpec} map."""
    specs: dict[str, SiteSpec] = {}
    for path in sorted(glob.glob(os.path.join(config_dir, "*.yaml"))):
        spec = load_site_spec(path)
        specs[spec.config_stem] = spec
    return specs


def select_specs(names: list[str] | None) -> list[SiteSpec]:
    """Resolve CLI site selectors (config stem, ISRaD name, or label) to specs."""
    specs = discover_site_specs()
    if not specs:
        raise FileNotFoundError(f"No site configs found under {CONFIG_DIR}")
    if not names:
        return list(specs.values())
    by_key: dict[str, SiteSpec] = {}
    for spec in specs.values():
        by_key[spec.config_stem] = spec
        by_key[spec.israd_name] = spec
        by_key[spec.label] = spec
    out: list[SiteSpec] = []
    for name in names:
        if name not in by_key:
            raise KeyError(
                f"Unknown site {name!r}. Available: {sorted(specs)}"
            )
        out.append(by_key[name])
    return out
