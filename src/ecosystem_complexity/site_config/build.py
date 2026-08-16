#!/usr/bin/env python
"""Build a site config YAML for the optimization and analysis apps."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

from ecosystem_complexity.site_config import build_site_config_dict, write_site_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", help="Existing site selector, tower id, or site name.")
    parser.add_argument("--tower-id", help="Flux tower id.")
    parser.add_argument("--lat", type=float, help="Tower latitude.")
    parser.add_argument("--lon", type=float, help="Tower longitude.")
    parser.add_argument("--biome", help="Biome string override.")
    parser.add_argument("--observation-path", choices=("bulk_resp", "fraction", "combined"))
    parser.add_argument("--template", default=os.path.join(_REPO_ROOT, "configs", "israd_multisite_3pool_config.yaml"))
    parser.add_argument("--out", help="Output YAML path. Defaults to configs/multisite/<stem>.yaml.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    stem, _ = build_site_config_dict(
        selector=args.selector,
        tower_id=args.tower_id,
        lat=args.lat,
        lon=args.lon,
        biome=args.biome,
        observation_path=args.observation_path,
        template_path=args.template,
    )
    out_path = args.out or os.path.join(_REPO_ROOT, "configs", "multisite", f"{stem}.yaml")
    dest = write_site_config(
        out_path,
        selector=args.selector,
        tower_id=args.tower_id,
        lat=args.lat,
        lon=args.lon,
        biome=args.biome,
        observation_path=args.observation_path,
        template_path=args.template,
    )
    print(dest)
    return 0
