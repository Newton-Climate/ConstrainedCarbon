#!/usr/bin/env python
"""Locate colocated ISRaD and flux-tower sites and export a CSV table."""

from __future__ import annotations

import argparse
import os

from ecosystem_complexity.fetch import locate_site


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux-tower", help="Tower id, tower name, or site name selector.")
    parser.add_argument("--lat", type=float, help="Query latitude.")
    parser.add_argument("--lon", type=float, help="Query longitude.")
    parser.add_argument("--biome", help="Biome substring filter.")
    parser.add_argument("--max-distance-km", type=float, default=50.0)
    parser.add_argument("--out", required=True, help="Output CSV path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    table = locate_site(
        flux_tower=args.flux_tower,
        lat=args.lat,
        lon=args.lon,
        biome=args.biome,
        max_distance_km=args.max_distance_km,
    )
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {len(table)} rows to {out_path}")
    return 0
