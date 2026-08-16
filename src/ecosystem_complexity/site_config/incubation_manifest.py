#!/usr/bin/env python3
"""Generate incubation-driven expansion configs from the exported manifest."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT / "notebooks" / "exports" / "incubation_config_manifest_20260719.csv"
)
DEFAULT_TEMPLATE = REPO_ROOT / "configs" / "expansion" / "nahuelbuta.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "configs" / "expansion"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"CSV manifest (default: {DEFAULT_MANIFEST.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help=f"YAML template (default: {DEFAULT_TEMPLATE.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR.relative_to(REPO_ROOT)})",
    )
    return parser.parse_args()


def _maybe_float(row: pd.Series, key: str) -> float | None:
    val = row.get(key)
    return None if pd.isna(val) else float(val)


def _maybe_str(row: pd.Series, key: str) -> str:
    val = row.get(key)
    return "" if pd.isna(val) else str(val).strip()


def main() -> int:
    args = _parse_args()
    manifest = pd.read_csv(args.manifest)
    template = yaml.safe_load(args.template.read_text()) or {}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for row in manifest.to_dict(orient="records"):
        cfg = copy.deepcopy(template)
        config_stem = str(row["config_stem"])
        forcing_kind = str(row["forcing_kind"])

        site = {
            "id": f"israd-expansion-{config_stem}",
            "name": str(row["label"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "biome": str(row["biome"]),
        }
        tower_id = _maybe_str(pd.Series(row), "tower_id")
        if tower_id:
            site["tower_id"] = tower_id
        for key in ("mat_c", "map_mm", "elevation_m"):
            val = _maybe_float(pd.Series(row), key)
            if val is not None:
                site[key] = val

        datasource = {
            "israd_name": str(row["site_name"]),
            "forcing_glob": str(row["forcing_glob"]),
            "forcing_kind": forcing_kind,
            "observation_path": str(row["observation_path"]),
        }
        er_path = _maybe_str(pd.Series(row), "er_observation_glob")
        if forcing_kind == "fluxcom" and er_path:
            datasource["er_observation_glob"] = er_path

        cfg["site"] = site
        cfg["datasource"] = datasource

        out_path = args.out_dir / f"{config_stem}.yaml"
        header = (
            "# Generated from notebooks/exports/incubation_config_manifest_20260719.csv\n"
            "# on 2026-07-19. Update the manifest and rerun this script to refresh.\n"
        )
        out_path.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(out_path.relative_to(REPO_ROOT))

    return 0
