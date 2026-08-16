#!/usr/bin/env python3
"""Download site-level FluxCom GPP series for expansion configs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ecosystem_complexity.data.fetch_fluxcom import (
    FLUXCOM_X_2021,
    FLUXCOM_X_2021_NEE,
    fetch_fluxcom_x_for_configs,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the official ICOS FLUXCOM-X 2021 daily global GPP and NEE "
            "grids, then write nearest-grid site CSVs for FluxCom-backed configs. "
            "ER is derived as GPP + NEE for each site-day."
        )
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help=(
            "site config paths; defaults to every yaml under configs/expansion/"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace any existing site CSVs under data/shared/fluxcom/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_paths = (
        [Path(p) for p in args.configs]
        if args.configs
        else sorted((_REPO_ROOT / "configs" / "expansion").glob("*.yaml"))
    )
    rows = fetch_fluxcom_x_for_configs(config_paths, overwrite=args.overwrite)
    if not rows:
        print("No configs with forcing_kind: fluxcom")
        return 0

    print("GPP source:", FLUXCOM_X_2021["landing_page"])
    print("NEE source:", FLUXCOM_X_2021_NEE["landing_page"])
    for row in rows:
        status = str(row["status"])
        forcing_path = str(row["forcing_output_path"])
        er_path = str(row["er_output_path"])
        if status == "skipped":
            print(f"- {row['site_id']}: kept existing {forcing_path} and {er_path}")
            continue
        mean_gpp = float(row["mean_gpp_gCm2day"])
        mean_er = float(row["mean_ER"])
        n_days = int(row["n_days"])
        grid_lat = float(row["grid_lat"])
        grid_lon = float(row["grid_lon"])
        print(
            f"- {row['site_id']}: wrote {forcing_path} and {er_path}  "
            f"({n_days} days, mean GPP {mean_gpp:.2f}, mean ER {mean_er:.2f}, "
            f"grid {grid_lat:.2f}, {grid_lon:.2f})"
        )
    return 0
