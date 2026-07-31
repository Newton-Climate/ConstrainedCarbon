#!/usr/bin/env python3
"""Stage one canonical input bundle and build manuscript Figures 8–10.

Raw inversion outputs remain in ``notebooks/exports``.  This script combines
them into ``<output-dir>/inputs`` and then passes only those staged files to
the three figure builders.  Re-running the command therefore regenerates both
the input bundle and the figures from the same declared sources.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_NB = _ROOT / "notebooks"
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from notebooks.paper_figs.fig_08 import make_figure_08
from notebooks.paper_figs.fig_09 import _load_new_site_tables, make_figure_09
from notebooks.paper_figs.fig_10 import make_figure_10
from notebooks.paper_figs.utils import close_or_show


def _parse_args() -> argparse.Namespace:
    exports = _NB / "exports"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_NB / "paper_figs" / "outputs" / "current_results",
    )
    parser.add_argument(
        "--pathway-information",
        nargs="+",
        type=Path,
        default=[
            exports / "cross_ecosystem_pathway_information_20260730.csv",
        ],
    )
    parser.add_argument(
        "--network-summary",
        type=Path,
        default=exports / "network_inversion_fluxcom_er_20260719" / "site_summary.csv",
    )
    parser.add_argument(
        "--warming-summary",
        type=Path,
        default=exports / "warming_vulnerability_fluxcom_er_20260719" / "site_warming_summary.csv",
    )
    parser.add_argument(
        "--new-sites",
        nargs="+",
        type=Path,
        default=[
            exports / "new_sites_incubation_20260719.csv",
            exports / "incubation_new_sites_runnable_20260719.csv",
            exports / "ecosystem_diversity_candidates_20260730.csv",
            exports / "maui_fluxcom_inversion_20260730.csv",
        ],
    )
    return parser.parse_args()


def _require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing figure-input source(s): " + ", ".join(missing))


def stage_inputs(args: argparse.Namespace) -> dict[str, Path]:
    _require([*args.pathway_information, args.network_summary, args.warming_summary, *args.new_sites])
    input_dir = args.output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    pathway = pd.concat([pd.read_csv(path) for path in args.pathway_information], ignore_index=True)
    pathway = pathway.drop_duplicates(["site", "pathway"], keep="last")
    new_sites = _load_new_site_tables([str(path) for path in args.new_sites])

    paths = {
        "pathway_information": input_dir / "pathway_information.csv",
        "network_summary": input_dir / "network_summary.csv",
        "warming_summary": input_dir / "warming_summary.csv",
        "new_sites": input_dir / "new_sites.csv",
    }
    pathway.to_csv(paths["pathway_information"], index=False)
    pd.read_csv(args.network_summary).to_csv(paths["network_summary"], index=False)
    pd.read_csv(args.warming_summary).to_csv(paths["warming_summary"], index=False)
    new_sites.to_csv(paths["new_sites"], index=False)
    (input_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pathway_information": [str(path) for path in args.pathway_information],
                "network_summary": str(args.network_summary),
                "warming_summary": str(args.warming_summary),
                "new_sites": [str(path) for path in args.new_sites],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    args = _parse_args()
    paths = stage_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, _ = make_figure_08(str(paths["pathway_information"]), output_dir=str(args.output_dir))
    close_or_show(fig, show=False)
    fig, _ = make_figure_09(
        str(paths["network_summary"]),
        str(paths["warming_summary"]),
        [str(paths["new_sites"])],
        output_dir=str(args.output_dir),
    )
    close_or_show(fig, show=False)
    fig, _ = make_figure_10(
        str(paths["network_summary"]),
        str(paths["warming_summary"]),
        [str(paths["new_sites"])],
        output_dir=str(args.output_dir),
    )
    close_or_show(fig, show=False)
    print(f"Staged inputs and figures in {args.output_dir}")


if __name__ == "__main__":
    main()
