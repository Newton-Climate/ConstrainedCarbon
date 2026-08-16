#!/usr/bin/env python
"""Analyze a site fit from exported artifacts or by re-running the site inversion."""

from __future__ import annotations

import argparse
import json
import os

from ecosystem_complexity.site_analysis import export_site_run, load_exported_analysis
from ecosystem_complexity.site_config import render_artifact_dir
from ecosystem_complexity.sites import discover_site_specs, load_site_spec, run_site_canonical


def _resolve_site(selector: str):
    if selector.endswith((".yaml", ".yml")) or os.path.sep in selector:
        return load_site_spec(os.path.abspath(selector))
    specs = discover_site_specs()
    for spec in specs.values():
        if selector in {spec.config_stem, spec.israd_name, spec.label, spec.tower_id}:
            return spec
    raise KeyError(f"Unknown site selector {selector!r}.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", nargs="?", help="Site selector or config path.")
    parser.add_argument("--export-dir", help="Artifact directory to write to or read from.")
    parser.add_argument("--from-artifacts", action="store_true", help="Only read existing exported artifacts.")
    parser.add_argument("--observation-path", choices=("bulk_resp", "fraction", "combined"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.from_artifacts:
        if not args.export_dir:
            raise SystemExit("--from-artifacts requires --export-dir")
        loaded = load_exported_analysis(args.export_dir)
        print(json.dumps(loaded["metrics"], indent=2))
        return 0

    if not args.site:
        raise SystemExit("Provide a site selector or use --from-artifacts.")

    spec = _resolve_site(args.site)
    export_dir = args.export_dir
    if export_dir is None:
        export_dir = render_artifact_dir(
            "results/{config_stem}",
            config_stem=spec.config_stem,
            site_id=spec.config_stem,
        )
    run = run_site_canonical(spec, observation_path=args.observation_path or spec.observation_path)
    if run.get("skipped"):
        raise SystemExit(f"{spec.label}: insufficient observations for analysis.")
    outputs = export_site_run(run, export_dir)
    print(json.dumps(outputs, indent=2))
    return 0
