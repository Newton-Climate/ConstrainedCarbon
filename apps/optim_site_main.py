#!/usr/bin/env python
"""Run the canonical OE soil-carbon inversion for any configured site.

Every site is defined entirely by a config YAML under ``configs/multisite/``
(see ``ecosystem_complexity.sites.multisite`` for the recipe those configs
encode), so adding a site needs a new YAML, not new code.

Sites may be named three ways, mixed freely:
  • a path to a config     configs/multisite/solling.yaml
  • a config stem          solling
  • an ISRaD site name     Solling

Examples
--------
    # one site, by config path
    python apps/optim_site_main.py configs/multisite/solling.yaml

    # two sites by stem, writing a summary table
    python apps/optim_site_main.py solling eml --out exports/run.csv

    # every configured site, 4 at a time
    python apps/optim_site_main.py --all --workers 4

    # every configured site, forcing the fraction observation path
    python apps/optim_site_main.py --all --observation-path fraction

    # what is available?
    python apps/optim_site_main.py --list

Exit status is non-zero if any requested site failed, so the command is usable
in a pipeline. Sites skipped for insufficient radiocarbon observations are
reported but are not failures.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Make the package importable from a plain checkout (no `pip install -e .`),
# matching how the notebooks/ scripts bootstrap themselves.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_APP_DIR)
_SRC = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ecosystem_complexity.sites import (  # noqa: E402
    SiteSpec,
    discover_site_specs,
    load_site_spec,
    run_site_canonical,
    run_sites,
    summary_row,
)
from ecosystem_complexity.site_analysis import export_site_run  # noqa: E402
from ecosystem_complexity.site_config import render_artifact_dir  # noqa: E402

OBSERVATION_PATHS = ("bulk_resp", "fraction", "combined")


def _resolve_specs(selectors: list[str]) -> list[SiteSpec]:
    """Resolve CLI selectors to SiteSpecs, accepting paths as well as names.

    ``select_specs`` in the library only understands names discovered under
    ``configs/multisite/``. Accepting a filesystem path too is what makes the
    CLI usable with a config kept outside that directory, so paths are loaded
    directly and only non-path selectors are looked up by name.
    """
    known = discover_site_specs()
    by_key: dict[str, SiteSpec] = {}
    for spec in known.values():
        by_key[spec.config_stem] = spec
        by_key[spec.israd_name] = spec
        by_key[spec.label] = spec

    specs: list[SiteSpec] = []
    for sel in selectors:
        if sel.endswith((".yaml", ".yml")) or os.path.sep in sel:
            path = os.path.abspath(sel)
            if not os.path.isfile(path):
                raise SystemExit(f"error: no such config file: {sel}")
            specs.append(load_site_spec(path))
            continue
        if sel not in by_key:
            raise SystemExit(
                f"error: unknown site {sel!r}\n"
                f"  known sites: {', '.join(sorted(known))}\n"
                f"  (or pass a path to a config YAML)"
            )
        specs.append(by_key[sel])
    return specs


def _print_available() -> None:
    specs = discover_site_specs()
    if not specs:
        raise SystemExit("error: no site configs found under configs/multisite/")
    width = max(len(s) for s in specs)
    print(f"{len(specs)} configured sites:\n")
    print(f"  {'STEM'.ljust(width)}  {'ISRAD NAME':<24} {'OBS PATH':<10} BIOME")
    for stem in sorted(specs):
        s = specs[stem]
        print(
            f"  {stem.ljust(width)}  {s.israd_name:<24} "
            f"{s.observation_path:<10} {s.biome}"
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="optim_site_main.py",
        description="Run the canonical OE inversion for one or more sites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sites may be given as a config path (configs/multisite/eml.yaml), "
            "a config stem (eml), or an ISRaD site name (EML)."
        ),
    )
    p.add_argument(
        "sites", nargs="*",
        help="site selectors: config path, config stem, or ISRaD name",
    )
    p.add_argument(
        "--all", action="store_true",
        help="run every config under configs/multisite/",
    )
    p.add_argument(
        "--list", action="store_true",
        help="list the configured sites and exit",
    )
    p.add_argument(
        "--observation-path", choices=OBSERVATION_PATHS, default=None,
        help=(
            "override the observation path each config declares "
            f"({'|'.join(OBSERVATION_PATHS)})"
        ),
    )
    p.add_argument(
        "--out", metavar="CSV",
        help="write the multi-site summary table to this CSV path",
    )
    p.add_argument(
        "-j", "--workers", type=int, default=1, metavar="N",
        help=(
            "run N sites concurrently in separate processes (default 1). "
            "Each worker builds its own model and JAX state, so memory scales "
            "with N; 4 is a reasonable default for a full --all sweep."
        ),
    )
    p.add_argument(
        "-q", "--quiet", action="store_true",
        help="suppress per-site progress logging",
    )
    p.add_argument(
        "--export-dir",
        help=(
            "artifact output directory for a single-site run; relative paths are "
            "resolved from the repo root and may include {config_stem}."
        ),
    )
    p.add_argument(
        "--no-export-artifacts",
        action="store_true",
        help="skip exporting matrices, diagnostics tables, and the site figure",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list:
        _print_available()
        return 0

    if not args.sites and not args.all:
        _build_parser().error(
            "give at least one site, or --all to run every config "
            "(--list shows what is available)"
        )
    if args.sites and args.all:
        _build_parser().error("--all cannot be combined with explicit site names")
    if args.workers < 1:
        _build_parser().error("--workers must be at least 1")

    # The drivers log their progress; route it to stdout so the CLI behaves the
    # way the old `python notebooks/sites/multisite_canonical.py` entry point did.
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    specs = (
        list(discover_site_specs().values()) if args.all
        else _resolve_specs(args.sites)
    )
    if not specs:
        raise SystemExit("error: no site configs found under configs/multisite/")

    workers = min(args.workers, len(specs))
    if workers > 1:
        print(f"Running {len(specs)} sites across {workers} worker processes…")
    # Reduce to summary rows in the worker: the raw result holds the compiled
    # model, which cannot be pickled back from a worker process. Doing it in
    # both modes keeps serial and parallel returning the same shape.
    single_site = len(specs) == 1 and workers == 1
    if single_site:
        raw = run_site_canonical(specs[0], observation_path=args.observation_path or specs[0].observation_path)
        if raw.get("skipped"):
            results = []
            failures = []
            summary_results: list[dict] = []
        else:
            results = [raw]
            failures = []
            summary_results = [summary_row(raw)]
    else:
        summary_results, failures = run_sites(
            specs,
            observation_path=args.observation_path,
            workers=workers,
            reduce=summary_row,
        )
        results = []

    n_skipped = len(specs) - len(summary_results) - len(failures)
    print(
        f"\n{len(summary_results)}/{len(specs)} sites inverted"
        + (f", {n_skipped} skipped (insufficient ¹⁴C obs)" if n_skipped else "")
        + (f", {len(failures)} failed" if failures else "")
    )
    for spec, exc in failures:
        print(f"  FAILED  {spec.label}: {exc}")

    if summary_results:
        import pandas as pd

        table = pd.DataFrame(summary_results)
        print()
        print(table.to_string(index=False))
        if args.out:
            out = os.path.abspath(args.out)
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            table.to_csv(out, index=False)
            print(f"\nSummary → {os.path.relpath(out, _REPO_ROOT)}")
        if single_site and results and not args.no_export_artifacts:
            raw = results[0]
            export_template = args.export_dir or str(
                raw["model"].config.output_raw.get("artifact_dir", "results/{config_stem}")
            )
            if os.path.isabs(export_template):
                export_dir = export_template
            else:
                export_dir = render_artifact_dir(
                    export_template,
                    config_stem=raw["spec"].config_stem,
                    site_id=raw["model"].config.site_id or raw["spec"].config_stem,
                )
            exports = export_site_run(raw, export_dir)
            print("\nArtifacts:")
            for label, path in exports.items():
                print(f"  {label:<18} {os.path.relpath(path, _REPO_ROOT)}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
