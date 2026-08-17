#!/usr/bin/env python3
"""``ecosys analyze`` — contract-aware post-hoc analysis workflows."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
for _path in (_REPO_ROOT / "notebooks", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_SUBVERBS = ("model", "network", "transit", "transit-vulnerability", "cross-ecosystem")
_OUTPUT_FLAGS = {
    "model": ("--export-dir",), "network": ("--outdir",),
    "transit": ("--out", "--figure", "--draws-out"),
    "transit-vulnerability": ("--outdir",),
    "cross-ecosystem": ("--output-dir", "--markdown-out"),
}


def _run(main_fn, argv: list[str]) -> int:
    """Call both modern ``main(argv)`` and older ``main()`` entry points."""
    try:
        rc = main_fn(argv)
    except TypeError:
        saved = sys.argv
        sys.argv = [main_fn.__module__, *argv]
        try:
            rc = main_fn()
        finally:
            sys.argv = saved
    return int(rc or 0)


def _value(argv: list[str], flag: str) -> str | None:
    for i, item in enumerate(argv):
        if item == flag and i + 1 < len(argv):
            return argv[i + 1]
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def _first_positional(argv: list[str]) -> str | None:
    value_flags = {"--site-set", "--observation-path", "--mode", "--workers", "--name"}
    skip = False
    for item in argv:
        if skip:
            skip = False
            continue
        if item.startswith("-"):
            skip = item in value_flags
            continue
        return item
    return None


def _common_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--outdir", default=None, help="root for contract outputs (default ./outputs)")
    parser.add_argument("--name", default=None, help="run name below the output root")
    return parser.parse_known_args(argv)


def _name_for(subverb: str, argv: list[str], requested: str | None) -> str:
    if requested:
        return requested
    if subverb == "model":
        site = _first_positional(argv)
        return Path(site).stem if site else "site_analysis"
    site_set = _value(argv, "--site-set")
    if site_set:
        return Path(site_set).stem
    return {"network": "network", "transit": "transit", "transit-vulnerability": "transit_vulnerability", "cross-ecosystem": "cross_ecosystem"}[subverb]


def _record_files(run) -> None:
    for path in run.root.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            run.record_output(str(path.relative_to(run.root)))


def _without_option(argv: list[str], flag: str) -> list[str]:
    """Remove one option (in either ``--flag value`` or ``--flag=value`` form)."""
    kept: list[str] = []
    skip = False
    for item in argv:
        if skip:
            skip = False
            continue
        if item == flag:
            skip = True
        elif item.startswith(f"{flag}="):
            continue
        else:
            kept.append(item)
    return kept


def _handler(subverb: str, argv: list[str], destination: Path):
    if subverb == "model":
        if "--from-artifacts" in argv:
            raise SystemExit("--from-artifacts reads an existing run; it cannot create a new contract run.")
        from ecosystem_complexity.site_analysis.analyze_run import main
        return main, [*argv, "--export-dir", str(destination)]
    if subverb == "network":
        from ecosystem_complexity.network.inversions import main
        return main, [*argv, "--outdir", str(destination)]
    if subverb == "transit":
        mode = _value(argv, "--mode")
        if mode not in {"intrinsic", "realized", "gradient"}:
            raise SystemExit("ecosys analyze transit requires --mode intrinsic, realized, or gradient")
        forwarded = _without_option(argv, "--mode")
        if mode == "intrinsic":
            from ecosystem_complexity.transit_time.network import main
            return main, [*forwarded, "--out", str(destination / "transit_times.csv"), "--figure", str(destination / "transit_times.png")]
        if mode == "realized":
            from ecosystem_complexity.transit_time.realized_network import main
            return main, [*forwarded, "--out", str(destination / "realized_transit_times.csv"), "--figure", str(destination / "realized_transit_times.png")]
        from ecosystem_complexity.transit_time.realized_gradient import main
        return main, [*forwarded, "--out", str(destination / "gradient_transit_times.csv"), "--draws-out", str(destination / "gradient_transit_draws.csv"), "--figure", str(destination / "gradient_transit_times.png")]
    if subverb == "transit-vulnerability":
        from ecosystem_complexity.network.transit_vulnerability import main
        return main, [*argv, "--outdir", str(destination)]
    from ecosystem_complexity.outputs.cross_ecosystem_summary import main
    return main, [*argv, "--output-dir", str(destination), "--markdown-out", str(destination / "report.md")]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(f"usage: ecosys analyze <subcommand> [--outdir DIR] [--name NAME] [args...]\nsubcommands: {', '.join(_SUBVERBS)}")
        return 0 if argv else 2
    subverb, supplied = argv[0], argv[1:]
    if subverb not in _SUBVERBS:
        raise SystemExit(f"unknown analyze subcommand: {subverb!r}")
    if "-h" in supplied or "--help" in supplied:
        if subverb == "transit" and _value(supplied, "--mode") is None:
            print("usage: ecosys analyze transit --mode {intrinsic,realized,gradient} [--outdir DIR] [--name NAME] [args...]")
            return 0
        handler, forwarded = _handler(subverb, supplied, Path("."))
        return _run(handler, [*forwarded, "--help"])
    common, forwarded = _common_args(supplied)
    present = [flag for flag in _OUTPUT_FLAGS[subverb] if _value(forwarded, flag) is not None]
    if present:
        raise SystemExit(f"{', '.join(present)} is controlled by ecosys analyze; use --outdir and optionally --name.")
    name = _name_for(subverb, forwarded, common.name)
    from ecosystem_complexity.outputs import attach_file_logger, open_run_dir
    run = open_run_dir(verb="analyze", subverb=subverb, name=name,
                       outdir=(Path(common.outdir) / name / "analyze" / subverb) if common.outdir else None,
                       inputs={"argv": supplied})
    handler, module_argv = _handler(subverb, forwarded, run.root)
    logger = attach_file_logger(run)
    try:
        result = _run(handler, module_argv)
        _record_files(run)
        print(json.dumps(run.finalize(), indent=2))
        return result
    finally:
        logging.getLogger().removeHandler(logger)
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
