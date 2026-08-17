#!/usr/bin/env python3
"""``ecosys report`` — contract-aware result synthesis workflows."""
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

_SUBVERBS = ("merge", "cross-ecosystem")


def _run(main_fn, argv: list[str]) -> int:
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


def _common_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--outdir", default=None, help="root for contract outputs (default ./outputs)")
    parser.add_argument("--name", default=None, help="run name below the output root")
    return parser.parse_known_args(argv)


def _record_files(run) -> None:
    for path in run.root.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            run.record_output(str(path.relative_to(run.root)))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: ecosys report <merge|cross-ecosystem> [--outdir DIR] [--name NAME] [args...]")
        return 0 if argv else 2
    subverb, supplied = argv[0], argv[1:]
    if subverb not in _SUBVERBS:
        raise SystemExit(f"unknown report subcommand: {subverb!r}")
    if "-h" in supplied or "--help" in supplied:
        if subverb == "merge":
            from ecosystem_complexity.network.merge_results import main as handler
        else:
            from ecosystem_complexity.outputs.cross_ecosystem_summary import main as handler
        return _run(handler, [*supplied, "--help"])
    common, forwarded = _common_args(supplied)
    name = common.name or ("merged_results" if subverb == "merge" else "cross_ecosystem")
    from ecosystem_complexity.outputs import attach_file_logger, open_run_dir
    run = open_run_dir(verb="report", subverb=subverb, name=name,
                       outdir=(Path(common.outdir) / name / "report" / subverb) if common.outdir else None,
                       inputs={"argv": supplied})
    if subverb == "merge":
        from ecosystem_complexity.network.merge_results import main as handler
        module_argv = [*forwarded, "--outdir", str(run.root)]
    else:
        if _value(forwarded, "--output-dir") or _value(forwarded, "--markdown-out"):
            raise SystemExit("--output-dir and --markdown-out are controlled by ecosys report.")
        from ecosystem_complexity.outputs.cross_ecosystem_summary import main as handler
        module_argv = [*forwarded, "--output-dir", str(run.root), "--markdown-out", str(run.root / "report.md")]
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
