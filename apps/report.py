#!/usr/bin/env python3
"""``ecosys report`` — cross-run report generators.

Subverbs
    merge            concat cross-biome result tables into one CSV
    cross-ecosystem  build the cross-ecosystem summary markdown/CSV bundle
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
_NB = _REPO_ROOT / "notebooks"
if str(_NB) not in sys.path:
    sys.path.insert(0, str(_NB))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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


def _cmd_merge(argv):
    from ecosystem_complexity.network.merge_results import main as m
    return _run(m, argv)


def _cmd_cross_ecosystem(argv):
    from ecosystem_complexity.outputs.cross_ecosystem_summary import main as m
    return _run(m, argv)


_HANDLERS = {
    "merge":           _cmd_merge,
    "cross-ecosystem": _cmd_cross_ecosystem,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(f"usage: ecosys report {{{'|'.join(_SUBVERBS)}}} [args...]",
              file=sys.stderr)
        return 0 if argv else 2
    sub, rest = argv[0], argv[1:]
    handler = _HANDLERS.get(sub)
    if handler is None:
        print(f"unknown report subverb: {sub!r}", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
