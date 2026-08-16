#!/usr/bin/env python3
"""``ecosys report`` — cross-run report generators.

Subverbs
    merge            concat cross-biome result tables into one CSV
                     (merge_cross_biome_results.py)
    cross-ecosystem  build the cross-ecosystem summary markdown/CSV bundle
                     (build_cross_ecosystem_summary.py)

Forwards argv to the underlying report modules unchanged.
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


_SUBVERBS = ("merge", "cross-ecosystem")


def _forward(module_name: str, argv: list[str]) -> int:
    mod = importlib.import_module(module_name)
    main = mod.main
    try:
        rc = main(argv)
    except TypeError:
        saved = sys.argv
        sys.argv = [module_name, *argv]
        try:
            rc = main()
        finally:
            sys.argv = saved
    return int(rc or 0)


_HANDLERS = {
    "merge":           lambda a: _forward("merge_cross_biome_results", a),
    "cross-ecosystem": lambda a: _forward("build_cross_ecosystem_summary", a),
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
