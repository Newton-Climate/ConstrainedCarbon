#!/usr/bin/env python3
"""``ecosys analyze`` — post-hoc analysis over exported inversion artifacts.

Subverbs
    model            per-site model export/reload (analyze_model.py)
    network          network-wide inversion + OE ladder aggregation
                       (analyze_network_inversions.py)
    transit          intrinsic / realized / gradient transit-time diagnostics
    transit-vulnerability
                     ridge / LOBO regression testing whether transit metrics
                     improve cross-biome vulnerability prediction
    cross-ecosystem  cross-ecosystem summary markdown/CSV
                       (build_cross_ecosystem_summary.py)

These subverbs are post-hoc consumers of the tables written by
``ecosys optimize``, ``ecosys warming``, and ``ecosys information``.
They currently forward argv to the underlying analysis modules
unchanged — a follow-up will rewrap them to write manifests under
``outputs/{name}/analyze/{subverb}/``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


_SUBVERBS = ("model", "network", "transit", "transit-vulnerability", "cross-ecosystem")


def _forward(module_name: str, argv: list[str]) -> int:
    """Import ``apps/<module_name>`` and invoke its ``main(argv)``."""
    import importlib
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


def _cmd_transit(argv: list[str]) -> int:
    """intrinsic|realized|gradient — dispatch on --mode."""
    parser = argparse.ArgumentParser(add_help=False, prog="ecosys analyze transit")
    parser.add_argument("--mode", required=True,
                        choices=("intrinsic", "realized", "gradient"))
    parser.add_argument("-h", "--help", action="store_true")
    known, rest = parser.parse_known_args(argv)
    module = {
        "intrinsic": "compute_transit_times",
        "realized": "realized_transit_all_sites",
        "gradient": "realized_transit_gradient",
    }[known.mode]
    if known.help:
        rest = [*rest, "--help"]
    return _forward(module, rest)


_HANDLERS = {
    "model":                 lambda a: _forward("analyze_model", a),
    "network":               lambda a: _forward("analyze_network_inversions", a),
    "transit":               _cmd_transit,
    "transit-vulnerability": lambda a: _forward("analyze_transit_vulnerability", a),
    "cross-ecosystem":       lambda a: _forward("build_cross_ecosystem_summary", a),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(f"usage: ecosys analyze {{{'|'.join(_SUBVERBS)}}} [args...]",
              file=sys.stderr)
        return 0 if argv else 2
    sub, rest = argv[0], argv[1:]
    handler = _HANDLERS.get(sub)
    if handler is None:
        print(f"unknown analyze subverb: {sub!r}", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
