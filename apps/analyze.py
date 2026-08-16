#!/usr/bin/env python3
"""``ecosys analyze`` — post-hoc analysis over exported inversion artifacts.

Subverbs
    model            per-site model export/reload
    network          network-wide inversion + OE ladder aggregation
    transit          intrinsic / realized / gradient transit-time diagnostics
    transit-vulnerability
                     ridge / LOBO regression testing whether transit metrics
                     improve cross-biome vulnerability prediction
    cross-ecosystem  cross-ecosystem summary markdown/CSV

These subverbs are post-hoc consumers of the tables written by
``ecosys optimize``, ``ecosys warming``, and ``ecosys information``.
They currently forward argv to the underlying analysis modules
unchanged — a follow-up will rewrap them to write manifests under
``outputs/{name}/analyze/{subverb}/``.
"""
from __future__ import annotations

import argparse
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


_SUBVERBS = ("model", "network", "transit", "transit-vulnerability", "cross-ecosystem")


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


def _cmd_transit(argv: list[str]) -> int:
    """intrinsic|realized|gradient — dispatch on --mode."""
    parser = argparse.ArgumentParser(add_help=False, prog="ecosys analyze transit")
    parser.add_argument("--mode", required=True,
                        choices=("intrinsic", "realized", "gradient"))
    parser.add_argument("-h", "--help", action="store_true")
    known, rest = parser.parse_known_args(argv)
    if known.help:
        rest = [*rest, "--help"]
    if known.mode == "intrinsic":
        from ecosystem_complexity.transit_time.network import main as m
    elif known.mode == "realized":
        from ecosystem_complexity.transit_time.realized_network import main as m
    else:
        from ecosystem_complexity.transit_time.realized_gradient import main as m
    return _run(m, rest)


def _cmd_model(argv):
    from ecosystem_complexity.site_analysis.analyze_run import main as m
    return _run(m, argv)


def _cmd_network(argv):
    from ecosystem_complexity.network.inversions import main as m
    return _run(m, argv)


def _cmd_transit_vulnerability(argv):
    from ecosystem_complexity.network.transit_vulnerability import main as m
    return _run(m, argv)


def _cmd_cross_ecosystem(argv):
    from ecosystem_complexity.outputs.cross_ecosystem_summary import main as m
    return _run(m, argv)


_HANDLERS = {
    "model":                 _cmd_model,
    "network":               _cmd_network,
    "transit":               _cmd_transit,
    "transit-vulnerability": _cmd_transit_vulnerability,
    "cross-ecosystem":       _cmd_cross_ecosystem,
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
