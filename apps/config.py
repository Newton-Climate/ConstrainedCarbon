#!/usr/bin/env python3
"""``ecosys config`` — config-file utilities.

Subverbs
    build       synthesize a new per-site config from a template + site metadata
    incubation  generate per-site incubation-experiment configs from an ISRaD
                extraction
    locate      look up a site's tower + coordinates in the co-location table

Forwards argv to the underlying utility modules unchanged. These are
config-authoring conveniences that do not produce experiment artifacts,
so they do not write to the ``outputs/`` tree.
"""
from __future__ import annotations

import logging
import sys

_SUBVERBS = ("build", "incubation", "locate")


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


def _cmd_build(argv):
    from ecosystem_complexity.site_config.build import main as m
    return _run(m, argv)


def _cmd_incubation(argv):
    from ecosystem_complexity.site_config.incubation_manifest import main as m
    return _run(m, argv)


def _cmd_locate(argv):
    from ecosystem_complexity.fetch.locate_cli import main as m
    return _run(m, argv)


_HANDLERS = {
    "build":      _cmd_build,
    "incubation": _cmd_incubation,
    "locate":     _cmd_locate,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(f"usage: ecosys config {{{'|'.join(_SUBVERBS)}}} [args...]",
              file=sys.stderr)
        return 0 if argv else 2
    sub, rest = argv[0], argv[1:]
    handler = _HANDLERS.get(sub)
    if handler is None:
        print(f"unknown config subverb: {sub!r}", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
