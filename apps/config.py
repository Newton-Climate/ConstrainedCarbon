#!/usr/bin/env python3
"""``ecosys config`` — config-file utilities.

Subverbs
    build       synthesize a new per-site config from a template + site metadata
                (build_site_config.py)
    incubation  generate per-site incubation-experiment configs from an ISRaD
                extraction (generate_incubation_configs.py)
    locate      look up a site's tower + coordinates in the co-location table
                (locate_site.py)

Forwards argv to the underlying utility modules unchanged. These are
config-authoring conveniences that do not produce experiment artifacts,
so they do not write to the ``outputs/`` tree.
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


_SUBVERBS = ("build", "incubation", "locate")


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
    "build":      lambda a: _forward("build_site_config", a),
    "incubation": lambda a: _forward("generate_incubation_configs", a),
    "locate":     lambda a: _forward("locate_site", a),
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
