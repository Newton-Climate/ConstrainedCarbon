"""Unified `ecosys` CLI.

Every subcommand takes a YAML config (or a site-set YAML) plus a small set of
CLI overrides. Under the hood each subcommand delegates to the existing
``apps/*.py`` module's ``main`` function; the CLI is the single entry point
and the ``apps/`` scripts are retained only as import targets until they are
folded in and deleted.

Usage:
    ecosys <command> [args...]

Commands:
    run        Canonical OE inversion for one site / site set.
    network    Network-wide inversion + OE ladder / Shapley aggregation.
    shapley    Per-parameter Shapley DFS across the network.
    warming    Standardized warming-response projections.
    transit    Transit-time diagnostics (intrinsic | realized | gradient).
    mcmc       MCMC sampler.
    analyze    Post-hoc diagnostics from exported artifacts.
    fetch      Download forcing data (flux | fluxcom).
    config     Config utilities (build | incubation | locate).
    report     Result merging / cross-ecosystem summary.
    model      Validate inputs or run the forward carbon model.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_APPS = _REPO_ROOT / "apps"
if str(_APPS) not in sys.path:
    sys.path.insert(0, str(_APPS))


def _delegate(module_name: str, argv: list[str]) -> int:
    """Import ``apps/<module_name>.py`` and invoke its ``main``.

    Handles both ``main(argv)`` and ``main()`` signatures by patching
    ``sys.argv`` for the latter.
    """
    module = importlib.import_module(module_name)
    main = module.main
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


# ── subcommand handlers ──────────────────────────────────────────────────────

def _cmd_optimize(argv: list[str]) -> int:
    return _delegate("optimize", argv)


def _cmd_network(argv: list[str]) -> int:
    return _delegate("analyze_network_inversions", argv)


def _cmd_shapley(argv: list[str]) -> int:
    # Legacy verb: forward to `information shapley`. The dedicated
    # `--sigma-rule` flag is now handled inside apps/information.py, so
    # no env-var monkey-patching happens here.
    return _delegate("information", ["shapley", *argv])


def _cmd_information(argv: list[str]) -> int:
    return _delegate("information", argv)


def _cmd_warming(argv: list[str]) -> int:
    return _delegate("warming", argv)


def _cmd_transit(argv: list[str]) -> int:
    # Legacy alias for `analyze transit --mode ...`. Kept because the
    # top-level `transit` verb is what earlier users typed.
    return _delegate("analyze", ["transit", *argv])


def _cmd_mcmc(argv: list[str]) -> int:
    return _delegate("mcmc", argv)


def _cmd_analyze(argv: list[str]) -> int:
    return _delegate("analyze", argv)


def _cmd_fetch(argv: list[str]) -> int:
    return _delegate("fetch", argv)


def _cmd_config(argv: list[str]) -> int:
    return _delegate("config", argv)


def _cmd_report(argv: list[str]) -> int:
    return _delegate("report", argv)


def _cmd_model(argv: list[str]) -> int:
    return _delegate("model", argv)


_COMMANDS = {
    "optimize": _cmd_optimize,
    "run": _cmd_optimize,  # legacy alias
    "information": _cmd_information,
    "shapley": _cmd_shapley,  # legacy alias → `information shapley`
    "network": _cmd_network,
    "warming": _cmd_warming,
    "transit": _cmd_transit,
    "mcmc": _cmd_mcmc,
    "analyze": _cmd_analyze,
    "fetch": _cmd_fetch,
    "config": _cmd_config,
    "report": _cmd_report,
    "model": _cmd_model,
}


def _print_usage() -> None:
    print(__doc__, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_usage()
        return 0 if args else 2
    cmd, rest = args[0], args[1:]
    handler = _COMMANDS.get(cmd)
    if handler is None:
        print(f"unknown command: {cmd!r}", file=sys.stderr)
        _print_usage()
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
