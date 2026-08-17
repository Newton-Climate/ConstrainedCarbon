#!/usr/bin/env python3
"""``ecosys mcmc`` — MCMC / Gaussian posterior sampling driver.

Resolves the ``mcmc:`` block from the anchor config (or a site-set's
first member), applies CLI overrides, and calls
``ecosystem_complexity.mcmc.chain.run_from_args`` directly. Routes
outputs under ``outputs/{name}/mcmc/`` per the shared output contract.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
_NB = _REPO_ROOT / "notebooks"
if str(_NB) not in sys.path:
    sys.path.insert(0, str(_NB))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ecosystem_complexity import mcmc as _mcmc  # noqa: E402
from ecosystem_complexity.model.configuration import load_config  # noqa: E402
from ecosystem_complexity.mcmc.chain import build_args_namespace, run_from_args  # noqa: E402
from ecosystem_complexity.outputs import open_run_dir, resolve_run_name  # noqa: E402

logger = logging.getLogger("ecosys.mcmc")

# YAML key -> package constant to override on ``ecosystem_complexity.mcmc``.
# Overrides land at parent-process level; subprocess workers under spawn
# will re-import fresh, so keep constants matched to per-run defaults.
_YAML_TO_CONST = {
    "rng_seed": "RNG_SEED",
    "posterior_draw_count": "POSTERIOR_DRAW_COUNT",
    "prior_draw_count": "PRIOR_DRAW_COUNT",
    "mc_iterations": "MC_ITERATIONS",
    "null_iterations": "NULL_ITERATIONS",
    "warming_horizon_years": "WARMING_HORIZON_YEARS",
    "warming_delta_c": "WARMING_DELTA_C",
}


def _apply_yaml_defaults(block: dict[str, Any]) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    for yaml_key, const_name in _YAML_TO_CONST.items():
        if yaml_key in block:
            value = block[yaml_key]
            setattr(_mcmc, const_name, value)
            applied[const_name] = value
    if "old_pools" in block:
        value = tuple(block["old_pools"])
        _mcmc.OLD_POOLS = value
        applied["OLD_POOLS"] = value
    return applied


def _load_anchor_mcmc_block(config_path: str) -> dict[str, Any]:
    return dict(load_config(config_path).mcmc_raw or {})


def _anchor_config_path(sites: list[str], site_set: str | None) -> str | None:
    if sites:
        first = sites[0]
        if first.endswith((".yaml", ".yml")) or os.path.sep in first:
            return os.path.abspath(first)
    if site_set:
        with open(site_set, encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        configs = payload.get("configs") or []
        if configs:
            first = configs[0]
            return first if os.path.isabs(first) else str(_REPO_ROOT / first)
    return None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ecosys mcmc",
        description="MCMC / posterior sampling — reads mcmc: from YAML.",
    )
    p.add_argument("sites", nargs="*",
                   help="anchor site config path (defaults from its mcmc: block)")
    p.add_argument("--site-set", metavar="YAML",
                   help="use the site-set's first entry as the anchor config")
    p.add_argument("--outdir", default=None,
                   help="root under which outputs land (default ./outputs/)")
    p.add_argument("--rng-seed", type=int, default=None)
    p.add_argument("--posterior-draws", type=int, default=None)
    p.add_argument("--prior-draws", type=int, default=None)
    p.add_argument("--mc-iterations", type=int, default=None)
    p.add_argument("--null-iterations", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--network-summary", default=None)
    p.add_argument("--warming-summary", default=None)
    p.add_argument("--new-sites", nargs="+", default=None)
    return p


def _resolve_overrides(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    return {
        "site_set": args.site_set,
        "network_summary": args.network_summary,
        "warming_summary": args.warming_summary,
        "new_sites": args.new_sites,
        "seed": args.rng_seed,
        "posterior_draws": args.posterior_draws,
        "prior_draws": args.prior_draws,
        "mc_iterations": args.mc_iterations,
        "null_iterations": args.null_iterations,
        "workers": args.workers,
        "output_dir": str(out_dir),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    anchor = _anchor_config_path(args.sites, args.site_set)
    block: dict[str, Any] = _load_anchor_mcmc_block(anchor) if anchor else {}

    if args.site_set:
        with open(args.site_set, encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        run_name = str(payload.get("name") or Path(args.site_set).stem)
    elif anchor:
        run_name = resolve_run_name(config=load_config(anchor))
    else:
        run_name = "mcmc_default"

    run = open_run_dir(
        verb="mcmc",
        name=run_name,
        outdir=(Path(args.outdir) / run_name / "mcmc") if args.outdir else None,
        inputs={
            "anchor_config": anchor,
            "site_set": args.site_set,
            "yaml_block": block,
        },
    )

    applied_from_yaml = _apply_yaml_defaults(block)
    run.add_manifest_field("mcmc_constants_from_yaml", applied_from_yaml)
    cli_overrides = {
        k: v for k, v in {
            "rng_seed": args.rng_seed,
            "posterior_draws": args.posterior_draws,
            "prior_draws": args.prior_draws,
            "mc_iterations": args.mc_iterations,
            "null_iterations": args.null_iterations,
        }.items() if v is not None
    }
    run.add_manifest_field("mcmc_cli_overrides", cli_overrides)
    if anchor:
        run.snapshot_config(load_config(anchor))
    run.finalize()

    resolved_args = build_args_namespace(_resolve_overrides(args, run.root))
    t0 = time.perf_counter()
    run_from_args(resolved_args)

    for p in run.root.rglob("*"):
        if p.is_file() and p.name not in {"manifest.json"}:
            run.record_output(str(p.relative_to(run.root)))
    run.add_manifest_field("wall_seconds", round(time.perf_counter() - t0, 1))
    run.finalize()

    print(f"  wrote {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
