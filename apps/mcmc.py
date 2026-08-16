#!/usr/bin/env python3
"""``ecosys mcmc`` — MCMC / Gaussian posterior sampling driver.

Thin adapter over the existing ``sample_mcmc`` pipeline. Its job is to:

1. Read the ``mcmc:`` block from the anchor config (or a site-set's first
   member) and use those values in place of ``sample_mcmc``'s
   module-level constants (``RNG_SEED``, ``MC_ITERATIONS``,
   ``POSTERIOR_DRAW_COUNT``, ``PRIOR_DRAW_COUNT``, ``NULL_ITERATIONS``,
   ``WARMING_HORIZON_YEARS``, ``WARMING_DELTA_C``, ``OLD_POOLS``). CLI
   flags override the YAML.
2. Route outputs under ``outputs/{name}/mcmc/`` following the shared
   output contract, with an initial ``manifest.json`` recording the
   resolved config.
3. Delegate the heavy lifting to ``sample_mcmc.main`` unchanged — the
   posterior sampler and regression pipeline are a paper-figure workflow
   that is out of scope to rewrite here.

Full end-to-end contract artifacts (``chains.npz``, ``chain_summary.parquet``,
``warming_samples.parquet``) will replace ``sample_mcmc``'s in-tree
paper-figure outputs in a follow-up rewrite.
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
_SRC = _REPO_ROOT / "src"
_NB = _REPO_ROOT / "notebooks"
for _p in (_REPO_ROOT, _SRC, _NB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ecosystem_complexity.config import load_config  # noqa: E402
from ecosystem_complexity.outputs import (  # noqa: E402
    open_run_dir,
    resolve_run_name,
)

logger = logging.getLogger("ecosys.mcmc")

# ---------------------------------------------------------------------------
# YAML → constant patching
# ---------------------------------------------------------------------------

# Names of the module-level constants in ``sample_mcmc`` that the YAML
# ``mcmc:`` block controls. Any key present in the block replaces the
# matching constant before ``sample_mcmc.main`` runs; CLI flags take
# precedence over both.
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
    """Patch ``sample_mcmc``'s module constants from a ``mcmc:`` block.

    Returns the (name → applied value) map for the manifest.
    """
    import apps.sample_mcmc as sm  # noqa: WPS433 — deliberate late import
    applied: dict[str, Any] = {}
    for yaml_key, const_name in _YAML_TO_CONST.items():
        if yaml_key in block:
            value = block[yaml_key]
            setattr(sm, const_name, value)
            applied[const_name] = value
    if "old_pools" in block:
        applied["OLD_POOLS"] = tuple(block["old_pools"])
        sm.OLD_POOLS = tuple(block["old_pools"])
    return applied


def _load_anchor_mcmc_block(config_path: str) -> dict[str, Any]:
    """Extract the ``mcmc:`` block from a per-site config YAML."""
    return dict(load_config(config_path).mcmc_raw or {})


def _anchor_config_path(sites: list[str], site_set: str | None) -> str | None:
    """Pick the config whose ``mcmc:`` block seeds defaults.

    Precedence: an explicit CLI site path > the site-set's first entry >
    None (fall back to sample_mcmc's built-in defaults).
    """
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
    p.add_argument("--rng-seed", type=int, default=None,
                   help="override mcmc.rng_seed from YAML")
    p.add_argument("--posterior-draws", type=int, default=None,
                   help="override mcmc.posterior_draw_count from YAML")
    p.add_argument("--prior-draws", type=int, default=None,
                   help="override mcmc.prior_draw_count from YAML")
    p.add_argument("--mc-iterations", type=int, default=None,
                   help="override mcmc.mc_iterations from YAML")
    p.add_argument("--null-iterations", type=int, default=None,
                   help="override mcmc.null_iterations from YAML")
    p.add_argument("--workers", type=int, default=None,
                   help="workers forwarded to sample_mcmc")
    # Downstream inputs the underlying pipeline requires
    p.add_argument("--network-summary", default=None)
    p.add_argument("--warming-summary", default=None)
    p.add_argument("--new-sites", nargs="+", default=None)
    return p


def _forwarded_argv(args: argparse.Namespace, out_dir: Path) -> list[str]:
    """Build the argv sample_mcmc.main will see, honoring CLI overrides."""
    a: list[str] = []
    if args.site_set:
        a += ["--site-set", args.site_set]
    if args.network_summary:
        a += ["--network-summary", args.network_summary]
    if args.warming_summary:
        a += ["--warming-summary", args.warming_summary]
    if args.new_sites:
        a += ["--new-sites", *args.new_sites]
    if args.rng_seed is not None:
        a += ["--seed", str(args.rng_seed)]
    if args.posterior_draws is not None:
        a += ["--posterior-draws", str(args.posterior_draws)]
    if args.prior_draws is not None:
        a += ["--prior-draws", str(args.prior_draws)]
    if args.mc_iterations is not None:
        a += ["--mc-iterations", str(args.mc_iterations)]
    if args.null_iterations is not None:
        a += ["--null-iterations", str(args.null_iterations)]
    if args.workers is not None:
        a += ["--workers", str(args.workers)]
    a += ["--output-dir", str(out_dir)]
    return a


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    anchor = _anchor_config_path(args.sites, args.site_set)
    block: dict[str, Any] = _load_anchor_mcmc_block(anchor) if anchor else {}

    # Resolve run name: the site-set name wins; otherwise the anchor config's
    # site.id; otherwise a generic fallback. Naming this way keeps every
    # MCMC run under a stable outputs root that co-locates with its
    # optimize/warming siblings for the same site or network.
    if args.site_set:
        with open(args.site_set, encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        run_name = str(payload.get("name") or Path(args.site_set).stem)
    elif anchor:
        run_name = resolve_run_name(config=load_config(anchor))
    else:
        run_name = "mcmc_default"

    # Open the run dir first so the manifest records the actual resolved
    # settings that sample_mcmc will use (YAML block ∪ CLI overrides).
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

    # Apply YAML defaults BEFORE _apply_cli_overrides so CLI wins on ties.
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
    run.finalize()  # write initial manifest; sample_mcmc adds its own files

    import apps.sample_mcmc as sm  # noqa: WPS433
    forwarded = _forwarded_argv(args, run.root)
    saved_argv = sys.argv
    sys.argv = ["sample_mcmc", *forwarded]
    t0 = time.perf_counter()
    try:
        sm.main()
    finally:
        sys.argv = saved_argv

    # Re-finalize to update finished_at and record files sample_mcmc wrote.
    for p in run.root.rglob("*"):
        if p.is_file() and p.name not in {"manifest.json"}:
            run.record_output(str(p.relative_to(run.root)))
    run.add_manifest_field("wall_seconds", round(time.perf_counter() - t0, 1))
    run.finalize()

    print(f"  wrote {run.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
