#!/usr/bin/env python3
"""``ecosys optimize`` — canonical OE soil-carbon inversion.

Runs the canonical inversion for one site, several sites, a site set, or a
sweep of pool-count/forcing variants, and writes each run's artifacts to
``./outputs/{name}/optimize/`` following the shared output contract in
:mod:`ecosystem_complexity.outputs`.

Selectors — mix freely
    * a path to a config     configs/multisite/solling.yaml
    * a config stem          solling
    * an ISRaD site name     Solling

Examples
    ecosys optimize configs/harvard_forest.yaml
    ecosys optimize --site-set configs/site_sets/full_network_41.yaml
    ecosys optimize --sweep configs/hf_pool_sweep/  -j 4

Every run writes ``manifest.json``, ``config.snapshot.yaml``,
``posterior.parquet``, ``diagnostics.json``, plus a ``summary.parquet``
at the site-set root when applicable. Sweeps land under
``optimize/sweep/{member_stem}/`` with a ``sweep_summary.parquet``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ecosystem_complexity.model.configuration import load_config  # noqa: E402
from ecosystem_complexity.outputs import (  # noqa: E402
    attach_file_logger,
    open_run_dir,
    resolve_run_name,
    write_json,
    write_npz,
    write_parquet,
)
from ecosystem_complexity.sites import (  # noqa: E402
    SiteSpec,
    discover_site_specs,
    load_site_spec,
    run_sites,
    summary_row,
)

logger = logging.getLogger("ecosys.optimize")

OBSERVATION_PATHS = ("bulk_resp", "fraction", "combined")
INCUBATION_DURATION_TYPES = ("<2 weeks", "<1 month", "<1 year", ">1 year")


# ---------------------------------------------------------------------------
# Site-set / sweep resolution
# ---------------------------------------------------------------------------


def _load_site_set(path: str) -> tuple[str, list[str], dict[str, Any]]:
    """Return ``(name, resolved_config_paths, run_options)`` from a site-set YAML."""
    with open(path, encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    name = str(payload.get("name") or Path(path).stem)
    configs = payload.get("configs")
    if not isinstance(configs, list) or not configs or not all(isinstance(p, str) for p in configs):
        raise SystemExit(f"error: {path}: expected a non-empty string list at 'configs'")
    resolved = [p if os.path.isabs(p) else str(_REPO_ROOT / p) for p in configs]
    missing = [p for p in resolved if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"error: {path}: missing config(s): {', '.join(missing)}")
    return name, resolved, dict(payload.get("run_options") or {})


def _resolve_sweep(cli_dir: str | None, anchor_config: str | None) -> tuple[list[str], str]:
    """Return ``(member_config_paths, sweep_kind)`` for --sweep.

    A directory given on the CLI wins; otherwise the anchor config's
    ``sweep:`` block is consulted (member_dir / member_glob).
    """
    if cli_dir:
        member_dir = Path(cli_dir)
        pattern = "*.yaml"
        kind = "cli"
    elif anchor_config:
        cfg = load_config(anchor_config)
        block = cfg.sweep_raw
        if not block:
            raise SystemExit(
                f"error: --sweep given without a directory and {anchor_config} has "
                f"no `sweep:` block."
            )
        member_dir = Path(block.get("member_dir") or "")
        pattern = str(block.get("member_glob") or "*.yaml")
        kind = str(block.get("kind") or "unspecified")
    else:
        raise SystemExit("error: --sweep needs a directory or an anchor config.")
    if not member_dir.is_absolute():
        member_dir = _REPO_ROOT / member_dir
    if not member_dir.is_dir():
        raise SystemExit(f"error: sweep member_dir does not exist: {member_dir}")
    members = sorted(str(p) for p in member_dir.glob(pattern))
    if not members:
        raise SystemExit(f"error: no configs matched {member_dir}/{pattern}")
    return members, kind


def _resolve_specs(selectors: list[str]) -> list[SiteSpec]:
    """Resolve CLI selectors (path, stem, ISRaD name) to SiteSpecs."""
    known = discover_site_specs()
    by_key: dict[str, SiteSpec] = {}
    for spec in known.values():
        by_key[spec.config_stem] = spec
        by_key[spec.israd_name] = spec
        by_key[spec.label] = spec

    specs: list[SiteSpec] = []
    for sel in selectors:
        if sel.endswith((".yaml", ".yml")) or os.path.sep in sel:
            path = os.path.abspath(sel)
            if not os.path.isfile(path):
                raise SystemExit(f"error: no such config file: {sel}")
            specs.append(load_site_spec(path))
            continue
        if sel not in by_key:
            raise SystemExit(
                f"error: unknown site {sel!r}\n"
                f"  known sites: {', '.join(sorted(known))}\n"
                f"  (or pass a path to a config YAML)"
            )
        specs.append(by_key[sel])
    return specs


# ---------------------------------------------------------------------------
# Per-site output writers
# ---------------------------------------------------------------------------


def _posterior_rows(result: dict) -> list[dict]:
    """Flatten optimized parameters into (name, value, prior_mean, prior_std) rows.

    We only surface the log_tau vector here: it is what every downstream
    consumer keys off and is directly interpretable as τ (days). log_f_transfer
    is preserved in the raw npz below.
    """
    params_opt = result["params_opt"]
    idx = result["model"].pool_index
    log_tau = np.array(params_opt.log_tau)
    tau_days = np.exp(log_tau)
    # Prior means come from the config's tau_prior_days; we don't wire the
    # per-parameter prior_std through here yet (needs deeper introspection).
    rows: list[dict] = []
    for i, name in enumerate(idx.pool_names):
        rows.append({
            "param": f"log_tau[{name}]",
            "value_log": float(log_tau[i]),
            "value_tau_days": float(tau_days[i]),
            "value_tau_years": float(tau_days[i] / 365.25),
        })
    return rows


def _write_site_outputs(result: dict, out_root: str | Path | None, verb_extra: dict[str, Any]) -> Path:
    """Write the per-site artifact set. Returns the run directory path."""
    spec: SiteSpec = result["spec"]
    config = load_config(spec.config_path)
    name = resolve_run_name(config=config)
    run = open_run_dir(
        verb="optimize",
        name=name,
        outdir=(Path(out_root) / name / "optimize") if out_root else None,
        inputs={
            "config_path": str(spec.config_path),
            "observation_path": result["observation_path"],
            "include_er_constraint": result.get("include_er_constraint", False),
            **verb_extra,
        },
    )
    run.snapshot_config(config)

    # posterior.parquet: parameter table
    write_parquet(run, "posterior.parquet", _posterior_rows(result))

    # posterior.npz: raw arrays for downstream information diagnostics
    params_opt = result["params_opt"]
    write_npz(
        run,
        "posterior.npz",
        log_tau=np.array(params_opt.log_tau),
        log_f_transfer=np.array(params_opt.log_f_transfer),
        converged=np.array(bool(result["converged"])),
        n_iter=np.array(int(result["n_iter"])),
    )

    # diagnostics.json: cost trace summary + constraint counts
    diag = {
        "cost0": float(result["cost0"]),
        "cost_final": float(result["cost_final"]),
        "converged": bool(result["converged"]),
        "n_iter": int(result["n_iter"]),
        "n_pool_blocks": int(result["n_pool_blocks"]),
        "n_resp_14C": int(result["n_resp"]),
        "n_incubation": int(result["n_incubation"]),
        "n_incubation_14c": int(result["n_incubation_14c"]),
        "n_fraction_12c": int(result["n_fraction_12c"]),
        "n_er_finite": int(result.get("n_er_finite") or 0),
        "n_cstock": int(result["n_cstock"]),
        "soc_source": str(result["soc_source"]),
        "soc_total_gCm2": float(result["soc_total_gCm2"]),
        "mean_gpp_gCm2yr": float(result["mean_gpp_gCm2yr"]),
    }
    write_json(run, "diagnostics.json", diag)

    # summary.parquet: one-row per-site rollup (same shape as summary_row)
    write_parquet(run, "summary.parquet", [summary_row(result)])

    run.add_manifest_field("site_id", spec.israd_name)
    run.add_manifest_field("biome", spec.biome)
    run.finalize()
    return run.root


def _write_site_set_summary(
    site_set_name: str,
    rows: list[dict],
    out_root: str | Path | None,
    verb_extra: dict[str, Any],
) -> Path:
    """Aggregate summary rows across a site set into one network table."""
    run = open_run_dir(
        verb="optimize",
        name=site_set_name,
        outdir=(Path(out_root) / site_set_name / "optimize") if out_root else None,
        inputs={"n_members": len(rows), **verb_extra},
    )
    write_parquet(run, "network_summary.parquet", rows)
    run.add_manifest_field("site_set", site_set_name)
    run.finalize()
    return run.root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ecosys optimize",
        description="Run the canonical OE inversion for one or more sites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("sites", nargs="*",
                   help="site selectors: config path, config stem, or ISRaD name")
    p.add_argument("--all", action="store_true",
                   help="run every config under configs/multisite/")
    p.add_argument("--site-set", metavar="YAML",
                   help="run the config list in a versioned site-set YAML")
    p.add_argument("--sweep", metavar="DIR", nargs="?", const="",
                   help="run each config in DIR as a sweep member; if the given "
                        "anchor config defines sweep: use that when DIR omitted")
    p.add_argument("--list", action="store_true", help="list configured sites and exit")
    p.add_argument("--observation-path", choices=OBSERVATION_PATHS, default=None,
                   help="override the observation path each config declares")
    p.add_argument("--outdir", metavar="DIR", default=None,
                   help="root under which outputs land (default ./outputs/)")
    p.add_argument("-j", "--workers", type=int, default=1, metavar="N",
                   help="run N sites concurrently in separate processes (default 1)")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="suppress per-site progress logging")
    p.add_argument("--include-incubation", action="store_true")
    p.add_argument("--include-incubation-14c", action="store_true")
    p.add_argument("--include-er", action="store_true")
    p.add_argument("--no-fraction-12c", dest="fraction_12c", action="store_false")
    p.set_defaults(fraction_12c=None)
    p.add_argument("--incubation-duration-type", action="append",
                   choices=INCUBATION_DURATION_TYPES, metavar="CLASS",
                   help="restrict incubation rows to one or more ISRaD duration classes")
    return p


def _print_available() -> int:
    specs = discover_site_specs()
    if not specs:
        raise SystemExit("error: no site configs found under configs/multisite/")
    width = max(len(s) for s in specs)
    print(f"{len(specs)} configured sites:\n")
    print(f"  {'STEM'.ljust(width)}  {'ISRAD NAME':<24} {'OBS PATH':<10} BIOME")
    for stem in sorted(specs):
        s = specs[stem]
        print(f"  {stem.ljust(width)}  {s.israd_name:<24} "
              f"{s.observation_path:<10} {s.biome}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list:
        return _print_available()

    exclusive = sum(bool(v) for v in (args.sites, args.all, args.site_set, args.sweep is not None))
    if exclusive == 0:
        _build_parser().error(
            "give at least one site, --all, --site-set, or --sweep "
            "(--list shows what is available)"
        )
    if exclusive > 1:
        _build_parser().error("site names, --all, --site-set, and --sweep are mutually exclusive")
    if args.workers < 1:
        _build_parser().error("--workers must be at least 1")
    if args.incubation_duration_type and not args.include_incubation:
        _build_parser().error("--incubation-duration-type requires --include-incubation")

    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s", stream=sys.stdout,
    )

    site_set_name: str | None = None
    verb_extra: dict[str, Any] = {
        "include_er": bool(args.include_er),
        "include_incubation": bool(args.include_incubation),
        "include_incubation_14c": bool(args.include_incubation_14c),
    }

    if args.all:
        specs = list(discover_site_specs().values())
    elif args.site_set:
        site_set_name, cfg_paths, run_options = _load_site_set(args.site_set)
        specs = _resolve_specs(cfg_paths)
        # Site-set run_options override CLI-defaults when the CLI didn't specify.
        if not args.include_er and run_options.get("include_er"):
            verb_extra["include_er"] = True
            args.include_er = True
        if not args.include_incubation and run_options.get("include_incubation"):
            verb_extra["include_incubation"] = True
            args.include_incubation = True
        if not args.include_incubation_14c and run_options.get("include_incubation_14c"):
            verb_extra["include_incubation_14c"] = True
            args.include_incubation_14c = True
    elif args.sweep is not None:
        anchor = args.sites[0] if args.sites else None
        members, sweep_kind = _resolve_sweep(args.sweep or None, anchor)
        specs = [load_site_spec(p) for p in members]
        # A sweep is named after the sweep directory unless one of the members
        # explicitly names an anchor site — that's the convention hf_pool_sweep used.
        site_set_name = Path(members[0]).parent.name
        verb_extra["sweep_kind"] = sweep_kind
    else:
        specs = _resolve_specs(args.sites)

    if not specs:
        raise SystemExit("error: no site configs to run")

    workers = min(args.workers, len(specs))
    if workers > 1:
        print(f"Running {len(specs)} sites across {workers} worker processes…")

    # Reduce inside the worker to a lightweight dict that survives pickling.
    # We keep enough to write the per-site artifacts on the parent side:
    # everything except the compiled model closure.
    def _reduce(result: dict) -> dict:
        # Drop the compiled EcosystemModel — closures don't pickle — but keep
        # everything else the per-site writer needs.
        r = dict(result)
        r.pop("model", None)
        r.pop("oe_result", None)  # holds JAX arrays; row/params are already extracted
        return r

    # When workers>1 the writer runs on the parent after collection; when
    # workers==1 the reduce is still applied for a uniform code path but the
    # model has to be rebuilt for _write_site_outputs (which reads config only).
    results, failures = run_sites(
        specs,
        observation_path=args.observation_path,
        include_er_constraint=args.include_er,
        include_incubation_constraint=args.include_incubation,
        include_incubation_14c_constraint=args.include_incubation_14c,
        incubation_duration_types=(
            frozenset(args.incubation_duration_type)
            if args.incubation_duration_type else None
        ),
        include_fraction_12c_constraint=args.fraction_12c,
        workers=workers,
        reduce=_reduce,
    )

    summary_rows: list[dict] = []
    for r in results:
        # Re-attach a minimal model pool_index for _posterior_rows.
        # Cheapest way: rebuild from config; that's a lightweight parse.
        from ecosystem_complexity.model.api import build_model  # noqa: WPS433
        r["model"] = build_model(r["spec"].config_path)
        run_dir = _write_site_outputs(r, args.outdir, verb_extra)
        summary_rows.append(summary_row(r))
        print(f"  wrote {run_dir}")

    if site_set_name and summary_rows:
        agg = _write_site_set_summary(site_set_name, summary_rows, args.outdir, verb_extra)
        print(f"  wrote site-set summary {agg}")

    n_skipped = len(specs) - len(results) - len(failures)
    print(
        f"\n{len(results)}/{len(specs)} sites inverted"
        + (f", {n_skipped} skipped (insufficient ¹⁴C obs)" if n_skipped else "")
        + (f", {len(failures)} failed" if failures else "")
    )
    for spec, exc in failures:
        print(f"  FAILED  {spec.label}: {exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
