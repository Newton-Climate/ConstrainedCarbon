#!/usr/bin/env python3
"""``ecosys warming`` — standardized warming-response projections.

Runs the canonical OE inversion at each site and projects the model
forward under a uniform temperature perturbation for ``horizon_years``.
For a site-set the per-site rows aggregate into a network table plus a
biome-group rollup.

Reads defaults from each config's optional ``warming:`` block; CLI flags
override that; hardcoded defaults are the last resort. This is the one
place ``warming.horizon_years`` / ``warming.warming_delta_c`` should
live for a reproducible experiment — passing them on the CLI is for ad-hoc runs.

Examples
    ecosys warming --site-set configs/site_sets/full_network_41.yaml
    ecosys warming harvard_forest --warming-delta-c 2.0 --horizon-years 50
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ecosystem_complexity.config import load_config  # noqa: E402
from ecosystem_complexity.outputs import (  # noqa: E402
    open_run_dir,
    resolve_run_name,
    write_json,
    write_parquet,
)
from ecosystem_complexity.sites.driver import run_site_canonical  # noqa: E402
from ecosystem_complexity.sites.spec import load_site_spec  # noqa: E402
from ecosystem_complexity.warming import project_warming_response  # noqa: E402

logger = logging.getLogger("ecosys.warming")

DEFAULT_HORIZON_YEARS = 100.0
DEFAULT_WARMING_DELTA_C = 4.0


# ---------------------------------------------------------------------------
# Config resolution: YAML warming: block > CLI flags > built-in defaults
# ---------------------------------------------------------------------------


def _resolve_warming_params(
    config_path: str,
    cli_horizon: float | None,
    cli_delta: float | None,
    cli_metric: str | None,
) -> dict[str, Any]:
    """Merge YAML/CLI/defaults into a single param dict for one config."""
    block = dict(load_config(config_path).warming_raw or {})
    return {
        "horizon_years": float(
            cli_horizon if cli_horizon is not None
            else block.get("horizon_years", DEFAULT_HORIZON_YEARS)
        ),
        "warming_delta_c": float(
            cli_delta if cli_delta is not None
            else block.get("warming_delta_c", DEFAULT_WARMING_DELTA_C)
        ),
        "metric": str(
            cli_metric if cli_metric is not None
            else block.get("metric", "vulnerability")
        ),
        "include_constraints": dict(block.get("include_constraints") or {}),
    }


# ---------------------------------------------------------------------------
# Site-set loader
# ---------------------------------------------------------------------------


def _load_site_set(path: str) -> tuple[str, list[str], dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    name = str(payload.get("name") or Path(path).stem)
    configs = payload.get("configs")
    if not isinstance(configs, list) or not configs or not all(isinstance(p, str) for p in configs):
        raise SystemExit(f"error: {path}: expected non-empty string list at 'configs'")
    resolved = [p if os.path.isabs(p) else str(_REPO_ROOT / p) for p in configs]
    missing = [p for p in resolved if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"error: {path}: missing config(s): {', '.join(missing)}")
    return name, resolved, dict(payload.get("run_options") or {})


def _biome_group(biome: str) -> str:
    b = biome.lower()
    if any(k in b for k in ("arctic", "tundra", "permafrost")):
        return "arctic_permafrost"
    if "boreal" in b:
        return "boreal"
    if any(k in b for k in ("peatland", "moss")):
        return "peatland"
    if "tropical" in b:
        return "tropical"
    if any(k in b for k in ("grassland", "mollisol", "mediterranean")):
        return "grassland_mediterranean"
    if "temperate" in b:
        return "temperate_forest"
    return "other"


# ---------------------------------------------------------------------------
# Worker: invert + project + write per-site artifacts
# ---------------------------------------------------------------------------


def _run_one(
    config_path: str,
    warming_params: dict[str, Any],
    include_er: bool,
    include_incubation: bool,
    include_incubation_14c: bool,
    outdir_root: str | None,
) -> dict[str, Any]:
    spec = load_site_spec(config_path)
    t0 = time.perf_counter()
    result = run_site_canonical(
        spec,
        observation_path=spec.observation_path,
        include_er_constraint=include_er,
        include_incubation_constraint=include_incubation,
        include_incubation_14c_constraint=include_incubation_14c,
    )
    if result.get("skipped"):
        return {"status": "skipped", "config": config_path, "label": spec.label,
                "runtime_s": time.perf_counter() - t0}

    metrics = project_warming_response(
        result["model"], result["forcing"], result["state0"], result["params_opt"],
        horizon_years=warming_params["horizon_years"],
        warming_delta_c=warming_params["warming_delta_c"],
    )

    row = {
        "config": config_path,
        "site": spec.israd_name, "label": spec.label,
        "tower_id": spec.tower_id, "biome": spec.biome,
        "biome_group": _biome_group(spec.biome),
        "forcing_kind": spec.forcing_kind,
        "observation_path": spec.observation_path,
        "converged": bool(result["converged"]),
        "dfs_total": float(np.trace(
            np.array(result["oe_result"].averaging_kernel, dtype=float)
        )),
        "mean_GPP_gCm2yr": float(result["mean_gpp_gCm2yr"]),
        "SOC_gCm2": float(result["soc_total_gCm2"]),
        "n_pool_blocks": int(result["n_pool_blocks"]),
        "n_resp": int(result["n_resp"]),
        "n_er_finite": int(result["n_er_finite"] or 0),
        "horizon_years": warming_params["horizon_years"],
        "warming_delta_c": warming_params["warming_delta_c"],
        "metric": warming_params["metric"],
        "runtime_s": time.perf_counter() - t0,
        **{k: float(v) for k, v in metrics.items()},
    }

    # Per-site artifacts under outputs/{site_id}/warming/
    config = load_config(config_path)
    name = resolve_run_name(config=config)
    run = open_run_dir(
        verb="warming",
        name=name,
        outdir=(Path(outdir_root) / name / "warming") if outdir_root else None,
        inputs={
            "config_path": config_path,
            **warming_params,
            "include_er": include_er,
            "include_incubation": include_incubation,
            "include_incubation_14c": include_incubation_14c,
        },
    )
    run.snapshot_config(config)
    write_parquet(run, "summary.parquet", [row])
    write_json(run, "diagnostics.json", {
        "converged": bool(result["converged"]),
        "n_iter": int(result["n_iter"]),
        "cost_final": float(result["cost_final"]),
        "dfs_total": row["dfs_total"],
        **{k: v for k, v in metrics.items()},
    })
    run.add_manifest_field("site_id", spec.israd_name)
    run.add_manifest_field("biome_group", row["biome_group"])
    run.finalize()

    return {"status": "ok", "row": row, "run_dir": str(run.root)}


# ---------------------------------------------------------------------------
# Aggregation across a site-set
# ---------------------------------------------------------------------------


def _biome_rollup(site_df: pd.DataFrame) -> pd.DataFrame:
    if site_df.empty:
        return pd.DataFrame()
    metric_cols = [c for c in (
        "frac_c_loss", "abs_c_loss_gCm2",
        "delta_rh_annual_mean_gCm2yr", "old_fraction_of_excess_rh",
    ) if c in site_df.columns]
    if not metric_cols:
        return pd.DataFrame()
    agg = (
        site_df.groupby("biome_group", dropna=False)[metric_cols]
        .agg(["mean", "median", "min", "max"])
    )
    agg.columns = ["__".join(str(p) for p in col if p) for col in agg.columns.to_flat_index()]
    return agg.reset_index()


def _write_site_set_summary(
    site_set_name: str, rows: list[dict], outdir_root: str | None,
    warming_params: dict[str, Any],
) -> Path:
    site_df = pd.DataFrame(rows)
    if not site_df.empty and "frac_c_loss" in site_df.columns:
        site_df = site_df.sort_values("frac_c_loss", ascending=False)
    run = open_run_dir(
        verb="warming",
        name=site_set_name,
        outdir=(Path(outdir_root) / site_set_name / "warming") if outdir_root else None,
        inputs={"n_members": len(rows), **warming_params},
    )
    write_parquet(run, "network_warming_summary.parquet", site_df)
    rollup = _biome_rollup(site_df)
    if not rollup.empty:
        write_parquet(run, "by_biome_group.parquet", rollup)
    run.add_manifest_field("site_set", site_set_name)
    run.finalize()
    return run.root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ecosys warming",
        description="Project warming response for one or more sites.",
    )
    p.add_argument("sites", nargs="*", help="config paths or site stems/names")
    p.add_argument("--site-set", metavar="YAML")
    p.add_argument("--outdir", default=None,
                   help="root under which outputs land (default ./outputs/)")
    p.add_argument("--horizon-years", type=float, default=None,
                   help="override warming.horizon_years from YAML")
    p.add_argument("--warming-delta-c", type=float, default=None,
                   help="override warming.warming_delta_c from YAML")
    p.add_argument("--metric", choices=("vulnerability", "transit"), default=None,
                   help="override warming.metric from YAML")
    p.add_argument("--include-er", action="store_true")
    p.add_argument("--include-incubation", action="store_true")
    p.add_argument("--include-incubation-14c", action="store_true")
    p.add_argument("-j", "--workers", type=int, default=1, metavar="N")
    return p


def _resolve_specs(selectors: list[str]) -> list[str]:
    """Return absolute config paths for CLI selectors (accepts paths or stems)."""
    from ecosystem_complexity.sites import discover_site_specs
    known = discover_site_specs()
    by_key = {}
    for spec in known.values():
        by_key[spec.config_stem] = spec.config_path
        by_key[spec.israd_name] = spec.config_path
        by_key[spec.label] = spec.config_path
    out: list[str] = []
    for sel in selectors:
        if sel.endswith((".yaml", ".yml")) or os.path.sep in sel:
            p = os.path.abspath(sel)
            if not os.path.isfile(p):
                raise SystemExit(f"error: no such config file: {sel}")
            out.append(p)
        elif sel in by_key:
            out.append(str(by_key[sel]))
        else:
            raise SystemExit(f"error: unknown site {sel!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if not args.sites and not args.site_set:
        _build_parser().error("give at least one site or --site-set")
    if args.sites and args.site_set:
        _build_parser().error("sites and --site-set are mutually exclusive")

    site_set_name: str | None = None
    if args.site_set:
        site_set_name, configs, run_options = _load_site_set(args.site_set)
        include_er = args.include_er or bool(run_options.get("include_er"))
        include_incub = args.include_incubation or bool(run_options.get("include_incubation"))
        include_incub14 = args.include_incubation_14c or bool(run_options.get("include_incubation_14c"))
    else:
        configs = _resolve_specs(args.sites)
        include_er = args.include_er
        include_incub = args.include_incubation
        include_incub14 = args.include_incubation_14c

    if not configs:
        raise SystemExit("error: no site configs to run")

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    workers = min(args.workers, len(configs)) if args.workers > 1 else 1

    def _do_one(cfg_path: str) -> dict[str, Any]:
        params = _resolve_warming_params(
            cfg_path, args.horizon_years, args.warming_delta_c, args.metric,
        )
        return _run_one(
            cfg_path, params, include_er, include_incub, include_incub14, args.outdir,
        )

    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        print(f"Running {len(configs)} sites across {workers} workers…")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            fut_to_path = {pool.submit(_do_one, p): p for p in configs}
            for fut in as_completed(fut_to_path):
                cfg = fut_to_path[fut]
                try:
                    payload = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append((cfg, repr(exc)))
                    print(f"  FAILED  {cfg}: {exc}")
                    continue
                if payload["status"] == "ok":
                    rows.append(payload["row"])
                    r = payload["row"]
                    print(f"  {r['label']}  frac_loss={r.get('frac_c_loss', float('nan')):.3f}  "
                          f"δRh={r.get('delta_rh_annual_mean_gCm2yr', float('nan')):.1f}")
                else:
                    print(f"  {payload.get('label', cfg)}: skipped")
    else:
        for cfg in configs:
            try:
                payload = _do_one(cfg)
            except Exception as exc:  # noqa: BLE001
                failures.append((cfg, repr(exc)))
                print(f"  FAILED  {cfg}: {exc}")
                continue
            if payload["status"] == "ok":
                rows.append(payload["row"])
                r = payload["row"]
                print(f"  {r['label']}  frac_loss={r.get('frac_c_loss', float('nan')):.3f}  "
                      f"δRh={r.get('delta_rh_annual_mean_gCm2yr', float('nan')):.1f}")
            else:
                print(f"  {payload.get('label', cfg)}: skipped")

    if site_set_name and rows:
        # Pick a representative warming_params dict for the manifest: they should
        # be identical across members unless CLI/YAML disagree.
        params = _resolve_warming_params(
            configs[0], args.horizon_years, args.warming_delta_c, args.metric,
        )
        agg_dir = _write_site_set_summary(site_set_name, rows, args.outdir, params)
        print(f"  wrote site-set summary {agg_dir}")

    print(f"\n{len(rows)}/{len(configs)} sites projected"
          + (f", {len(failures)} failed" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
