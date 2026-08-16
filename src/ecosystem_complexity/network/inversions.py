#!/usr/bin/env python3
"""Run network-wide site inversions and aggregate OE diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]

from ecosystem_complexity.oe_diagnostics import (
    cumulative_ladder_from_context,
    constraint_orthogonality_from_context,
    oe_ladder_context,
    shapley_dfs_attribution_from_context,
)
from ecosystem_complexity.sites.driver import (
    OPT_FIELDS,
    run_site_canonical,
)
from ecosystem_complexity.sites.spec import load_site_spec


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


def _config_paths(include_expansion: bool, site_set: str | None = None) -> list[str]:
    if site_set:
        with open(site_set, encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        paths = payload.get("configs")
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
            raise ValueError(f"{site_set}: expected a non-empty string list at 'configs'")
        root = _REPO_ROOT
        resolved = [str(Path(p) if Path(p).is_absolute() else root / p) for p in paths]
        missing = [p for p in resolved if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"{site_set}: missing config(s): {', '.join(missing)}")
        return resolved
    paths = sorted(str(p) for p in Path("configs/multisite").glob("*.yaml"))
    if include_expansion:
        paths += sorted(str(p) for p in Path("configs/expansion").glob("*.yaml"))
    return paths


def _analyze_one(
    config_path: str,
    include_er_constraint: bool,
    include_incubation_constraint: bool,
    include_incubation_14c_constraint: bool,
) -> dict:
    spec = load_site_spec(config_path)
    t0 = time.perf_counter()
    result = run_site_canonical(
        spec,
        observation_path=spec.observation_path,
        include_er_constraint=include_er_constraint,
        include_incubation_constraint=include_incubation_constraint,
        include_incubation_14c_constraint=include_incubation_14c_constraint,
    )
    if result.get("skipped"):
        return {
            "status": "skipped",
            "config": config_path,
            "label": spec.label,
            "runtime_s": time.perf_counter() - t0,
        }

    oe = result["oe_result"]
    ak = np.array(oe.averaging_kernel, dtype=float)
    y_obs = np.array(oe.y_obs, dtype=float)
    y_opt = np.array(oe.y_opt, dtype=float)
    y_prior = np.array(oe.y_prior, dtype=float)

    ctx = oe_ladder_context(
        result["model"],
        result["forcing"],
        result["state0"],
        result["params_opt"],
        result["obs_full"],
        OPT_FIELDS,
        extra_obs_blocks=result["pool_blocks"],
    )
    ladder = cumulative_ladder_from_context(ctx)
    orth = constraint_orthogonality_from_context(ctx)
    shapley = shapley_dfs_attribution_from_context(ctx)
    shapley_map = {row["family"]: row for row in shapley}
    ladder_map = {row["added_family"]: row for row in ladder}
    dominant_family = max(
        (row for row in shapley if row["n_obs"] > 0),
        key=lambda row: row["shapley_dfs"],
    )["family"]

    resid = y_opt - y_obs
    prior_resid = y_prior - y_obs
    rmse = float(np.sqrt(np.mean(resid**2))) if resid.size else float("nan")
    prior_rmse = (
        float(np.sqrt(np.mean(prior_resid**2))) if prior_resid.size else float("nan")
    )

    site_row = {
        "config": config_path,
        "site": spec.israd_name,
        "label": spec.label,
        "biome": spec.biome,
        "biome_group": _biome_group(spec.biome),
        "forcing_kind": spec.forcing_kind,
        "observation_path": result["observation_path"],
        "include_er_constraint": include_er_constraint,
        "include_incubation_constraint": include_incubation_constraint,
        "include_incubation_14c_constraint": include_incubation_14c_constraint,
        "tower_id": spec.tower_id,
        "n_obs_total": int(y_obs.size),
        "n_state": int(len(oe.state_names)),
        "dfs_total": float(np.trace(ak)),
        "dfs_diag_mean": float(np.mean(np.diag(ak))),
        "dfs_diag_max": float(np.max(np.diag(ak))),
        "rmse_obs_space": rmse,
        "rmse_obs_space_prior": prior_rmse,
        "rmse_improvement_frac": (
            (prior_rmse - rmse) / prior_rmse
            if np.isfinite(prior_rmse) and prior_rmse > 0.0
            else float("nan")
        ),
        "converged": bool(result["converged"]),
        "n_iter": int(result["n_iter"]),
        "cost0": float(result["cost0"]),
        "cost_final": float(result["cost_final"]),
        "mean_GPP_gCm2yr": float(result["mean_gpp_gCm2yr"]),
        "SOC_gCm2": float(result["soc_total_gCm2"]),
        "n_pool_blocks": int(result["n_pool_blocks"]),
        "n_resp": int(result["n_resp"]),
        "n_er_finite": int(result["n_er_finite"]),
        "n_obs_cstocks": int(ctx["n_obs_per_family"].get("C_stocks", 0)),
        "n_obs_bulk14C": int(ctx["n_obs_per_family"].get("bulk_14C", 0)),
        "n_obs_fraction14C": int(ctx["n_obs_per_family"].get("fraction_14C", 0)),
        "n_obs_resp14C": int(ctx["n_obs_per_family"].get("resp_14C", 0)),
        "n_obs_ER_annual": int(ctx["n_obs_per_family"].get("ER_annual", 0)),
        "ladder_inc_cstocks": float(
            ladder_map.get("C_stocks", {}).get("dfs_increment", 0.0)
        ),
        "ladder_inc_bulk14C": float(
            ladder_map.get("bulk_14C", {}).get("dfs_increment", 0.0)
        ),
        "ladder_inc_fraction14C": float(
            ladder_map.get("fraction_14C", {}).get("dfs_increment", 0.0)
        ),
        "ladder_inc_resp14C": float(
            ladder_map.get("resp_14C", {}).get("dfs_increment", 0.0)
        ),
        "ladder_inc_ER_annual": float(
            ladder_map.get("ER_annual", {}).get("dfs_increment", 0.0)
        ),
        "shapley_cstocks": float(
            shapley_map.get("C_stocks", {}).get("shapley_dfs", 0.0)
        ),
        "shapley_bulk14C": float(
            shapley_map.get("bulk_14C", {}).get("shapley_dfs", 0.0)
        ),
        "shapley_fraction14C": float(
            shapley_map.get("fraction_14C", {}).get("shapley_dfs", 0.0)
        ),
        "shapley_resp14C": float(
            shapley_map.get("resp_14C", {}).get("shapley_dfs", 0.0)
        ),
        "shapley_ER_annual": float(
            shapley_map.get("ER_annual", {}).get("shapley_dfs", 0.0)
        ),
        "shapley_share_cstocks": float(
            shapley_map.get("C_stocks", {}).get("shapley_share", 0.0)
        ),
        "shapley_share_bulk14C": float(
            shapley_map.get("bulk_14C", {}).get("shapley_share", 0.0)
        ),
        "shapley_share_fraction14C": float(
            shapley_map.get("fraction_14C", {}).get("shapley_share", 0.0)
        ),
        "shapley_share_resp14C": float(
            shapley_map.get("resp_14C", {}).get("shapley_share", 0.0)
        ),
        "shapley_share_ER_annual": float(
            shapley_map.get("ER_annual", {}).get("shapley_share", 0.0)
        ),
        "dominant_family": dominant_family,
        "stock_unique_frac": float(orth["uniqueness_a"]),
        "c14_unique_frac": float(orth["uniqueness_b"]),
        "stock_c14_redundancy": float(orth["redundancy"]),
        "mean_principal_angle_deg": float(orth["mean_principal_angle_deg"]),
        "min_principal_angle_deg": float(orth["min_principal_angle_deg"]),
        "runtime_s": time.perf_counter() - t0,
    }
    for pool_name, tau in result["tau_years"].items():
        site_row[f"tau_{pool_name}"] = float(tau)

    ladder_rows = [
        {
            "site": spec.israd_name,
            "label": spec.label,
            "biome_group": _biome_group(spec.biome),
            "forcing_kind": spec.forcing_kind,
            **{
                k: (
                    float(v) if isinstance(v, (np.floating, float))
                    else int(v) if isinstance(v, (np.integer, int))
                    else v
                )
                for k, v in rung.items()
                if k != "dfs_per_param"
            },
        }
        for rung in ladder
    ]
    shapley_rows = [
        {
            "site": spec.israd_name,
            "label": spec.label,
            "biome_group": _biome_group(spec.biome),
            "forcing_kind": spec.forcing_kind,
            **{
                k: (
                    float(v) if isinstance(v, (np.floating, float))
                    else int(v) if isinstance(v, (np.integer, int))
                    else v
                )
                for k, v in row.items()
            },
        }
        for row in shapley
    ]

    return {
        "status": "ok",
        "config": config_path,
        "label": spec.label,
        "site_row": site_row,
        "ladder_rows": ladder_rows,
        "shapley_rows": shapley_rows,
    }


def _aggregate(site_df: pd.DataFrame) -> dict:
    agg: dict = {}
    if site_df.empty:
        return agg
    def _records(frame: pd.DataFrame) -> list[dict]:
        flat = frame.copy()
        flat.columns = [
            col if isinstance(col, str) else "__".join(str(part) for part in col if part)
            for col in flat.columns.to_flat_index()
        ]
        return flat.reset_index().to_dict(orient="records")

    agg["by_biome_group"] = _records(
        site_df.groupby("biome_group")[
            [
                "dfs_total",
                "shapley_share_cstocks",
                "shapley_share_bulk14C",
                "shapley_share_fraction14C",
                "shapley_share_resp14C",
                "shapley_share_ER_annual",
                "stock_unique_frac",
                "c14_unique_frac",
                "stock_c14_redundancy",
                "mean_GPP_gCm2yr",
            ]
        ]
        .agg(["count", "mean", "median"])
        .round(4)
    )
    agg["by_forcing_kind"] = _records(
        site_df.groupby("forcing_kind")[
            [
                "dfs_total",
                "shapley_share_cstocks",
                "shapley_share_bulk14C",
                "shapley_share_fraction14C",
                "shapley_share_resp14C",
                "shapley_share_ER_annual",
                "stock_unique_frac",
                "c14_unique_frac",
                "stock_c14_redundancy",
            ]
        ]
        .agg(["count", "mean", "median"])
        .round(4)
    )
    agg["dominant_family_by_biome"] = (
        site_df.groupby(["biome_group", "dominant_family"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict(orient="records")
    )
    return agg


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run canonical inversions across the configured network and "
            "aggregate OE information-content diagnostics."
        )
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of site-level worker processes (default 4)",
    )
    p.add_argument(
        "--include-expansion",
        action="store_true",
        help="include configs/expansion/*.yaml in addition to configs/multisite/",
    )
    p.add_argument(
        "--site-set",
        help="YAML with an explicit 'configs' list; overrides the default network selection",
    )
    p.add_argument(
        "--include-er-constraint",
        action="store_true",
        help="add annual tower ER as an OE observation block where available",
    )
    p.add_argument("--include-incubation-constraint", action="store_true")
    p.add_argument("--include-incubation-14c-constraint", action="store_true")
    p.add_argument(
        "--outdir",
        default="notebooks/exports/network_inversion_20260718",
        help="output directory for CSV/JSON summaries",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.ERROR)

    if args.site_set and args.include_expansion:
        raise SystemExit("--site-set and --include-expansion are mutually exclusive")
    config_paths = _config_paths(args.include_expansion, args.site_set)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    site_rows: list[dict] = []
    ladder_rows: list[dict] = []
    shapley_rows: list[dict] = []
    failures: list[dict] = []

    max_workers = max(1, min(args.workers, len(config_paths)))
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(
                _analyze_one,
                path,
                args.include_er_constraint,
                args.include_incubation_constraint,
                args.include_incubation_14c_constraint,
            ): path
            for path in config_paths
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            path = futures[fut]
            try:
                payload = fut.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {"config": path, "label": Path(path).stem, "status": "failed", "error": repr(exc)}
                )
                print(f"[{i}/{len(futures)}] {path} :: FAILED {exc!r}", flush=True)
                continue

            status = payload["status"]
            label = payload["label"]
            if status == "ok":
                row = payload["site_row"]
                site_rows.append(row)
                ladder_rows.extend(payload["ladder_rows"])
                shapley_rows.extend(payload["shapley_rows"])
                print(
                    f"[{i}/{len(futures)}] {label} :: DFS={row['dfs_total']:.2f} "
                    f"dominant={row['dominant_family']}",
                    flush=True,
                )
            else:
                failures.append(payload)
                print(f"[{i}/{len(futures)}] {label} :: {status}", flush=True)

    site_df = pd.DataFrame(site_rows)
    ladder_df = pd.DataFrame(ladder_rows)
    shapley_df = pd.DataFrame(shapley_rows)
    fail_df = pd.DataFrame(failures)

    site_df.to_csv(outdir / "site_summary.csv", index=False)
    ladder_df.to_csv(outdir / "ladder_summary.csv", index=False)
    shapley_df.to_csv(outdir / "shapley_summary.csv", index=False)
    fail_df.to_csv(outdir / "failures.csv", index=False)
    with (outdir / "aggregate_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_aggregate(site_df), fh, indent=2)

    print(
        f"Wrote {len(site_df)} sites, {len(ladder_df)} ladder rows, "
        f"{len(shapley_df)} shapley rows to {outdir}",
        flush=True,
    )
    return 0 if fail_df.empty else 1
