#!/usr/bin/env python3
"""Compute and plot intrinsic mean transit time for optimized network sites.

The repeatable default joins MAP turnover times retained in the canonical
network summary and the incubation-expansion summary with the configured
transfer topology.  ``--recover-map`` is
available when a full re-optimization is warranted to recover MAP transfers;
that is intentionally not the default because transfer logits are not retained
by the normal site-summary export.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import jax.nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecosystem_complexity.sites.driver import run_site_canonical
from ecosystem_complexity.sites.spec import load_site_spec
from ecosystem_complexity.transit_time import intrinsic_mean_transit_time
from ecosystem_complexity.api import build_model

COLORS = {
    "arctic_permafrost": "#355C7D", "boreal": "#6C8E4E", "peatland": "#7A4E7A",
    "temperate_forest": "#C06C2B", "grassland_mediterranean": "#C89B2B",
    "tropical": "#2C8C7B", "other": "#808080",
}


def biome_group(biome: str) -> str:
    b = biome.lower()
    if any(k in b for k in ("arctic", "tundra", "permafrost")): return "arctic_permafrost"
    if "boreal" in b: return "boreal"
    if any(k in b for k in ("peatland", "moss")): return "peatland"
    if "tropical" in b: return "tropical"
    if any(k in b for k in ("grassland", "mollisol", "mediterranean")): return "grassland_mediterranean"
    if "temperate" in b or "conifer" in b: return "temperate_forest"
    return "other"


def _one(config_path: str, include_er: bool) -> dict:
    spec = load_site_spec(config_path)
    result = run_site_canonical(spec, observation_path=spec.observation_path, include_er_constraint=include_er)
    if result.get("skipped"):
        return {"status": "skipped", "config": config_path, "site": spec.israd_name}
    model, params = result["model"], result["params_opt"]
    n = len(model.pool_index)
    weights = np.zeros(n)
    ext = model.config.external_inputs
    if ext is not None and ext.enabled:
        target = [model.pool_index[name] for name in ext.partition]
        part = np.asarray(jax.nn.softmax(params.log_external_input_partition), dtype=float)
        weights[target] = part
    else:
        weights[0] = 1.0
    mtt_days, by_pool = intrinsic_mean_transit_time(params.log_tau, params.log_f_transfer, weights)
    tau = np.exp(np.asarray(params.log_tau, dtype=float)) / 365.25
    return {
        "status": "ok", "config": config_path, "site": spec.israd_name, "label": spec.label,
        "biome": spec.biome, "biome_group": biome_group(spec.biome), "converged": result["converged"],
        "mean_transit_time_yr": mtt_days / 365.25,
        "transit_from_active_yr": by_pool[0] / 365.25,
        "transit_from_slow_yr": by_pool[1] / 365.25,
        "transit_from_passive_yr": by_pool[2] / 365.25,
        "tau_active_yr": tau[0], "tau_slow_yr": tau[1], "tau_passive_yr": tau[2],
    }


def _from_summary(row: pd.Series) -> dict:
    """Fast, reproducible structural MTT from retained MAP turnover times."""
    spec = load_site_spec(str(row.config))
    model = build_model(spec.config_path)
    n = len(model.pool_index)
    def tau_value(soil_name: str, expansion_name: str) -> float:
        value = row.get(soil_name, np.nan)
        if pd.isna(value):
            value = row.get(expansion_name, np.nan)
        return float(value)

    tau_years = np.array([
        tau_value("tau_soil_active", "tau_active_yr"),
        tau_value("tau_soil_slow", "tau_slow_yr"),
        tau_value("tau_soil_passive", "tau_passive_yr"),
    ])
    if not np.isfinite(tau_years).all() or (tau_years <= 0).any():
        raise ValueError(f"Missing valid optimized turnover times for {spec.israd_name}")
    ext = model.config.external_inputs
    weights = np.zeros(n)
    if ext is not None and ext.enabled:
        for pool, fraction in ext.partition.items():
            weights[model.pool_index[pool]] = float(fraction)
    else:
        weights[0] = 1.0
    # Build-time transfer rules are the transfer topology retained by the
    # network export; optimize_oe's MAP logits were not persisted historically.
    from ecosystem_complexity.state import make_default_params
    params = make_default_params(model.config)
    mtt_days, by_pool = intrinsic_mean_transit_time(
        np.log(tau_years * 365.25), np.asarray(params.log_f_transfer), weights
    )
    return {
        "status": "ok", "config": row.config, "site": spec.israd_name, "label": spec.label,
        "biome": spec.biome, "biome_group": biome_group(spec.biome), "converged": bool(row.converged),
        "mean_transit_time_yr": mtt_days / 365.25,
        "transit_from_active_yr": by_pool[0] / 365.25,
        "transit_from_slow_yr": by_pool[1] / 365.25,
        "transit_from_passive_yr": by_pool[2] / 365.25,
        "tau_active_yr": tau_years[0], "tau_slow_yr": tau_years[1], "tau_passive_yr": tau_years[2],
        "transfer_source": "configured topology; MAP tau",
        "inversion_scope": row.get("inversion_scope", "direct warming"),
    }


def plot(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.2), constrained_layout=True)
    ax = axes[0]
    for group, sub in df.groupby("biome_group", observed=True):
        direct = sub[sub.inversion_scope == "direct warming"]
        expansion = sub[sub.inversion_scope != "direct warming"]
        ax.scatter(direct.tau_passive_yr, direct.mean_transit_time_yr, s=62, color=COLORS[group],
                   edgecolor="white", linewidth=.7, label=group.replace("_", " / "))
        ax.scatter(expansion.tau_passive_yr, expansion.mean_transit_time_yr, s=74, marker="s",
                   facecolor="white", edgecolor=COLORS[group], linewidth=1.8)
    for row in df.nlargest(3, "mean_transit_time_yr").itertuples():
        ax.annotate(row.site, (row.tau_passive_yr, row.mean_transit_time_yr), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel="Optimized passive-pool turnover time (years)", ylabel="Intrinsic mean transit time (years)", title="Mean transit time reflects turnover and transfer pathways")
    ax.legend(frameon=False, fontsize=8, title="Biome")
    ax.grid(alpha=.2, which="both")
    ax = axes[1]
    ordered = df.sort_values("mean_transit_time_yr")
    y = np.arange(len(ordered))
    ax.hlines(y, 0, ordered.mean_transit_time_yr, color="0.8", lw=1)
    ax.scatter(ordered.mean_transit_time_yr, y, c=[COLORS[g] for g in ordered.biome_group], s=52, edgecolor="white", linewidth=.5)
    ax.set(yticks=y, yticklabels=ordered.site, xlabel="Intrinsic mean transit time (years)", title="Input-weighted transit times across 34 optimized ecosystems")
    ax.grid(axis="x", alpha=.2)
    fig.text(
        0.5, -0.02,
        "Reference-environment calculation: MAP turnover times with configured transfer fractions.",
        ha="center", fontsize=9, color="0.3",
    )
    axes[0].scatter([], [], marker="s", s=55, facecolor="white", edgecolor="0.25", linewidth=1.3,
                    label="incubation expansion")
    axes[0].legend(frameon=False, fontsize=8, title="Biome / site set")
    fig.savefig(out, dpi=250, bbox_inches="tight")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--site-summary", default=str(ROOT / "notebooks/exports/network_inversion_fluxcom_er_20260719/site_summary.csv"))
    p.add_argument("--expansion-summary", default=str(ROOT / "notebooks/exports/incubation_new_sites_runnable_20260719.csv"))
    p.add_argument("--out", default=str(ROOT / "notebooks/exports/optimized_ecosystem_transit_times_20260730.csv"))
    p.add_argument("--figure", default=str(ROOT / "notebooks/paper_figs/outputs/current_results/figures/figure_11_transit_times.png"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--recover-map", action="store_true", help="rerun each inversion to recover MAP transfer fractions (slow)")
    args = p.parse_args()
    direct = pd.read_csv(args.site_summary).copy()
    direct["inversion_scope"] = "direct warming"
    expansion = pd.read_csv(args.expansion_summary).copy()
    config_by_site = {
        "La Campana": "configs/expansion/la_campana.yaml",
        "Biadaski": "configs/expansion/biadaski.yaml",
        "Treynor": "configs/expansion/treynor.yaml",
        "Dinesen": "configs/expansion/dinesen.yaml",
        "Nelson Farm": "configs/expansion/nelson_farm.yaml",
        "CZ_1981burn": "configs/expansion/cz_1981burn.yaml",
        "Trumbore Musick": "configs/expansion/trumbore_musick.yaml",
        "Trumbore Falbrook": "configs/expansion/trumbore_falbrook.yaml",
        "Trumbore Ahwahnee": "configs/expansion/trumbore_ahwahnee.yaml",
        "CZ_1930burn": "configs/expansion/cz_1930burn.yaml",
    }
    expansion["config"] = expansion["site"].map(config_by_site)
    expansion["inversion_scope"] = "incubation expansion"
    if expansion["config"].isna().any():
        raise ValueError("No config mapping for one or more incubation-expansion sites")
    summary = pd.concat([direct, expansion], ignore_index=True, sort=False)
    rows, failures = [], []
    if args.recover_map:
        jobs = summary[["config", "include_er_constraint"]].drop_duplicates().itertuples(index=False, name=None)
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(_one, path, bool(er)): path for path, er in jobs}
            for fut in as_completed(futures):
                try:
                    payload = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append({"config": futures[fut], "error": repr(exc)})
                    continue
                if payload["status"] == "ok": rows.append(payload)
                else: failures.append(payload)
    else:
        for row in summary.itertuples(index=False):
            try:
                rows.append(_from_summary(pd.Series(row._asdict())))
            except Exception as exc:  # noqa: BLE001
                failures.append({"config": row.config, "error": repr(exc)})
    df = pd.DataFrame(rows).sort_values("mean_transit_time_yr")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    plot(df, Path(args.figure))
    if failures:
        pd.DataFrame(failures).to_csv(Path(args.out).with_name("transit_time_failures.csv"), index=False)
    print(f"Wrote {len(df)} sites to {args.out} and {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
