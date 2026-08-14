"""Shapley DFS attribution across sites carrying ISRaD density fractions.

Answers "how much information does each observation family — including the new
fraction_12C rung — contribute to the joint posterior, averaged over every
possible constraint ordering?". Writes:

    notebooks/exports/multisite_shapley_with_frac12c_20260731.csv
    notebooks/paper_figs/outputs/current_results/figures/fraction_12c_shapley.{png,pdf,svg}

Run:
    python notebooks/export_fraction_12c_shapley.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_NB_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_NB_ROOT)
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_REPO_ROOT)

from ecosystem_complexity.oe_diagnostics import (
    oe_ladder_context,
    shapley_dfs_attribution_from_context,
)
from ecosystem_complexity.sites import (
    OPT_FIELDS,
    discover_site_specs,
    run_site_canonical,
)

FAMILIES = (
    "C_stocks", "bulk_14C", "fraction_14C", "resp_14C", "fraction_12C",
)
FAMILY_LABELS = {
    "C_stocks":     "¹²C stocks",
    "bulk_14C":     "bulk Δ¹⁴C",
    "fraction_14C": "fraction Δ¹⁴C",
    "resp_14C":     "respired Δ¹⁴C",
    "fraction_12C": "fraction ¹²C (new)",
}
FAMILY_COLORS = {
    "C_stocks":     "#8C8C8C",
    "bulk_14C":     "#4A7C59",
    "fraction_14C": "#2E5D8A",
    "resp_14C":     "#B24E4E",
    "fraction_12C": "#D4A574",
}

OUT_CSV = os.path.join(
    _NB_ROOT, "exports", "multisite_shapley_with_frac12c_20260731.csv"
)
FIG_DIR = os.path.join(
    _NB_ROOT, "paper_figs", "outputs", "current_results", "figures"
)


def _shapley_one(stem: str, spec) -> list[dict]:
    print(f"── {spec.label} ({stem}) ──")
    res = run_site_canonical(
        spec, observation_path="combined", include_fraction_12c_constraint=True,
    )
    if res.get("skipped") or not res.get("converged"):
        print("  skipped or non-converged.")
        return []
    ctx = oe_ladder_context(
        res["model"], res["forcing"], res["state0"], res["params_opt"],
        res["obs_full"], OPT_FIELDS, extra_obs_blocks=res["pool_blocks"],
    )
    rows = shapley_dfs_attribution_from_context(ctx, families=FAMILIES)
    out = []
    for r in rows:
        out.append({
            "site": spec.israd_name,
            "config_stem": stem,
            "biome": spec.biome,
            "family": r["family"],
            "n_obs": r["n_obs"],
            "shapley_dfs": round(r["shapley_dfs"], 4),
            "shapley_share": round(r["shapley_share"], 4)
                if np.isfinite(r["shapley_share"]) else None,
            "dfs_standalone": round(r["dfs_standalone"], 4),
            "dfs_unique_last": round(r["dfs_unique_last"], 4),
            "dfs_joint_total": round(r["dfs_joint_total"], 4),
        })
    for r in out:
        print(
            f"  {r['family']:<14s} n={r['n_obs']:>2d}  "
            f"Shapley DFS={r['shapley_dfs']:>5.2f}  "
            f"share={r['shapley_share'] if r['shapley_share'] is not None else float('nan'):>5.1%}  "
            f"(alone={r['dfs_standalone']:.2f}, unique={r['dfs_unique_last']:.2f})"
        )
    return out


def _plot(df: pd.DataFrame, out_stem: str) -> None:
    fired_sites = (
        df[(df.family == "fraction_12C") & (df.n_obs > 0)]
        .sort_values("shapley_dfs", ascending=False)["site"].tolist()
    )
    if not fired_sites:
        print("No sites with a firing fraction_12C rung — no figure written.")
        return
    sub = df[df.site.isin(fired_sites)].copy()
    pivot = sub.pivot(index="site", columns="family", values="shapley_dfs").reindex(
        index=fired_sites, columns=list(FAMILIES)
    ).fillna(0.0)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    y_pos = np.arange(len(fired_sites))
    left = np.zeros(len(fired_sites))
    for fam in FAMILIES:
        vals = pivot[fam].to_numpy()
        ax.barh(y_pos, vals, left=left, color=FAMILY_COLORS[fam],
                edgecolor="white", linewidth=0.5, label=FAMILY_LABELS[fam])
        left = left + vals
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fired_sites, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Shapley DFS (degrees of freedom for signal)")
    ax.set_title(
        "Shapley DFS attribution — ISRaD density-fraction ¹²C is a major "
        "information source at the fraction sites",
        fontsize=10.5, loc="left",
    )
    ax.axvline(5.0, color="0.4", linestyle=":", linewidth=0.8)
    ax.text(5.02, len(fired_sites) - 0.4, "state-vector rank = 5",
            fontsize=7.5, color="0.35", va="center")
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        path = os.path.join(FIG_DIR, f"{out_stem}.{ext}")
        fig.savefig(path)
        print(f"Wrote {path}")
    plt.close(fig)


def main(names: list[str] | None = None) -> None:
    specs = discover_site_specs()
    if names:
        wanted = {n.lower() for n in names}
        specs = {k: v for k, v in specs.items() if k.lower() in wanted}
    rows: list[dict] = []
    for stem, spec in specs.items():
        try:
            rows.extend(_shapley_one(stem, spec))
        except Exception as exc:  # noqa: BLE001 — never let one site abort the sweep
            print(f"  ERROR at {spec.label}: {exc}")
    if not rows:
        print("No Shapley rows produced.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(df)} rows, {df['site'].nunique()} sites)")
    os.makedirs(FIG_DIR, exist_ok=True)
    _plot(df, "fraction_12c_shapley")


if __name__ == "__main__":
    main(sys.argv[1:])
