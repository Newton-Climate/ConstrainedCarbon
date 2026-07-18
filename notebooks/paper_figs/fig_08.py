from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from notebooks.paper_figs.utils import coerce_table, finalize_figure, panelize, setup_figure_config
else:
    from .utils import coerce_table, finalize_figure, panelize, setup_figure_config


PATH_LABELS = {
    "bulk_resp": "Bulk + respired 14C",
    "fraction": "Fraction 14C",
    "combined": "Combined (fraction + bulk + stocks)",
}
PATH_COLORS = {
    "bulk_resp": "#9C4F2D",
    "fraction": "#3F7754",
    "combined": "#3A5A8C",
}
# Pathways rendered in the pooled panels (C, D), in draw order.
POOLED_PATHWAYS = ("bulk_resp", "fraction", "combined")
REQUIRED = {
    "site", "biome", "pathway", "n_14c_blocks", "dfs_tau",
    "ur_tau_mean", "ur_tau_active", "ur_tau_slow", "ur_tau_passive",
}


def _paired_table(df: pd.DataFrame) -> pd.DataFrame:
    # Panels A/B compare fraction vs bulk+respired specifically; restrict to those
    # two pathways and keep only sites where BOTH are present (robust to a third
    # 'combined' pathway also being in the table).
    two = df[df["pathway"].isin(["fraction", "bulk_resp"])]
    have_both = two.groupby("site")["pathway"].nunique()
    paired = two[two["site"].isin(have_both[have_both == 2].index)]
    values = [
        "dfs_tau", "ur_tau_mean", "ur_tau_active", "ur_tau_slow", "ur_tau_passive",
    ]
    wide = paired.pivot(index=["site", "biome"], columns="pathway", values=values)
    wide.columns = [f"{metric}_{pathway}" for metric, pathway in wide.columns]
    wide = wide.reset_index()
    wide["dfs_tau_ratio"] = wide["dfs_tau_fraction"] / wide["dfs_tau_bulk_resp"]
    for pool in ("active", "slow", "passive"):
        wide[f"ur_delta_{pool}"] = (
            wide[f"ur_tau_{pool}_fraction"] - wide[f"ur_tau_{pool}_bulk_resp"]
        )
    return wide.sort_values("dfs_tau_ratio")


def _site_order(df: pd.DataFrame) -> list[str]:
    means = df.groupby("site", sort=False)["dfs_tau"].mean().sort_values()
    return list(means.index)


def make_figure_08(
    pathway_information: str | pd.DataFrame,
    output_dir: str = "outputs",
    config_path: str | None = None,
):
    setup_figure_config(config_path)
    info = coerce_table(pathway_information, "pathway_information")
    assert info is not None
    missing = REQUIRED - set(info.columns)
    if missing:
        raise ValueError(f"pathway information is missing columns: {sorted(missing)}")

    paired = _paired_table(info)
    site_order = _site_order(info)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.0))
    panelize(axes)

    ax = axes[0, 0]
    y = np.arange(len(paired))
    ax.axvspan(0.9, 1.1, color="0.9", zorder=0)
    ax.axvline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.scatter(paired["dfs_tau_ratio"], y, s=65, color="#2E5D4A", zorder=3)
    for yi, value in zip(y, paired["dfs_tau_ratio"]):
        ax.plot([1.0, value], [yi, yi], color="0.45", linewidth=1.2, zorder=1)
        ax.text(value, yi + 0.17, f"{value:.2f}", fontsize=8, ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels(paired["site"])
    ax.set_xlabel("Turnover DFS ratio: fraction / bulk + respired")
    ax.set_title("Pathway equivalence is site-specific")
    ax.text(0.73, 0.97, "±10% equivalence band", transform=ax.transAxes,
            ha="center", va="top", fontsize=8)

    ax = axes[0, 1]
    delta_cols = ["ur_delta_active", "ur_delta_slow", "ur_delta_passive"]
    mat = paired.set_index("site")[delta_cols].to_numpy(dtype=float) * 100.0
    vmax = max(20.0, float(np.nanmax(np.abs(mat))))
    im = ax.imshow(mat, cmap="BrBG", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Active", "Slow", "Passive"])
    ax.set_yticks(np.arange(len(paired)))
    ax.set_yticklabels(paired["site"])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if abs(mat[i, j]) > 0.55 * vmax else "black"
            ax.text(j, i, f"{mat[i, j]:+.0f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_title("Fraction minus bulk + respired uncertainty reduction")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Percentage-point difference")

    ax = axes[1, 0]
    ypos = {site: i for i, site in enumerate(site_order)}
    present = [p for p in POOLED_PATHWAYS if (info["pathway"] == p).any()]
    span = 0.26
    offsets = {
        p: (span * (i - (len(present) - 1) / 2) if len(present) > 1 else 0.0)
        for i, p in enumerate(present)
    }
    for pathway in present:
        sub = info[info["pathway"] == pathway]
        ys = [ypos[site] + offsets[pathway] for site in sub["site"]]
        ax.scatter(
            sub["dfs_tau"], ys, s=48, color=PATH_COLORS[pathway],
            label=PATH_LABELS[pathway], alpha=0.9,
        )
    ax.set_yticks(np.arange(len(site_order)))
    labels = [f"{site}  |  {info.loc[info.site == site, 'biome'].iloc[0]}" for site in site_order]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Turnover degrees of freedom for signal")
    ax.set_title("Expanded cross-ecosystem sample")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1, 1]
    markers = {"bulk_resp": "o", "fraction": "D", "combined": "s"}
    for pathway in present:
        sub = info[info["pathway"] == pathway]
        ax.scatter(
            sub["n_14c_blocks"], sub["dfs_tau"], s=55,
            marker=markers[pathway], color=PATH_COLORS[pathway],
            label=PATH_LABELS[pathway], alpha=0.9,
        )
        label_sites = {"Howland Forest", "Harvard Forest", "EML", "Appi forest", "CZ_Old_Black_Spruce"}
        for row in sub[sub["site"].isin(label_sites)].itertuples(index=False):
            ax.annotate(str(row.site), (row.n_14c_blocks, row.dfs_tau),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Number of radiocarbon constraint blocks")
    ax.set_ylabel("Turnover degrees of freedom for signal")
    ax.set_title("Constraint geometry matters more than count alone")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    equivalent = int(paired["dfs_tau_ratio"].between(0.9, 1.1).sum())
    alt_text = (
        f"Four-panel comparison of radiocarbon pathways across {info.site.nunique()} ecosystems. "
        f"{equivalent} of {len(paired)} paired sites have fraction-to-bulk-plus-respired turnover DFS "
        "within ten percent. Pool-level differences show that equivalence depends on which turnover "
        "modes the available fractions represent, and the expanded site panel shows substantial "
        "between-ecosystem variation."
    )
    caption = (
        "Ecosystem dependence of radiocarbon information pathways. (A) Within-site ratio of turnover "
        "degrees of freedom for signal (DFS) from fraction radiocarbon versus bulk-soil plus respired "
        "radiocarbon; shading marks a descriptive ±10% equivalence interval. (B) Difference in posterior "
        "turnover-time uncertainty reduction between pathways for active, slow, and passive pools. "
        "(C) Turnover DFS for all locally forced sites, including sites with only one available pathway. "
        "(D) Turnover DFS versus the number of radiocarbon constraint blocks. Every inversion uses the "
        "same three-pool optimal-estimation method and a site-specific steady-state SOC prior with 25% uncertainty."
    )
    biome_summary = (
        info.groupby(["biome", "pathway"], as_index=False)
        .agg(n_sites=("site", "nunique"), dfs_tau_mean=("dfs_tau", "mean"),
             uncertainty_reduction_mean=("ur_tau_mean", "mean"))
    )
    finalize_figure(
        fig,
        "figure_08",
        output_dir,
        {"pathway_information": info, "paired_pathway_comparison": paired, "biome_summary": biome_summary},
        alt_text,
        "Figure 8",
        caption,
    )
    return fig, axes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pathway-information", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    make_figure_08(args.pathway_information, output_dir=args.output_dir, config_path=args.config)


if __name__ == "__main__":
    main()
