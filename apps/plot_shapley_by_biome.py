#!/usr/bin/env python3
"""Plot mean Shapley attribution by biome group from network inversion output."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SHAPLEY_COLS = [
    "shapley_share_cstocks",
    "shapley_share_bulk14C",
    "shapley_share_fraction14C",
    "shapley_share_resp14C",
]

LABELS = {
    "shapley_share_cstocks": "C stocks",
    "shapley_share_bulk14C": "Bulk 14C",
    "shapley_share_fraction14C": "Fraction 14C",
    "shapley_share_resp14C": "Respired 14C",
}

COLORS = {
    "shapley_share_cstocks": "#6A8E54",
    "shapley_share_bulk14C": "#4C78A8",
    "shapley_share_fraction14C": "#F2C14E",
    "shapley_share_resp14C": "#E45756",
}

GROUP_ORDER = [
    "arctic_permafrost",
    "boreal",
    "peatland",
    "temperate_forest",
    "tropical",
    "grassland_mediterranean",
    "other",
]

GROUP_LABELS = {
    "arctic_permafrost": "Arctic / permafrost",
    "boreal": "Boreal",
    "peatland": "Peatland",
    "temperate_forest": "Temperate forest",
    "tropical": "Tropical",
    "grassland_mediterranean": "Grassland / Mediterranean",
    "other": "Other",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot mean Shapley shares by biome group from site_summary.csv."
    )
    p.add_argument(
        "--site-summary",
        default="notebooks/exports/network_inversion_20260718/site_summary.csv",
        help="input site_summary.csv path",
    )
    p.add_argument(
        "--outdir",
        default="notebooks/exports/network_inversion_20260718",
        help="output directory for figure and aggregated table",
    )
    return p


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("biome_group", as_index=False)
        .agg(
            n_sites=("site", "count"),
            mean_dfs=("dfs_total", "mean"),
            **{col: (col, "mean") for col in SHAPLEY_COLS},
        )
        .copy()
    )
    out["biome_group"] = pd.Categorical(
        out["biome_group"], categories=GROUP_ORDER, ordered=True
    )
    out = out.sort_values("biome_group").reset_index(drop=True)
    out["biome_label"] = out["biome_group"].map(GROUP_LABELS).fillna(out["biome_group"])
    out["biome_label_n"] = out.apply(
        lambda r: f"{r['biome_label']} (n={int(r['n_sites'])})", axis=1
    )
    return out


def _plot_panel(ax, agg: pd.DataFrame, title: str) -> None:
    left = pd.Series(0.0, index=agg.index)
    for col in SHAPLEY_COLS:
        vals = agg[col].fillna(0.0)
        ax.barh(
            agg["biome_label_n"],
            vals,
            left=left,
            color=COLORS[col],
            edgecolor="white",
            linewidth=0.8,
            label=LABELS[col],
        )
        left = left + vals

    for i, row in agg.iterrows():
        ax.text(
            1.01,
            i,
            f"DFS={row['mean_dfs']:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    ax.set_xlim(0.0, 1.12)
    ax.set_xlabel("Mean Shapley share")
    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.invert_yaxis()


def main() -> int:
    args = _build_parser().parse_args()
    site_summary = Path(args.site_summary)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(site_summary)

    subsets = [
        ("All completed sites", df.copy(), "all_sites"),
        (
            "Tower-backed bulk_resp subset",
            df[(df["forcing_kind"] == "daily") & (df["observation_path"] == "bulk_resp")].copy(),
            "tower_bulk_resp",
        ),
    ]

    agg_frames = []
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10.5, 8.2),
        constrained_layout=True,
    )

    for ax, (title, sub, subset_name) in zip(axes, subsets, strict=True):
        agg = _aggregate(sub)
        agg["subset"] = subset_name
        agg_frames.append(agg)
        _plot_panel(ax, agg, title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
    )
    fig.suptitle(
        "Shapley attribution by biome group",
        fontsize=15,
        weight="bold",
        y=1.03,
    )

    png_path = outdir / "shapley_by_biome_group.png"
    pdf_path = outdir / "shapley_by_biome_group.pdf"
    csv_path = outdir / "shapley_by_biome_group_summary.csv"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    pd.concat(agg_frames, ignore_index=True).to_csv(csv_path, index=False)

    print(png_path)
    print(pdf_path)
    print(csv_path)
    return 0
