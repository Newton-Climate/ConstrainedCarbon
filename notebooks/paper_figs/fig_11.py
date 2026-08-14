"""Two-axis synthesis of turnover separation, transit time, and vulnerability."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from notebooks.paper_figs.fig_09 import BIOME_GROUP_COLORS, BIOME_GROUP_LABELS, BIOME_GROUP_ORDER
    from notebooks.paper_figs.utils import finalize_figure, setup_figure_config
else:
    from .fig_09 import BIOME_GROUP_COLORS, BIOME_GROUP_LABELS, BIOME_GROUP_ORDER
    from .utils import finalize_figure, setup_figure_config


ROOT = Path(__file__).resolve().parents[2]


def _correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for x, y, label in [
        ("turnover_separation", "old_fraction_of_excess_rh", "separation vs old-C share"),
        ("turnover_separation", "frac_c_loss", "separation vs fractional C loss"),
        ("environmental_protection", "old_fraction_of_excess_rh", "protection vs old-C share"),
        ("environmental_protection", "frac_c_loss", "protection vs fractional C loss"),
        ("turnover_separation", "environmental_protection", "structure vs protection"),
    ]:
        sample = df[[x, y]].dropna()
        rows.append(
            {
                "relationship": label,
                "n": len(sample),
                "pearson_r": sample[x].corr(sample[y], method="pearson"),
                "spearman_rho": sample[x].corr(sample[y], method="spearman"),
            }
        )
    return pd.DataFrame(rows)


def build_figure(metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.2, 6.8), constrained_layout=True)
    x = metrics["turnover_separation"].to_numpy(dtype=float)
    y = metrics["environmental_protection"].to_numpy(dtype=float)
    sizes = 28 + 48 * np.clip(metrics["dfs_total"].to_numpy(dtype=float), 0.0, 2.5)
    scatter = ax.scatter(
        x,
        y,
        c=metrics["frac_c_loss"],
        s=sizes,
        cmap="YlOrRd",
        vmin=float(metrics["frac_c_loss"].min()),
        vmax=float(metrics["frac_c_loss"].max()),
        edgecolor="white",
        linewidth=0.75,
        zorder=3,
    )
    for biome in BIOME_GROUP_ORDER:
        sub = metrics[metrics["biome_group"] == biome]
        if sub.empty:
            continue
        ax.scatter([], [], color=BIOME_GROUP_COLORS[biome], label=BIOME_GROUP_LABELS.get(biome, biome), s=34)
        ax.scatter(
            sub["turnover_separation"],
            sub["environmental_protection"],
            facecolors="none",
            edgecolors=BIOME_GROUP_COLORS[biome],
            linewidth=1.15,
            s=28 + 48 * np.clip(sub["dfs_total"], 0.0, 2.5),
            zorder=4,
        )
    ax.axvline(float(np.median(x)), color="0.45", linestyle="--", linewidth=0.9, zorder=1)
    ax.axhline(0.0, color="0.45", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(float(np.min(x)) + 0.01, 0.035, "P > 1: environmental protection", fontsize=8, color="0.32")
    ax.text(float(np.min(x)) + 0.01, -0.30, "P < 1: accelerated exit", fontsize=8, color="0.32")
    ax.text(float(np.median(x)) + 0.01, float(np.max(y)) - 0.03, "higher turnover separation", fontsize=8, color="0.32")

    label_sites = set(metrics.nlargest(2, "frac_c_loss")["site"]).union(metrics.nlargest(1, "realised_to_intrinsic")["site"]).union(metrics.nsmallest(1, "realised_to_intrinsic")["site"])
    for row in metrics[metrics["site"].isin(label_sites)].itertuples(index=False):
        ax.annotate(str(row.site), (row.turnover_separation, row.environmental_protection), xytext=(5, 4), textcoords="offset points", fontsize=7)

    ax.set_title("Structural turnover separation and environmental exposure are distinct axes", loc="left", fontweight="bold")
    ax.set_xlabel(r"Turnover separation, $\log_{10}(\tau_{passive}/\tau_{active})$")
    ax.set_ylabel(r"Environmental exposure, $\log_{10}(P)$  ($P=T_{realized}/T_{intrinsic}$)")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    cbar.set_label("Fractional soil-C loss under warming")
    for dfs in [0.5, 1.5, 2.5]:
        ax.scatter([], [], s=28 + 48 * dfs, facecolor="0.7", edgecolor="white", label=f"DFS {dfs:.1f}")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.17, 1.0), frameon=False, fontsize=7, title="Biome outline / point size")
    legend.get_title().set_fontsize(8)
    return fig


def make_figure_11(
    transit_vulnerability_metrics: str | pd.DataFrame,
    output_dir: str = "outputs",
    config_path: str | None = None,
):
    """Build Figure 11 through the shared manuscript-figure interface."""
    setup_figure_config(config_path)
    metrics = (
        pd.read_csv(transit_vulnerability_metrics)
        if isinstance(transit_vulnerability_metrics, (str, Path))
        else transit_vulnerability_metrics.copy()
    )
    metrics = metrics.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["turnover_separation", "environmental_protection", "frac_c_loss", "dfs_total"]
    )
    correlations = _correlation_table(metrics)
    fig = build_figure(metrics)
    finalize_figure(
        fig,
        "figure_11_turnover_transit_vulnerability",
        output_dir,
        {"site_metrics": metrics, "correlations": correlations},
        "Two-axis summary across 30 direct-warming ecosystems. The x-axis is passive-to-active turnover separation; the y-axis is the log realized-to-intrinsic transit-time ratio, with zero indicating no environmental modification. Point colour denotes fractional soil-carbon loss under warming, point size denotes total DFS, and outline colour denotes biome group.",
        "Turnover separation, transit time, and vulnerability",
        "Turnover separation captures the inferred structural age contrast, while the realized-to-intrinsic ratio captures recurring environmental protection or acceleration. Warming loss and DFS are encoded without treating either as a substitute for the transit metrics.",
    )
    return fig, fig.axes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default=str(ROOT / "notebooks/exports/cross_biome_with_ma_harvard_20260731/transit_vulnerability_site_metrics.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "notebooks/paper_figs/outputs/cross_biome_with_ma_harvard_20260731"))
    args = parser.parse_args()
    fig, _ = make_figure_11(args.metrics, args.output_dir)
    metrics = pd.read_csv(args.metrics)
    correlations = _correlation_table(metrics)
    print(correlations.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
