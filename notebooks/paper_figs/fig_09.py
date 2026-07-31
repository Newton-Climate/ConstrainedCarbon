from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from notebooks.paper_figs.io import load_table
    from notebooks.paper_figs.utils import (
        coerce_table,
        finalize_figure,
        panelize,
        setup_figure_config,
        standard_figure_parser,
    )
else:
    from .io import load_table
    from .utils import coerce_table, finalize_figure, panelize, setup_figure_config, standard_figure_parser


BIOME_GROUP_ORDER = [
    "arctic_permafrost",
    "boreal",
    "peatland",
    "temperate_forest",
    "grassland_mediterranean",
    "tropical",
]
BIOME_GROUP_LABELS = {
    "arctic_permafrost": "Arctic / permafrost",
    "boreal": "Boreal",
    "peatland": "Peatland",
    "temperate_forest": "Temperate forest",
    "grassland_mediterranean": "Grassland / Mediterranean",
    "tropical": "Tropical",
    "other": "Other",
}
BIOME_GROUP_COLORS = {
    "arctic_permafrost": "#355C7D",
    "boreal": "#6C8E4E",
    "peatland": "#7A4E7A",
    "temperate_forest": "#C06C2B",
    "grassland_mediterranean": "#C89B2B",
    "tropical": "#2C8C7B",
    "other": "#808080",
}
FAMILY_ORDER = [
    "shapley_share_cstocks",
    "shapley_share_bulk14C",
    "shapley_share_fraction14C",
    "shapley_share_resp14C",
    "shapley_share_ER_annual",
]
FAMILY_LABELS = {
    "shapley_share_cstocks": "C stocks",
    "shapley_share_bulk14C": "Bulk 14C",
    "shapley_share_fraction14C": "Fraction 14C",
    "shapley_share_resp14C": "Respired 14C",
    "shapley_share_ER_annual": "Annual ER",
}
FAMILY_COLORS = {
    "shapley_share_cstocks": "#8C8C8C",
    "shapley_share_bulk14C": "#9C4F2D",
    "shapley_share_fraction14C": "#3F7754",
    "shapley_share_resp14C": "#3A5A8C",
    "shapley_share_ER_annual": "#7A9E9F",
}


def _biome_group(biome: str) -> str:
    b = str(biome).lower()
    if any(k in b for k in ("arctic", "tundra", "permafrost")):
        return "arctic_permafrost"
    if "boreal" in b:
        return "boreal"
    if any(k in b for k in ("peatland", "moss")):
        return "peatland"
    if "tropical" in b:
        return "tropical"
    if any(k in b for k in ("grassland", "mollisol", "mediterranean", "cropland", "shrubland")):
        return "grassland_mediterranean"
    if "temperate" in b or "conifer" in b:
        return "temperate_forest"
    return "other"


def _load_new_site_tables(paths: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = load_table(path).copy()
        frame["source_file"] = Path(path).stem
        frames.append(frame)
    if not frames:
        raise ValueError("At least one new-site table is required.")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(
        ["site", "n_incubation", "J_final", "source_file"],
        ascending=[True, False, True, True],
    )
    return merged.drop_duplicates("site", keep="first").reset_index(drop=True)


def build_cross_ecosystem_tables(
    network_summary: str | pd.DataFrame,
    warming_summary: str | pd.DataFrame,
    new_sites: list[str] | tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    network = coerce_table(network_summary, "network_summary")
    warming = coerce_table(warming_summary, "warming_summary")
    assert network is not None
    assert warming is not None
    new_df = _load_new_site_tables(list(new_sites))

    warm_cols = [
        "site",
        "frac_c_loss",
        "abs_c_loss_gCm2",
        "delta_rh_annual_mean_gCm2yr",
        "old_fraction_of_excess_rh",
    ]
    direct = network.merge(warming[warm_cols], on="site", how="left", validate="one_to_one")
    direct["has_direct_warming"] = direct["frac_c_loss"].notna()
    direct["source_set"] = "network"
    direct["tau_active_yr"] = direct["tau_soil_active"]
    direct["tau_slow_yr"] = direct["tau_soil_slow"]
    direct["tau_passive_yr"] = direct["tau_soil_passive"]

    extra = new_df.loc[~new_df["site"].isin(direct["site"])].copy()
    extra["biome_group"] = extra["biome"].map(_biome_group)
    extra["has_direct_warming"] = False
    extra["source_set"] = "incubation_expansion"
    extra["dfs_total"] = np.nan
    extra["dominant_family"] = "incubation_only"
    extra["frac_c_loss"] = np.nan
    extra["abs_c_loss_gCm2"] = np.nan
    extra["delta_rh_annual_mean_gCm2yr"] = np.nan
    extra["old_fraction_of_excess_rh"] = np.nan
    extra["mean_GPP_gCm2yr"] = pd.to_numeric(extra["mean_GPP_gCm2yr"], errors="coerce")
    extra["SOC_gCm2"] = pd.to_numeric(extra["SOC_gCm2"], errors="coerce")

    all_cols = [
        "site",
        "label",
        "biome",
        "biome_group",
        "source_set",
        "has_direct_warming",
        "dfs_total",
        "dominant_family",
        "mean_GPP_gCm2yr",
        "SOC_gCm2",
        "tau_active_yr",
        "tau_slow_yr",
        "tau_passive_yr",
        "frac_c_loss",
        "abs_c_loss_gCm2",
        "delta_rh_annual_mean_gCm2yr",
        "old_fraction_of_excess_rh",
        "n_pool_blocks",
        "n_resp",
        "n_incubation",
    ]
    for col in all_cols:
        if col not in direct.columns:
            direct[col] = np.nan
        if col not in extra.columns:
            extra[col] = np.nan

    all_sites = pd.concat([direct[all_cols], extra[all_cols]], ignore_index=True)
    all_sites["biome_group"] = pd.Categorical(
        all_sites["biome_group"],
        categories=BIOME_GROUP_ORDER + ["other"],
        ordered=True,
    )
    all_sites = all_sites.sort_values(
        ["biome_group", "has_direct_warming", "tau_passive_yr", "tau_slow_yr", "site"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    biome_vulnerability = (
        direct.loc[direct["has_direct_warming"]]
        .groupby("biome_group", observed=True, as_index=False)
        .agg(
            n_sites=("site", "nunique"),
            dfs_mean=("dfs_total", "mean"),
            frac_loss_mean=("frac_c_loss", "mean"),
            frac_loss_median=("frac_c_loss", "median"),
            frac_loss_min=("frac_c_loss", "min"),
            frac_loss_max=("frac_c_loss", "max"),
            abs_loss_mean=("abs_c_loss_gCm2", "mean"),
            old_share_mean=("old_fraction_of_excess_rh", "mean"),
            old_share_median=("old_fraction_of_excess_rh", "median"),
        )
    )
    biome_vulnerability["biome_label"] = biome_vulnerability["biome_group"].map(BIOME_GROUP_LABELS)

    biome_shapley = (
        network.groupby("biome_group", observed=True, as_index=False)[FAMILY_ORDER]
        .mean()
        .sort_values("biome_group")
    )
    biome_shapley["biome_label"] = biome_shapley["biome_group"].map(BIOME_GROUP_LABELS)

    biome_coverage = (
        all_sites.groupby(["biome_group", "has_direct_warming"], observed=True)
        .size()
        .rename("n_sites")
        .reset_index()
    )
    biome_coverage["biome_label"] = biome_coverage["biome_group"].map(BIOME_GROUP_LABELS)

    return {
        "all_sites_union": all_sites,
        "direct_warming_sites": direct.loc[direct["has_direct_warming"]].copy(),
        "biome_vulnerability_summary": biome_vulnerability,
        "biome_shapley_summary": biome_shapley,
        "biome_coverage_summary": biome_coverage,
    }


def _annotate_points(ax, df: pd.DataFrame, xcol: str, ycol: str, sites: list[str]) -> None:
    sub = df[df["site"].isin(sites)]
    for row in sub.itertuples(index=False):
        ax.annotate(
            str(row.site),
            (getattr(row, xcol), getattr(row, ycol)),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
        )


def make_figure_09(
    network_summary: str | pd.DataFrame,
    warming_summary: str | pd.DataFrame,
    new_sites: list[str] | tuple[str, ...],
    output_dir: str = "outputs",
    config_path: str | None = None,
):
    setup_figure_config(config_path)
    tables = build_cross_ecosystem_tables(network_summary, warming_summary, new_sites)
    all_sites = tables["all_sites_union"]
    direct = tables["direct_warming_sites"]
    vuln = tables["biome_vulnerability_summary"]
    shapley = tables["biome_shapley_summary"]
    n_all_sites = all_sites["site"].nunique()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    panelize(axes)

    ax = axes[0, 0]
    for biome_group in BIOME_GROUP_ORDER:
        sub = all_sites[all_sites["biome_group"] == biome_group]
        if sub.empty:
            continue
        color = BIOME_GROUP_COLORS[biome_group]
        direct_sub = sub[sub["has_direct_warming"]]
        extra_sub = sub[~sub["has_direct_warming"]]
        if not direct_sub.empty:
            ax.scatter(
                direct_sub["tau_slow_yr"],
                direct_sub["tau_passive_yr"],
                s=70,
                color=color,
                alpha=0.9,
            )
        if not extra_sub.empty:
            ax.scatter(
                extra_sub["tau_slow_yr"],
                extra_sub["tau_passive_yr"],
                s=95,
                facecolors="none",
                edgecolors=color,
                linewidths=1.4,
                marker="^",
            )
    label_sites = [
        "Dinesen",
        "Treynor",
        "Trumbore Musick",
        "Trumbore Ahwahnee",
        "EML",
        "Adventdalen Valley",
        "CZ_1964burn_NSA",
        "Nahuelbuta",
    ]
    _annotate_points(ax, all_sites, "tau_slow_yr", "tau_passive_yr", label_sites)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Slow-pool turnover time (years)")
    ax.set_ylabel("Passive-pool turnover time (years)")
    ax.set_title(f"All {n_all_sites} ecosystems occupy a common multi-pool turnover space")
    biome_handles = [
        mpatches.Patch(color=BIOME_GROUP_COLORS[key], label=BIOME_GROUP_LABELS[key])
        for key in BIOME_GROUP_ORDER
        if (all_sites["biome_group"] == key).any()
    ]
    style_handles = [
        mlines.Line2D([], [], color="0.25", marker="o", linestyle="None", markersize=7, label="Direct warming output"),
        mlines.Line2D([], [], color="0.25", marker="^", markerfacecolor="white", linestyle="None", markersize=7, label="Turnover-only expansion"),
    ]
    legend1 = ax.legend(handles=biome_handles, loc="lower right", fontsize=8, title="Biome group")
    ax.add_artist(legend1)
    ax.legend(handles=style_handles, loc="upper left", fontsize=8)
    ax.grid(alpha=0.25, which="both")

    ax = axes[0, 1]
    max_abs = float(direct["abs_c_loss_gCm2"].max())
    sizes = 30 + 220 * np.power(direct["abs_c_loss_gCm2"] / max_abs, 0.8)
    for biome_group in BIOME_GROUP_ORDER:
        sub = direct[direct["biome_group"] == biome_group]
        if sub.empty:
            continue
        ax.scatter(
            sub["dfs_total"],
            sub["frac_c_loss"],
            s=sizes[sub.index],
            color=BIOME_GROUP_COLORS[biome_group],
            alpha=0.8,
            edgecolors="0.2",
            linewidths=0.5,
            label=BIOME_GROUP_LABELS[biome_group],
        )
    outlier_sites = list(
        pd.Index(direct.nlargest(4, "frac_c_loss")["site"]).union(direct.nlargest(4, "abs_c_loss_gCm2")["site"])
    )
    _annotate_points(ax, direct, "dfs_total", "frac_c_loss", outlier_sites)
    ax.set_xlabel("Constrainability (total DFS)")
    ax.set_ylabel("Fractional carbon loss under warming")
    ax.set_title("Vulnerability is not just a function of constrainability")
    ax.grid(alpha=0.25)
    size_levels = [1000.0, 3000.0, 6000.0]
    size_handles = [
        ax.scatter([], [], s=30 + 220 * np.power(level / max_abs, 0.8), color="white", edgecolors="0.2")
        for level in size_levels
    ]
    ax.legend(
        size_handles,
        [f"{int(level):,} gC m$^{{-2}}$ loss" for level in size_levels],
        fontsize=8,
        title="Absolute loss",
        loc="upper right",
    )

    ax = axes[1, 0]
    vuln = vuln.sort_values("frac_loss_mean", ascending=False).reset_index(drop=True)
    y = np.arange(len(vuln))
    for yi, row in enumerate(vuln.itertuples(index=False)):
        sub = direct[direct["biome_group"] == row.biome_group]
        jitter = np.linspace(-0.14, 0.14, max(len(sub), 1))
        ax.scatter(
            sub["frac_c_loss"],
            yi + jitter[: len(sub)],
            s=38,
            color=BIOME_GROUP_COLORS[str(row.biome_group)],
            alpha=0.6,
        )
        ax.scatter(
            row.frac_loss_mean,
            yi,
            s=95,
            color=BIOME_GROUP_COLORS[str(row.biome_group)],
            edgecolors="0.15",
            linewidths=0.7,
            marker="D",
        )
        ax.hlines(yi, row.frac_loss_min, row.frac_loss_max, color=BIOME_GROUP_COLORS[str(row.biome_group)], linewidth=1.4, alpha=0.8)
        ax.text(
            max(vuln["frac_loss_max"]) + 0.015,
            yi,
            f"old-RH share {row.old_share_mean:.2f}",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(vuln["biome_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Fractional carbon loss under warming")
    ax.set_title("Boreal is most fractionally vulnerable; permafrost has the oldest excess RH")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1, 1]
    shapley = shapley.sort_values(
        "biome_group",
        key=lambda s: pd.Categorical(s, categories=BIOME_GROUP_ORDER, ordered=True),
    ).reset_index(drop=True)
    y = np.arange(len(shapley))
    left = np.zeros(len(shapley))
    for family in FAMILY_ORDER:
        vals = shapley[family].to_numpy(dtype=float)
        ax.barh(
            y,
            vals,
            left=left,
            color=FAMILY_COLORS[family],
            label=FAMILY_LABELS[family],
            height=0.68,
        )
        left = left + vals
    ax.set_yticks(y)
    ax.set_yticklabels(shapley["biome_label"])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mean Shapley share of total DFS")
    ax.set_title("Constrainability shifts with observation family, not just ecosystem")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    alt_text = (
        f"Four-panel synthesis across {n_all_sites} site-level inversions, including {direct['site'].nunique()} sites with direct warming-response outputs and "
        "10 additional turnover-only expansion sites. Panel A places all ecosystems in slow- versus passive-pool turnover "
        "space and shows that additional incubation-constrained sites extend the old-carbon tail rather than defining a new "
        "turnover regime. Panel B shows that fractional warming loss varies widely at similar total degrees of freedom for "
        "signal, with marker size indicating absolute carbon loss. Panel C summarizes direct warming vulnerability by biome "
        "group, showing the highest fractional losses in boreal systems and the highest old-carbon contribution to excess "
        "respiration in arctic and permafrost systems. Panel D shows that the mean attribution of constrainability differs "
        "across biome groups, with low-information sites remaining stock-dominated while bulk and respired radiocarbon carry "
        "more of the information where richer data exist."
    )
    caption = (
        "Cross-ecosystem synthesis of 14C-constrained optimal-estimation inversions. (A) Slow- versus passive-pool turnover "
        f"times for all {n_all_sites} site-level inversions; filled circles indicate sites with direct warming-response diagnostics and "
        "open triangles indicate additional incubation/expansion inversions. (B) Fractional carbon loss under the standardized "
        "warming experiment versus total degrees of freedom for signal (DFS) for the 24 sites with direct warming outputs; "
        "marker size scales with absolute carbon loss. (C) Biome-group summary of warming vulnerability, with site-level values, "
        "within-biome ranges, and mean old-carbon share of excess heterotrophic respiration. (D) Mean Shapley attribution of "
        "DFS across observation families by biome group, highlighting that constrainability depends most strongly on the available "
        "observation geometry rather than on biome identity alone."
    )
    finalize_figure(
        fig,
        "figure_09",
        output_dir,
        tables,
        alt_text,
        "Figure 9",
        caption,
    )
    return fig, axes


def main() -> None:
    parser = standard_figure_parser(__doc__ or "Figure 9")
    parser.add_argument("--network-summary", required=True)
    parser.add_argument("--warming-summary", required=True)
    parser.add_argument("--new-sites", nargs="+", required=True)
    args = parser.parse_args()
    make_figure_09(
        args.network_summary,
        args.warming_summary,
        args.new_sites,
        output_dir=args.output_dir,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
