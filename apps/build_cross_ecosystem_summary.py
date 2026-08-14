#!/usr/bin/env python3
"""Build a cross-ecosystem summary figure and markdown report from current inversion outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from notebooks.paper_figs.fig_09 import BIOME_GROUP_LABELS, build_cross_ecosystem_tables, make_figure_09  # noqa: E402
from notebooks.paper_figs.utils import close_or_show  # noqa: E402


def _default_paths() -> tuple[Path, Path, list[Path]]:
    exports = _REPO_ROOT / "notebooks" / "exports"
    return (
        exports / "network_inversion_fluxcom_er_20260719" / "site_summary.csv",
        exports / "warming_vulnerability_fluxcom_er_20260719" / "site_warming_summary.csv",
        [
            exports / "new_sites_incubation_20260719.csv",
            exports / "incubation_new_sites_runnable_20260719.csv",
        ],
    )


def _site_table_markdown(df: pd.DataFrame, cols: list[str]) -> str:
    table = df.loc[:, cols].copy()
    return table.to_markdown(index=False)


def build_report_text(
    figure_path: Path,
    tables: dict[str, pd.DataFrame],
    network_path: Path,
    warming_path: Path,
    new_site_paths: list[Path],
) -> str:
    all_sites = tables["all_sites_union"]
    direct = tables["direct_warming_sites"]
    vuln = tables["biome_vulnerability_summary"].sort_values("frac_loss_mean", ascending=False)
    shapley = tables["biome_shapley_summary"].copy()
    network_display = network_path.relative_to(_REPO_ROOT).as_posix()
    warming_display = warming_path.relative_to(_REPO_ROOT).as_posix()
    new_site_display = ", ".join(f"`{p.relative_to(_REPO_ROOT).as_posix()}`" for p in new_site_paths)

    dfs = direct["dfs_total"]
    incubation_only = all_sites.loc[~all_sites["has_direct_warming"]].copy()
    top_constrained = direct.nlargest(6, "dfs_total")[
        ["site", "biome_group", "dfs_total", "dominant_family"]
    ].copy()
    top_constrained["biome_group"] = top_constrained["biome_group"].map(BIOME_GROUP_LABELS)
    top_constrained["dfs_total"] = top_constrained["dfs_total"].map(lambda x: f"{x:.2f}")

    top_vulnerable = direct.nlargest(6, "frac_c_loss")[
        ["site", "biome_group", "frac_c_loss", "abs_c_loss_gCm2", "old_fraction_of_excess_rh"]
    ].copy()
    top_vulnerable["biome_group"] = top_vulnerable["biome_group"].map(BIOME_GROUP_LABELS)
    top_vulnerable["frac_c_loss"] = top_vulnerable["frac_c_loss"].map(lambda x: f"{x:.3f}")
    top_vulnerable["abs_c_loss_gCm2"] = top_vulnerable["abs_c_loss_gCm2"].map(lambda x: f"{x:,.0f}")
    top_vulnerable["old_fraction_of_excess_rh"] = top_vulnerable["old_fraction_of_excess_rh"].map(lambda x: f"{x:.2f}")

    incubation_tail = incubation_only.nlargest(6, "tau_passive_yr")[
        ["site", "biome_group", "tau_slow_yr", "tau_passive_yr", "n_incubation"]
    ].copy()
    incubation_tail["biome_group"] = incubation_tail["biome_group"].map(BIOME_GROUP_LABELS)
    incubation_tail["tau_slow_yr"] = incubation_tail["tau_slow_yr"].map(lambda x: f"{x:.1f}")
    incubation_tail["tau_passive_yr"] = incubation_tail["tau_passive_yr"].map(lambda x: f"{x:.1f}")
    incubation_tail["n_incubation"] = incubation_tail["n_incubation"].fillna(0).astype(int)

    shapley["dominant_constraint"] = shapley[
        [
            "shapley_share_cstocks",
            "shapley_share_bulk14C",
            "shapley_share_fraction14C",
            "shapley_share_resp14C",
            "shapley_share_ER_annual",
        ]
    ].idxmax(axis=1)
    shapley["dominant_constraint"] = shapley["dominant_constraint"].map(
        {
            "shapley_share_cstocks": "C stocks",
            "shapley_share_bulk14C": "Bulk 14C",
            "shapley_share_fraction14C": "Fraction 14C",
            "shapley_share_resp14C": "Respired 14C",
            "shapley_share_ER_annual": "Annual ER",
        }
    )
    shapley["biome_group"] = shapley["biome_group"].map(BIOME_GROUP_LABELS)
    shapley["bulk+respired_share"] = (shapley["shapley_share_bulk14C"] + shapley["shapley_share_resp14C"]).map(
        lambda x: f"{x:.2f}"
    )
    shapley["stock_share"] = shapley["shapley_share_cstocks"].map(lambda x: f"{x:.2f}")
    shapley = shapley[["biome_group", "dominant_constraint", "stock_share", "bulk+respired_share"]]

    old_share_line = ", ".join(
        f"{row.biome_label}: {row.old_share_mean:.2f}"
        for row in vuln.sort_values("old_share_mean", ascending=False).itertuples(index=False)
    )

    return f"""# Cross-Ecosystem 14C Inversion Summary

Figure: ![Cross-ecosystem summary](../paper_figs/outputs/cross_ecosystem_summary/figures/figure_09.png)

## Scope

- Total unique inverted site-ecosystems: **{all_sites['site'].nunique()}**
- Sites with direct warming-vulnerability outputs: **{direct['site'].nunique()}**
- Additional turnover-only expansion sites: **{incubation_only['site'].nunique()}**
- Direct-warming DFS range: **{dfs.min():.2f} to {dfs.max():.2f}** with median **{dfs.median():.2f}**
- Source tables: `{network_display}`, `{warming_display}`, {new_site_display}

## Main Results

- The full union sits in a common three-pool turnover regime: active turnover remains near 2 years, while slow and passive pools span the ecological gradient.
- Warming vulnerability is strongest in **boreal systems fractionally** and in **arctic/permafrost systems for old-carbon release and absolute loss**.
- Constrainability is driven primarily by observation family rather than biome identity. Bulk 14C, respired 14C, and annual ER carry most of the leverage where they exist.
- The added incubation-expansion sites mostly extend the **old-C tail** rather than creating a new turnover cluster.
- Mean old-carbon share of excess RH by biome group: {old_share_line}.

## Biome-Level Vulnerability

{_site_table_markdown(
    vuln.assign(
        biome_label=vuln["biome_label"],
        frac_loss_mean=vuln["frac_loss_mean"].map(lambda x: f"{x:.3f}"),
        abs_loss_mean=vuln["abs_loss_mean"].map(lambda x: f"{x:,.0f}"),
        old_share_mean=vuln["old_share_mean"].map(lambda x: f"{x:.2f}"),
        dfs_mean=vuln["dfs_mean"].map(lambda x: f"{x:.2f}"),
    )[["biome_label", "n_sites", "dfs_mean", "frac_loss_mean", "abs_loss_mean", "old_share_mean"]],
    ["biome_label", "n_sites", "dfs_mean", "frac_loss_mean", "abs_loss_mean", "old_share_mean"],
)}

## Most Constrained Direct-Warming Sites

{_site_table_markdown(top_constrained, ["site", "biome_group", "dfs_total", "dominant_family"])}

## Most Vulnerable Direct-Warming Sites

{_site_table_markdown(top_vulnerable, ["site", "biome_group", "frac_c_loss", "abs_c_loss_gCm2", "old_fraction_of_excess_rh"])}

## Long-Tail Expansion Sites

These sites do not yet have direct warming projections in this summary, but they most strongly extend the old-carbon turnover tail.

{_site_table_markdown(incubation_tail, ["site", "biome_group", "tau_slow_yr", "tau_passive_yr", "n_incubation"])}

## Constraint Structure by Biome

{_site_table_markdown(shapley, ["biome_group", "dominant_constraint", "stock_share", "bulk+respired_share"])}

## Interpretation

- This ensemble is large enough to show that old-carbon vulnerability is not restricted to permafrost. Boreal, temperate, and tropical systems all mobilize meaningfully old carbon once warming increases RH.
- Several weakly constrained sites remain stock-dominated, so the main limit on inference is still missing radiocarbon geometry, not ecosystem diversity.
- The expansion sites make the passive-pool tail broader and older, which increases confidence that the long-lived tail is a general ecosystem property rather than a four-site artifact.
- The strongest next step is to run standardized warming projections for the turnover-only expansion sites with the largest passive tails, especially `Dinesen`, `Treynor`, `Trumbore Ahwahnee`, `Trumbore Musick`, and `Nahuelbuta`.
"""


def parse_args() -> argparse.Namespace:
    network_path, warming_path, new_site_paths = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-summary", default=str(network_path))
    parser.add_argument("--warming-summary", default=str(warming_path))
    parser.add_argument("--new-sites", nargs="+", default=[str(p) for p in new_site_paths])
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "notebooks" / "paper_figs" / "outputs" / "cross_ecosystem_summary"),
    )
    parser.add_argument(
        "--markdown-out",
        default=str(_REPO_ROOT / "notebooks" / "exports" / "cross_ecosystem_summary_20260719.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, _ = make_figure_09(
        args.network_summary,
        args.warming_summary,
        args.new_sites,
        output_dir=str(output_dir),
    )
    close_or_show(fig, show=False)

    tables = build_cross_ecosystem_tables(
        args.network_summary,
        args.warming_summary,
        args.new_sites,
    )
    figure_path = output_dir / "figures" / "figure_09.png"
    report = build_report_text(
        figure_path,
        tables,
        Path(args.network_summary),
        Path(args.warming_summary),
        [Path(p) for p in args.new_sites],
    )
    markdown_path = Path(args.markdown_out)
    markdown_path.write_text(report.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {figure_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
