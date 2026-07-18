from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from notebooks.paper_figs.utils import (
        coerce_table,
        finalize_figure,
        panelize,
        select_observation_subset,
        setup_figure_config,
        standard_figure_parser,
        summarize_quantiles,
    )
    from notebooks.paper_figs.validation import validate_information_table, validate_posterior_table
else:
    from .utils import (
        coerce_table,
        finalize_figure,
        panelize,
        select_observation_subset,
        setup_figure_config,
        standard_figure_parser,
        summarize_quantiles,
    )
    from .validation import validate_information_table, validate_posterior_table


def _resolved_modes(info: pd.DataFrame) -> pd.DataFrame:
    df = (
        info.groupby("ecosystem", sort=False)["averaging_kernel_diagonal"]
        .sum()
        .rename("effective_resolved_modes")
        .reset_index()
    )
    return df


def make_figure_03(
    posterior: str | pd.DataFrame,
    information_metrics: str | pd.DataFrame,
    topology_comparison: str | pd.DataFrame | None = None,
    output_dir: str = "outputs",
    config_path: str | None = None,
    summary_subset: str | None = None,
):
    cfg = setup_figure_config(config_path)
    posterior_df = coerce_table(posterior, "posterior")
    info_df = coerce_table(information_metrics, "information_metrics")
    topo_df = coerce_table(topology_comparison, "topology_comparison")
    assert posterior_df is not None and info_df is not None
    validate_posterior_table(posterior_df)
    validate_information_table(info_df)
    subset = select_observation_subset(posterior_df, cfg, summary_subset, label="posterior table")
    posterior_df = posterior_df[posterior_df["observation_subset"] == subset].copy()
    info_df = info_df[info_df["observation_subset"] == subset].copy()
    if info_df.empty:
        raise ValueError(
            f"information_metrics does not contain observation subset {subset!r}, "
            "which was selected from the posterior table."
        )

    ecosystems = [e for e in cfg.ecosystem_order if e in posterior_df["ecosystem"].unique()]
    if not ecosystems:
        ecosystems = list(posterior_df["ecosystem"].drop_duplicates())
    modes = [m for m in cfg.turnover_mode_labels if m in posterior_df["mode"].unique()]
    if not modes:
        modes = list(posterior_df["mode"].drop_duplicates())

    posterior_summary = summarize_quantiles(posterior_df, "turnover_time_years", ["ecosystem", "mode"])
    resolved_df = _resolved_modes(info_df)
    boundary_rows = []
    for _, row in posterior_summary.iterrows():
        prior_median = np.nan
        prior_lo = np.nan
        prior_hi = np.nan
        boundary = False
        if {"prior_lower_bound", "prior_upper_bound"} <= set(posterior_df.columns):
            mask = (
                (posterior_df["ecosystem"] == row["ecosystem"])
                & (posterior_df["mode"] == row["mode"])
            )
            sub = posterior_df.loc[mask]
            lower = float(sub["prior_lower_bound"].iloc[0])
            upper = float(sub["prior_upper_bound"].iloc[0])
            boundary = bool((row["q05"] <= lower) or (row["q95"] >= upper))
            prior_lo = lower
            prior_hi = upper
        boundary_rows.append(
            {
                "ecosystem": row["ecosystem"],
                "mode": row["mode"],
                "posterior_median_years": row["median"],
                "posterior_q05_years": row["q05"],
                "posterior_q95_years": row["q95"],
                "prior_median_years": prior_median,
                "prior_q05_years": prior_lo,
                "prior_q95_years": prior_hi,
                "boundary_contact_indicator": boundary,
            }
        )
    diag_df = pd.DataFrame(boundary_rows)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), gridspec_kw={"width_ratios": [1.6, 0.8, 1.1]})
    panelize(axes)

    ax = axes[0]
    ypos = []
    labels = []
    y = 0
    for eco in ecosystems:
        for mode in modes:
            sub = posterior_summary[(posterior_summary["ecosystem"] == eco) & (posterior_summary["mode"] == mode)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            ax.hlines(y, row["q05"], row["q95"], color="0.35", linewidth=1.2)
            ax.hlines(y, row["q25"], row["q75"], color="0.15", linewidth=3.2)
            ax.scatter(row["median"], y, marker="o", s=34, color="white", edgecolors="0.1", zorder=3)
            ypos.append(y)
            labels.append(f"{eco} | {mode}")
            y += 1
        y += 0.4
    ax.set_xscale("log")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Turnover time (years)")
    ax.set_title("Posterior Turnover-Time Distributions")

    ax = axes[1]
    rd = resolved_df.set_index("ecosystem").reindex(ecosystems).dropna().reset_index()
    ax.scatter(rd["effective_resolved_modes"], np.arange(len(rd)), marker="D", s=55, facecolors="white", edgecolors="0.1")
    ax.set_yticks(np.arange(len(rd)))
    ax.set_yticklabels(rd["ecosystem"])
    ax.set_xlabel("Effective resolved modes\n(sum of averaging-kernel diagonal)")
    ax.set_title("Resolved Turnover Modes")
    ax.invert_yaxis()

    ax = axes[2]
    if topo_df is None:
        ax.text(
            0.5,
            0.5,
            "No topology-comparison table supplied.\nPanel intentionally omitted.",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
    else:
        require = {"ecosystem", "topology_label", "score_difference"}
        missing = require - set(topo_df.columns)
        if missing:
            raise ValueError(f"topology comparison table is missing columns: {sorted(missing)}")
        for eco in ecosystems:
            sub = topo_df[topo_df["ecosystem"] == eco]
            ax.plot(sub["score_difference"], sub["topology_label"], marker="o", label=eco)
        ax.axvline(0, color="0.55", linestyle=":")
        ax.set_xlabel("Relative score difference")
        ax.set_title("Topology Support")
        ax.legend(fontsize=8)

    alt_text = (
        f"Figure 3 summarizes posterior turnover structure and effective model complexity for the {subset} observation subset. Panel A shows median, 50 percent, "
        "and 95 percent credible intervals for turnover time on a logarithmic axis by ecosystem and turnover mode. Panel B "
        "shows the effective number of resolved modes, defined here as the sum of averaging-kernel diagonal values across the "
        "mode set. Panel C either shows topology-comparison support when supplied or explicitly reports that topology support "
        "is unavailable rather than inventing values. The principal conclusion is that available observations resolve only a "
        "limited number of effective turnover modes and the resolved complexity differs among ecosystems."
    )
    caption = (
        f"Posterior turnover structure and effective model complexity for the {subset} observation subset. (A) Posterior turnover-time distributions summarized by "
        "the median, 50% credible interval, and 95% credible interval on a logarithmic axis. (B) Effective number of resolved "
        "turnover modes, defined here as the sum of averaging-kernel diagonal values across the turnover-mode block. "
        "(C) Relative support for alternative model topologies when a topology-comparison table is supplied; otherwise the "
        "panel is intentionally omitted. Boundary-contact indicators are exported in the diagnostic CSV."
    )
    finalize_figure(
        fig,
        "figure_03",
        output_dir,
        {
            "posterior_turnover_summary": posterior_summary,
            "resolved_modes_summary": resolved_df,
            "turnover_diagnostics": diag_df,
            "selected_subset": pd.DataFrame([{"observation_subset": subset}]),
        },
        alt_text,
        "Figure 3",
        caption,
    )
    return fig, axes


def main() -> None:
    parser = standard_figure_parser(__doc__ or "Figure 3")
    parser.add_argument("--posterior", required=True)
    parser.add_argument("--information-metrics", required=True)
    parser.add_argument("--topology-comparison", default=None)
    parser.add_argument("--summary-subset", default=None)
    args = parser.parse_args()
    make_figure_03(
        args.posterior,
        args.information_metrics,
        topology_comparison=args.topology_comparison,
        output_dir=args.output_dir,
        config_path=args.config,
        summary_subset=args.summary_subset,
    )


if __name__ == "__main__":
    main()
