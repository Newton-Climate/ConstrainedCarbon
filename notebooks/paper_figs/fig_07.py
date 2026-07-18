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
        add_one_to_one,
        coerce_table,
        finalize_figure,
        panelize,
        setup_figure_config,
        standard_figure_parser,
        summarize_quantiles,
    )
else:
    from .utils import add_one_to_one, coerce_table, finalize_figure, panelize, setup_figure_config, standard_figure_parser, summarize_quantiles


def make_figure_07(
    cesm_comparison: str | pd.DataFrame,
    output_dir: str = "outputs",
    config_path: str | None = None,
):
    cfg = setup_figure_config(config_path)
    df = coerce_table(cesm_comparison, "cesm_comparison")
    assert df is not None
    required = {
        "ecosystem",
        "model_source",
        "draw_or_member",
        "mode",
        "turnover_time_years",
        "carbon_stock",
        "baseline_respiration",
        "warm_respiration",
        "cumulative_carbon_loss",
        "slow_carbon_fraction_of_excess_respiration",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CESM comparison table is missing required columns: {sorted(missing)}")

    ecosystems = [e for e in cfg.ecosystem_order if e in df["ecosystem"].unique()]
    if not ecosystems:
        ecosystems = list(df["ecosystem"].drop_duplicates())
    modes = [m for m in cfg.turnover_mode_labels if m in df["mode"].unique()]
    if not modes:
        modes = list(df["mode"].drop_duplicates())

    turn = summarize_quantiles(df, "turnover_time_years", ["ecosystem", "model_source", "mode"])
    agg = (
        df.groupby(["ecosystem", "model_source", "draw_or_member"], sort=False)[
            ["carbon_stock", "baseline_respiration", "cumulative_carbon_loss", "slow_carbon_fraction_of_excess_respiration"]
        ]
        .sum()
        .reset_index()
    )
    obs = agg[agg["model_source"] == "observation_constrained"].groupby("ecosystem").median(numeric_only=True).reset_index()
    cesm = agg[agg["model_source"] == "CESM"].groupby("ecosystem").median(numeric_only=True).reset_index()
    comp = obs.merge(cesm, on="ecosystem", suffixes=("_obscon", "_cesm"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panelize(axes)

    ax = axes[0, 0]
    y = 0
    yticks = []
    ylabels = []
    for eco in ecosystems:
        for mode in modes:
            for source, marker in [("observation_constrained", "o"), ("CESM", "s")]:
                sub = turn[(turn["ecosystem"] == eco) & (turn["mode"] == mode) & (turn["model_source"] == source)]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                ax.hlines(y, row["q05"], row["q95"], color="0.35", linewidth=1.2)
                ax.scatter(row["median"], y, marker=marker, s=34, facecolors="white", edgecolors="0.1")
                yticks.append(y)
                ylabels.append(f"{eco} | {mode} | {source}")
                y += 1
        y += 0.5
    ax.set_xscale("log")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Turnover time (years)")
    ax.set_title("Turnover-Time Spectrum")

    ax = axes[0, 1]
    ax.scatter(comp["carbon_stock_cesm"], comp["carbon_stock_obscon"], marker="o", s=50, label="Soil C stock")
    ax.scatter(comp["baseline_respiration_cesm"], comp["baseline_respiration_obscon"], marker="s", s=50, label="Baseline respiration")
    for row in comp.itertuples(index=False):
        ax.text(row.carbon_stock_cesm, row.carbon_stock_obscon, str(row.ecosystem), fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("CESM")
    ax.set_ylabel("Observation-constrained")
    ax.set_title("Present-Day Stocks and Fluxes")
    add_one_to_one(ax)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.scatter(comp["cumulative_carbon_loss_cesm"], comp["cumulative_carbon_loss_obscon"], marker="D", s=54, facecolors="white", edgecolors="0.1")
    for row in comp.itertuples(index=False):
        ax.text(row.cumulative_carbon_loss_cesm, row.cumulative_carbon_loss_obscon, str(row.ecosystem), fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("CESM cumulative carbon loss")
    ax.set_ylabel("Observation-constrained cumulative carbon loss")
    ax.set_title("Warming Response")
    add_one_to_one(ax)

    ax = axes[1, 1]
    ax.scatter(
        comp["slow_carbon_fraction_of_excess_respiration_cesm"],
        comp["slow_carbon_fraction_of_excess_respiration_obscon"],
        marker="^",
        s=54,
        facecolors="white",
        edgecolors="0.1",
    )
    for row in comp.itertuples(index=False):
        ax.text(
            row.slow_carbon_fraction_of_excess_respiration_cesm,
            row.slow_carbon_fraction_of_excess_respiration_obscon,
            str(row.ecosystem),
            fontsize=8,
            ha="left",
            va="bottom",
        )
    ax.set_xlabel("CESM slow/old fraction of excess respiration")
    ax.set_ylabel("Observation-constrained slow/old fraction")
    ax.set_title("Origin of Warming-Induced Respiration")
    add_one_to_one(ax)

    alt_text = (
        "Figure 7 is an illustrative CESM comparison. Panel A compares turnover-time spectra between the observation-constrained model "
        "and CESM, keeping the two sources visually distinct. Panel B compares present-day total stocks and baseline respiration on one-to-one "
        "axes. Panel C compares warming-response magnitudes, and Panel D compares the fraction of excess respiration arising from slow or old "
        "carbon. The principal conclusion is that similar present-day stocks or fluxes do not guarantee similar turnover structure or the same "
        "attribution of future carbon release."
    )
    caption = (
        "Illustrative CESM comparison. (A) Turnover-time spectrum for the observation-constrained model and CESM. (B) Present-day total soil-carbon "
        "stock and baseline respiration. (C) Warming-response magnitude under matched comparison assumptions. (D) Fraction of excess respiration "
        "originating from slow or old carbon. Agreement in present-day bulk metrics does not guarantee agreement in turnover structure or the "
        "origin of future carbon release."
    )
    finalize_figure(
        fig,
        "figure_07",
        output_dir,
        {
            "turnover_spectrum_summary": turn,
            "cesm_obscon_comparison": comp,
        },
        alt_text,
        "Figure 7",
        caption,
    )
    return fig, axes


def main() -> None:
    parser = standard_figure_parser(__doc__ or "Figure 7")
    parser.add_argument("--cesm-output", required=True)
    args = parser.parse_args()
    make_figure_07(args.cesm_output, output_dir=args.output_dir, config_path=args.config)


if __name__ == "__main__":
    main()
