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
    )
    from notebooks.paper_figs.validation import (
        validate_information_table,
        validate_posterior_table,
        validate_warming_table,
    )
else:
    from .utils import (
        coerce_table,
        finalize_figure,
        panelize,
        select_observation_subset,
        setup_figure_config,
        standard_figure_parser,
    )
    from .validation import validate_information_table, validate_posterior_table, validate_warming_table


def _build_offsets(obs: pd.DataFrame) -> pd.DataFrame:
    if {"ecosystem", "observation_type", "delta14c_permil"} - set(obs.columns):
        raise ValueError("Figure 5 requires ecosystem, observation_type, and delta14c_permil columns in the observation table.")
    bulk = obs[obs["observation_type"] == "bulk_soil"].groupby("ecosystem")["delta14c_permil"].mean().rename("stored_delta14c")
    resp = obs[obs["observation_type"] == "respired_carbon"].groupby("ecosystem")["delta14c_permil"].mean().rename("respired_delta14c")
    df = pd.concat([bulk, resp], axis=1).dropna().reset_index()
    df["delta14c_offset"] = df["respired_delta14c"] - df["stored_delta14c"]
    return df


def make_figure_05(
    observations: str | pd.DataFrame,
    posterior: str | pd.DataFrame,
    information_metrics: str | pd.DataFrame,
    warming_output: str | pd.DataFrame,
    output_dir: str = "outputs",
    config_path: str | None = None,
    summary_subset: str | None = None,
):
    cfg = setup_figure_config(config_path)
    obs = coerce_table(observations, "observations")
    posterior_df = coerce_table(posterior, "posterior")
    info = coerce_table(information_metrics, "information_metrics")
    warm = coerce_table(warming_output, "warming_output")
    assert obs is not None and posterior_df is not None and info is not None and warm is not None
    validate_posterior_table(posterior_df)
    validate_information_table(info)
    validate_warming_table(warm)
    subset = select_observation_subset(posterior_df, cfg, summary_subset, label="posterior table")
    posterior_df = posterior_df[posterior_df["observation_subset"] == subset].copy()
    info = info[info["observation_subset"] == subset].copy()
    warm = warm[warm["observation_subset"] == subset].copy()
    if info.empty or warm.empty:
        raise ValueError(
            f"Figure 5 selected observation subset {subset!r}, but one or more required tables do not contain it."
        )

    offsets = _build_offsets(obs)
    slow_obs = (
        info[info["mode"] == "slow"]
        .groupby("ecosystem", sort=False)["averaging_kernel_diagonal"]
        .mean()
        .rename("slow_mode_averaging_kernel_diagonal")
        .reset_index()
    )
    slow_frac = (
        posterior_df[posterior_df["mode"] == "slow"]
        .groupby("ecosystem", sort=False)["respiration_fraction"]
        .mean()
        .rename("slow_mode_respiration_fraction")
        .reset_index()
    )
    posterior_df["log10_turnover"] = np.log10(posterior_df["turnover_time_years"].to_numpy(dtype=float))
    slow_unc = (
        posterior_df[posterior_df["mode"] == "slow"]
        .groupby("ecosystem", sort=False)["log10_turnover"]
        .std()
        .rename("slow_mode_log10_turnover_sd")
        .reset_index()
    )
    horizon = int(warm["year"].max())
    warm_h = warm[warm["year"] == horizon].copy()
    warm_h["carbon_loss"] = warm_h["control_carbon"] - warm_h["warm_carbon"]
    warm_h = warm_h[warm_h["mode"] == "slow"]
    warm_unc = (
        warm_h.groupby("ecosystem", sort=False)["carbon_loss"]
        .agg(lambda x: float(np.quantile(x, 0.95) - np.quantile(x, 0.05)))
        .rename("slow_mode_warming_ci_width")
        .reset_index()
    )

    merged = offsets.merge(slow_obs, on="ecosystem").merge(slow_frac, on="ecosystem").merge(slow_unc, on="ecosystem").merge(warm_unc, on="ecosystem")
    ecosystems = [e for e in cfg.ecosystem_order if e in merged["ecosystem"].unique()]
    if not ecosystems:
        ecosystems = list(merged["ecosystem"].drop_duplicates())
    merged = merged.set_index("ecosystem").loc[ecosystems].reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panelize(axes)
    ycols = [
        ("slow_mode_averaging_kernel_diagonal", "Slow-mode observability"),
        ("slow_mode_respiration_fraction", "Slow-mode respiration fraction"),
        ("slow_mode_log10_turnover_sd", "SD of log10 slow turnover"),
        ("slow_mode_warming_ci_width", "95% CI width of slow-mode warming loss"),
    ]
    for ax, (col, ylabel) in zip(np.ravel(axes), ycols):
        ax.scatter(merged["delta14c_offset"], merged[col], marker="D", s=60, facecolors="white", edgecolors="0.1")
        for row in merged.itertuples(index=False):
            ax.text(row.delta14c_offset, getattr(row, col), str(row.ecosystem), fontsize=8, ha="left", va="bottom")
        ax.axvline(0, color="0.55", linestyle=":")
        ax.set_xlabel("Respired Δ$^{14}$C - stored Δ$^{14}$C (‰)")
        ax.set_ylabel(ylabel)
    axes[0, 0].set_title("Offset vs Slow-Mode Observability")
    axes[0, 1].set_title("Offset vs Slow-Mode Respiration Share")
    axes[1, 0].set_title("Offset vs Slow-Mode Turnover Uncertainty")
    axes[1, 1].set_title("Offset vs Warming Uncertainty")

    alt_text = (
        f"Figure 5 relates the stored-respired radiocarbon offset to slow-mode observability, slow-mode respiration share, "
        "slow-mode turnover uncertainty, and slow-mode warming-loss uncertainty. Each point is an ecosystem labeled directly. "
        f"Slow-mode summaries are taken from the {subset} observation subset. "
        "The principal conclusion is that ecosystems with larger separation between stored and respired radiocarbon can also show "
        "weaker present-day observability of old or slow carbon, but the script leaves the interpretation numerical rather than "
        "asserting a universal age conversion from the offset."
    )
    caption = (
        "Radiocarbon offset and old-carbon observability. The radiocarbon offset is defined as respired Δ14C minus stored Δ14C. "
        "Panels relate that offset to slow-mode observability, the baseline fraction of respiration from the slow mode, uncertainty "
        f"in slow-mode turnover time, and uncertainty in slow-mode warming loss for the {subset} observation subset. Ecosystems are labeled directly."
    )
    finalize_figure(
        fig,
        "figure_05",
        output_dir,
        {
            "radiocarbon_offset_observability": merged,
            "selected_subset": pd.DataFrame([{"observation_subset": subset}]),
        },
        alt_text,
        "Figure 5",
        caption,
    )
    return fig, axes


def main() -> None:
    parser = standard_figure_parser(__doc__ or "Figure 5")
    parser.add_argument("--observations", required=True)
    parser.add_argument("--posterior", required=True)
    parser.add_argument("--information-metrics", required=True)
    parser.add_argument("--warming-output", required=True)
    parser.add_argument("--summary-subset", default=None)
    args = parser.parse_args()
    make_figure_05(
        args.observations,
        args.posterior,
        args.information_metrics,
        args.warming_output,
        output_dir=args.output_dir,
        config_path=args.config,
        summary_subset=args.summary_subset,
    )


if __name__ == "__main__":
    main()
