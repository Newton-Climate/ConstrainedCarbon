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
    )
    from notebooks.paper_figs.validation import validate_information_table
else:
    from .utils import add_one_to_one, coerce_table, finalize_figure, panelize, setup_figure_config, standard_figure_parser
    from .validation import validate_information_table


def _subset_total(info: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return info.groupby(["ecosystem", "observation_subset"], sort=False)[value_col].sum().reset_index()


def _subset_ticklabels(df: pd.DataFrame, subset_order: list[str]) -> list[str]:
    if "observation_subset_label" not in df.columns:
        return subset_order
    labels = (
        df[["observation_subset", "observation_subset_label"]]
        .drop_duplicates()
        .set_index("observation_subset")["observation_subset_label"]
    )
    return [str(labels.get(subset, subset)) for subset in subset_order]


def _resolve_incremental_subsets(subsets: list[str]) -> tuple[str, str, str]:
    candidates = {
        "baseline": ["stocks_fluxes", "stocks_only", "stocks"],
        "soil": ["stocks_fluxes_soil14c", "stocks_soil14c", "soil14c_only"],
        "resp": ["stocks_fluxes_respired14c", "stocks_resp14c", "resp14c_only"],
    }
    resolved: dict[str, str] = {}
    for key, names in candidates.items():
        for name in names:
            if name in subsets:
                resolved[key] = name
                break
    missing = [key for key in ("baseline", "soil", "resp") if key not in resolved]
    if missing:
        raise ValueError(
            "Figure 4 could not identify baseline/soil14C/respired14C subsets from the "
            f"available subsets: {subsets}"
        )
    return resolved["baseline"], resolved["soil"], resolved["resp"]


def _derive_incremental(info: pd.DataFrame) -> pd.DataFrame:
    total = _subset_total(info, "information_gain_nats")
    available = list(total["observation_subset"].drop_duplicates())
    baseline_subset, soil_subset, resp_subset = _resolve_incremental_subsets(available)
    rows = []
    for eco, grp in total.groupby("ecosystem", sort=False):
        sub = grp.set_index("observation_subset")["information_gain_nats"]
        base = sub.get(baseline_subset, np.nan)
        soil = sub.get(soil_subset, np.nan)
        resp = sub.get(resp_subset, np.nan)
        rows.append(
            {
                "ecosystem": eco,
                "incremental_soil14c_nats": soil - base if np.isfinite(base) and np.isfinite(soil) else np.nan,
                "incremental_respired14c_nats": resp - base if np.isfinite(base) and np.isfinite(resp) else np.nan,
                "baseline_subset": baseline_subset,
                "soil_subset": soil_subset,
                "respired_subset": resp_subset,
            }
        )
    return pd.DataFrame(rows)


def make_figure_04(
    information_metrics: str | pd.DataFrame,
    averaging_kernel_matrix: str | pd.DataFrame | None = None,
    output_dir: str = "outputs",
    config_path: str | None = None,
):
    cfg = setup_figure_config(config_path)
    info = coerce_table(information_metrics, "information_metrics")
    kernel = coerce_table(averaging_kernel_matrix, "averaging_kernel_matrix")
    assert info is not None
    validate_information_table(info)

    ecosystems = [e for e in cfg.ecosystem_order if e in info["ecosystem"].unique()]
    if not ecosystems:
        ecosystems = list(info["ecosystem"].drop_duplicates())
    subset_order = [s for s in cfg.observation_subset_order if s in info["observation_subset"].unique()]
    if not subset_order:
        subset_order = list(info["observation_subset"].drop_duplicates())
    subset_ticks = _subset_ticklabels(info, subset_order)
    mode_order = [m for m in cfg.turnover_mode_labels if m in info["mode"].unique()]
    if not mode_order:
        mode_order = list(info["mode"].drop_duplicates())

    dfs_df = _subset_total(info, "degrees_of_freedom")
    unc_df = info.copy()
    unc_df["posterior_log10_sd"] = np.log10(np.maximum(unc_df["posterior_sd"].to_numpy(dtype=float), 1e-12))
    incr_df = _derive_incremental(info)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    panelize(axes)

    ax = axes[0, 0]
    for eco in ecosystems:
        sub = dfs_df[dfs_df["ecosystem"] == eco].set_index("observation_subset").reindex(subset_order)
        ax.plot(np.arange(len(subset_order)), sub["degrees_of_freedom"], marker="o", label=eco)
    ax.set_xticks(np.arange(len(subset_order)))
    ax.set_xticklabels(subset_ticks, rotation=30, ha="right")
    ax.set_ylabel("Degrees of freedom for signal")
    ax.set_title("Degrees of Freedom by Observation Subset")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    if kernel is None:
        ax.text(0.5, 0.5, "No averaging-kernel matrix supplied.\nPanel intentionally omitted.", ha="center", va="center")
        ax.set_axis_off()
    else:
        require = {"site", "param_name", "param_index"} <= set(kernel.columns)
        if not require:
            raise ValueError("averaging-kernel matrix table must contain at least site, param_name, and param_index columns.")
        site = ecosystems[0]
        sub = kernel[kernel["site"] == site]
        val_cols = [c for c in sub.columns if c.startswith("log_tau") or c.startswith("log_f_transfer")]
        mat = sub[val_cols].to_numpy(dtype=float)[: min(5, len(sub)), : min(5, len(val_cols))]
        im = ax.imshow(mat, cmap="BrBG", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_xticklabels([str(c) for c in val_cols[: mat.shape[1]]], rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(list(sub["param_name"].iloc[: mat.shape[0]]), fontsize=8)
        ax.set_title(f"Averaging Kernels ({site})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    for eco in ecosystems:
        sub = unc_df[unc_df["ecosystem"] == eco]
        for mode in mode_order:
            sm = sub[sub["mode"] == mode].set_index("observation_subset").reindex(subset_order)
            ax.plot(np.arange(len(subset_order)), sm["posterior_log10_sd"], marker="o", label=f"{eco} | {mode}")
    ax.set_xticks(np.arange(len(subset_order)))
    ax.set_xticklabels(subset_ticks, rotation=30, ha="right")
    ax.set_ylabel("log10 posterior SD")
    ax.set_title("Posterior Uncertainty by Turnover Mode")

    ax = axes[1, 1]
    ax.scatter(incr_df["incremental_respired14c_nats"], incr_df["incremental_soil14c_nats"], marker="D", s=60, facecolors="white", edgecolors="0.1")
    for row in incr_df.itertuples(index=False):
        ax.text(row.incremental_respired14c_nats, row.incremental_soil14c_nats, str(row.ecosystem), fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("Incremental info gain from respired 14C (nats)")
    ax.set_ylabel("Incremental info gain from soil 14C (nats)")
    ax.set_title("Complementarity or Redundancy")
    add_one_to_one(ax)
    baseline_subset = str(incr_df["baseline_subset"].iloc[0]) if not incr_df.empty else "baseline"

    alt_text = (
        "Figure 4 summarizes observation-specific constraints on turnover. Panel A shows degrees of freedom for signal by ecosystem and "
        "observation subset, ordered from less complete to more complete subsets. Panel B shows averaging kernels when a matrix table is "
        "supplied and otherwise explicitly omits the panel. Panel C shows posterior uncertainty by turnover mode on a log-transformed "
        "uncertainty scale. Panel D compares the incremental information gained from adding soil radiocarbon versus respired radiocarbon "
        f"relative to the {baseline_subset} baseline subset. The principal conclusion is that different observation types constrain different parts "
        "of the turnover spectrum and can be complementary rather than redundant."
    )
    caption = (
        "Observation-specific constraints on turnover. (A) Degrees of freedom for signal by ecosystem and observation subset. "
        "(B) Averaging-kernel matrix display when the full matrix is supplied. (C) Posterior uncertainty by turnover mode, shown here as "
        "the logarithm of posterior standard deviation. (D) Incremental information gain from adding soil radiocarbon or respired radiocarbon "
        f"relative to the same {baseline_subset} baseline subset."
    )
    finalize_figure(
        fig,
        "figure_04",
        output_dir,
        {
            "dfs_by_subset": dfs_df,
            "posterior_uncertainty_by_mode": unc_df,
            "incremental_information_gain": incr_df,
        },
        alt_text,
        "Figure 4",
        caption,
    )
    return fig, axes


def main() -> None:
    parser = standard_figure_parser(__doc__ or "Figure 4")
    parser.add_argument("--information-metrics", required=True)
    parser.add_argument("--averaging-kernel-matrix", default=None)
    args = parser.parse_args()
    make_figure_04(
        args.information_metrics,
        averaging_kernel_matrix=args.averaging_kernel_matrix,
        output_dir=args.output_dir,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
