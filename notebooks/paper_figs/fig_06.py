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
    from notebooks.paper_figs.validation import validate_warming_table
else:
    from .utils import add_one_to_one, coerce_table, finalize_figure, panelize, setup_figure_config, standard_figure_parser, summarize_quantiles
    from .validation import validate_warming_table


def _derive_warming_metrics(warm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = warm.copy()
    df["carbon_loss"] = df["control_carbon"] - df["warm_carbon"]
    df["excess_respiration"] = df["warm_respiration"] - df["control_respiration"]
    total = (
        df.groupby(["ecosystem", "observation_subset", "draw", "year"], sort=False)[["carbon_loss", "excess_respiration"]]
        .sum()
        .reset_index()
    )
    slow = (
        df[df["mode"] == "slow"]
        .groupby(["ecosystem", "observation_subset", "draw", "year"], sort=False)["excess_respiration"]
        .sum()
        .rename("slow_excess_respiration")
        .reset_index()
    )
    merged = total.merge(slow, on=["ecosystem", "observation_subset", "draw", "year"], how="left")
    merged["slow_fraction_of_excess_respiration"] = merged["slow_excess_respiration"] / merged["excess_respiration"].replace(0, np.nan)
    return df, merged


def _reduction(sd: float, base: float) -> float:
    """Fractional posterior-uncertainty reduction of a subset relative to baseline."""
    return 1 - sd / base if np.isfinite(base) and np.isfinite(sd) and base > 0 else np.nan


def _subset_value(summary: pd.DataFrame, eco: str, subset: str) -> float:
    sub = summary[(summary["ecosystem"] == eco) & (summary["observation_subset"] == subset)]
    if sub.empty:
        return np.nan
    return float(sub["sd"].iloc[0])


def _subset_ticklabels(df: pd.DataFrame, subset_order: list[str]) -> list[str]:
    if "observation_subset_label" not in df.columns:
        return subset_order
    labels = (
        df[["observation_subset", "observation_subset_label"]]
        .drop_duplicates()
        .set_index("observation_subset")["observation_subset_label"]
    )
    return [str(labels.get(subset, subset)) for subset in subset_order]


def _resolve_value_subsets(subsets: list[str]) -> tuple[str, str, str, str]:
    candidates = {
        "baseline": ["stocks_fluxes", "stocks_only", "stocks"],
        "soil": ["stocks_fluxes_soil14c", "stocks_soil14c", "soil14c_only"],
        "resp": ["stocks_fluxes_respired14c", "stocks_resp14c", "resp14c_only"],
        "combined": ["all_observations", "stocks_fluxes_soil_respired14c", "stocks_soil_resp14c"],
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
            "Figure 6 could not identify baseline/soil14C/respired14C subsets from the "
            f"available subsets: {subsets}"
        )
    # combined is optional — fall back to the soil subset (no extra panel encoding) if absent.
    resolved.setdefault("combined", resolved["soil"])
    return resolved["baseline"], resolved["soil"], resolved["resp"], resolved["combined"]


def make_figure_06(
    warming_output: str | pd.DataFrame,
    output_dir: str = "outputs",
    config_path: str | None = None,
    horizon_year: int | None = None,
):
    cfg = setup_figure_config(config_path)
    warm = coerce_table(warming_output, "warming_output")
    assert warm is not None
    validate_warming_table(warm)
    _, derived = _derive_warming_metrics(warm)

    if horizon_year is None:
        horizon_year = max(cfg.warming_years)
    use = derived[derived["year"] == horizon_year].copy()
    ecosystems = [e for e in cfg.ecosystem_order if e in use["ecosystem"].unique()]
    if not ecosystems:
        ecosystems = list(use["ecosystem"].drop_duplicates())
    subsets = [s for s in cfg.observation_subset_order if s in use["observation_subset"].unique()]
    if not subsets:
        subsets = list(use["observation_subset"].drop_duplicates())
    subset_ticks = _subset_ticklabels(use, subsets)
    baseline_subset, soil_subset, resp_subset, combined_subset = _resolve_value_subsets(subsets)

    loss_summary = summarize_quantiles(use, "carbon_loss", ["ecosystem", "observation_subset"])
    slow_summary = summarize_quantiles(use, "slow_fraction_of_excess_respiration", ["ecosystem", "observation_subset"])
    rows = []
    for eco in ecosystems:
        loss_sd = loss_summary[loss_summary["ecosystem"] == eco]
        base = _subset_value(loss_sd, eco, baseline_subset)
        soil = _subset_value(loss_sd, eco, soil_subset)
        resp = _subset_value(loss_sd, eco, resp_subset)
        comb = _subset_value(loss_sd, eco, combined_subset)

        soil_v = _reduction(soil, base)
        resp_v = _reduction(resp, base)
        comb_v = _reduction(comb, base)
        best_single = np.nanmax([soil_v, resp_v]) if np.isfinite([soil_v, resp_v]).any() else np.nan
        rows.append(
            {
                "ecosystem": eco,
                "respired14c_value": resp_v,
                "soil14c_value": soil_v,
                "combined_value": comb_v,
                "combined_minus_best_single": comb_v - best_single if np.isfinite(comb_v) and np.isfinite(best_single) else np.nan,
                "baseline_subset": baseline_subset,
                "soil_subset": soil_subset,
                "respired_subset": resp_subset,
                "combined_subset": combined_subset,
            }
        )
    value_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1.45, 1.25, 1.0]})
    panelize(axes)
    for ax, summary_df, value_col, title, ylabel in [
        (axes[0], loss_summary, "carbon_loss", f"Total Carbon Loss at {horizon_year} Years", "Carbon loss"),
        (axes[1], slow_summary, "slow_fraction_of_excess_respiration", f"Slow Fraction of Excess Respiration at {horizon_year} Years", "Slow fraction"),
    ]:
        x = np.arange(len(subsets))
        for i, eco in enumerate(ecosystems):
            sub = summary_df[summary_df["ecosystem"] == eco].set_index("observation_subset").reindex(subsets)
            offset = (i - (len(ecosystems) - 1) / 2) * 0.12
            ax.errorbar(
                x + offset,
                sub["median"],
                yerr=np.vstack([sub["median"] - sub["q05"], sub["q95"] - sub["median"]]),
                fmt="o",
                capsize=3,
                label=eco if ax is axes[0] else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(subset_ticks, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=8)

    ax = axes[2]
    has_combined = combined_subset != soil_subset and value_df["combined_value"].notna().any()
    if has_combined:
        sc = ax.scatter(
            value_df["respired14c_value"], value_df["soil14c_value"],
            c=value_df["combined_value"], cmap="viridis", vmin=0.0, vmax=1.0,
            marker="D", s=90, edgecolors="0.1", zorder=3,
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Combined value (all obs)")
        for row in value_df.itertuples(index=False):
            ax.text(row.respired14c_value, row.soil14c_value,
                    f"{row.ecosystem}\nΔ{row.combined_minus_best_single:+.2f}",
                    fontsize=7, ha="left", va="bottom")
        ax.set_title("Value of Soil, Respired, and Combined Radiocarbon")
    else:
        ax.scatter(value_df["respired14c_value"], value_df["soil14c_value"], marker="D", s=64, facecolors="white", edgecolors="0.1")
        for row in value_df.itertuples(index=False):
            ax.text(row.respired14c_value, row.soil14c_value, str(row.ecosystem), fontsize=8, ha="left", va="bottom")
        ax.set_title("Relative Value of Soil and Respired Radiocarbon")
    ax.set_xlabel("Respired-14C value")
    ax.set_ylabel("Soil-14C value")
    add_one_to_one(ax)

    alt_text = (
        f"Figure 6 summarizes warming-vulnerability constraints at a {horizon_year}-year horizon. Panel A shows posterior carbon-loss "
        "medians with 95 percent credible intervals by ecosystem and observation subset. Panel B shows the slow-mode fraction of excess "
        "respiration with the same subset ordering. Panel C compares the uncertainty reduction from adding respired radiocarbon or soil "
        f"radiocarbon relative to the {baseline_subset} baseline subset. Points above the one-to-one line indicate greater value from soil "
        "radiocarbon, while points below indicate greater value from respired radiocarbon. Marker color encodes the combined value from using "
        "all observation types together, and each label reports the combined value minus the better single radiocarbon type, i.e. the marginal "
        "gain from combining."
    )
    caption = (
        f"Observation-subset constraints on warming vulnerability at a {horizon_year}-year horizon under the standardized warming experiment. "
        "Panel A shows posterior carbon loss by ecosystem and observation subset. Panel B shows the fraction of excess respiration attributed "
        "to the slow mode. Panel C compares the value of adding soil radiocarbon or respired radiocarbon, defined as one minus posterior "
        "uncertainty with the added observation type divided by the posterior uncertainty for the same baseline subset; marker color shows the "
        "combined value from all observation types and the annotation gives its marginal gain over the better single type (small or negative "
        "gains indicate redundant rather than complementary information)."
    )
    finalize_figure(
        fig,
        "figure_06",
        output_dir,
        {
            "warming_draw_level_metrics": use,
            "warming_loss_summary": loss_summary,
            "warming_slow_fraction_summary": slow_summary,
            "radiocarbon_value_summary": value_df,
        },
        alt_text,
        "Figure 6",
        caption,
    )
    return fig, axes


def main() -> None:
    parser = standard_figure_parser(__doc__ or "Figure 6")
    parser.add_argument("--warming-output", required=True)
    parser.add_argument("--horizon-year", type=int, default=None)
    args = parser.parse_args()
    make_figure_06(
        args.warming_output,
        output_dir=args.output_dir,
        config_path=args.config,
        horizon_year=args.horizon_year,
    )


if __name__ == "__main__":
    main()
