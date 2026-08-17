from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecosystem_complexity.synthesis.biomes import BIOME_GROUP_COLORS, BIOME_GROUP_LABELS, BIOME_GROUP_ORDER

from .cross_ecosystem import build_cross_ecosystem_tables
from .utils import coerce_table, finalize_figure, panelize, setup_figure_config, standard_figure_parser


def _load_observed_offsets(network: pd.DataFrame) -> pd.DataFrame:
    from ecosystem_complexity.model.api import build_model
    from ecosystem_complexity.data.israd_14c import build_bulk_14C_blocks, build_resp_14C_obs
    from ecosystem_complexity.data.parsers import attach_atm14C
    from ecosystem_complexity.data.parsers_14C import load_full_14C_record
    from ecosystem_complexity.data.paths import GRAVEN_PATH, HUA_PATH, INTCAL_PATH
    from ecosystem_complexity.sites.driver import load_site_forcing, resolve_forcing_file
    from ecosystem_complexity.sites.spec import load_site_spec

    rows: list[dict] = []
    for cfg in network["config"]:
        spec = load_site_spec(cfg)
        if spec.observation_path != "bulk_resp":
            continue
        model = build_model(spec.config_path)
        forcing_path = resolve_forcing_file(spec)
        forcing = load_site_forcing(spec, forcing_path, model)
        hemisphere = "NH" if spec.lat >= 0 else "SH"
        years_daily, d14c_daily = load_full_14C_record(
            hua_path=HUA_PATH,
            graven_path=GRAVEN_PATH,
            intcal_path=INTCAL_PATH,
            hemisphere=hemisphere,
            start_year=1500.0,
            end_year=2025.0,
        )
        forcing = attach_atm14C(forcing, d14c_daily, years_daily)
        bulk_blocks = build_bulk_14C_blocks(spec.israd_name, forcing.time, model)
        resp = np.array(build_resp_14C_obs(spec.israd_name, forcing.time), dtype=float)

        bulk_vals: list[float] = []
        for block in bulk_blocks:
            vals = np.array(block.y, dtype=float).ravel()
            bulk_vals.extend(vals[np.isfinite(vals)].tolist())
        resp_vals = resp[np.isfinite(resp)]
        if not bulk_vals or resp_vals.size == 0:
            continue
        rows.append(
            {
                "site": spec.israd_name,
                "obs_bulk_mean": float(np.mean(bulk_vals)),
                "obs_resp_mean": float(np.mean(resp_vals)),
                "obs_offset_resp_minus_bulk": float(np.mean(resp_vals) - np.mean(bulk_vals)),
                "n_bulk_vals": int(len(bulk_vals)),
                "n_resp_vals": int(resp_vals.size),
            }
        )
    return pd.DataFrame(rows)


def _spearman(df: pd.DataFrame, x: str, y: str) -> float:
    sample = df[[x, y]].dropna()
    return float(sample[x].corr(sample[y], method="spearman"))


def _pearson(df: pd.DataFrame, x: str, y: str) -> float:
    sample = df[[x, y]].dropna()
    return float(sample[x].corr(sample[y], method="pearson"))


def _annotate_corr(ax, df: pd.DataFrame, x: str, y: str, xpos: float = 0.04, ypos: float = 0.96) -> None:
    sample = df[[x, y]].dropna()
    if sample.empty:
        return
    rs = _spearman(sample, x, y)
    rp = _pearson(sample, x, y)
    ax.text(
        xpos,
        ypos,
        f"n = {len(sample)}\n$\\rho_s$ = {rs:.2f}\nr = {rp:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.85", "boxstyle": "round,pad=0.25"},
    )


def _add_fit(ax, df: pd.DataFrame, x: str, y: str) -> None:
    sample = df[[x, y]].dropna()
    if len(sample) < 3:
        return
    slope, intercept = np.polyfit(sample[x].to_numpy(dtype=float), sample[y].to_numpy(dtype=float), 1)
    xfit = np.linspace(float(sample[x].min()), float(sample[x].max()), 100)
    ax.plot(xfit, slope * xfit + intercept, color="0.15", linewidth=1.3, zorder=2)


def _scatter_by_biome(ax, df: pd.DataFrame, x: str, y: str, annotate_sites: list[str] | None = None) -> None:
    for biome_group in BIOME_GROUP_ORDER:
        sub = df[df["biome_group"] == biome_group]
        if sub.empty:
            continue
        ax.scatter(
            sub[x],
            sub[y],
            s=54,
            color=BIOME_GROUP_COLORS[biome_group],
            alpha=0.88,
            edgecolors="white",
            linewidths=0.55,
            zorder=3,
        )
    if annotate_sites:
        sub = df[df["site"].isin(annotate_sites)]
        for row in sub.itertuples(index=False):
            ax.annotate(
                str(row.site),
                (getattr(row, x), getattr(row, y)),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )


def _format_interval(median: float, q025: float, q975: float, digits: int = 2) -> str:
    return f"{median:.{digits}f} [{q025:.{digits}f}, {q975:.{digits}f}]"


def _find_relation(summary: pd.DataFrame, relationship: str) -> pd.Series:
    match = summary[summary["relationship"] == relationship]
    if match.empty:
        raise KeyError(f"Missing regression summary row for {relationship}")
    return match.iloc[0]


def _find_predictor(summary: pd.DataFrame, predictor: str) -> pd.Series:
    match = summary[summary["predictor"] == predictor]
    if match.empty:
        raise KeyError(f"Missing predictor summary row for {predictor}")
    return match.iloc[0]


def _plot_regression_band(
    ax,
    samples: pd.DataFrame,
    relationship: str,
    x_min: float,
    x_max: float,
    color: str = "0.15",
    alpha: float = 0.18,
) -> None:
    grp = samples[samples["relationship"] == relationship]
    if grp.empty:
        return
    xfit = np.linspace(x_min, x_max, 150)
    slopes = grp["slope"].to_numpy(dtype=float)
    intercepts = grp["intercept"].to_numpy(dtype=float)
    yfit = intercepts[:, None] + slopes[:, None] * xfit[None, :]
    ax.fill_between(
        xfit,
        np.quantile(yfit, 0.025, axis=0),
        np.quantile(yfit, 0.975, axis=0),
        color=color,
        alpha=alpha,
        zorder=1,
    )
    ax.plot(xfit, np.quantile(yfit, 0.5, axis=0), color=color, linewidth=1.5, zorder=2)


def _posterior_error_scatter(
    ax,
    df: pd.DataFrame,
    x_med: str,
    x_lo: str,
    x_hi: str,
    y_med: str,
    y_lo: str | None = None,
    y_hi: str | None = None,
    *,
    annotate_sites: list[str] | None = None,
) -> None:
    for biome_group in BIOME_GROUP_ORDER:
        sub = df[df["biome_group"] == biome_group]
        if sub.empty:
            continue
        for row in sub.itertuples(index=False):
            x = float(getattr(row, x_med))
            xerr = np.array(
                [
                    [x - float(getattr(row, x_lo))],
                    [float(getattr(row, x_hi)) - x],
                ]
            )
            if y_lo is not None and y_hi is not None:
                y = float(getattr(row, y_med))
                yerr = np.array(
                    [
                        [y - float(getattr(row, y_lo))],
                        [float(getattr(row, y_hi)) - y],
                    ]
                )
            else:
                y = float(getattr(row, y_med))
                yerr = None
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor=BIOME_GROUP_COLORS[biome_group],
                elinewidth=0.8,
                alpha=0.35,
                zorder=1,
            )
            marker = "^" if row.source_set == "incubation_expansion" else "o"
            face = "white" if marker == "^" else BIOME_GROUP_COLORS[biome_group]
            ax.scatter(
                x,
                y,
                s=62 if marker == "^" else 58,
                facecolors=face,
                edgecolors=BIOME_GROUP_COLORS[biome_group],
                linewidths=0.8 if marker == "^" else 0.55,
                marker=marker,
                zorder=3,
            )
    if annotate_sites:
        sub = df[df["site"].isin(annotate_sites)]
        for row in sub.itertuples(index=False):
            ax.annotate(
                str(row.site),
                (getattr(row, x_med), getattr(row, y_med)),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )


def _annotate_relation(ax, row: pd.Series, *, include_slope: bool = True, extra: str | None = None) -> None:
    parts = [f"n = {int(row['n_sites'])}"]
    if include_slope:
        parts.append(
            "slope = "
            + _format_interval(
                float(row["slope_median"]),
                float(row["slope_q025"]),
                float(row["slope_q975"]),
                digits=2,
            )
        )
    parts.append(
        "r = "
        + _format_interval(
            float(row["pearson_r_median"]),
            float(row["pearson_r_q025"]),
            float(row["pearson_r_q975"]),
            digits=2,
        )
    )
    parts.append(
        "$\\rho_s$ = "
        + _format_interval(
            float(row["spearman_rho_median"]),
            float(row["spearman_rho_q025"]),
            float(row["spearman_rho_q975"]),
            digits=2,
        )
    )
    if extra:
        parts.append(extra)
    ax.text(
        0.03,
        0.97,
        "\n".join(parts),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.85", "boxstyle": "round,pad=0.25"},
    )


def build_coupling_tables(
    network_summary: str | pd.DataFrame,
    warming_summary: str | pd.DataFrame,
    new_sites: list[str] | tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    tables = build_cross_ecosystem_tables(network_summary, warming_summary, new_sites)
    direct = tables["direct_warming_sites"].copy()
    all_sites = tables["all_sites_union"].copy()

    for df in (direct, all_sites):
        df["log_tau_passive_active"] = np.log10(df["tau_passive_yr"] / df["tau_active_yr"])
        df["log_tau_slow_active"] = np.log10(df["tau_slow_yr"] / df["tau_active_yr"])
        df["log_tau_passive_slow"] = np.log10(df["tau_passive_yr"] / df["tau_slow_yr"])

    offset_df = _load_observed_offsets(pd.DataFrame(network_summary) if isinstance(network_summary, pd.DataFrame) else pd.read_csv(network_summary))
    direct_offset = direct.merge(offset_df, on="site", how="left")

    corr_specs = [
        ("log_tau_passive_active", "old_fraction_of_excess_rh", "turnover separation", "old-carbon share"),
        ("log_tau_passive_active", "frac_c_loss", "turnover separation", "fractional C loss"),
        ("log_tau_passive_active", "dfs_total", "turnover separation", "constrainability"),
        ("obs_offset_resp_minus_bulk", "old_fraction_of_excess_rh", "observed respired-bulk offset", "old-carbon share"),
        ("obs_offset_resp_minus_bulk", "frac_c_loss", "observed respired-bulk offset", "fractional C loss"),
        ("obs_offset_resp_minus_bulk", "dfs_total", "observed respired-bulk offset", "constrainability"),
    ]
    corr_rows: list[dict] = []
    for x, y, x_label, y_label in corr_specs:
        frame = direct_offset if x.startswith("obs_offset") else direct
        sample = frame[[x, y]].dropna()
        corr_rows.append(
            {
                "x": x,
                "y": y,
                "x_label": x_label,
                "y_label": y_label,
                "n": int(len(sample)),
                "pearson_r": _pearson(sample, x, y) if len(sample) >= 2 else np.nan,
                "spearman_r": _spearman(sample, x, y) if len(sample) >= 2 else np.nan,
            }
        )

    split_cutoff = float(direct["log_tau_passive_active"].median())
    summary = direct.assign(
        separation_group=np.where(
            direct["log_tau_passive_active"] >= split_cutoff,
            "high separation",
            "low separation",
        )
    )
    median_split = (
        summary.groupby("separation_group", as_index=False)
        .agg(
            n_sites=("site", "nunique"),
            median_frac_loss=("frac_c_loss", "median"),
            median_old_share=("old_fraction_of_excess_rh", "median"),
            median_dfs=("dfs_total", "median"),
            median_abs_loss=("abs_c_loss_gCm2", "median"),
        )
    )
    return {
        "all_sites_coupling": all_sites,
        "direct_sites_coupling": direct,
        "direct_sites_with_offsets": direct_offset,
        "coupling_correlations": pd.DataFrame(corr_rows),
        "coupling_median_split": median_split,
        "coupling_cutoff": pd.DataFrame([{"metric": "log_tau_passive_active", "median_cutoff": split_cutoff}]),
    }


def make_figure_10_from_posterior_analysis(
    site_metrics: str | pd.DataFrame,
    regression_samples: str | pd.DataFrame,
    regression_summary: str | pd.DataFrame,
    leave_one_out: str | pd.DataFrame,
    predictor_comparison: str | pd.DataFrame,
    predicted_percentiles: str | pd.DataFrame,
    output_dir: str = "outputs",
    config_path: str | None = None,
):
    setup_figure_config(config_path)
    metrics = coerce_table(site_metrics, "site_metrics")
    samples = coerce_table(regression_samples, "regression_samples")
    summary = coerce_table(regression_summary, "regression_summary")
    loo = coerce_table(leave_one_out, "leave_one_out")
    predictors = coerce_table(predictor_comparison, "predictor_comparison")
    percentiles = coerce_table(predicted_percentiles, "predicted_percentiles")
    assert metrics is not None
    assert samples is not None
    assert summary is not None
    assert loo is not None
    assert predictors is not None
    assert percentiles is not None

    metrics = metrics.sort_values(["biome_group", "site"]).reset_index(drop=True)
    y_map = {key: i for i, key in enumerate(BIOME_GROUP_ORDER)}
    cutoff = float(metrics["turnover_separation_median"].median())
    rel_old = _find_relation(summary, "old_fraction_of_excess_rh__vs__turnover_separation")
    rel_loss = _find_relation(summary, "frac_c_loss__vs__turnover_separation")
    rel_dfs = _find_relation(summary, "dfs_total__vs__turnover_separation")
    pred_gap = _find_predictor(predictors, "obs_offset_resp_minus_bulk")

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.8))
    panelize(axes)

    ax = axes[0, 0]
    for biome_group in BIOME_GROUP_ORDER:
        sub = metrics[metrics["biome_group"] == biome_group].copy()
        if sub.empty:
            continue
        jitter = np.linspace(-0.18, 0.18, len(sub)) if len(sub) > 1 else np.array([0.0])
        yvals = y_map[biome_group] + jitter
        x = sub["turnover_separation_median"].to_numpy(dtype=float)
        xerr = np.vstack(
            [
                x - sub["turnover_separation_q025"].to_numpy(dtype=float),
                sub["turnover_separation_q975"].to_numpy(dtype=float) - x,
            ]
        )
        marker = np.where(sub["source_set"].to_numpy() == "incubation_expansion", "^", "o")
        for j, row in enumerate(sub.itertuples(index=False)):
            face = "white" if row.source_set == "incubation_expansion" else BIOME_GROUP_COLORS[biome_group]
            ax.errorbar(
                x[j],
                yvals[j],
                xerr=xerr[:, [j]],
                fmt="none",
                ecolor=BIOME_GROUP_COLORS[biome_group],
                elinewidth=0.9,
                alpha=0.4,
                zorder=1,
            )
            ax.scatter(
                x[j],
                yvals[j],
                s=62 if marker[j] == "o" else 84,
                facecolors=face,
                edgecolors=BIOME_GROUP_COLORS[biome_group],
                linewidths=1.0,
                marker=marker[j],
                zorder=3,
            )
    for site in ["Dinesen", "Treynor", "Trumbore Ahwahnee", "Howland Forest", "EML", "Harvard Forest"]:
        row = metrics[metrics["site"] == site]
        if row.empty:
            continue
        ax.annotate(
            site,
            (float(row["turnover_separation_median"].iloc[0]), y_map[str(row["biome_group"].iloc[0])] + 0.12),
            fontsize=7,
        )
    ax.axvline(cutoff, color="0.4", linestyle="--", linewidth=1.0)
    ax.set_yticks(np.arange(len(BIOME_GROUP_ORDER)))
    ax.set_yticklabels([BIOME_GROUP_LABELS[key] for key in BIOME_GROUP_ORDER])
    ax.set_xlabel(r"$\log_{10}(\tau_{passive} / \tau_{active})$")
    ax.set_title("Posterior turnover separation across the 34-site network")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        handles=[
            mlines.Line2D([], [], color="0.2", marker="o", linestyle="None", markersize=6, label="Network sites"),
            mlines.Line2D([], [], color="0.2", marker="^", markerfacecolor="white", linestyle="None", markersize=7, label="Expansion sites"),
        ],
        loc="lower right",
        fontsize=8,
    )

    ax = axes[0, 1]
    _posterior_error_scatter(
        ax,
        metrics.dropna(subset=["old_fraction_of_excess_rh_median"]),
        "turnover_separation_median",
        "turnover_separation_q025",
        "turnover_separation_q975",
        "old_fraction_of_excess_rh_median",
        "old_fraction_of_excess_rh_q025",
        "old_fraction_of_excess_rh_q975",
        annotate_sites=["Adventdalen Valley", "EML", "Howland Forest", "Harvard Forest"],
    )
    _plot_regression_band(
        ax,
        samples,
        "old_fraction_of_excess_rh__vs__turnover_separation",
        float(metrics["turnover_separation_q025"].min()),
        float(metrics["turnover_separation_q975"].max()),
    )
    _annotate_relation(
        ax,
        rel_old,
        extra=(
            "LOO slope/r/$\\rho_s$ = "
            f"[{loo['slope'].min():.2f}, {loo['slope'].max():.2f}] / "
            f"[{loo['pearson_r'].min():.2f}, {loo['pearson_r'].max():.2f}] / "
            f"[{loo['spearman_rho'].min():.2f}, {loo['spearman_rho'].max():.2f}]"
        ),
    )
    ax.set_xlabel(r"$\log_{10}(\tau_{passive} / \tau_{active})$")
    ax.set_ylabel("Old fraction of excess RH")
    ax.set_title("Posterior propagation preserves a strong old-RH relationship")
    ax.grid(alpha=0.25)

    ax = axes[0, 2]
    _posterior_error_scatter(
        ax,
        metrics.dropna(subset=["frac_c_loss_median"]),
        "turnover_separation_median",
        "turnover_separation_q025",
        "turnover_separation_q975",
        "frac_c_loss_median",
        "frac_c_loss_q025",
        "frac_c_loss_q975",
        annotate_sites=["CZ_1964burn_NSA", "EML", "Harvard Forest", "Willow Creek"],
    )
    _plot_regression_band(
        ax,
        samples,
        "frac_c_loss__vs__turnover_separation",
        float(metrics["turnover_separation_q025"].min()),
        float(metrics["turnover_separation_q975"].max()),
    )
    _annotate_relation(ax, rel_loss, include_slope=False)
    ax.set_xlabel(r"$\log_{10}(\tau_{passive} / \tau_{active})$")
    ax.set_ylabel("Fractional carbon loss")
    ax.set_title("Separated pools do not imply the largest fractional losses")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    _posterior_error_scatter(
        ax,
        metrics.dropna(subset=["dfs_total"]),
        "turnover_separation_median",
        "turnover_separation_q025",
        "turnover_separation_q975",
        "dfs_total",
        annotate_sites=["Howland Forest", "EML", "CA_Mollisol", "AZ_Mollisol"],
    )
    _plot_regression_band(
        ax,
        samples,
        "dfs_total__vs__turnover_separation",
        float(metrics["turnover_separation_q025"].min()),
        float(metrics["turnover_separation_q975"].max()),
    )
    _annotate_relation(ax, rel_dfs, include_slope=False, extra="Interpretation: null relationship")
    ax.set_xlabel(r"$\log_{10}(\tau_{passive} / \tau_{active})$")
    ax.set_ylabel("Total DFS")
    ax.set_title("Constrainability is largely independent of turnover separation")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    gap_sample = metrics.dropna(subset=["obs_offset_resp_minus_bulk", "old_fraction_of_excess_rh_median"])
    for biome_group in BIOME_GROUP_ORDER:
        sub = gap_sample[gap_sample["biome_group"] == biome_group]
        if sub.empty:
            continue
        y = sub["old_fraction_of_excess_rh_median"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - sub["old_fraction_of_excess_rh_q025"].to_numpy(dtype=float),
                sub["old_fraction_of_excess_rh_q975"].to_numpy(dtype=float) - y,
            ]
        )
        ax.errorbar(
            sub["obs_offset_resp_minus_bulk"],
            y,
            yerr=yerr,
            fmt="none",
            ecolor=BIOME_GROUP_COLORS[biome_group],
            elinewidth=0.8,
            alpha=0.35,
            zorder=1,
        )
        ax.scatter(
            sub["obs_offset_resp_minus_bulk"],
            y,
            s=56,
            color=BIOME_GROUP_COLORS[biome_group],
            edgecolors="white",
            linewidths=0.55,
            zorder=3,
        )
    if len(gap_sample) >= 2:
        x = gap_sample["obs_offset_resp_minus_bulk"].to_numpy(dtype=float)
        xfit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
        ax.plot(
            xfit,
            float(pred_gap["intercept_median"]) + float(pred_gap["slope_median"]) * xfit,
            color="0.15",
            linewidth=1.3,
            zorder=2,
        )
    _annotate_relation(ax, pred_gap, extra=f"smaller-sample comparison (n = {int(pred_gap['n_sites'])})")
    ax.set_xlabel(r"Observed respired $\Delta^{14}$C - bulk $\Delta^{14}$C (‰)")
    ax.set_ylabel("Old fraction of excess RH")
    ax.set_title(rf"Raw $\Delta^{{14}}$C gap comparison (n = {int(pred_gap['n_sites'])})")
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    metric_order = [
        ("frac_c_loss", "Fractional C loss"),
        ("old_fraction_of_excess_rh", "Old fraction of excess RH"),
        ("dfs_total", "Total DFS"),
    ]
    ypos = np.arange(len(metric_order))[::-1]
    for y, (metric, label) in zip(ypos, metric_order):
        low = percentiles[(percentiles["metric"] == metric) & (percentiles["turnover_percentile"] == "p20")].iloc[0]
        high = percentiles[(percentiles["metric"] == metric) & (percentiles["turnover_percentile"] == "p80")].iloc[0]
        ax.plot(
            [float(low["predicted_median"]), float(high["predicted_median"])],
            [y, y],
            color="0.75",
            linewidth=1.2,
            zorder=1,
        )
        ax.errorbar(
            float(low["predicted_median"]),
            y,
            xerr=np.array(
                [
                    [float(low["predicted_median"]) - float(low["predicted_q025"])],
                    [float(low["predicted_q975"]) - float(low["predicted_median"])],
                ]
            ),
            fmt="o",
            mfc="white",
            mec="0.2",
            ecolor="0.45",
            ms=6,
            elinewidth=1.0,
            zorder=3,
        )
        ax.errorbar(
            float(high["predicted_median"]),
            y,
            xerr=np.array(
                [
                    [float(high["predicted_median"]) - float(high["predicted_q025"])],
                    [float(high["predicted_q975"]) - float(high["predicted_median"])],
                ]
            ),
            fmt="o",
            mfc="0.15",
            mec="0.15",
            ecolor="0.25",
            ms=6,
            elinewidth=1.0,
            zorder=3,
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels([label for _, label in metric_order])
    ax.set_title("Predicted values at the 20th and 80th turnover-separation percentiles")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        handles=[
            mlines.Line2D([], [], color="0.2", marker="o", markerfacecolor="white", linestyle="None", markersize=6, label="20th percentile"),
            mlines.Line2D([], [], color="0.15", marker="o", linestyle="None", markersize=6, label="80th percentile"),
        ],
        loc="lower right",
        fontsize=8,
    )

    fig.tight_layout()
    alt_text = (
        "Six-panel posterior-propagated summary of turnover separation and warming vulnerability across the 34-site figure-10 site set. "
        "Panel A shows posterior median turnover separation with 95 percent intervals for each site. Panel B shows posterior uncertainty "
        "in both turnover separation and old-fraction response, with a posterior-propagated regression and a leave-one-site-out range. "
        "Panel C shows the weaker negative relationship between turnover separation and fractional carbon loss. Panel D shows the near-null "
        "relationship between turnover separation and total DFS. Panel E compares the observed respired-minus-bulk radiocarbon gap against "
        "old-fraction response for the smaller subset of sites where that raw gap is available. Panel F compares regression-predicted values "
        "at the 20th and 80th percentiles of turnover separation, with 95 percent intervals for each predicted metric."
    )
    caption = (
        "Posterior-propagated strengthening of the turnover-separation result. (A) Site-level turnover separation with posterior 95 percent "
        "intervals across the 34-site Figure 10 universe. (B) The old fraction of excess heterotrophic respiration remains strongly related "
        "to turnover separation after posterior propagation, with uncertainty shown on both axes and the leave-one-site-out range noted in "
        "the annotation. (C) Fractional carbon loss remains a weaker and negative function of turnover separation. (D) Total DFS remains "
        "approximately unrelated to turnover separation, consistent with a null constrainability relationship. (E) The raw observed "
        "respired-minus-bulk radiocarbon gap is shown for the smaller available subset and is not overinterpreted. (F) Instead of isolated "
        "high- versus low-separation points, predicted values are shown at the 20th and 80th percentiles of turnover separation with "
        "posterior 95 percent intervals."
    )
    csv_map = {
        "posterior_site_metrics": metrics,
        "posterior_regression_samples": samples,
        "posterior_regression_summary": summary,
        "leave_one_out": loo,
        "predictor_comparison": predictors,
        "predicted_percentiles": percentiles,
    }
    finalize_figure(
        fig,
        "figure_10",
        output_dir,
        csv_map,
        alt_text,
        "Figure 10",
        caption,
    )
    return fig, axes


def make_figure_10(
    network_summary: str | pd.DataFrame,
    warming_summary: str | pd.DataFrame,
    new_sites: list[str] | tuple[str, ...],
    output_dir: str = "outputs",
    config_path: str | None = None,
):
    setup_figure_config(config_path)
    tables = build_coupling_tables(network_summary, warming_summary, new_sites)
    all_sites = tables["all_sites_coupling"]
    direct = tables["direct_sites_coupling"]
    offset = tables["direct_sites_with_offsets"]
    split = tables["coupling_median_split"]
    cutoff = float(tables["coupling_cutoff"]["median_cutoff"].iloc[0])
    n_all_sites = all_sites["site"].nunique()

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.5))
    panelize(axes)

    ax = axes[0, 0]
    y_map = {key: i for i, key in enumerate(BIOME_GROUP_ORDER)}
    for biome_group in BIOME_GROUP_ORDER:
        sub = all_sites[all_sites["biome_group"] == biome_group].copy()
        if sub.empty:
            continue
        jitter = np.linspace(-0.18, 0.18, len(sub)) if len(sub) > 1 else np.array([0.0])
        direct_sub = sub[sub["has_direct_warming"]]
        extra_sub = sub[~sub["has_direct_warming"]]
        if not direct_sub.empty:
            yvals = y_map[biome_group] + jitter[: len(direct_sub)]
            ax.scatter(
                direct_sub["log_tau_passive_active"],
                yvals,
                s=54,
                color=BIOME_GROUP_COLORS[biome_group],
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )
        if not extra_sub.empty:
            yvals = y_map[biome_group] + jitter[len(direct_sub) : len(direct_sub) + len(extra_sub)]
            ax.scatter(
                extra_sub["log_tau_passive_active"],
                yvals,
                s=82,
                facecolors="none",
                edgecolors=BIOME_GROUP_COLORS[biome_group],
                linewidths=1.4,
                marker="^",
                zorder=3,
            )
    for site in ["Dinesen", "Treynor", "Trumbore Ahwahnee", "Howland Forest", "EML", "Harvard Forest"]:
        row = all_sites[all_sites["site"] == site]
        if row.empty:
            continue
        ax.annotate(
            site,
            (float(row["log_tau_passive_active"].iloc[0]), y_map[str(row["biome_group"].iloc[0])] + 0.12),
            fontsize=7,
        )
    ax.axvline(cutoff, color="0.4", linestyle="--", linewidth=1.0)
    ax.set_yticks(np.arange(len(BIOME_GROUP_ORDER)))
    ax.set_yticklabels([BIOME_GROUP_LABELS[key] for key in BIOME_GROUP_ORDER])
    ax.set_xlabel(r"$\log_{10}(\tau_{passive} / \tau_{active})$")
    ax.set_title(f"Turnover separation spans the full {n_all_sites}-site network")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        handles=[
            mlines.Line2D([], [], color="0.2", marker="o", linestyle="None", markersize=6, label="Direct warming sites"),
            mlines.Line2D([], [], color="0.2", marker="^", markerfacecolor="white", linestyle="None", markersize=7, label="Expansion sites"),
        ],
        loc="lower right",
        fontsize=8,
    )

    ax = axes[0, 1]
    _scatter_by_biome(ax, direct, "log_tau_passive_active", "old_fraction_of_excess_rh", ["Adventdalen Valley", "EML", "Howland Forest", "Harvard Forest"])
    _add_fit(ax, direct, "log_tau_passive_active", "old_fraction_of_excess_rh")
    _annotate_corr(ax, direct, "log_tau_passive_active", "old_fraction_of_excess_rh")
    ax.set_xlabel(r"$\log_{10}(\tau_{passive} / \tau_{active})$")
    ax.set_ylabel("Old fraction of excess RH")
    ax.set_title("Separated pools mobilize older carbon under warming")
    ax.grid(alpha=0.25)

    ax = axes[0, 2]
    _scatter_by_biome(ax, direct, "log_tau_passive_active", "frac_c_loss", ["CZ_1964burn_NSA", "EML", "Harvard Forest", "Willow Creek"])
    _add_fit(ax, direct, "log_tau_passive_active", "frac_c_loss")
    _annotate_corr(ax, direct, "log_tau_passive_active", "frac_c_loss")
    ax.set_xlabel(r"$\log_{10}(\tau_{passive} / \tau_{active})$")
    ax.set_ylabel("Fractional carbon loss")
    ax.set_title("Separated pools do not lose the largest fraction of C")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    _scatter_by_biome(ax, direct, "log_tau_passive_active", "dfs_total", ["Howland Forest", "EML", "CA_Mollisol", "AZ_Mollisol"])
    _add_fit(ax, direct, "log_tau_passive_active", "dfs_total")
    _annotate_corr(ax, direct, "log_tau_passive_active", "dfs_total")
    ax.set_xlabel(r"$\log_{10}(\tau_{passive} / \tau_{active})$")
    ax.set_ylabel("Total DFS")
    ax.set_title("Turnover coupling is largely independent of constrainability")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    offset_sample = offset.dropna(subset=["obs_offset_resp_minus_bulk", "old_fraction_of_excess_rh"]).copy()
    _scatter_by_biome(ax, offset_sample, "obs_offset_resp_minus_bulk", "old_fraction_of_excess_rh", ["Adventdalen Valley", "FLONA", "EML", "Howland Forest"])
    _add_fit(ax, offset_sample, "obs_offset_resp_minus_bulk", "old_fraction_of_excess_rh")
    _annotate_corr(ax, offset_sample, "obs_offset_resp_minus_bulk", "old_fraction_of_excess_rh")
    ax.set_xlabel(r"Observed respired $\Delta^{14}$C - bulk $\Delta^{14}$C (‰)")
    ax.set_ylabel("Old fraction of excess RH")
    ax.set_title("The raw 14C gap alone is a weaker predictor")
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    metric_rows = [
        ("median_frac_loss", "Fractional C loss"),
        ("median_old_share", "Old fraction of excess RH"),
        ("median_dfs", "Total DFS"),
    ]
    ypos = np.arange(len(metric_rows))[::-1]
    low = split[split["separation_group"] == "low separation"].iloc[0]
    high = split[split["separation_group"] == "high separation"].iloc[0]
    for y, (col, label) in zip(ypos, metric_rows):
        ax.plot([low[col], high[col]], [y, y], color="0.7", linewidth=1.2, zorder=1)
        ax.scatter(low[col], y, s=62, facecolors="white", edgecolors="0.2", linewidths=1.0, zorder=3)
        ax.scatter(high[col], y, s=62, color="0.15", zorder=3)
        ax.text(low[col], y - 0.14, f"{low[col]:.2f}", ha="center", va="top", fontsize=8)
        ax.text(high[col], y + 0.14, f"{high[col]:.2f}", ha="center", va="bottom", fontsize=8, color="0.1")
    ax.set_yticks(ypos)
    ax.set_yticklabels([label for _, label in metric_rows])
    ax.set_xlim(0.0, max(1.0, float(high["median_dfs"]) + 0.15))
    ax.set_title("High-separation systems shift toward older, not larger, losses")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        handles=[
            mlines.Line2D([], [], color="0.2", marker="o", markerfacecolor="white", linestyle="None", markersize=6, label="Low separation"),
            mlines.Line2D([], [], color="0.15", marker="o", linestyle="None", markersize=6, label="High separation"),
        ],
        loc="lower right",
        fontsize=8,
    )

    fig.tight_layout()
    alt_text = (
        "Six-panel summary figure about turnover coupling and vulnerability across the expanded 14C-constrained site network. "
        f"Panel A shows the distribution of passive-to-active turnover separation across all {n_all_sites} sites, grouped by biome and distinguishing "
        "direct warming sites from expansion-only sites. Panels B through D show that larger passive-active separation is strongly associated "
        "with a larger old-carbon share of warming-induced respiration, negatively associated with fractional carbon loss, and nearly unrelated "
        "to total degrees of freedom for signal. Panel E shows that the observed respired-minus-bulk radiocarbon offset is a weaker predictor "
        "than posterior turnover separation. Panel F compares low- and high-separation median systems and shows that high-separation systems "
        "have older warming losses but not larger fractional losses."
    )
    caption = (
        "Turnover coupling separates the age of vulnerability from the amount of vulnerability across the 14C-constrained site network. "
        f"(A) Passive-to-active turnover separation for all {n_all_sites} inversions, including the expansion-only sites. (B) Sites with more strongly "
        "separated passive and active pools contribute a larger old-carbon fraction to warming-induced excess heterotrophic respiration. "
        "(C) The same separation metric is negatively associated with total fractional carbon loss under warming. (D) Turnover separation is "
        "largely independent of total constrainability (DFS), implying that the result is not a simple artifact of better-resolved sites. "
        "(E) The observed respired-minus-bulk radiocarbon gap is directionally consistent but weaker, indicating that a simple bulk-versus-"
        "respired age offset does not replace the posterior turnover structure. (F) Median low- versus high-separation systems show the same "
        "pattern directly: high-separation systems mobilize older carbon while losing a smaller fraction of total carbon."
    )
    finalize_figure(
        fig,
        "figure_10",
        output_dir,
        tables,
        alt_text,
        "Figure 10",
        caption,
    )
    return fig, axes


def main() -> None:
    parser = standard_figure_parser(__doc__ or "Figure 10")
    parser.add_argument("--network-summary")
    parser.add_argument("--warming-summary")
    parser.add_argument("--new-sites", nargs="+")
    parser.add_argument("--posterior-site-metrics")
    parser.add_argument("--posterior-regression-samples")
    parser.add_argument("--posterior-regression-summary")
    parser.add_argument("--leave-one-out")
    parser.add_argument("--predictor-comparison")
    parser.add_argument("--predicted-percentiles")
    args = parser.parse_args()
    if args.posterior_site_metrics:
        if not all(
            [
                args.posterior_regression_samples,
                args.posterior_regression_summary,
                args.leave_one_out,
                args.predictor_comparison,
                args.predicted_percentiles,
            ]
        ):
            raise ValueError("Posterior Figure 10 mode requires all posterior analysis tables.")
        make_figure_10_from_posterior_analysis(
            args.posterior_site_metrics,
            args.posterior_regression_samples,
            args.posterior_regression_summary,
            args.leave_one_out,
            args.predictor_comparison,
            args.predicted_percentiles,
            output_dir=args.output_dir,
            config_path=args.config,
        )
        return
    if not (args.network_summary and args.warming_summary and args.new_sites):
        raise ValueError("Legacy Figure 10 mode requires --network-summary, --warming-summary, and --new-sites.")
    make_figure_10(
        args.network_summary,
        args.warming_summary,
        args.new_sites,
        output_dir=args.output_dir,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
