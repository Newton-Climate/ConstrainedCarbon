"""Statistical rollups over collected MCMC posterior + prior draws."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2 or y.size < 2:
        return (np.nan, np.nan)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _quantiles(values: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan, np.nan)
    return (
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.5)),
        float(np.quantile(vals, 0.975)),
    )


def _summarize_site_draws(
    posterior_df: pd.DataFrame,
    *,
    payload: dict,
    posterior_kind: str,
    dfs_total: float,
    gap: dict[str, float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "site": payload["site"],
        "label": payload["label"],
        "tower_id": payload["tower_id"],
        "biome": payload["biome"],
        "biome_group": payload["biome_group"],
        "source_set": payload["source_set"],
        "config": payload["config"],
        "posterior_kind": posterior_kind,
        "n_draws": int(len(posterior_df)),
        "dfs_total": dfs_total,
        **gap,
    }
    for col in (
        "tau_active_yr",
        "tau_slow_yr",
        "tau_passive_yr",
        "turnover_separation",
        "frac_c_loss",
        "abs_c_loss_gCm2",
        "old_fraction_of_excess_rh",
    ):
        q025, median, q975 = _quantiles(posterior_df[col].to_numpy(dtype=float))
        row[f"{col}_q025"] = q025
        row[f"{col}_median"] = median
        row[f"{col}_q975"] = q975
    row["has_observed_gap"] = bool(np.isfinite(row["obs_offset_resp_minus_bulk"]))
    return row


def _regression_samples(
    draws: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_iter: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    by_site = {site: grp.reset_index(drop=True) for site, grp in draws.groupby("site", sort=True)}
    rows: list[dict[str, object]] = []
    for i in range(n_iter):
        sample_rows = []
        for site, grp in by_site.items():
            take = int(rng.integers(0, len(grp)))
            sample_rows.append(grp.iloc[take])
        sample = pd.DataFrame(sample_rows)[["site", x_col, y_col]].dropna()
        x = sample[x_col].to_numpy(dtype=float)
        y = sample[y_col].to_numpy(dtype=float)
        slope, intercept = _slope_intercept(x, y)
        rows.append(
            {
                "relationship": f"{y_col}__vs__{x_col}",
                "iteration": i,
                "n": int(len(sample)),
                "slope": slope,
                "intercept": intercept,
                "pearson_r": _pearson(x, y),
                "spearman_rho": float(spearmanr(x, y, nan_policy="omit").statistic) if len(sample) >= 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _summarize_stat_frame(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, grp in df.groupby(group_col, sort=False):
        row = {group_col: key, "n_iterations": int(len(grp))}
        for col in ("slope", "intercept", "pearson_r", "spearman_rho"):
            q025, median, q975 = _quantiles(grp[col].to_numpy(dtype=float))
            row[f"{col}_q025"] = q025
            row[f"{col}_median"] = median
            row[f"{col}_q975"] = q975
        row["n_sites"] = int(pd.to_numeric(grp["n"], errors="coerce").median())
        rows.append(row)
    return pd.DataFrame(rows)


def _leave_one_out(site_metrics: pd.DataFrame) -> pd.DataFrame:
    use = site_metrics[["site", "turnover_separation_median", "old_fraction_of_excess_rh_median"]].dropna()
    rows = []
    for site in use["site"]:
        sample = use[use["site"] != site]
        x = sample["turnover_separation_median"].to_numpy(dtype=float)
        y = sample["old_fraction_of_excess_rh_median"].to_numpy(dtype=float)
        slope, _intercept = _slope_intercept(x, y)
        rows.append(
            {
                "excluded_site": site,
                "n_sites": int(len(sample)),
                "slope": slope,
                "pearson_r": _pearson(x, y),
                "spearman_rho": float(spearmanr(x, y, nan_policy="omit").statistic) if len(sample) >= 2 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("excluded_site").reset_index(drop=True)


def _predictor_comparison(
    site_metrics: pd.DataFrame,
    posterior_draws: pd.DataFrame,
    n_iter: int,
    seed: int,
) -> pd.DataFrame:
    specs = [
        ("turnover_separation", "old_fraction_of_excess_rh"),
        ("dfs_total", "old_fraction_of_excess_rh"),
        ("obs_offset_resp_minus_bulk", "old_fraction_of_excess_rh"),
    ]
    rows: list[dict[str, object]] = []
    for i, (x_col, y_col) in enumerate(specs):
        if x_col == "turnover_separation":
            stat_samples = _regression_samples(
                posterior_draws.rename(columns={"turnover_separation": x_col, "old_fraction_of_excess_rh": y_col}),
                x_col,
                y_col,
                n_iter,
                seed + i,
            )
            summary = _summarize_stat_frame(stat_samples, "relationship").iloc[0].to_dict()
            rows.append(
                {
                    "predictor": x_col,
                    "response": y_col,
                    "n_sites": int(stat_samples["n"].median()),
                    **summary,
                }
            )
            continue

        stat_samples = _regression_samples(
            posterior_draws.rename(columns={"old_fraction_of_excess_rh": y_col}),
            x_col,
            y_col,
            n_iter,
            seed + i,
        )
        stat_samples = stat_samples.dropna(subset=["slope", "pearson_r", "spearman_rho"])
        summary = _summarize_stat_frame(stat_samples, "relationship").iloc[0].to_dict()
        rows.append(
            {
                "predictor": x_col,
                "response": y_col,
                "n_sites": int(stat_samples["n"].median()) if not stat_samples.empty else 0,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def _prior_structural_null(
    prior_draws: pd.DataFrame,
    observed_r: float,
    n_iter: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    by_site = {site: grp.reset_index(drop=True) for site, grp in prior_draws.groupby("site", sort=True)}
    rows = []
    for i in range(n_iter):
        sample_rows = []
        for site, grp in by_site.items():
            take = int(rng.integers(0, len(grp)))
            sample_rows.append(grp.iloc[take])
        sample = pd.DataFrame(sample_rows).dropna(subset=["turnover_separation", "old_fraction_of_excess_rh"])
        x = sample["turnover_separation"].to_numpy(dtype=float)
        y = sample["old_fraction_of_excess_rh"].to_numpy(dtype=float)
        rows.append(
            {
                "iteration": i,
                "n_sites": int(len(sample)),
                "pearson_r": _pearson(x, y),
                "spearman_rho": float(spearmanr(x, y, nan_policy="omit").statistic) if len(sample) >= 2 else np.nan,
            }
        )
    null_df = pd.DataFrame(rows)
    abs_null = np.abs(null_df["pearson_r"].to_numpy(dtype=float))
    p_emp = float((1.0 + np.sum(abs_null >= abs(observed_r))) / (len(abs_null) + 1.0))
    percentile = float(100.0 * np.mean(null_df["pearson_r"].to_numpy(dtype=float) <= observed_r))
    summary = pd.DataFrame(
        [
            {
                "null_definition": (
                    "Independent prior draws per site from the saved OE prior for "
                    "the optimized parameter vector (log_tau and log_f_transfer), "
                    "propagated through the existing +4 C, 100-year warming experiment."
                ),
                "observed_pearson_r": observed_r,
                "null_pearson_r_q025": float(np.quantile(null_df["pearson_r"], 0.025)),
                "null_pearson_r_median": float(np.quantile(null_df["pearson_r"], 0.5)),
                "null_pearson_r_q975": float(np.quantile(null_df["pearson_r"], 0.975)),
                "null_spearman_rho_q025": float(np.quantile(null_df["spearman_rho"], 0.025)),
                "null_spearman_rho_median": float(np.quantile(null_df["spearman_rho"], 0.5)),
                "null_spearman_rho_q975": float(np.quantile(null_df["spearman_rho"], 0.975)),
                "observed_percentile": percentile,
                "empirical_p_two_sided": p_emp,
                "n_iterations": int(n_iter),
            }
        ]
    )
    return null_df, summary


def _predicted_percentile_table(regression_samples: pd.DataFrame, site_metrics: pd.DataFrame) -> pd.DataFrame:
    x20 = float(np.quantile(site_metrics["turnover_separation_median"].dropna(), 0.2))
    x80 = float(np.quantile(site_metrics["turnover_separation_median"].dropna(), 0.8))
    rows = []
    rel_to_metric = {
        "old_fraction_of_excess_rh__vs__turnover_separation": "old_fraction_of_excess_rh",
        "frac_c_loss__vs__turnover_separation": "frac_c_loss",
        "dfs_total__vs__turnover_separation": "dfs_total",
    }
    for rel, metric in rel_to_metric.items():
        grp = regression_samples[regression_samples["relationship"] == rel]
        for label, xval in (("p20", x20), ("p80", x80)):
            pred = grp["intercept"].to_numpy(dtype=float) + grp["slope"].to_numpy(dtype=float) * xval
            q025, median, q975 = _quantiles(pred)
            rows.append(
                {
                    "relationship": rel,
                    "metric": metric,
                    "turnover_percentile": label,
                    "turnover_separation_value": xval,
                    "predicted_q025": q025,
                    "predicted_median": median,
                    "predicted_q975": q975,
                }
            )
    return pd.DataFrame(rows)
