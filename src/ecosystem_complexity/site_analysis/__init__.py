"""Standardized site-fit artifact export, metrics, and plotting."""

from __future__ import annotations

import json
import math
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecosystem_complexity.inference._helpers import build_oe_prior_sigma
from ecosystem_complexity.inference.diagnostics import (
    constraint_orthogonality_from_context,
    cumulative_ladder_from_context,
    oe_gain_matrix_diagnostics,
    oe_ladder_context,
    oe_style_ablation,
    shapley_dfs_attribution_from_context,
)
from ecosystem_complexity.model.state import make_default_params
from ecosystem_complexity.synthesis.analysis import compute_age_diagnostics


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _rows_for_families(
    rows_by_family: dict[str, list[int]],
    families: tuple[str, ...],
) -> list[int]:
    rows: list[int] = []
    for family in families:
        rows.extend(rows_by_family.get(family, []))
    return sorted(rows)


def _dfs_from_rows(k_tilde: np.ndarray, rows: list[int]) -> tuple[float, np.ndarray]:
    n_state = k_tilde.shape[1]
    if not rows:
        return 0.0, np.zeros(n_state)
    kt = k_tilde[rows, :]
    information = kt.T @ kt
    averaging_kernel = np.linalg.solve(information + np.eye(n_state), information)
    return float(np.trace(averaging_kernel)), np.diag(averaging_kernel)


def _ablation_from_context(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = [
        ("C_stocks", ("C_stocks",)),
        ("pool_delta14C", ("bulk_14C", "fraction_14C")),
        ("resp_delta14C", ("resp_14C",)),
        ("C_stocks+pool_delta14C", ("C_stocks", "bulk_14C", "fraction_14C")),
        (
            "C_stocks+pool_delta14C+resp_delta14C",
            ("C_stocks", "bulk_14C", "fraction_14C", "resp_14C"),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for label, families in scenarios:
        active_rows = _rows_for_families(ctx["rows_by_family"], families)
        dfs, dfs_diag = _dfs_from_rows(ctx["k_tilde"], active_rows)
        rows.append(
            {
                "scenario": label,
                "families": list(families),
                "n_obs": len(active_rows),
                "dfs_total": dfs,
                "dfs_per_param": dfs_diag.tolist(),
            }
        )
    return rows


def compute_information_metrics(
    *,
    K: np.ndarray,
    Se_diag: np.ndarray,
    Sx: np.ndarray,
    averaging_kernel: np.ndarray,
    y_obs: np.ndarray,
    y_prior: np.ndarray,
    y_opt: np.ndarray,
    cost_final: float,
    rows_by_family: dict[str, list[int]] | None = None,
    k_tilde: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute fit-quality and information-content metrics from exported arrays."""
    n_obs = int(y_obs.shape[0])
    n_params = int(Sx.shape[0])
    dof_nominal = max(n_obs - n_params, 1)
    residual_prior = y_obs - y_prior
    residual_opt = y_obs - y_opt
    weighted_residual_opt = residual_opt / np.sqrt(np.maximum(Se_diag, 1e-30))
    fim = (K.T / np.maximum(Se_diag, 1e-30)) @ K

    metrics: dict[str, Any] = {
        "n_obs": n_obs,
        "n_params": n_params,
        "dof_nominal": dof_nominal,
        "cost_final": float(cost_final),
        "reduced_chi2": float(cost_final / dof_nominal),
        "rmse_prior": float(np.sqrt(np.mean(residual_prior**2))),
        "rmse_opt": float(np.sqrt(np.mean(residual_opt**2))),
        "weighted_rmse_opt": float(np.sqrt(np.mean(weighted_residual_opt**2))),
        "dfs_total": float(np.trace(averaging_kernel)),
        "averaging_kernel_trace": float(np.trace(averaging_kernel)),
        "posterior_variance_trace": float(np.trace(Sx)),
        "gain_matrix_rank": int(np.linalg.matrix_rank(Sx @ (K.T / np.maximum(Se_diag, 1e-30)))),
        "jacobian_rank": int(np.linalg.matrix_rank(K)),
        "fim_condition_number": float(np.linalg.cond(fim + np.eye(fim.shape[0]) * 1e-12)),
    }

    if rows_by_family is not None and k_tilde is not None:
        ctx = {
            "k_tilde": k_tilde,
            "rows_by_family": rows_by_family,
            "n_obs_per_family": {family: len(rows) for family, rows in rows_by_family.items()},
            "n_state": int(k_tilde.shape[1]),
            "state_names": [],
        }
        metrics["constraint_ladder"] = cumulative_ladder_from_context(ctx)
        metrics["shapley"] = shapley_dfs_attribution_from_context(ctx)
        metrics["ablation"] = _ablation_from_context(ctx)
        metrics["orthogonality"] = constraint_orthogonality_from_context(ctx)

    return metrics


def analyze_site_run(site_run: dict[str, Any]) -> dict[str, Any]:
    """Compute standardized diagnostics for a completed canonical site fit."""
    required = {"model", "forcing", "state0", "params_opt", "obs_full", "oe_result"}
    missing = required.difference(site_run)
    if missing:
        raise KeyError(f"site_run is missing required keys: {sorted(missing)}")

    model = site_run["model"]
    forcing = site_run["forcing"]
    # Reuse the state supplied to ``optimize_oe``.  Its forward operator
    # analytically replaces C12 at each parameter value while retaining this
    # state's Δ14C initialization.  ``state_at_map`` has a different Δ14C
    # initialization, so using it here reconstructs a different Jacobian and
    # makes the exported DFS disagree with the fitted OE result.
    state0 = site_run["state0"]
    params_opt = site_run["params_opt"]
    obs_full = site_run["obs_full"]
    opt_fields = tuple(site_run.get("opt_fields", ()))
    extra_blocks = list(site_run.get("extra_blocks", site_run.get("pool_blocks", [])))
    oe_result = site_run["oe_result"]

    gain = oe_gain_matrix_diagnostics(
        model,
        forcing,
        state0,
        params_opt,
        obs_full,
        opt_fields=opt_fields,
        extra_obs_blocks=extra_blocks,
    )
    # The exported diagnostics are a reconstruction of the OE linearisation.
    # Do not silently publish a second, inconsistent averaging kernel if either
    # code path changes its state construction or observation operator.
    fitted_kernel = np.asarray(oe_result.averaging_kernel, dtype=float)
    if not np.allclose(
        np.asarray(gain["averaging_kernel"], dtype=float),
        fitted_kernel,
        rtol=1e-5,
        # Equivalent JAX Jacobian evaluations can differ at ~1e-6 because of
        # floating-point reduction order; this still rejects a changed
        # linearization state or observation operator.
        atol=2e-6,
    ):
        raise RuntimeError(
            "Exported OE diagnostics do not reproduce the fitted averaging "
            "kernel. Use the inversion's linearization state and forward "
            "operator before exporting DFS."
        )
    ladder_ctx = oe_ladder_context(
        model,
        forcing,
        state0,
        params_opt,
        obs_full,
        opt_fields=opt_fields,
        extra_obs_blocks=extra_blocks,
    )
    ablation = oe_style_ablation(
        model,
        forcing,
        state0,
        params_opt,
        obs_full,
        opt_fields=opt_fields,
        extra_obs_blocks=extra_blocks,
    )
    ladder = cumulative_ladder_from_context(ladder_ctx)
    shapley = shapley_dfs_attribution_from_context(ladder_ctx)
    orthogonality = constraint_orthogonality_from_context(ladder_ctx)

    params0 = make_default_params(model.config)
    sa_sigma = np.array(build_oe_prior_sigma(model.config, params0, opt_fields), dtype=float)
    sa_diag = sa_sigma**2

    metrics = compute_information_metrics(
        K=np.asarray(gain["K"], dtype=float),
        Se_diag=np.asarray(gain["Se_diag"], dtype=float),
        Sx=np.asarray(gain["Sx"], dtype=float),
        averaging_kernel=np.asarray(gain["averaging_kernel"], dtype=float),
        y_obs=np.asarray(gain["y_obs"], dtype=float),
        y_prior=np.asarray(gain["y_prior"], dtype=float),
        y_opt=np.asarray(gain["y_opt"], dtype=float),
        cost_final=float(np.asarray(oe_result.cost_history)[-1]),
        rows_by_family=dict(ladder_ctx["rows_by_family"]),
        k_tilde=np.asarray(ladder_ctx["k_tilde"], dtype=float),
    )
    metrics["ablation"] = [
        {"scenario": key, **value} for key, value in ablation.items()
    ]
    metrics["constraint_ladder"] = ladder
    metrics["shapley"] = shapley
    metrics["orthogonality"] = orthogonality

    return {
        "gain": gain,
        "ladder_context": ladder_ctx,
        "metrics": metrics,
        "sa_diag": sa_diag,
        "sa_sigma": sa_sigma,
    }


def plot_site_run(site_run: dict[str, Any], analysis: dict[str, Any], output_path: str) -> str:
    """Create a standardized 5-panel site summary figure."""
    out_prior = site_run["out_prior"]
    out_opt = site_run["out_opt"]
    forcing = site_run["forcing"]
    obs_full = site_run["obs_full"]
    params_prior = site_run["params_prior"]
    params_opt = site_run["params_opt"]
    model = site_run["model"]

    gain = analysis["gain"]
    metrics = analysis["metrics"]
    ablation = pd.DataFrame(metrics["ablation"])
    ladder = pd.DataFrame(metrics["constraint_ladder"])

    age_prior = compute_age_diagnostics(out_prior, params_prior, model)
    age_opt = compute_age_diagnostics(out_opt, params_opt, model)

    time_years = 1970.0 + np.asarray(forcing.time, dtype=float) / 365.25
    total_c_prior = np.asarray(out_prior.C12).sum(axis=1)
    total_c_opt = np.asarray(out_opt.C12).sum(axis=1)
    obs_resp = np.asarray(obs_full.delta14C_resp) if obs_full.delta14C_resp is not None else None

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.1])
    ax_gpp = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_resp = fig.add_subplot(gs[1, 0])
    ax_fit = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[2, :])

    ax_gpp.plot(time_years, np.asarray(forcing.GPP_obs) * 365.25, color="forestgreen", lw=1.1)
    ax_gpp.set_title("GPP Forcing", loc="left")
    ax_gpp.set_ylabel("gC m$^{-2}$ yr$^{-1}$")
    ax_gpp.set_xlabel("Year")
    ax_gpp.grid(alpha=0.3, lw=0.4)

    ax_c.plot(time_years, total_c_prior, label="Prior", color="0.5", ls=":")
    ax_c.plot(time_years, total_c_opt, label="MAP", color="tab:blue")
    ax_c.set_title("Total Soil Carbon", loc="left")
    ax_c.set_ylabel("gC m$^{-2}$")
    ax_c.set_xlabel("Year")
    ax_c.grid(alpha=0.3, lw=0.4)
    ax_c.legend(framealpha=0.9, fontsize=8)

    ax_resp.plot(time_years, age_prior.respired_delta14C, color="0.5", ls=":", label="Prior")
    ax_resp.plot(time_years, age_opt.respired_delta14C, color="tab:red", label="MAP")
    if obs_resp is not None:
        mask = np.isfinite(obs_resp)
        ax_resp.scatter(time_years[mask], obs_resp[mask], s=18, c="black", alpha=0.55, label="Obs")
    ax_resp.set_title("Respired Δ$^{14}$C", loc="left")
    ax_resp.set_ylabel("‰")
    ax_resp.set_xlabel("Year")
    ax_resp.grid(alpha=0.3, lw=0.4)
    ax_resp.legend(framealpha=0.9, fontsize=8)

    ax_fit.scatter(np.asarray(gain["y_obs"]), np.asarray(gain["y_opt"]), s=16, alpha=0.75)
    lo = float(min(np.min(gain["y_obs"]), np.min(gain["y_opt"])))
    hi = float(max(np.max(gain["y_obs"]), np.max(gain["y_opt"])))
    ax_fit.plot([lo, hi], [lo, hi], color="0.3", ls=":")
    ax_fit.set_title("Observation Fit", loc="left")
    ax_fit.set_xlabel("Observed")
    ax_fit.set_ylabel("Posterior prediction")
    ax_fit.grid(alpha=0.3, lw=0.4)

    x0 = np.arange(len(ablation))
    x1 = np.arange(len(ladder))
    width = 0.42
    ax_info.bar(x0 - width / 2.0, ablation["dfs_total"], width, label="Ablation DFS", color="tab:blue", alpha=0.8)
    ax_info.bar(
        np.linspace(len(ablation) + 1, len(ablation) + len(ladder), len(ladder)),
        ladder["dfs_cumulative"],
        width,
        label="Cumulative ladder DFS",
        color="tab:orange",
        alpha=0.8,
    )
    tick_positions = list(x0) + list(np.linspace(len(ablation) + 1, len(ablation) + len(ladder), len(ladder)))
    tick_labels = list(ablation["scenario"]) + list(ladder["label"])
    ax_info.set_xticks(tick_positions)
    ax_info.set_xticklabels(tick_labels, rotation=20, ha="right", fontsize=8)
    ax_info.set_ylabel("Degrees of freedom for signal")
    ax_info.set_title(
        f"Information Content  |  reduced χ²={metrics['reduced_chi2']:.2f}, RMSE={metrics['rmse_opt']:.2f}",
        loc="left",
    )
    ax_info.grid(alpha=0.3, lw=0.4, axis="y")
    ax_info.legend(framealpha=0.9, fontsize=8)

    site_label = site_run["spec"].label if "spec" in site_run else model.config.site_name
    fig.suptitle(site_label, fontsize=13)
    fig.tight_layout()
    dest = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def export_site_run(site_run: dict[str, Any], export_dir: str) -> dict[str, str]:
    """Export matrices, tables, JSON summary, and a diagnostic figure."""
    analysis = analyze_site_run(site_run)
    gain = analysis["gain"]
    ladder_ctx = analysis["ladder_context"]
    metrics = analysis["metrics"]

    out_dir = os.path.abspath(export_dir)
    os.makedirs(out_dir, exist_ok=True)

    matrices_path = os.path.join(out_dir, "fit_matrices.npz")
    np.savez(
        matrices_path,
        K=np.asarray(gain["K"], dtype=float),
        Se_diag=np.asarray(gain["Se_diag"], dtype=float),
        Sx=np.asarray(gain["Sx"], dtype=float),
        averaging_kernel=np.asarray(gain["averaging_kernel"], dtype=float),
        gain_matrix=np.asarray(gain["gain_matrix"], dtype=float),
        y_obs=np.asarray(gain["y_obs"], dtype=float),
        y_prior=np.asarray(gain["y_prior"], dtype=float),
        y_opt=np.asarray(gain["y_opt"], dtype=float),
        sa_diag=np.asarray(analysis["sa_diag"], dtype=float),
        k_tilde=np.asarray(ladder_ctx["k_tilde"], dtype=float),
    )

    obs_csv = os.path.join(out_dir, "observations.csv")
    pd.DataFrame(gain["obs_annotations"]).to_csv(obs_csv, index=False)

    ladder_csv = os.path.join(out_dir, "constraint_ladder.csv")
    pd.DataFrame(metrics["constraint_ladder"]).to_csv(ladder_csv, index=False)

    shapley_csv = os.path.join(out_dir, "shapley.csv")
    pd.DataFrame(metrics["shapley"]).to_csv(shapley_csv, index=False)

    ablation_csv = os.path.join(out_dir, "ablation.csv")
    pd.DataFrame(metrics["ablation"]).to_csv(ablation_csv, index=False)

    summary_path = os.path.join(out_dir, "summary.json")
    summary = {
        "site_label": site_run["spec"].label if "spec" in site_run else site_run["model"].config.site_name,
        "config_path": site_run.get("config_path"),
        "observation_path": site_run.get("observation_path"),
        "state_names": list(gain["subset_state_names"]) if gain.get("subset_state_names") else list(site_run["oe_result"].state_names),
        "all_state_names": list(site_run["oe_result"].state_names),
        "constraint_labels": list(gain["constraint_labels"]),
        "rows_by_family": {k: list(v) for k, v in ladder_ctx["rows_by_family"].items()},
        "n_obs_per_family": {k: int(v) for k, v in ladder_ctx["n_obs_per_family"].items()},
        "metrics": metrics,
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)

    figure_path = plot_site_run(site_run, analysis, os.path.join(out_dir, "site_diagnostics.png"))
    return {
        "export_dir": out_dir,
        "matrices": matrices_path,
        "observations": obs_csv,
        "constraint_ladder": ladder_csv,
        "shapley": shapley_csv,
        "ablation": ablation_csv,
        "summary": summary_path,
        "figure": figure_path,
    }


def load_exported_analysis(export_dir: str) -> dict[str, Any]:
    """Load exported matrices and recompute metrics without re-running the fit."""
    out_dir = os.path.abspath(export_dir)
    matrices = np.load(os.path.join(out_dir, "fit_matrices.npz"))
    with open(os.path.join(out_dir, "summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    metrics = compute_information_metrics(
        K=np.asarray(matrices["K"], dtype=float),
        Se_diag=np.asarray(matrices["Se_diag"], dtype=float),
        Sx=np.asarray(matrices["Sx"], dtype=float),
        averaging_kernel=np.asarray(matrices["averaging_kernel"], dtype=float),
        y_obs=np.asarray(matrices["y_obs"], dtype=float),
        y_prior=np.asarray(matrices["y_prior"], dtype=float),
        y_opt=np.asarray(matrices["y_opt"], dtype=float),
        cost_final=float(summary["metrics"]["cost_final"]),
        rows_by_family={k: list(v) for k, v in summary["rows_by_family"].items()},
        k_tilde=np.asarray(matrices["k_tilde"], dtype=float),
    )
    return {"summary": summary, "metrics": metrics}
