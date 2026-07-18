"""
four_site_ak_gain_figures.py — 4-site AK and OE gain-matrix figures.

Runs the four canonical site inversions and exports:
  - a 5×5 averaging-kernel subset for the fitted parameter block
  - a 5×n_obs OE gain matrix for the same retrieved-parameter subset
  - a long annotated CSV for the plotted 5-parameter gain matrix

How to read the gain figure:
  - 2×2 layout: one site per panel
  - rows: the retrieved parameters being plotted
  - columns: scalar OE constraints in canonical observation-vector order
  - cell values: signed OE gain entries from G = Sx K^T Se^-1
  - larger magnitudes: stronger local retrieval sensitivity
  - sign: whether a positive perturbation in the constraint pushes the
    retrieved parameter up or down

Run from the repository root:
    python notebooks/four_site_ak_gain_figures.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NB_ROOT = os.path.join(_SCRIPT_ROOT, "notebooks")
_SRC_ROOT = os.path.join(_SCRIPT_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_SCRIPT_ROOT)

from ecosystem_complexity.oe_diagnostics import (  # noqa: E402
    fit_param_subset_labels,
    oe_gain_matrix_diagnostics,
)
from gain_obs_metadata import (  # noqa: E402
    _short_constraint_label,
    descriptive_labels_for_site,
)
from sites.canonical import run_barrow_canonical, run_hf_canonical  # noqa: E402
from sites.eight_mile_lake import run_eml_canonical  # noqa: E402
from sites.howland_forest import run_howland_canonical  # noqa: E402

C_BODY = "#2A2A2A"
C_PRIMARY = "#1F3A2E"


def _compute_site_diagnostics(site_runs: list[tuple[str, str, object]]) -> list[dict]:
    rows: list[dict] = []
    for site_label, site_id, run_fn in site_runs:
        data = run_fn()
        diag = oe_gain_matrix_diagnostics(
            data["model"],
            data["forcing"],
            data["state0_obs"],
            data["params_opt"],
            data["obs_full"],
            data["opt_fields"],
            extra_obs_blocks=data["extra_blocks"],
        )
        rows.append(
            {
                "site": site_label,
                "site_id": site_id,
                "data": data,
                "diag": diag,
            }
        )
    return rows


def _build_gain_annotated_rows(site_diags: list[dict]) -> list[dict]:
    rows: list[dict] = []
    param_labels = fit_param_subset_labels()
    for row in site_diags:
        diag = row["diag"]
        gain = diag["subset_gain_matrix"]
        obs_annotations = diag["obs_annotations"]
        for i, (param_name, param_label) in enumerate(
            zip(diag["subset_state_names"], param_labels)
        ):
            abs_rank = np.argsort(np.abs(gain[i]))[::-1]
            rank_lookup = {int(obs_idx): rank + 1 for rank, obs_idx in enumerate(abs_rank)}
            for obs in obs_annotations:
                obs_index = int(obs["obs_index"])
                rows.append(
                    {
                        "site": row["site"],
                        "site_id": row["site_id"],
                        "param_index_subset": i,
                        "param_name": param_name,
                        "param_label_short": param_label,
                        "obs_index": obs_index,
                        "obs_block_name": obs["obs_block_name"],
                        "obs_family": obs["obs_family"],
                        "obs_label_short": _short_constraint_label(obs["obs_label_full"]),
                        "obs_label_full": obs["obs_label_full"],
                        "y_obs": obs["y_obs"],
                        "y_prior": obs["y_prior"],
                        "y_opt": obs["y_opt"],
                        "obs_sigma": obs["obs_sigma"],
                        "obs_variance": obs["obs_variance"],
                        "gain_value": float(gain[i, obs_index]),
                        "abs_gain_value": float(abs(gain[i, obs_index])),
                        "rank_within_param_by_abs_gain": int(rank_lookup[obs_index]),
                    }
                )
    return rows


def _plot_ak_figure(site_diags: list[dict], out_path: str) -> None:
    param_labels = fit_param_subset_labels()
    panels = [row["diag"]["subset_averaging_kernel"] for row in site_diags]
    vmax = max(float(np.max(np.abs(a))) for a in panels)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("BrBG")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    im = None
    for ax, row in zip(axes, site_diags):
        A = row["diag"]["subset_averaging_kernel"]
        im = ax.imshow(A, cmap=cmap, norm=norm, aspect="equal")
        n = A.shape[0]
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(param_labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(param_labels, fontsize=10)
        ax.set_xlabel("True parameter", fontsize=10, color=C_BODY)
        ax.set_ylabel("Retrieved parameter", fontsize=10, color=C_BODY)
        ax.set_title(
            f"{row['site']} ({row['site_id']})\n"
            f"trace(A subset) = {np.trace(A):.3f}",
            fontsize=11,
            color=C_PRIMARY,
            fontweight="bold",
        )
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0)
        for i in range(n):
            for j in range(n):
                v = A[i, j]
                txt_color = "white" if abs(v) > 0.45 * vmax else C_BODY
                ax.text(
                    j,
                    i,
                    f"{v:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=txt_color,
                )

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Averaging-kernel value", fontsize=10, color=C_BODY)
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle(
        "Canonical fitted-parameter averaging-kernel subset across all sites",
        fontsize=14,
        color=C_PRIMARY,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"saved {out_path}")


def _plot_gain_figure(site_diags: list[dict], out_path: str) -> None:
    param_labels = fit_param_subset_labels()
    panels = [row["diag"]["subset_gain_matrix"] for row in site_diags]
    vmax = max(float(np.max(np.abs(g))) for g in panels)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    fig, axes = plt.subplots(2, 2, figsize=(28, 18))
    axes = axes.ravel()
    im = None
    for ax, row in zip(axes, site_diags):
        G = row["diag"]["subset_gain_matrix"]
        labels = descriptive_labels_for_site(row)
        # Font shrinks as the column count grows so long "pool · date" labels
        # stay legible on the densest panel (Harvard Forest, ~50 columns).
        n_cols = len(labels)
        xfont = 6.5 if n_cols <= 20 else (5.0 if n_cols <= 35 else 3.8)
        im = ax.imshow(G, cmap=cmap, norm=norm, aspect="auto")
        ax.set_yticks(np.arange(len(param_labels)))
        ax.set_yticklabels(param_labels, fontsize=10)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=xfont)
        ax.set_xlabel(
            "Observation (Δ¹⁴C measurement · pool · date, or C stock · pool)",
            fontsize=10,
            color=C_BODY,
        )
        ax.set_ylabel("Retrieved parameter", fontsize=10, color=C_BODY)
        ax.set_title(
            f"{row['site']} ({row['site_id']})\n"
            f"G shape = {G.shape[0]}×{G.shape[1]}",
            fontsize=11,
            color=C_PRIMARY,
            fontweight="bold",
        )
        ax.set_yticks(np.arange(-0.5, G.shape[0], 1), minor=True)
        ax.grid(which="minor", axis="y", color="white", linewidth=1.1)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0)

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.018, pad=0.02)
    cbar.set_label("OE gain value", fontsize=10, color=C_BODY)
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle(
        "Canonical fitted-parameter OE gain matrix across all sites",
        fontsize=14,
        color=C_PRIMARY,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.965))
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"saved {out_path}")


def main() -> None:
    export_dir = os.path.join(_NB_ROOT, "exports")
    os.makedirs(export_dir, exist_ok=True)
    site_runs = [
        ("Harvard Forest", "US-Ha1", run_hf_canonical),
        ("Barrow", "US-A10", run_barrow_canonical),
        ("Howland Forest", "US-Ho1", run_howland_canonical),
        ("Eight-mile Lake", "US-EML", run_eml_canonical),
    ]
    print("Running canonical site inversions for AK / gain figures…")
    print(
        "Gain figure: rows are the retrieved fitted parameters, columns are scalar "
        "constraints in canonical OE order, and signed colors show how a "
        "positive constraint perturbation shifts the retrieved parameter."
    )
    site_diags = _compute_site_diagnostics(site_runs)
    _plot_ak_figure(
        site_diags,
        os.path.join(_NB_ROOT, "four_site_averaging_kernel_fit_params.png"),
    )
    _plot_gain_figure(
        site_diags,
        os.path.join(_NB_ROOT, "four_site_gain_matrix_fit_params_vs_constraints.pdf"),
    )
    gain_rows = _build_gain_annotated_rows(site_diags)
    gain_df = pd.DataFrame(gain_rows)
    gain_csv_path = os.path.join(
        export_dir, "four_site_gain_matrix_annotated_subset.csv"
    )
    gain_df.to_csv(gain_csv_path, index=False)
    print(f"saved {gain_csv_path}")


if __name__ == "__main__":
    main()
