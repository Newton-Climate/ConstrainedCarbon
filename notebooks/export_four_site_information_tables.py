"""
export_four_site_information_tables.py — Export four-site OE information tables.

Runs the current canonical OE analyses for:
  - Harvard Forest (US-Ha1)
  - Barrow (US-A10)
  - Eight-mile Lake (US-EML)
  - Howland Forest (US-Ho1)

and writes:
  1. a combined one-constraint-at-a-time ladder CSV
  2. a combined annotated averaging-kernel summary CSV
  3. one labeled averaging-kernel matrix CSV per site

Run from the repository root:
    python notebooks/export_four_site_information_tables.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NB_ROOT = os.path.join(_SCRIPT_ROOT, "notebooks")
_SRC_ROOT = os.path.join(_SCRIPT_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_SCRIPT_ROOT)

from sites.canonical import run_hf_canonical, run_barrow_canonical, site_information_analysis
from sites.eight_mile_lake import run_eml_canonical
from sites.howland_forest import run_howland_canonical


def _slug(label: str) -> str:
    return label.lower().replace(" ", "_").replace("-", "_")


def _group_lookup(param_groups: dict[str, list[int]] | None, n_params: int) -> list[str]:
    groups = ["unassigned"] * n_params
    for group_name, idxs in (param_groups or {}).items():
        for i in idxs:
            if 0 <= i < n_params:
                groups[i] = group_name
    return groups


def _build_kernel_summary_rows(site_label: str, site_id: str, analysis: dict) -> list[dict]:
    dof = analysis["dof_full"]
    post = analysis["post_full"]
    param_names = dof.param_names or post.param_names or []
    n_params = len(param_names)
    groups = _group_lookup(dof.param_groups, n_params)
    A = np.array(dof.averaging_kernel, dtype=float)
    dfs_per = np.array(dof.dfs_per_param, dtype=float)
    prior_sigma = np.array(post.prior_sigma, dtype=float)
    post_sigma = np.array(post.posterior_sigma, dtype=float)
    ur = np.array(post.uncertainty_reduction, dtype=float)

    rows: list[dict] = []
    for i, name in enumerate(param_names):
        row = A[i, :]
        abs_row = np.abs(row)
        if abs_row.size > 1:
            offdiag = abs_row.copy()
            offdiag[i] = -np.inf
            j = int(np.argmax(offdiag))
            strongest_cross = param_names[j]
            strongest_cross_val = float(row[j])
        else:
            strongest_cross = ""
            strongest_cross_val = np.nan
        rows.append({
            "site": site_label,
            "site_id": site_id,
            "param_index": i,
            "param_name": name,
            "param_group": groups[i],
            "dfs_per_param": float(dfs_per[i]),
            "ak_diag": float(A[i, i]),
            "prior_sigma": float(prior_sigma[i]),
            "posterior_sigma": float(post_sigma[i]),
            "uncertainty_reduction_fraction": float(ur[i]),
            "uncertainty_reduction_percent": float(100.0 * ur[i]),
            "strongest_cross_parameter": strongest_cross,
            "strongest_cross_kernel_value": strongest_cross_val,
        })
    return rows


def _build_kernel_matrix_df(site_label: str, site_id: str, analysis: dict) -> pd.DataFrame:
    dof = analysis["dof_full"]
    param_names = dof.param_names or []
    A = np.array(dof.averaging_kernel, dtype=float)
    groups = _group_lookup(dof.param_groups, len(param_names))
    df = pd.DataFrame(A, columns=param_names)
    df.insert(0, "param_group", groups)
    df.insert(0, "param_name", param_names)
    df.insert(0, "param_index", np.arange(len(param_names), dtype=int))
    df.insert(0, "site_id", site_id)
    df.insert(0, "site", site_label)
    return df


def main() -> None:
    export_dir = os.path.join(_NB_ROOT, "exports")
    os.makedirs(export_dir, exist_ok=True)

    site_runs = [
        ("Harvard Forest", "US-Ha1", run_hf_canonical),
        ("Barrow", "US-A10", run_barrow_canonical),
        ("Eight-mile Lake", "US-EML", run_eml_canonical),
        ("Howland Forest", "US-Ho1", run_howland_canonical),
    ]

    ladder_rows: list[dict] = []
    kernel_summary_rows: list[dict] = []

    for site_label, site_id, run_fn in site_runs:
        data = run_fn()
        analysis = site_information_analysis(data)

        for rank, rung in enumerate(
            sorted(analysis["ladder"], key=lambda r: (-float(r["dfs"]), r["label"])),
            start=1,
        ):
            ladder_rows.append({
                "site": site_label,
                "site_id": site_id,
                "rank_within_site": rank,
                "constraint_label": rung["label"],
                "obs_type": rung["obs_type"],
                "n_obs": int(rung["n_obs"]),
                "dfs": float(rung["dfs"]),
            })

        kernel_summary_rows.extend(_build_kernel_summary_rows(site_label, site_id, analysis))

        kernel_matrix = _build_kernel_matrix_df(site_label, site_id, analysis)
        kernel_matrix_path = os.path.join(
            export_dir, f"{_slug(site_label)}_averaging_kernel_matrix.csv"
        )
        kernel_matrix.to_csv(kernel_matrix_path, index=False)
        print(f"Saved matrix:   {kernel_matrix_path}")

    ladder_df = pd.DataFrame(ladder_rows)
    ladder_path = os.path.join(export_dir, "four_site_constraint_ladder.csv")
    ladder_df.to_csv(ladder_path, index=False)

    kernel_summary_df = pd.DataFrame(kernel_summary_rows)
    kernel_summary_path = os.path.join(
        export_dir, "four_site_annotated_averaging_kernel_summary.csv"
    )
    kernel_summary_df.to_csv(kernel_summary_path, index=False)

    print(f"Saved ladder:   {ladder_path}")
    print(f"Saved summary:  {kernel_summary_path}")


if __name__ == "__main__":
    main()
