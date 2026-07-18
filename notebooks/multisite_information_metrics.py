"""
multisite_information_metrics.py — Compute DFS and uncertainty reduction for
the multisite canonical inversions.

Runs the current multisite inversions site-by-site, derives information metrics
from each site's `OEResult`, and writes:

    notebooks/exports/multisite_information_metrics.csv
    notebooks/exports/multisite_information_metrics_long.csv

The summary CSV has one row per site. The long CSV has one row per fitted
parameter per site with prior/posterior sigma, uncertainty reduction, and the
averaging-kernel diagonal contribution.

Run:
    python notebooks/multisite_information_metrics.py
    python notebooks/multisite_information_metrics.py --sites FLONA ZF2
    python notebooks/multisite_information_metrics.py --max-workers 3
"""
from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd

_NB_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_NB_ROOT)
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ecosystem_complexity._oe_helpers import build_oe_prior_sigma
from ecosystem_complexity.state import make_default_params
from sites.multisite_canonical import (
    OPT_FIELDS,
    discover_site_specs,
    run_site_canonical,
)

OUT_SUMMARY = os.path.join(_NB_ROOT, "exports", "multisite_information_metrics.csv")
OUT_LONG = os.path.join(_NB_ROOT, "exports", "multisite_information_metrics_long.csv")


def _summarize_indices(names: list[str], prefix: str) -> list[int]:
    return [i for i, name in enumerate(names) if name.startswith(prefix)]


def _safe_mean(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _run_one(site_name: str) -> tuple[dict, list[dict]]:
    spec = discover_site_specs()[site_name]
    result = run_site_canonical(spec, observation_path=spec.observation_path)
    if result.get("skipped"):
        return (
            {
                "site": spec.israd_name,
                "label": spec.label,
                "tower_id": spec.tower_id,
                "biome": spec.biome,
                "skipped": True,
                "converged": False,
            },
            [],
        )

    oe = result["oe_result"]
    A = np.array(oe.averaging_kernel, dtype=float)
    Sx = np.array(oe.Sx, dtype=float)
    state_names = list(oe.state_names)
    prior_sigma = np.array(
        build_oe_prior_sigma(
            result["model"].config,
            make_default_params(result["model"].config),
            tuple(OPT_FIELDS),
        ),
        dtype=float,
    )
    post_sigma = np.sqrt(np.clip(np.diag(Sx), 0.0, None))
    ur = 1.0 - post_sigma / prior_sigma
    dfs_diag = np.diag(A)

    tau_idx = _summarize_indices(state_names, "log_tau[")
    transfer_idx = _summarize_indices(state_names, "log_f_transfer[")

    summary = {
        "site": spec.israd_name,
        "label": spec.label,
        "tower_id": spec.tower_id,
        "biome": spec.biome,
        "skipped": False,
        "converged": bool(result["converged"]),
        "n_iter": int(result["n_iter"]),
        "n_obs": int(np.array(oe.y_obs).shape[0]),
        "n_cstock": int(result["n_cstock"]),
        "n_pool_blocks": int(result["n_pool_blocks"]),
        "n_resp": int(result["n_resp"]),
        "mean_GPP_gCm2yr": float(result["mean_gpp_gCm2yr"]),
        "SOC_gCm2": float(result["soc_total_gCm2"]),
        "J0": float(result["cost0"]),
        "J_final": float(result["cost_final"]),
        "dfs_total": float(np.trace(A)),
        "dfs_tau": float(np.sum(dfs_diag[tau_idx])),
        "dfs_transfer": float(np.sum(dfs_diag[transfer_idx])),
        "ur_mean": _safe_mean(ur),
        "ur_tau_mean": _safe_mean(ur[tau_idx]),
        "ur_transfer_mean": _safe_mean(ur[transfer_idx]),
        "ur_tau_active": float(ur[tau_idx[0]]) if len(tau_idx) > 0 else float("nan"),
        "ur_tau_slow": float(ur[tau_idx[1]]) if len(tau_idx) > 1 else float("nan"),
        "ur_tau_passive": float(ur[tau_idx[2]]) if len(tau_idx) > 2 else float("nan"),
    }

    rows: list[dict] = []
    for i, name in enumerate(state_names):
        rows.append(
            {
                "site": spec.israd_name,
                "label": spec.label,
                "tower_id": spec.tower_id,
                "biome": spec.biome,
                "state_name": name,
                "prior_sigma": float(prior_sigma[i]),
                "posterior_sigma": float(post_sigma[i]),
                "uncertainty_reduction": float(ur[i]),
                "averaging_kernel_diagonal": float(dfs_diag[i]),
            }
        )

    return summary, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sites", nargs="*", default=sorted(discover_site_specs()),
        help="Config stems to run (default: every config in configs/multisite/).",
    )
    parser.add_argument("--max-workers", type=int, default=3, help="Process count for parallel execution.")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)
    ctx = mp.get_context("spawn")

    summary_rows: list[dict] = []
    long_rows: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_run_one, site_name): site_name for site_name in args.sites}
        for fut in concurrent.futures.as_completed(futures):
            site_name = futures[fut]
            try:
                summary, rows = fut.result()
            except Exception as exc:  # noqa: BLE001
                summary_rows.append(
                    {
                        "site": site_name,
                        "label": site_name,
                        "skipped": False,
                        "converged": False,
                        "error": str(exc),
                    }
                )
                continue
            summary_rows.append(summary)
            long_rows.extend(rows)

    summary_df = pd.DataFrame(summary_rows).sort_values("label")
    long_df = pd.DataFrame(long_rows).sort_values(["label", "state_name"])
    summary_df.to_csv(OUT_SUMMARY, index=False)
    long_df.to_csv(OUT_LONG, index=False)

    print(f"Saved summary → {os.path.relpath(OUT_SUMMARY, _REPO_ROOT)}")
    print(f"Saved long    → {os.path.relpath(OUT_LONG, _REPO_ROOT)}")
    if not summary_df.empty:
        cols = [
            "site",
            "converged",
            "dfs_total",
            "dfs_tau",
            "ur_tau_mean",
            "ur_tau_active",
            "ur_tau_slow",
            "ur_tau_passive",
            "J_final",
        ]
        cols = [c for c in cols if c in summary_df.columns]
        print(summary_df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
