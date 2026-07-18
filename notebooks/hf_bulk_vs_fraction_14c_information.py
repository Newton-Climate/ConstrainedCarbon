"""
hf_bulk_vs_fraction_14c_information.py — What does bulk ¹⁴C add?

Runs the canonical Harvard Forest 3-pool OE inversion three ways, changing only
the ISRaD pool-Δ¹⁴C constraint while holding C stocks and respired Δ¹⁴C fixed:

    fraction   density fractions  (free light→active, occluded→slow, heavy→passive)
    bulk       whole-soil layer Δ¹⁴C (McFarlane 2013 H1–H5) binned to pools by depth
    both       fraction + bulk together

For each run we report (i) the DFS-by-observation-type ablation at the MAP
estimate, (ii) the one-constraint-at-a-time ladder, and (iii) the posterior
uncertainty on each pool turnover time log τ.  The comparison isolates the
*information content* of bulk ¹⁴C:

  • bulk vs fraction   — which decomposition resolves more of the 3-pool system,
                          and which pools it pins;
  • both vs fraction   — the *marginal* information bulk adds on top of fractions.

Run from the repository root:
    python notebooks/hf_bulk_vs_fraction_14c_information.py

Output
------
  notebooks/hf_bulk_vs_fraction_14c_information.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# ── Paths ───────────────────────────────────────────────────────────────────
_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NB_ROOT = os.path.join(_SCRIPT_ROOT, "notebooks")
_SRC_ROOT = os.path.join(_SCRIPT_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_SCRIPT_ROOT)

import jax.numpy as jnp  # noqa: E402

from sites.canonical import (  # noqa: E402
    run_hf_canonical,
    _canonical_prior_sigma,
)
from ecosystem_complexity.oe_diagnostics import (  # noqa: E402
    oe_constraint_ladder,
    oe_style_ablation,
)
from ecosystem_complexity.information import (  # noqa: E402
    OBS_C_STOCKS, OBS_POOL_D14C, OBS_RESP_D14C,
)

_SOURCES = ["fraction", "bulk", "both"]
_SOURCE_LABELS = {
    "fraction": "Fraction Δ¹⁴C\n(+ resp + stocks)",
    "bulk":     "Bulk Δ¹⁴C\n(+ resp + stocks)",
    "both":     "Fraction + Bulk\n(+ resp + stocks)",
}
_SOURCE_COLORS = {"fraction": "tab:green", "bulk": "tab:orange", "both": "0.35"}
_TOTAL_KEY = "C_stocks+pool_delta14C+resp_delta14C"
_POOLS = ["soil_active", "soil_slow", "soil_passive"]
_POOL_SHORT = {"soil_active": "active", "soil_slow": "slow", "soil_passive": "passive"}


# ════════════════════════════════════════════════════════════════════════════
# Metrics per run
# ════════════════════════════════════════════════════════════════════════════

def _tau_sigmas(data: dict, prior_sigma: np.ndarray) -> dict[str, tuple[float, float]]:
    """Return {pool: (prior_sigma_logtau, posterior_sigma_logtau)}.

    OE state names are positional (``log_tau[0]``, ``log_tau[1]``, …) in
    pool-index order, so we resolve each pool through the model's PoolIndex.
    """
    oe = data["oe_result"]
    names = list(oe.state_names)
    pool_index = data["idx"]
    post_sigma = np.sqrt(np.clip(np.diag(np.array(oe.Sx)), 0.0, None))
    out: dict[str, tuple[float, float]] = {}
    for pool in _POOLS:
        key = f"log_tau[{int(pool_index[pool])}]"
        if key in names:
            i = names.index(key)
            out[pool] = (float(prior_sigma[i]), float(post_sigma[i]))
    return out


def _chi2(data: dict) -> tuple[float, int, int, float]:
    oe = data["oe_result"]
    J = float(np.array(oe.cost_history)[-1])
    n_obs = int(oe.y_obs.shape[0])
    n_par = int(oe.x_opt.shape[0])
    dof = max(n_obs - n_par, 1)
    return J, n_obs, n_par, J / dof


def analyse_run(source: str) -> dict:
    print(f"\n{'='*78}\n  RUN: pool ¹⁴C source = {source!r}\n{'='*78}")
    data = run_hf_canonical(pool_14c_source=source)

    abl = oe_style_ablation(
        data["model"], data["forcing"], data["state_at_map"],
        data["params_opt"], data["obs_full"],
        opt_fields=tuple(data["opt_fields"]),
        extra_obs_blocks=data.get("extra_blocks", []),
    )
    ladder = oe_constraint_ladder(
        data["model"], data["forcing"], data["state_at_map"],
        data["params_opt"], data["obs_full"],
        opt_fields=tuple(data["opt_fields"]),
        extra_obs_blocks=data.get("extra_blocks", []),
    )
    prior_sigma = _canonical_prior_sigma(data["model"], data["opt_fields"])
    tau_sig = _tau_sigmas(data, prior_sigma)
    J, n_obs, n_par, chi2_red = _chi2(data)

    return dict(
        source=source, data=data, ablation=abl, ladder=ladder,
        tau_sig=tau_sig, chi2=(J, n_obs, n_par, chi2_red),
        dfs_total=abl[_TOTAL_KEY]["dfs_total"],
        dfs_pool14c=abl["pool_delta14C"]["dfs_total"],
        dfs_stocks=abl["C_stocks"]["dfs_total"],
        dfs_resp=abl["resp_delta14C"]["dfs_total"],
    )


# ════════════════════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════════════════════

def print_summary(runs: dict[str, dict]) -> None:
    print(f"\n{'#'*78}\n#  SUMMARY — information content of the pool-¹⁴C constraint\n{'#'*78}")

    print(f"\n{'Metric':<34s}" + "".join(f"{s:>14s}" for s in _SOURCES))
    print("  " + "─" * 74)
    rows = [
        ("Total DFS (all obs)",        lambda r: f"{r['dfs_total']:.3f}"),
        ("  pool Δ¹⁴C alone (DFS)",     lambda r: f"{r['dfs_pool14c']:.3f}"),
        ("  C stocks alone (DFS)",      lambda r: f"{r['dfs_stocks']:.3f}"),
        ("  resp Δ¹⁴C alone (DFS)",     lambda r: f"{r['dfs_resp']:.3f}"),
        ("χ² / DOF",                    lambda r: f"{r['chi2'][3]:.2f}"),
        ("n_obs",                       lambda r: f"{r['chi2'][1]:d}"),
    ]
    for name, fn in rows:
        print(f"{name:<34s}" + "".join(f"{fn(runs[s]):>14s}" for s in _SOURCES))

    # Posterior σ on log τ per pool (lower = better constrained)
    print(f"\n{'Posterior σ(log τ)  [prior→post]':<34s}"
          + "".join(f"{s:>14s}" for s in _SOURCES))
    print("  " + "─" * 74)
    for pool in _POOLS:
        cells = []
        for s in _SOURCES:
            pr, po = runs[s]["tau_sig"].get(pool, (np.nan, np.nan))
            cells.append(f"{po:.3f}")
        pr0 = runs["fraction"]["tau_sig"].get(pool, (np.nan,))[0]
        print(f"  τ_{_POOL_SHORT[pool]:<10s}(prior {pr0:.2f}) " + "".join(f"{c:>14s}" for c in cells))

    # Marginal information: bulk on top of fractions (per-block ladder deltas)
    print(f"\n{'-'*78}\n  MARGINAL: what do the *bulk* blocks add on top of fractions?")
    print("  (per-block DFS in the 'both' run; bulk blocks are the extra info)\n")
    both_ladder = runs["both"]["ladder"]
    for row in both_ladder:
        if row["label"].startswith("israd_bulk") or row["label"].startswith("israd_"):
            tag = "BULK" if row["label"].startswith("israd_bulk") else "frac"
            print(f"    [{tag}] {row['label']:<28s}  n={row['n_obs']}  DFS(alone)={row['dfs']:.3f}")

    d_frac = runs["fraction"]["dfs_total"]
    d_both = runs["both"]["dfs_total"]
    print(f"\n  Total DFS  fraction-only = {d_frac:.3f}   fraction+bulk = {d_both:.3f}"
          f"   →  bulk adds {d_both - d_frac:+.3f} DFS on top of fractions")
    d_bulk = runs["bulk"]["dfs_total"]
    print(f"  Total DFS  bulk-only     = {d_bulk:.3f}   (vs fraction-only {d_frac:.3f})")


# ════════════════════════════════════════════════════════════════════════════
# Figure
# ════════════════════════════════════════════════════════════════════════════

def make_figure(runs: dict[str, dict], out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    ax_dfs, ax_tau, ax_ur = axes

    # (a) DFS by obs type, grouped by source
    obs_types = [OBS_C_STOCKS, OBS_POOL_D14C, OBS_RESP_D14C, "total"]
    type_labels = ["C stocks", "pool Δ¹⁴C", "resp Δ¹⁴C", "ALL combined"]
    x = np.arange(len(obs_types))
    w = 0.26
    for j, s in enumerate(_SOURCES):
        r = runs[s]
        vals = [r["dfs_stocks"], r["dfs_pool14c"], r["dfs_resp"], r["dfs_total"]]
        ax_dfs.bar(x + (j - 1) * w, vals, w, color=_SOURCE_COLORS[s],
                   alpha=0.88, label=_SOURCE_LABELS[s].replace("\n", " "))
    ax_dfs.set_xticks(x)
    ax_dfs.set_xticklabels(type_labels, fontsize=9, rotation=12, ha="right")
    ax_dfs.set_ylabel("Degrees of freedom for signal (DFS)", fontsize=9)
    ax_dfs.set_title("(a) DFS by observation type", fontsize=10, loc="left")
    ax_dfs.legend(fontsize=8, framealpha=0.9)
    ax_dfs.grid(axis="y", lw=0.4, alpha=0.4)
    ax_dfs.set_ylim(bottom=0)

    # (b) Posterior σ(log τ) per pool: prior vs each source
    x_p = np.arange(len(_POOLS))
    prior_vals = [runs["fraction"]["tau_sig"].get(p, (np.nan,))[0] for p in _POOLS]
    ax_tau.bar(x_p - 1.5 * w, prior_vals, w, color="0.8", alpha=0.9, label="Prior")
    for j, s in enumerate(_SOURCES):
        post_vals = [runs[s]["tau_sig"].get(p, (np.nan, np.nan))[1] for p in _POOLS]
        ax_tau.bar(x_p + (j - 0.5) * w, post_vals, w, color=_SOURCE_COLORS[s],
                   alpha=0.88, label=_SOURCE_LABELS[s].replace("\n", " "))
    ax_tau.set_xticks(x_p)
    ax_tau.set_xticklabels([f"τ_{_POOL_SHORT[p]}" for p in _POOLS], fontsize=9)
    ax_tau.set_ylabel("Posterior σ (log τ)  — lower = better", fontsize=9)
    ax_tau.set_title("(b) Turnover-time uncertainty per pool", fontsize=10, loc="left")
    ax_tau.legend(fontsize=7.5, framealpha=0.9)
    ax_tau.grid(axis="y", lw=0.4, alpha=0.4)
    ax_tau.set_ylim(bottom=0)

    # (c) Uncertainty reduction (%) on log τ per pool
    for j, s in enumerate(_SOURCES):
        ur = []
        for p in _POOLS:
            pr, po = runs[s]["tau_sig"].get(p, (np.nan, np.nan))
            ur.append((1 - po / pr) * 100 if pr and np.isfinite(pr) else np.nan)
        ax_ur.bar(x_p + (j - 1) * w, ur, w, color=_SOURCE_COLORS[s], alpha=0.88,
                  label=_SOURCE_LABELS[s].replace("\n", " "))
    ax_ur.set_xticks(x_p)
    ax_ur.set_xticklabels([f"τ_{_POOL_SHORT[p]}" for p in _POOLS], fontsize=9)
    ax_ur.set_ylabel("Uncertainty reduction on log τ (%)", fontsize=9)
    ax_ur.set_title("(c) How much each source pins each pool", fontsize=10, loc="left")
    ax_ur.legend(fontsize=7.5, framealpha=0.9)
    ax_ur.grid(axis="y", lw=0.4, alpha=0.4)
    ax_ur.set_ylim(bottom=0)

    fig.suptitle(
        "Harvard Forest — information content of bulk vs. density-fraction ¹⁴C\n"
        "3-pool OE inversion, respired Δ¹⁴C + C stocks held fixed",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    runs = {s: analyse_run(s) for s in _SOURCES}
    print_summary(runs)
    out_path = os.path.join(_NB_ROOT, "hf_bulk_vs_fraction_14c_information.png")
    make_figure(runs, out_path)
