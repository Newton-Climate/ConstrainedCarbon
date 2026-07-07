"""
cross_site_information.py — Side-by-side DFS-by-observation-type comparison
between Harvard Forest (US-Ha1) and Barrow, Alaska (US-A10).

Both sites use the single canonical OE inversion from sites/canonical.py
(3-pool model, optimize_oe with field set (log_tau, log_f_transfer),
full obs vector including C stocks plus radiocarbon constraints).  The
current canonical Barrow run uses two individual stock constraints
(`soil_active`, `soil_passive`) and does not include the older overlapping
sum constraints. The DFS-by-obs-type
ablation uses the time-resolved Jacobian at the MAP analytical-SS state,
which makes the "all-obs" DFS match OEResult.averaging_kernel exactly.

Run from the repository root:
    python notebooks/cross_site_information.py

Output
------
  notebooks/cross_site_observation_type_dfs_comparison.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ───────────────────────────────────────────────────────────────────
_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NB_ROOT = os.path.join(_SCRIPT_ROOT, "notebooks")
_SRC_ROOT = os.path.join(_SCRIPT_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_SCRIPT_ROOT)

from sites.canonical import (
    run_hf_canonical, run_barrow_canonical, oe_style_ablation,
)
from ecosystem_complexity.information import (
    OBS_C_STOCKS, OBS_POOL_D14C, OBS_RESP_D14C,
)


# ════════════════════════════════════════════════════════════════════════════
# Ablation per site
# ════════════════════════════════════════════════════════════════════════════

def _ablate(label: str, data: dict) -> dict:
    print(f"\n[{label}] OE-style ablation at MAP estimate…")
    return oe_style_ablation(
        data["model"], data["forcing"], data["state_at_map"],
        data["params_opt"], data["obs_full"],
        opt_fields=tuple(data["opt_fields"]),
        extra_obs_blocks=data.get("extra_blocks", []),
    )


_SINGLE_OBS_TYPES = (OBS_C_STOCKS, OBS_POOL_D14C, OBS_RESP_D14C)


def _single_type_dfs(ablation: dict) -> dict[str, float]:
    return {label: row["dfs_total"]
            for label, row in ablation.items()
            if label in _SINGLE_OBS_TYPES}


def _total_dfs(ablation: dict) -> float:
    full_label = max(ablation, key=lambda label: label.count("+"))
    return ablation[full_label]["dfs_total"]


def _validate_canonical_setup(label: str, data: dict) -> None:
    """Guard against stale / statistically misleading canonical setups."""
    extra_names = [b.name for b in data.get("extra_blocks", [])]

    if label == "Harvard Forest":
        if data["obs_full"].delta14C_obs:
            raise RuntimeError(
                "Harvard Forest canonical run still has standard pool Δ14C "
                "observations populated; this would risk double-counting the "
                "ISRaD fraction constraints."
            )
        if not any(name.startswith("israd_") for name in extra_names):
            raise RuntimeError(
                "Harvard Forest canonical run is missing the named ISRaD "
                "fraction blocks expected by the DFS comparison."
            )

    if label == "Barrow":
        legacy_sum = [name for name in extra_names if name.startswith("c_sum_")]
        if legacy_sum:
            raise RuntimeError(
                "Barrow canonical run still includes legacy sum constraints; "
                "the cross-site DFS comparison assumes those correlated stock "
                "sums are omitted."
            )
        if len(data.get("c_pools_obs", {})) < 2:
            raise RuntimeError(
                "Barrow canonical run is missing the individual active/passive "
                "stock constraints used by the current comparison."
            )


def _print_stock_identifiability_note(label: str, data: dict, ablation: dict) -> None:
    """Make underdetermined stock-only subproblems explicit in the console output."""
    stock_only = ablation.get("C_stocks")
    if stock_only is None:
        return

    n_stock_obs = int(stock_only["n_obs"])
    n_pools = len(data["idx"].pool_names)
    dfs = float(stock_only["dfs_total"])
    print(f"\n[{label}] stock-only identifiability check")
    print(f"  stock observations = {n_stock_obs}   pools = {n_pools}   stock-only DFS = {dfs:.3f}")

    if n_stock_obs < n_pools:
        print(
            "  NOTE: stock data alone are underdetermined for the 3-pool split. "
            "Interpret the stock-only DFS as information on a constrained subspace, "
            "not as independent identification of every pool stock."
        )


# ════════════════════════════════════════════════════════════════════════════
# Figure
# ════════════════════════════════════════════════════════════════════════════

_OBS_TYPE_ORDER  = [OBS_C_STOCKS, OBS_POOL_D14C, OBS_RESP_D14C]
_OBS_TYPE_LABELS = {
    OBS_C_STOCKS:  "C stocks",
    OBS_POOL_D14C: "Pool Δ¹⁴C",
    OBS_RESP_D14C: "Respired Δ¹⁴C",
}
_OBS_TYPE_COLORS = {
    OBS_C_STOCKS:  "tab:blue",
    OBS_POOL_D14C: "tab:green",
    OBS_RESP_D14C: "tab:red",
}


def _draw_site_panel(ax, title: str, single_dfs: dict, total_dfs: float):
    obs_types = _OBS_TYPE_ORDER
    vals = [single_dfs.get(t, 0.0) for t in obs_types]
    colors = [_OBS_TYPE_COLORS[t] for t in obs_types]
    x = np.arange(len(obs_types))
    ax.bar(x, vals, 0.55, color=colors, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals + [total_dfs]) * 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, color="0.25")
    ax.axhline(total_dfs, lw=1.0, linestyle=":", color="black", alpha=0.6,
               label=f"All obs combined: {total_dfs:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([_OBS_TYPE_LABELS[t] for t in obs_types],
                       fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("DFS  (obs type alone)", fontsize=9)
    ax.set_title(title, fontsize=9, loc="left")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    ax.set_ylim(0, max(total_dfs, max(vals)) * 1.25 + 1e-6)


def make_figure(hf_abl: dict, br_abl: dict, out_path: str) -> None:
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(
        1, 3, figure=fig, width_ratios=[1.0, 1.0, 0.9],
        wspace=0.42, left=0.07, right=0.97, top=0.85, bottom=0.16,
    )
    ax_hf  = fig.add_subplot(gs[0, 0])
    ax_br  = fig.add_subplot(gs[0, 1])
    ax_cmp = fig.add_subplot(gs[0, 2])

    hf_single = _single_type_dfs(hf_abl)
    br_single = _single_type_dfs(br_abl)
    hf_total  = _total_dfs(hf_abl)
    br_total  = _total_dfs(br_abl)

    _draw_site_panel(ax_hf, "(a) Harvard Forest (US-Ha1)\n3-pool model",
                     hf_single, hf_total)
    _draw_site_panel(ax_br, "(b) Barrow, Alaska (US-A10)\n3-pool permafrost",
                     br_single, br_total)

    obs_types = _OBS_TYPE_ORDER
    x = np.arange(len(obs_types))
    w = 0.36
    hf_frac = [hf_single.get(t, 0.0) / hf_total if hf_total > 0 else 0.0
               for t in obs_types]
    br_frac = [br_single.get(t, 0.0) / br_total if br_total > 0 else 0.0
               for t in obs_types]
    ax_cmp.bar(x - w / 2, hf_frac, w, color="seagreen", alpha=0.85,
               label=f"HF  (total DFS = {hf_total:.2f})")
    ax_cmp.bar(x + w / 2, br_frac, w, color="purple", alpha=0.85,
               label=f"Barrow  (total DFS = {br_total:.2f})")
    for i, (hv, bv) in enumerate(zip(hf_frac, br_frac)):
        ax_cmp.text(i - w / 2, hv + 0.01, f"{hv:.2f}",
                    ha="center", va="bottom", fontsize=8, color="0.25")
        ax_cmp.text(i + w / 2, bv + 0.01, f"{bv:.2f}",
                    ha="center", va="bottom", fontsize=8, color="0.25")
    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels([_OBS_TYPE_LABELS[t] for t in obs_types],
                           fontsize=9, rotation=15, ha="right")
    ax_cmp.set_ylabel("Single-obs DFS  /  total DFS", fontsize=9)
    ax_cmp.set_title("(c) Relative information content\nper observation type",
                     fontsize=9, loc="left")
    ax_cmp.legend(fontsize=8, framealpha=0.9, loc="upper right")
    ax_cmp.tick_params(axis="y", labelsize=8)
    ax_cmp.grid(axis="y", lw=0.4, alpha=0.4)
    ax_cmp.set_ylim(0, max(max(hf_frac), max(br_frac), 0.5) * 1.30)

    fig.suptitle(
        "Cross-site comparison of ¹⁴C and C-stock information content\n"
        "Canonical 3-pool OE inversion — utility of obs type is ecosystem-specific",
        fontsize=12,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    hf_data = run_hf_canonical()
    _validate_canonical_setup("Harvard Forest", hf_data)
    hf_abl  = _ablate("Harvard Forest", hf_data)
    _print_stock_identifiability_note("Harvard Forest", hf_data, hf_abl)

    br_data = run_barrow_canonical()
    _validate_canonical_setup("Barrow", br_data)
    br_abl  = _ablate("Barrow", br_data)
    _print_stock_identifiability_note("Barrow", br_data, br_abl)

    # ── Fit quality diagnostics (reduced χ² and DOF) ────────────────────
    import jax.numpy as jnp
    print("\n══ Fit quality (reduced χ²) ═══════════════════════════════════════")
    for label, data in [("HF", hf_data), ("Barrow", br_data)]:
        oe = data["oe_result"]
        ch = jnp.array(oe.cost_history)
        J_final = float(ch[-1])
        n_obs   = int(oe.y_obs.shape[0])
        n_params = int(oe.x_opt.shape[0])
        dof = max(n_obs - n_params, 1)
        chi2_red = J_final / dof
        print(f"  {label:<8s}  J_final = {J_final:7.2f}   n_obs = {n_obs:3d}   "
              f"n_params = {n_params:3d}   DOF = {dof:3d}   χ²/DOF = {chi2_red:6.2f}")
        if dof <= 6:
            print(
                "            low nominal DOF — interpret reduced χ² cautiously, "
                "especially for Barrow."
            )

    print("\n══ Summary ════════════════════════════════════════════════════════")
    hf_single = _single_type_dfs(hf_abl)
    br_single = _single_type_dfs(br_abl)
    hf_total  = _total_dfs(hf_abl)
    br_total  = _total_dfs(br_abl)
    print(f"\n  {'Obs type':<20s}  {'HF DFS':>10s}  {'Barrow DFS':>12s}")
    print("  " + "─" * 50)
    for t in _OBS_TYPE_ORDER:
        print(f"  {_OBS_TYPE_LABELS[t]:<20s}  "
              f"{hf_single.get(t, 0.0):10.3f}  {br_single.get(t, 0.0):12.3f}")
    print(f"  {'All combined':<20s}  {hf_total:10.3f}  {br_total:12.3f}")

    out_path = os.path.join(_NB_ROOT, "cross_site_observation_type_dfs_comparison.png")
    make_figure(hf_abl, br_abl, out_path)
