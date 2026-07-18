"""
harvard_forest_gpp_ladder.py — Harvard Forest OE constraint ladder under two
GPP-forcing treatments.

Repeats the canonical Harvard Forest (US-Ha1) 3-pool OE inversion for two
forcing scenarios and compares the information content of the constraint ladder:

  • "flux-tower GPP"  — the observed daily GPP time series drives the transient
                        forward model (time-varying carbon input).
  • "mean GPP"        — GPP held constant at its record-mean value (the same
                        long-term mean, but with the seasonal/interannual
                        variability removed).

For each treatment we re-optimise the inversion, then evaluate at that MAP:
  (a) the one-constraint-at-a-time DFS ladder,
  (b) DFS by cumulative observation subset (ablation),
  (c) per-parameter uncertainty reduction (prior σ → posterior σ),
  (d) the averaging kernel over the fitted (τ, f_transfer) subset.

The point of the comparison: because both treatments share the SAME mean GPP,
the steady-state operating point is identical; only the transient trajectory
differs. So any difference in DFS / uncertainty reduction / averaging kernel is
information carried by the *time structure* of GPP, not its magnitude.

Run from the repository root:
    <env-python> notebooks/harvard_forest_gpp_ladder.py

Output
------
  notebooks/harvard_forest_gpp_ladder.png
  notebooks/exports/harvard_forest_gpp_ladder_metrics.csv
"""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NB_ROOT = os.path.join(_SCRIPT_ROOT, "notebooks")
_SRC_ROOT = os.path.join(_SCRIPT_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_SCRIPT_ROOT)

from ecosystem_complexity.api import optimize_oe, run_model  # noqa: E402
from ecosystem_complexity.state import make_default_params  # noqa: E402
from ecosystem_complexity._oe_helpers import build_oe_prior_sigma  # noqa: E402
from ecosystem_complexity.oe_utils import (  # noqa: E402
    build_mean_ss_modifier,
    ss_state_for_params,
)
from ecosystem_complexity.oe_diagnostics import (  # noqa: E402
    oe_constraint_ladder,
    oe_style_ablation,
    oe_gain_matrix_diagnostics,
    fit_param_subset_labels,
    classify_block,
)
from sites.canonical import run_hf_canonical  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# Forcing variants and per-variant OE fit
# ════════════════════════════════════════════════════════════════════════════

def make_mean_gpp_forcing(forcing):
    """Return a copy of ``forcing`` with GPP_obs flattened to its record mean."""
    gpp = np.array(forcing.GPP_obs, dtype=float)
    mean_gpp = float(np.nanmean(gpp))
    gpp_flat = jnp.full_like(forcing.GPP_obs, mean_gpp)
    return forcing._replace(GPP_obs=gpp_flat), mean_gpp


def fit_variant(model, forcing, state0, obs_full, extra_blocks, opt_fields, label):
    """Re-optimise the OE inversion under a given forcing; return MAP + SS state."""
    print(f"\n── optimise_oe under [{label}] forcing ──")
    result = optimize_oe(
        model, forcing, obs_full, state0=state0,
        fields=opt_fields, extra_obs_blocks=extra_blocks,
    )
    ch = np.array(result.cost_history)
    print(f"   J {ch[0]:.2f} → {ch[-1]:.2f}  "
          f"({result.n_iter} iter, converged={result.converged})")
    params_opt = result.params_opt
    state_at_map = ss_state_for_params(model, forcing, state0, params_opt)
    return params_opt, state_at_map


def variant_diagnostics(model, forcing, state_at_map, params_opt,
                        obs_full, extra_blocks, opt_fields):
    """Ladder, ablation, and gain-matrix (AK / Sx) diagnostics at the MAP."""
    ladder = oe_constraint_ladder(
        model, forcing, state_at_map, params_opt, obs_full,
        opt_fields=tuple(opt_fields), extra_obs_blocks=extra_blocks,
    )
    ablation = oe_style_ablation(
        model, forcing, state_at_map, params_opt, obs_full,
        opt_fields=tuple(opt_fields), extra_obs_blocks=extra_blocks,
    )
    gain = oe_gain_matrix_diagnostics(
        model, forcing, state_at_map, params_opt, obs_full,
        opt_fields=tuple(opt_fields), extra_obs_blocks=extra_blocks,
    )
    return dict(ladder=ladder, ablation=ablation, gain=gain)


# ════════════════════════════════════════════════════════════════════════════
# Plot
# ════════════════════════════════════════════════════════════════════════════

_OBS_COLORS = {
    "C_stocks": "tab:blue",
    "pool_delta14C": "tab:green",
    "resp_delta14C": "tab:red",
}


def make_figure(tv, mean, prior_sigma, subset_labels, out_path,
                gpp_series_years, gpp_series, mean_gpp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(
        2, 3, figure=fig, hspace=0.42, wspace=0.34,
        left=0.06, right=0.97, top=0.90, bottom=0.09,
    )
    ax_gpp = fig.add_subplot(gs[0, 0])
    ax_lad = fig.add_subplot(gs[0, 1])
    ax_abl = fig.add_subplot(gs[0, 2])
    ax_ur = fig.add_subplot(gs[1, 0])
    ax_ak_tv = fig.add_subplot(gs[1, 1])
    ax_ak_mn = fig.add_subplot(gs[1, 2])

    C_TV, C_MN = "#1f77b4", "#d62728"

    # (a) The GPP forcing, both treatments
    ax_gpp.plot(gpp_series_years, gpp_series, lw=0.5, color=C_TV, alpha=0.7,
                label="Flux-tower GPP (daily)")
    ax_gpp.axhline(mean_gpp, lw=2.0, color=C_MN, label=f"Mean GPP = {mean_gpp:.2f}")
    ax_gpp.set_ylabel("GPP (gC m⁻² d⁻¹)", fontsize=9)
    ax_gpp.set_xlabel("Year", fontsize=9)
    ax_gpp.set_title("(a) GPP forcing treatments", fontsize=10, loc="left")
    ax_gpp.legend(fontsize=8, framealpha=0.9)
    ax_gpp.grid(alpha=0.3, lw=0.4)

    # (b) One-constraint-at-a-time ladder — grouped bars
    lad_tv, lad_mn = tv["ladder"], mean["ladder"]
    labels = [r["label"] for r in lad_tv]
    types = [r["obs_type"] for r in lad_tv]
    dfs_tv = [r["dfs"] for r in lad_tv]
    dfs_mn = [r["dfs"] for r in lad_mn]
    x = np.arange(len(labels))
    w = 0.4
    ax_lad.bar(x - w / 2, dfs_tv, w, color=C_TV, alpha=0.85, label="Flux-tower GPP")
    ax_lad.bar(x + w / 2, dfs_mn, w, color=C_MN, alpha=0.85, label="Mean GPP")
    short = [f"{classify_block(l).replace('_delta14C','·Δ¹⁴C').replace('_',' ')}\n{l}"
             if False else l for l in labels]
    ax_lad.set_xticks(x)
    ax_lad.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
    ax_lad.set_ylabel("DFS (single constraint)", fontsize=9)
    ax_lad.set_title("(b) One-constraint-at-a-time DFS ladder", fontsize=10, loc="left")
    ax_lad.legend(fontsize=8, framealpha=0.9)
    ax_lad.grid(axis="y", alpha=0.3, lw=0.4)

    # (c) Cumulative-subset ablation DFS
    scen = list(tv["ablation"].keys())
    dfs_abl_tv = [tv["ablation"][s]["dfs_total"] for s in scen]
    dfs_abl_mn = [mean["ablation"][s]["dfs_total"] for s in scen]
    xa = np.arange(len(scen))
    ax_abl.bar(xa - w / 2, dfs_abl_tv, w, color=C_TV, alpha=0.85, label="Flux-tower GPP")
    ax_abl.bar(xa + w / 2, dfs_abl_mn, w, color=C_MN, alpha=0.85, label="Mean GPP")
    for i, (a, b) in enumerate(zip(dfs_abl_tv, dfs_abl_mn)):
        ax_abl.text(i, max(a, b) + 0.01, f"Δ{a-b:+.3f}", ha="center",
                    va="bottom", fontsize=6.5, color="0.3")
    scen_lab = [s.replace("_delta14C", "·Δ¹⁴C").replace("+", "\n+") for s in scen]
    ax_abl.set_xticks(xa)
    ax_abl.set_xticklabels(scen_lab, fontsize=6.5)
    ax_abl.set_ylabel("Total DFS", fontsize=9)
    ax_abl.set_title("(c) DFS by cumulative obs subset", fontsize=10, loc="left")
    ax_abl.legend(fontsize=8, framealpha=0.9)
    ax_abl.grid(axis="y", alpha=0.3, lw=0.4)

    # (d) Per-parameter uncertainty reduction (subset)
    sub_tv = tv["gain"]["subset_indices"]
    post_tv = np.sqrt(np.clip(np.diag(tv["gain"]["Sx"])[sub_tv], 0, None))
    post_mn = np.sqrt(np.clip(np.diag(mean["gain"]["Sx"])[sub_tv], 0, None))
    prior_sub = prior_sigma[sub_tv]
    ur_tv = (1 - post_tv / prior_sub) * 100
    ur_mn = (1 - post_mn / prior_sub) * 100
    xp = np.arange(len(subset_labels))
    ax_ur.bar(xp - w / 2, ur_tv, w, color=C_TV, alpha=0.85, label="Flux-tower GPP")
    ax_ur.bar(xp + w / 2, ur_mn, w, color=C_MN, alpha=0.85, label="Mean GPP")
    ax_ur.set_xticks(xp)
    ax_ur.set_xticklabels(subset_labels, fontsize=9)
    ax_ur.set_ylabel("Uncertainty reduction (%)", fontsize=9)
    ax_ur.set_title("(d) Posterior uncertainty reduction", fontsize=10, loc="left")
    ax_ur.legend(fontsize=8, framealpha=0.9)
    ax_ur.grid(axis="y", alpha=0.3, lw=0.4)
    ax_ur.set_ylim(0, 100)

    # (e), (f) Averaging kernels (subset) for each treatment
    ak_tv = tv["gain"]["subset_averaging_kernel"]
    ak_mn = mean["gain"]["subset_averaging_kernel"]
    for ax, ak, title, dfs in (
        (ax_ak_tv, ak_tv, "(e) Averaging kernel — flux-tower GPP",
         np.trace(ak_tv)),
        (ax_ak_mn, ak_mn, "(f) Averaging kernel — mean GPP",
         np.trace(ak_mn)),
    ):
        im = ax.imshow(ak, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(subset_labels)))
        ax.set_yticks(range(len(subset_labels)))
        ax.set_xticklabels(subset_labels, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(subset_labels, fontsize=8)
        for i in range(ak.shape[0]):
            for j in range(ak.shape[1]):
                ax.text(j, i, f"{ak[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(ak[i, j]) < 0.6 else "white")
        ax.set_title(f"{title}\ntrace (DFS on subset) = {dfs:.3f}",
                     fontsize=9, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Harvard Forest (US-Ha1) — OE constraint ladder: flux-tower (time-varying) "
        "vs. mean GPP forcing", fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    data = run_hf_canonical()
    model = data["model"]
    forcing_tv = data["forcing"]
    state0 = data["state0_obs"]
    obs_full = data["obs_full"]
    extra_blocks = data["extra_blocks"]
    opt_fields = data["opt_fields"]
    time_years = data["time_years"]

    # Mean-GPP forcing (same record mean, no time structure)
    forcing_mn, mean_gpp = make_mean_gpp_forcing(forcing_tv)
    print(f"\nMean GPP over record = {mean_gpp:.3f} gC m⁻² d⁻¹")

    # Re-optimise under each treatment
    p_tv, s_tv = fit_variant(model, forcing_tv, state0, obs_full,
                             extra_blocks, opt_fields, "flux-tower GPP")
    p_mn, s_mn = fit_variant(model, forcing_mn, state0, obs_full,
                             extra_blocks, opt_fields, "mean GPP")

    print("\nMAP log_tau (flux-tower):", np.array(p_tv.log_tau))
    print("MAP log_tau (mean GPP)  :", np.array(p_mn.log_tau))

    tv = variant_diagnostics(model, forcing_tv, s_tv, p_tv,
                             obs_full, extra_blocks, opt_fields)
    mean = variant_diagnostics(model, forcing_mn, s_mn, p_mn,
                               obs_full, extra_blocks, opt_fields)

    prior_sigma = np.array(
        build_oe_prior_sigma(model.config, make_default_params(model.config),
                             tuple(opt_fields)), dtype=float,
    )
    subset_labels = fit_param_subset_labels()

    # ── Console comparison ──────────────────────────────────────────────────
    def _total_dfs(v):
        return v["ablation"]["C_stocks+pool_delta14C+resp_delta14C"]["dfs_total"]
    print("\n" + "═" * 70)
    print("SUMMARY — flux-tower (time-varying) vs mean GPP")
    print("═" * 70)
    print(f"  Total DFS (all obs) : {_total_dfs(tv):.4f}  vs  {_total_dfs(mean):.4f}"
          f"   (Δ = {_total_dfs(tv)-_total_dfs(mean):+.4f})")
    print(f"  Subset AK trace     : {np.trace(tv['gain']['subset_averaging_kernel']):.4f}"
          f"  vs  {np.trace(mean['gain']['subset_averaging_kernel']):.4f}")

    # ── Export metrics CSV ──────────────────────────────────────────────────
    import csv
    exports = os.path.join(_NB_ROOT, "exports")
    os.makedirs(exports, exist_ok=True)
    csv_path = os.path.join(exports, "harvard_forest_gpp_ladder_metrics.csv")
    sub = tv["gain"]["subset_indices"]
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["metric", "parameter", "flux_tower_gpp", "mean_gpp", "delta"])
        # ladder DFS
        for r_tv, r_mn in zip(tv["ladder"], mean["ladder"]):
            wtr.writerow(["ladder_dfs", r_tv["label"], f"{r_tv['dfs']:.6f}",
                          f"{r_mn['dfs']:.6f}", f"{r_tv['dfs']-r_mn['dfs']:+.6f}"])
        # ablation DFS
        for s in tv["ablation"]:
            a, b = tv["ablation"][s]["dfs_total"], mean["ablation"][s]["dfs_total"]
            wtr.writerow(["ablation_dfs", s, f"{a:.6f}", f"{b:.6f}", f"{a-b:+.6f}"])
        # uncertainty reduction per subset param
        post_tv = np.sqrt(np.clip(np.diag(tv["gain"]["Sx"])[sub], 0, None))
        post_mn = np.sqrt(np.clip(np.diag(mean["gain"]["Sx"])[sub], 0, None))
        for i, name in enumerate(tv["gain"]["subset_state_names"]):
            ur_tv = (1 - post_tv[i] / prior_sigma[sub][i]) * 100
            ur_mn = (1 - post_mn[i] / prior_sigma[sub][i]) * 100
            wtr.writerow(["uncertainty_reduction_pct", name, f"{ur_tv:.4f}",
                          f"{ur_mn:.4f}", f"{ur_tv-ur_mn:+.4f}"])
        # AK diagonal per subset param
        ak_tv = np.diag(tv["gain"]["subset_averaging_kernel"])
        ak_mn = np.diag(mean["gain"]["subset_averaging_kernel"])
        for i, name in enumerate(tv["gain"]["subset_state_names"]):
            wtr.writerow(["ak_diagonal", name, f"{ak_tv[i]:.6f}",
                          f"{ak_mn[i]:.6f}", f"{ak_tv[i]-ak_mn[i]:+.6f}"])
    print(f"Metrics CSV saved → {csv_path}")

    out_path = os.path.join(_NB_ROOT, "harvard_forest_gpp_ladder.png")
    make_figure(tv, mean, prior_sigma, subset_labels, out_path,
                time_years, np.array(forcing_tv.GPP_obs), mean_gpp)


if __name__ == "__main__":
    main()
