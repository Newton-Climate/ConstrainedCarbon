"""
headline_figures.py — Custom presentation figures.

Produces:
  notebooks/canonical_posterior_turnover_time_comparison.png
      Side-by-side posterior τ ± σ bars for Harvard Forest and Barrow.
  notebooks/model_complexity_information_saturation.png
      DFS vs. model complexity rungs.
  notebooks/canonical_respired_delta14c_model_observation_fits.png
      Respired Δ¹⁴C simulations vs observations at both sites.
"""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_SCRIPT_ROOT, "src"))
sys.path.insert(0, os.path.join(_SCRIPT_ROOT, "notebooks"))
os.chdir(_SCRIPT_ROOT)

from sites.canonical import run_hf_canonical, run_barrow_canonical
from ecosystem_complexity.analysis import compute_resp_delta14C

# Forest palette
C_PRIMARY   = "#1F3A2E"
C_SECONDARY = "#4A7C59"
C_ACCENT    = "#D4A574"
C_PRIOR     = "#BFBFBF"
C_POST_HF   = "#4A7C59"
C_POST_BR   = "#1F3A2E"
C_BODY      = "#2A2A2A"
C_MUTED     = "#6B6B6B"
C_BGLIGHT   = "#F7F4ED"


def _fetch_posterior(data: dict) -> dict:
    tau_prior = np.exp(np.array(data["params_prior"].log_tau)) / 365.0
    tau_post  = np.exp(np.array(data["params_opt"].log_tau)) / 365.0
    Sx_diag = np.array(jnp.diag(data["oe_result"].Sx))
    sigma_log = np.sqrt(np.abs(Sx_diag[:len(tau_post)]))
    tau_lo = tau_post * np.exp(-sigma_log)
    tau_hi = tau_post * np.exp(+sigma_log)
    return dict(prior=tau_prior, post=tau_post, lo=tau_lo, hi=tau_hi,
                pool_names=data["idx"].pool_names)


def _fetch_resp_fit(data: dict) -> dict:
    yrs = 1970.0 + np.array(data["forcing"].time) / 365.25
    resp_sim = np.array(compute_resp_delta14C(data["out_opt"], data["params_opt"]))
    resp_obs = np.array(data["delta14C_resp"])
    m = np.isfinite(resp_obs)
    return dict(time=yrs, sim=resp_sim, obs=resp_obs, mask=m)


def make_posterior_tau_figure(hf, br, out_path: str) -> None:
    fig = plt.figure(figsize=(13, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.28,
                             left=0.07, right=0.97, top=0.86, bottom=0.10)
    sites = [("Harvard Forest (US-Ha1)", hf, C_POST_HF),
             ("Barrow, Alaska (US-A10)", br, C_POST_BR)]

    for ax_i, (title, d, c_post) in enumerate(sites):
        ax = fig.add_subplot(gs[0, ax_i])
        pools = d["pool_names"]
        x = np.arange(len(pools))
        w = 0.38
        ax.bar(x - w/2, d["prior"], w, color=C_PRIOR, label="Prior τ",
               edgecolor=C_MUTED, lw=0.6)
        ax.bar(x + w/2, d["post"], w, color=c_post, label="Posterior τ", alpha=0.92)
        # ±1σ error bars
        ax.errorbar(x + w/2, d["post"],
                    yerr=[d["post"] - d["lo"], d["hi"] - d["post"]],
                    fmt="none", color=C_PRIMARY, capsize=4, lw=1.4)
        # Annotate ratios above
        for i, (p, q) in enumerate(zip(d["prior"], d["post"])):
            ratio = q / p
            ax.text(x[i] + w/2, d["hi"][i] * 1.05, f"{ratio:.2f}×",
                    ha="center", va="bottom", fontsize=9, color=C_BODY,
                    fontweight="bold")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("soil_", "") for p in pools],
                           fontsize=11, fontweight="bold")
        ax.set_ylabel("Turnover time τ (years, log scale)", fontsize=11)
        ax.set_title(title, fontsize=12, color=C_PRIMARY, fontweight="bold",
                     loc="left", pad=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", lw=0.4, alpha=0.4)
        ax.legend(fontsize=10, framealpha=0.9, loc="upper left")

    fig.suptitle("Posterior turnover times: data-constrained τ vs. literature prior  ·  ±1σ from OE posterior",
                 fontsize=13, color=C_PRIMARY, fontweight="bold")
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"saved {out_path}")


def make_resp_fit_figure(hf, br, out_path: str) -> None:
    fig = plt.figure(figsize=(13, 5.5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.20,
                             left=0.07, right=0.97, top=0.85, bottom=0.13)
    sites = [("Harvard Forest", hf, C_POST_HF, "NWN, 1996–2010"),
             ("Barrow, Alaska", br, C_POST_BR, "Surface, 2012–2014")]
    for ax_i, (title, d, color, src) in enumerate(sites):
        ax = fig.add_subplot(gs[0, ax_i])
        ax.axhline(0, lw=0.4, color="0.4", linestyle=":")
        ax.plot(d["time"], d["sim"], color=color, lw=1.7,
                label="Model posterior", zorder=3)
        ax.scatter(d["time"][d["mask"]], d["obs"][d["mask"]],
                   s=42, color=C_ACCENT, edgecolors=C_PRIMARY,
                   lw=0.8, label="Observed", zorder=4)
        resid = d["sim"][d["mask"]] - d["obs"][d["mask"]]
        rmse = float(np.sqrt(np.mean(resid**2)))
        bias = float(np.mean(resid))
        n = int(d["mask"].sum())
        ax.text(0.02, 0.97,
                f"n = {n}\nRMSE = {rmse:.1f} ‰\nbias = {bias:+.1f} ‰",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=10, color=C_PRIMARY,
                bbox=dict(boxstyle="round,pad=0.4", fc=C_BGLIGHT,
                          ec=C_ACCENT, lw=1))
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Respired CO₂ Δ¹⁴C (‰)", fontsize=11)
        ax.set_title(f"{title} — {src}", fontsize=12, color=C_PRIMARY,
                     fontweight="bold", loc="left", pad=10)
        ax.tick_params(labelsize=9)
        ax.grid(lw=0.3, alpha=0.35)
        ax.legend(fontsize=10, framealpha=0.9, loc="lower left")
        # Auto x-range to data + a margin
        valid_t = d["time"][d["mask"]]
        if valid_t.size:
            xpad = (valid_t.max() - valid_t.min()) * 0.12 + 1
            ax.set_xlim(valid_t.min() - xpad, valid_t.max() + xpad)

    fig.suptitle("Posterior model reproduces observed respired Δ¹⁴C at both sites",
                 fontsize=13, color=C_PRIMARY, fontweight="bold")
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"saved {out_path}")


def make_complexity_ladder_figure(hf, br, out_path: str) -> None:
    """DFS by obs-type combinations (the IT ablation result) presented as a clean stair-step."""
    from sites.canonical import oe_style_ablation
    abl_hf = oe_style_ablation(
        hf["model"], hf["forcing"], hf["state_at_map"],
        hf["params_opt"], hf["obs_full"],
        opt_fields=tuple(hf["opt_fields"]),
        extra_obs_blocks=hf.get("extra_blocks", []),
    )
    abl_br = oe_style_ablation(
        br["model"], br["forcing"], br["state_at_map"],
        br["params_opt"], br["obs_full"],
        opt_fields=tuple(br["opt_fields"]),
        extra_obs_blocks=br.get("extra_blocks", []),
    )

    fig = plt.figure(figsize=(13, 5.5))
    gs = gridspec.GridSpec(1, 1, figure=fig,
                           left=0.10, right=0.97, top=0.85, bottom=0.18)
    ax = fig.add_subplot(gs[0, 0])

    rungs = [
        ("C stocks",                              ["C_stocks"]),
        ("C stocks +\npool Δ¹⁴C",                  ["C_stocks", "pool_delta14C"]),
        ("C stocks + pool Δ¹⁴C +\nresp Δ¹⁴C  (all)", ["C_stocks", "pool_delta14C", "resp_delta14C"]),
    ]

    def _dfs(abl, types):
        # Look up the matching scenario
        key = ""
        if types == ["C_stocks"]:
            key = "C_stocks"
        elif types == ["C_stocks", "pool_delta14C"]:
            key = "C_stocks+pool_delta14C"
        else:
            key = "C_stocks+pool_delta14C+resp_delta14C"
        return abl[key]["dfs_total"]

    x = np.arange(len(rungs))
    w = 0.38
    hf_vals = [_dfs(abl_hf, t) for _, t in rungs]
    br_vals = [_dfs(abl_br, t) for _, t in rungs]

    ax.bar(x - w/2, hf_vals, w, color=C_POST_HF, label="Harvard Forest",
           edgecolor=C_PRIMARY, lw=0.5)
    ax.bar(x + w/2, br_vals, w, color=C_POST_BR, label="Barrow, Alaska",
           edgecolor=C_PRIMARY, lw=0.5)

    for i, (h, b) in enumerate(zip(hf_vals, br_vals)):
        ax.text(x[i] - w/2, h + 0.07, f"{h:.2f}",
                ha="center", va="bottom", fontsize=10, color=C_PRIMARY,
                fontweight="bold")
        ax.text(x[i] + w/2, b + 0.07, f"{b:.2f}",
                ha="center", va="bottom", fontsize=10, color=C_PRIMARY,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rungs], fontsize=11)
    ax.set_ylabel("Degrees of Freedom for Signal (DFS)", fontsize=12)
    ax.set_title("Adding observation types — diminishing returns above ~3 DFS for a 3-pool model",
                 fontsize=12, color=C_PRIMARY, fontweight="bold", loc="left", pad=12)
    ax.set_ylim(0, 4)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    ax.legend(fontsize=11, framealpha=0.9, loc="upper left")

    # Annotation: max useful information
    ax.axhline(3.0, lw=0.8, linestyle="--", color=C_ACCENT, alpha=0.7)
    ax.text(2.45, 3.05, "3 params = 3-pool model dimensionality",
            fontsize=9, color=C_ACCENT, ha="right", style="italic")

    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    print("Running canonical inversions…")
    hf_data = run_hf_canonical()
    br_data = run_barrow_canonical()

    hf_tau = _fetch_posterior(hf_data)
    br_tau = _fetch_posterior(br_data)
    hf_resp = _fetch_resp_fit(hf_data)
    br_resp = _fetch_resp_fit(br_data)

    out_dir = os.path.join(_SCRIPT_ROOT, "notebooks")
    make_posterior_tau_figure(hf_tau, br_tau,
                               os.path.join(out_dir, "canonical_posterior_turnover_time_comparison.png"))
    make_resp_fit_figure(hf_resp, br_resp,
                          os.path.join(out_dir, "canonical_respired_delta14c_model_observation_fits.png"))
    make_complexity_ladder_figure(hf_data, br_data,
                                   os.path.join(out_dir, "model_complexity_information_saturation.png"))
