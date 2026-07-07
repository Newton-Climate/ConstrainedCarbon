"""
analyze_clm.py — Compare CESM2 turnover at four sites against the OE posterior.

This reads the global CESM2 historical files in ``data/cmip``, selects the
nearest grid cell for the four canonical analysis sites, computes annual means,
and compares implied pool turnover times against the radiocarbon-constrained OE
posterior.

Output: notebooks/clm/clm_comparison.png
"""
from __future__ import annotations

import os
import sys

import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_SCRIPT_ROOT, "src"))
sys.path.insert(0, os.path.join(_SCRIPT_ROOT, "notebooks"))
os.chdir(_SCRIPT_ROOT)

from clm.cmip_global import SITE_SPECS, load_site_cesm
from sites.canonical import run_barrow_canonical, run_hf_canonical
from sites.eight_mile_lake import run_eml_canonical
from sites.howland_forest import run_howland_canonical

OUT_PATH = os.path.join(_SCRIPT_ROOT, "notebooks", "clm", "clm_comparison.png")

C_PRIMARY = "#1F3A2E"
C_MUTED = "#6B6B6B"
C_CESM = "#B85042"


def derived_tau(data: dict) -> dict:
    """Implied turnover times τ = pool C / Rh, in years."""
    rh = data["rh"]
    safe_rh = np.where(rh > 1e-3, rh, np.nan)
    tau = {
        "fast": data["cSoilFast"] / safe_rh if data.get("cSoilFast") is not None else None,
        "medium": data["cSoilMedium"] / safe_rh if data.get("cSoilMedium") is not None else None,
        "slow": data["cSoilSlow"] / safe_rh if data.get("cSoilSlow") is not None else None,
        "bulk": data["cSoil"] / safe_rh if data.get("cSoil") is not None else None,
    }
    return tau


def _fetch_oe(data: dict) -> dict:
    tau = np.exp(np.array(data["params_opt"].log_tau)) / 365.0
    s_diag = np.array(jnp.diag(data["oe_result"].Sx))[: len(tau)]
    sigma_log = np.sqrt(np.abs(s_diag))
    return {
        "pool_names": data["idx"].pool_names,
        "tau": tau,
        "tau_lo": tau * np.exp(-sigma_log),
        "tau_hi": tau * np.exp(+sigma_log),
    }


def fetch_oe_posteriors() -> dict[str, dict]:
    return {
        "harvard_forest": _fetch_oe(run_hf_canonical()),
        "barrow": _fetch_oe(run_barrow_canonical()),
        "howland_forest": _fetch_oe(run_howland_canonical()),
        "eight_mile_lake": _fetch_oe(run_eml_canonical()),
    }


def make_figure(cesm: dict[str, dict], oe: dict[str, dict], out_path: str) -> None:
    fig = plt.figure(figsize=(15.0, 9.5))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.28, left=0.05, right=0.98, top=0.90, bottom=0.08)

    for col, site in enumerate(SITE_SPECS):
        data = cesm[site.key]

        ax = fig.add_subplot(gs[0, col])
        yrs = data["years"]
        ax.plot(yrs, data["cSoilFast"], color="#4A7C59", lw=1.4, label="cSoilFast")
        ax.plot(yrs, data["cSoilMedium"], color="#D4A574", lw=1.4, label="cSoilMedium")
        ax.plot(yrs, data["cSoilSlow"], color="#1F3A2E", lw=1.4, label="cSoilSlow")
        ax.plot(yrs, data["cSoil"], color=C_MUTED, lw=1.0, linestyle=":", label="cSoil total")
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("Carbon stock (gC m⁻²)", fontsize=9)
        ax.set_title(
            f"({chr(ord('a') + col)}) {site.label} ({site.code})\n"
            f"cell=({data['cell_lat']:.2f}, {data['cell_lon']:.2f})  dist={data['dist_km']:.0f} km",
            fontsize=9,
            color=C_PRIMARY,
            fontweight="bold",
            loc="left",
            pad=8,
        )
        ax.grid(lw=0.3, alpha=0.4)
        ax.legend(fontsize=7, framealpha=0.92, loc="upper left")

        ax = fig.add_subplot(gs[1, col])
        cesm_tau = derived_tau(data)
        recent = (data["years"] >= 2005) & (data["years"] <= 2014)
        cesm_vals = [float(np.nanmean(cesm_tau[key][recent])) for key in ("fast", "medium", "slow")]
        oe_post = oe[site.key]
        x = np.arange(3)
        w = 0.36
        ax.bar(x - w / 2, cesm_vals, w, color=C_CESM, alpha=0.85, edgecolor=C_PRIMARY, lw=0.5, label="CESM2 implied τ")
        ax.bar(x + w / 2, oe_post["tau"], w, color=C_PRIMARY, alpha=0.85, edgecolor=C_PRIMARY, lw=0.5, label="OE posterior τ ± 1σ")
        ax.errorbar(
            x + w / 2,
            oe_post["tau"],
            yerr=[oe_post["tau"] - oe_post["tau_lo"], oe_post["tau_hi"] - oe_post["tau"]],
            fmt="none",
            color=C_PRIMARY,
            capsize=4,
            lw=1.4,
        )
        for i in range(3):
            if np.isfinite(cesm_vals[i]) and oe_post["tau"][i] > 0.0:
                ratio = cesm_vals[i] / oe_post["tau"][i]
                top = max(cesm_vals[i], oe_post["tau"][i])
                ax.text(x[i], top * 1.45, f"{ratio:.2f}×", ha="center", va="bottom", fontsize=9, color=C_PRIMARY, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(["active", "slow", "passive"], fontsize=9)
        ax.set_ylabel("Turnover time τ (years, log)", fontsize=9)
        ax.set_title(f"({chr(ord('e') + col)}) CESM2 vs OE posterior τ", fontsize=9, color=C_PRIMARY, fontweight="bold", loc="left", pad=8)
        ax.grid(axis="y", lw=0.3, alpha=0.4)
        ax.legend(fontsize=7, framealpha=0.92, loc="upper left")

    fig.suptitle(
        "CESM2 / CLM5 global-grid turnover vs 14C-constrained posterior at the four analysis sites",
        fontsize=13,
        color=C_PRIMARY,
        fontweight="bold",
    )
    plt.savefig(out_path, dpi=165, bbox_inches="tight")
    print(f"saved {out_path}")


def main():
    print("Loading CESM2 global fields at the four analysis sites…")
    cesm = {site.key: load_site_cesm(site.key) for site in SITE_SPECS}
    for site in SITE_SPECS:
        data = cesm[site.key]
        print(f"  {site.short_label:<8s} cell=({data['cell_lat']:.2f}, {data['cell_lon']:.2f})"
              f" dist={data['dist_km']:.0f} km  Rh_2005-2014={np.nanmean(data['rh'][(data['years'] >= 2005) & (data['years'] <= 2014)]):.1f}")

    print("\nRunning canonical OE inversions for posterior τ…")
    oe = fetch_oe_posteriors()
    make_figure(cesm, oe, OUT_PATH)


if __name__ == "__main__":
    main()
