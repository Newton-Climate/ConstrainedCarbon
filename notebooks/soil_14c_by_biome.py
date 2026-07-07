"""
soil_14c_by_biome.py — Slide-5 figure.

Two-panel comparison of soil radiocarbon by Köppen-Geiger climate biome:

  (a) Bulk surface soil Δ¹⁴C  (0–30 cm midpoint, ISRaD layer table)
  (b) Soil-respiration Δ¹⁴C   (ISRaD flux table, soil emissions)

Annotates where Harvard Forest and Barrow sit on each distribution.

Output: notebooks/soil_delta14c_biome_distribution_with_sites.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISRAD_DIR  = os.path.join(_SCRIPT_ROOT, "data/shared/israd")
ISRAD_FLUX = os.path.join(ISRAD_DIR, "ISRaD_extra_flat_flux_v 2.6.6.2024-01-25.csv")
ISRAD_LYR  = os.path.join(ISRAD_DIR, "ISRaD_extra_flat_layer_v 2.6.6.2024-01-25.csv")
OUT_PATH   = os.path.join(_SCRIPT_ROOT, "notebooks", "soil_delta14c_biome_distribution_with_sites.png")

# ── Palette ──────────────────────────────────────────────────────────────────
C_PRIMARY   = "#1F3A2E"
C_SECONDARY = "#4A7C59"
C_ACCENT    = "#D4A574"
C_BODY      = "#2A2A2A"
C_MUTED     = "#6B6B6B"
C_BGLIGHT   = "#F7F4ED"

BIOME_COLORS = {
    "Tropical": "#B85042",
    "Temperate": "#4A7C59",
    "Boreal":    "#2C5F2D",
    "Tundra":    "#5A88A8",
}


def _classify_biome(kg_long: str) -> str | None:
    if not isinstance(kg_long, str):
        return None
    s = kg_long.lower()
    if s.startswith("polar"):    return "Tundra"
    if s.startswith("tropical"): return "Tropical"
    if s.startswith("temperate"): return "Temperate"
    if s.startswith("cold"):
        return "Temperate" if "hot summer" in s else "Boreal"
    return None


def _load_resp() -> pd.DataFrame:
    df = pd.read_csv(ISRAD_FLUX, low_memory=False)
    m = (df["flx_14c"].notna()
         & (df["flx_pathway"] == "soil emission")
         & df["flx_ecosystem_component"].isin(["ecosystem", "heterotrophic"]))
    df = df.loc[m, ["flx_14c", "pro_KG_present_long", "pro_lat", "pro_long"]].copy()
    df["biome"] = df["pro_KG_present_long"].apply(_classify_biome)
    return df.dropna(subset=["biome", "flx_14c"])


def _load_bulk_surface() -> pd.DataFrame:
    df = pd.read_csv(ISRAD_LYR, low_memory=False)
    m = (df["lyr_14c"].notna() & df["pro_KG_present_long"].notna()
         & df["lyr_top"].notna() & df["lyr_bot"].notna())
    df = df.loc[m, ["lyr_14c", "lyr_top", "lyr_bot",
                    "pro_KG_present_long", "pro_lat", "pro_long"]].copy()
    df["depth_mid"] = 0.5 * (df["lyr_top"].astype(float) + df["lyr_bot"].astype(float))
    df = df[(df["depth_mid"] >= 0) & (df["depth_mid"] <= 30)]   # surface 0–30 cm
    df["biome"] = df["pro_KG_present_long"].apply(_classify_biome)
    return df.dropna(subset=["biome", "lyr_14c"])


def _draw_panel(ax, biome_data: dict[str, np.ndarray], order: list[str],
                title: str, ylabel: str, hf_val: float, br_val: float,
                show_n_top: bool = True) -> None:
    rng = np.random.default_rng(42)

    box_data = [biome_data[b] for b in order]
    bp = ax.boxplot(
        box_data, positions=range(len(order)), widths=0.55,
        patch_artist=True, showfliers=False,
        medianprops=dict(color=C_PRIMARY, lw=1.8),
        whiskerprops=dict(color=C_MUTED, lw=1.0),
        capprops=dict(color=C_MUTED, lw=1.0),
        boxprops=dict(lw=0),
    )
    for patch, b in zip(bp["boxes"], order):
        patch.set_facecolor(BIOME_COLORS[b])
        patch.set_alpha(0.55)

    # Establish y-limits before plotting jitter so text positions stay sane
    all_vals = np.concatenate([d for d in box_data if len(d)])
    ymin, ymax = np.percentile(all_vals, [1, 99])
    pad = 0.08 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad * 1.6)

    for i, (b, data) in enumerate(zip(order, box_data)):
        if len(data) == 0:
            continue
        x = i + (rng.random(len(data)) - 0.5) * 0.32
        ax.scatter(x, data, s=7, color=BIOME_COLORS[b], alpha=0.28,
                   edgecolors="none", zorder=2)
        if show_n_top:
            ax.text(i, ax.get_ylim()[1] * 0.97, f"n = {len(data)}",
                    ha="center", va="top", fontsize=9, color=C_PRIMARY,
                    fontweight="bold")

    # HF / Barrow markers
    hf_x = order.index("Temperate") if "Temperate" in order else 1
    br_x = order.index("Tundra") if "Tundra" in order else len(order) - 1
    ax.scatter([hf_x + 0.32], [hf_val], s=110, marker="*",
               color=C_ACCENT, edgecolors=C_PRIMARY, linewidths=1.1, zorder=10)
    ax.scatter([br_x + 0.32], [br_val], s=110, marker="*",
               color=C_ACCENT, edgecolors=C_PRIMARY, linewidths=1.1, zorder=10)
    ax.annotate(f"HF\n{hf_val:+.0f}‰", xy=(hf_x + 0.32, hf_val),
                xytext=(hf_x + 0.55, hf_val + 0.32 * (ymax - ymin)),
                fontsize=8, color=C_PRIMARY, ha="left",
                arrowprops=dict(arrowstyle="-", color=C_PRIMARY, lw=0.7))
    ax.annotate(f"Barrow\n{br_val:+.0f}‰", xy=(br_x + 0.32, br_val),
                xytext=(br_x - 0.85, br_val - 0.25 * (ymax - ymin)),
                fontsize=8, color=C_PRIMARY, ha="left",
                arrowprops=dict(arrowstyle="-", color=C_PRIMARY, lw=0.7))

    ax.axhline(0, lw=0.4, color="0.5", linestyle=":", alpha=0.7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=11, color=C_BODY)
    ax.set_ylabel(ylabel, fontsize=11, color=C_BODY)
    ax.set_title(title, fontsize=12, color=C_PRIMARY, fontweight="bold",
                 loc="left", pad=10)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9)


def main():
    bulk = _load_bulk_surface()
    resp = _load_resp()

    order = ["Tropical", "Temperate", "Boreal", "Tundra"]
    order_bulk = [b for b in order if b in bulk["biome"].unique()]
    order_resp = [b for b in order if b in resp["biome"].unique()]

    bulk_by = {b: bulk.loc[bulk["biome"] == b, "lyr_14c"].values
                for b in order_bulk}
    resp_by = {b: resp.loc[resp["biome"] == b, "flx_14c"].values
                for b in order_resp}

    fig = plt.figure(figsize=(13.0, 5.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.22,
                            left=0.07, right=0.97, top=0.85, bottom=0.13)
    ax_b = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    # Site overlays — values from our canonical inversions / data:
    #   HF bulk surface (ISRaD H1–H5, 0–30 cm)  ≈  +80 ‰
    #   Barrow bulk surface (Vaughn 2018, 0–15 cm) ≈ −200 ‰  (mean across 9 profiles)
    #   HF respired (NWN 1996–2010 mean)        ≈ +90 ‰
    #   Barrow respired (Vaughn surface mean)    ≈ +18 ‰
    _draw_panel(
        ax_b, bulk_by, order_bulk,
        title=f"(a) Bulk surface soil Δ¹⁴C  (0–30 cm,  N = {sum(len(v) for v in bulk_by.values())})",
        ylabel="Bulk soil Δ¹⁴C (‰)",
        hf_val=80.0, br_val=-200.0,
    )
    _draw_panel(
        ax_r, resp_by, order_resp,
        title=f"(b) Soil-respiration Δ¹⁴C  (ISRaD flux table,  N = {sum(len(v) for v in resp_by.values())})",
        ylabel="Respired CO₂ Δ¹⁴C (‰)",
        hf_val=90.0, br_val=18.0,
    )

    # Shared legend at the bottom
    legend_patches = [Patch(facecolor=BIOME_COLORS[b], label=b, alpha=0.7)
                       for b in order_bulk]
    legend_patches.append(
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=C_ACCENT,
                   markeredgecolor=C_PRIMARY, markersize=12,
                   label="This study (HF, Barrow)"),
    )
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(legend_patches), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Both stored and respired soil ¹⁴C vary systematically by biome — and they tell different stories",
        fontsize=13, color=C_PRIMARY, fontweight="bold",
    )
    plt.savefig(OUT_PATH, dpi=170, bbox_inches="tight")
    print(f"saved {OUT_PATH}")

    # Print medians
    print("\nMedian Δ¹⁴C by biome:")
    print(f"{'biome':<10s}  {'bulk (0-30 cm)':>16s}  {'respired':>12s}  ratio_stored_younger")
    for b in order:
        bv = bulk_by.get(b, np.array([]))
        rv = resp_by.get(b, np.array([]))
        if len(bv) == 0 or len(rv) == 0:
            continue
        b_med = np.median(bv); r_med = np.median(rv)
        gap = r_med - b_med
        print(f"  {b:<10s}  {b_med:+7.1f} ‰ (n={len(bv):4d})  {r_med:+6.1f} ‰ (n={len(rv):3d})  "
              f"resp − bulk = {gap:+.0f} ‰")


if __name__ == "__main__":
    main()
