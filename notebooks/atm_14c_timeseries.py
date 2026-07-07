"""
atm_14c_timeseries.py — Slide-4 figure.

Atmospheric Δ¹⁴C from 1900–2025 (IntCal20 pre-1950 + Hua 2021 / Graven 2017
post-bomb), with key events annotated and a shaded band marking the ISRaD
soil-flux sampling window.

Output: notebooks/atmospheric_delta14c_sampling_windows.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_SCRIPT_ROOT, "src"))
os.chdir(_SCRIPT_ROOT)

import pandas as pd

C_PRIMARY   = "#1F3A2E"
C_SECONDARY = "#4A7C59"
C_ACCENT    = "#D4A574"
C_BODY      = "#2A2A2A"
C_MUTED     = "#6B6B6B"
C_BGLIGHT   = "#F7F4ED"

HUA    = "data/shared/atm_14C/Hua_2021.csv"
GRAVEN = "data/shared/atm_14C/Graven_2017.csv"
INTCAL = "data/shared/atm_14C/intcal20.14c"

OUT = os.path.join(_SCRIPT_ROOT, "notebooks", "atmospheric_delta14c_sampling_windows.png")


def _splice(hua_col: str, graven_col: str):
    """Splice Graven (pre-1941) + Hua (post-1941) into a single record."""
    graven = pd.read_csv(GRAVEN).dropna(subset=["Date", graven_col])
    hua = pd.read_csv(HUA).dropna(subset=["Year.AD", hua_col])
    g_mask = (graven["Date"] >= 1900) & (graven["Date"] < 1941)
    h_mask = (hua["Year.AD"] >= 1941) & (hua["Year.AD"] <= 2025)
    yrs  = np.concatenate([graven.loc[g_mask, "Date"].values,
                            hua.loc[h_mask, "Year.AD"].values]).astype(float)
    vals = np.concatenate([graven.loc[g_mask, graven_col].values,
                            hua.loc[h_mask, hua_col].values]).astype(float)
    order = np.argsort(yrs)
    return yrs[order], vals[order]


def main():
    yrs_nh, d14c_nh = _splice("NH14C", "NHc14")
    yrs_sh, d14c_sh = _splice("SH14C", "SHc14")

    fig = plt.figure(figsize=(13.0, 5.8))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.4],
                            wspace=0.18, left=0.07, right=0.97, top=0.86, bottom=0.13)

    # ── (a) Full record 1900–2025 with bomb spike ───────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(yrs_nh, d14c_nh, color=C_PRIMARY, lw=2.0, label="Northern Hemisphere", zorder=3)
    ax.plot(yrs_sh, d14c_sh, color=C_SECONDARY, lw=1.4,
            linestyle="--", alpha=0.85, label="Southern Hemisphere", zorder=2)

    # ISRaD sampling window
    ax.axvspan(1979, 2018, color=C_ACCENT, alpha=0.15, zorder=1,
               label="ISRaD soil-flux sampling window")
    # Bomb test ban annotation
    ax.axvline(1963, color=C_PRIMARY, lw=0.7, linestyle=":", alpha=0.7)
    peak = np.nanmax(d14c_nh)
    peak_yr = yrs_nh[np.nanargmax(d14c_nh)]
    ax.annotate(
        f"Bomb peak\n{peak:.0f} ‰  ({peak_yr:.0f})",
        xy=(peak_yr, peak),
        xytext=(peak_yr + 11, peak - 100),
        fontsize=9, color=C_PRIMARY, ha="left",
        arrowprops=dict(arrowstyle="-", color=C_PRIMARY, lw=0.8),
    )
    ax.annotate(
        "Partial Test Ban\n1963",
        xy=(1963, 600), xytext=(1922, 620),
        fontsize=9, color=C_MUTED, ha="left",
        arrowprops=dict(arrowstyle="-", color=C_MUTED, lw=0.6),
    )

    ax.axhline(0, lw=0.5, color="0.4", linestyle=":")
    ax.set_xlim(1900, 2025)
    ax.set_xlabel("Year", fontsize=11, color=C_BODY)
    ax.set_ylabel("Atmospheric Δ¹⁴C (‰)", fontsize=11, color=C_BODY)
    ax.set_title("(a) Full record  1900–2025", fontsize=12, color=C_PRIMARY,
                 fontweight="bold", loc="left", pad=10)
    ax.grid(lw=0.3, alpha=0.4)
    ax.legend(fontsize=9, framealpha=0.92, loc="upper right")

    # ── (b) Modern decline 1990–2025 (zoom) ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    m_nh = (yrs_nh >= 1990) & (yrs_nh <= 2025)
    m_sh = (yrs_sh >= 1990) & (yrs_sh <= 2025)
    ax2.plot(yrs_nh[m_nh], d14c_nh[m_nh], color=C_PRIMARY, lw=2.2,
             label="Northern Hemisphere", zorder=3)
    ax2.plot(yrs_sh[m_sh], d14c_sh[m_sh], color=C_SECONDARY, lw=1.5,
             linestyle="--", alpha=0.85, label="Southern Hemisphere", zorder=2)

    # Annotate HF + Barrow sampling windows
    ax2.axvspan(1996, 2010, color=C_ACCENT, alpha=0.22, zorder=1,
                 label="HF NWN resp. window")
    ax2.axvspan(2012, 2014, color="#5A88A8", alpha=0.25, zorder=1,
                 label="Barrow surface emissions")

    # Annotate slope: rate of modern decline
    mask_modern = (yrs_nh >= 2000) & (yrs_nh <= 2020)
    slope = np.polyfit(yrs_nh[mask_modern], d14c_nh[mask_modern], 1)[0]
    ax2.text(0.97, 0.93, f"Modern decline:\n{slope:+.1f} ‰ yr⁻¹  (2000–2020)",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=10, color=C_PRIMARY,
             bbox=dict(boxstyle="round,pad=0.4", fc=C_BGLIGHT,
                       ec=C_ACCENT, lw=1))

    ax2.axhline(0, lw=0.5, color="0.4", linestyle=":")
    ax2.set_xlim(1990, 2025)
    ax2.set_xlabel("Year", fontsize=11, color=C_BODY)
    ax2.set_ylabel("Atmospheric Δ¹⁴C (‰)", fontsize=11, color=C_BODY)
    ax2.set_title("(b) Modern decline  1990–2025  (our sampling windows)",
                  fontsize=12, color=C_PRIMARY, fontweight="bold",
                  loc="left", pad=10)
    ax2.grid(lw=0.3, alpha=0.4)
    ax2.legend(fontsize=9, framealpha=0.92, loc="upper right")

    fig.suptitle(
        "Atmospheric Δ¹⁴C — the clock that times our soil-C inversions",
        fontsize=14, color=C_PRIMARY, fontweight="bold",
    )
    plt.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
