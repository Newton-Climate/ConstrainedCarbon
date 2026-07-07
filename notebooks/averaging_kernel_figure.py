"""
averaging_kernel_figure.py — Side-by-side τ-block averaging kernels.

Runs the canonical 3-pool OE inversion at Harvard Forest and Barrow and renders
the turnover-time block of the Rodgers (2000) averaging kernel A = Sₓ (KᵀSₑ⁻¹K)
for each site.  The diagonal of A is the degrees-of-freedom-for-signal (DFS)
contribution of each parameter (0 = prior-dominated, 1 = fully data-resolved);
off-diagonal terms show how the retrieval of one turnover time smooths into the
others.  trace(A) over the full parameter vector equals the site total DFS.

Run from the repository root:
    python notebooks/averaging_kernel_figure.py

Output
------
  notebooks/averaging_kernel_tau_comparison.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_SCRIPT_ROOT, "src"))
sys.path.insert(0, os.path.join(_SCRIPT_ROOT, "notebooks"))
os.chdir(_SCRIPT_ROOT)

from sites.canonical import run_hf_canonical, run_barrow_canonical

# Forest palette (matches the deck)
C_PRIMARY = "#1F3A2E"
C_BODY = "#2A2A2A"
C_MUTED = "#6B6B6B"

_POOL_LABEL = {
    "soil_active":  r"$\tau_{\mathrm{active}}$",
    "soil_slow":    r"$\tau_{\mathrm{slow}}$",
    "soil_passive": r"$\tau_{\mathrm{passive}}$",
}


def _tau_block(data: dict) -> tuple[np.ndarray, list[str], float]:
    res = data["oe_result"]
    A = np.array(res.averaging_kernel)
    names = list(res.state_names)
    tau_idx = [i for i, n in enumerate(names) if n.startswith("log_tau")]
    A_tau = A[np.ix_(tau_idx, tau_idx)]
    pool_names = list(data["idx"].pool_names)
    labels = [_POOL_LABEL.get(p, p) for p in pool_names]
    total_dfs = float(np.trace(A))
    return A_tau, labels, total_dfs


def make_figure(hf: dict, br: dict, out_path: str) -> None:
    hf_A, labels, hf_dfs = _tau_block(hf)
    br_A, _, br_dfs = _tau_block(br)

    fig = plt.figure(figsize=(12.2, 5.4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.055], wspace=0.28,
                           left=0.07, right=0.93, top=0.82, bottom=0.14)

    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=1.0)
    cmap = plt.get_cmap("BrBG")

    panels = [
        ("Harvard Forest  (US-Ha1)", hf_A, hf_dfs, "#4A7C59"),
        ("Barrow, Alaska  (US-A10)", br_A, br_dfs, C_PRIMARY),
    ]

    im = None
    for col, (title, A_tau, total_dfs, c_title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(A_tau, cmap=cmap, norm=norm, aspect="equal")
        n = A_tau.shape[0]
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_yticklabels(labels, fontsize=13)
        ax.set_xlabel("True parameter", fontsize=11, color=C_BODY)
        if col == 0:
            ax.set_ylabel("Retrieved parameter", fontsize=11, color=C_BODY)
        # Grid lines between cells
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0)
        # Annotate cells
        for i in range(n):
            for j in range(n):
                v = A_tau[i, j]
                is_diag = i == j
                txt_color = "white" if v > 0.55 else C_BODY
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=13 if is_diag else 11,
                        fontweight="bold" if is_diag else "normal",
                        color=txt_color)
        ax.set_title(f"{title}\ntotal DFS = {total_dfs:.2f}   ·   "
                     f"diag(A$_\\tau$) = {np.trace(A_tau):.2f}",
                     fontsize=12.5, color=c_title, fontweight="bold",
                     loc="center", pad=12)

    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Averaging-kernel value\n(diagonal = DFS: 0 = prior, 1 = data-resolved)",
                 fontsize=10, color=C_BODY)
    cb.ax.tick_params(labelsize=9)

    fig.suptitle("Averaging kernels resolve $\\tau_{\\mathrm{active}}$ fully at "
                 "both sites, but $\\tau_{\\mathrm{slow}}$ and "
                 "$\\tau_{\\mathrm{passive}}$ trade off — especially at Barrow",
                 fontsize=13.5, color=C_PRIMARY, fontweight="bold", y=0.97)
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    print("Running canonical inversions for averaging-kernel figure…")
    hf_data = run_hf_canonical()
    br_data = run_barrow_canonical()
    out = os.path.join(_SCRIPT_ROOT, "notebooks",
                       "averaging_kernel_tau_comparison.png")
    make_figure(hf_data, br_data, out)
