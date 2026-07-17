"""
bulk_respired_on_par_with_fraction_14c.py — Headline figure:
bulk + respired ¹⁴C is on par with density-fraction ¹⁴C.

Mechanism.  The three model pools (active / slow / passive) each carry one
Δ¹⁴C.  Different observations are different *weighted projections* of that
pool vector:

    fraction Δ¹⁴C   ≈ direct read of each pool (chemistry separates them)
    bulk Δ¹⁴C       = STOCK-weighted mix   Σ (C_i / ΣC)·Δ_i     → slow + passive
    respired Δ¹⁴C   = FLUX-weighted mix    Σ (C_i/τ_i / Σ)·Δ_i  → active

Bulk grabs the slow/mineral-dominated end of the turnover spectrum; respired
grabs the fast end.  Together they span the same 3-pool space fractions
resolve directly — so bulk+respired recovers the fraction-equivalent
information.  At Barrow no density fractions exist in ISRaD at all, so
bulk+respired is the *only* option — and still nearly saturates the 3-pool
model.

Quantified as degrees of freedom for signal (DFS), resolved per pool via the
OE Jacobian at the MAP estimate (diagonal of the averaging kernel restricted
to each τ).

Run from the repository root:
    python notebooks/bulk_respired_on_par_with_fraction_14c.py

Output
------
  notebooks/bulk_respired_on_par_with_fraction_14c.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NB_ROOT = os.path.join(_SCRIPT_ROOT, "notebooks")
_SRC_ROOT = os.path.join(_SCRIPT_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_SCRIPT_ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from sites.canonical import run_hf_canonical, run_barrow_canonical  # noqa: E402
from ecosystem_complexity._oe_helpers import (  # noqa: E402
    _build_obs_blocks, _analytical_c12_ss, _build_sa_diag,
)
from ecosystem_complexity.api import run_model  # noqa: E402
from ecosystem_complexity.oe_utils import build_mean_ss_modifier  # noqa: E402
from ecosystem_complexity.optimizer import params_to_vector, vector_to_params  # noqa: E402
from ecosystem_complexity.state import make_default_params  # noqa: E402
from ecosystem_complexity.oe_diagnostics import classify_block  # noqa: E402
from ecosystem_complexity.sensitivity import (  # noqa: E402
    OBS_C_STOCKS, OBS_POOL_D14C, OBS_RESP_D14C,
)

_POOLS = ["soil_active", "soil_slow", "soil_passive"]
_POOL_SHORT = ["active", "slow", "passive"]

# Presentation palette (matches build_ecosystem_complexity_presentation.js)
_C_FRAC = "#4A7C59"   # moss green   — fractions (benchmark)
_C_BULK = "#D4A574"   # warm sand    — bulk (stock-weighted)
_C_RESP = "#B5502E"   # radiocarbon red — respired (flux-weighted)
_C_SUM  = "#1F3A2E"   # deep forest  — bulk + respired


# ════════════════════════════════════════════════════════════════════════════
# Per-pool DFS for arbitrary observation-type subsets (OE Jacobian at MAP)
# ════════════════════════════════════════════════════════════════════════════

def _build_ctx(data: dict) -> dict:
    model, forcing, state0 = data["model"], data["forcing"], data["state_at_map"]
    observations = data["obs_full"]
    extra = data.get("extra_blocks", [])
    opt_fields = tuple(data["opt_fields"])
    params_opt = data["params_opt"]

    inv = getattr(model.config, "inversion_raw", {}) or {}
    sp = float(inv.get("sigma_pool_14C", 5.0))
    sr = float(inv.get("sigma_resp_14C", 10.0))
    sc = float(inv.get("sigma_carbon_gCm2", 1000.0))
    blocks = _build_obs_blocks(observations, model, sp, sr, sc,
                               f_hetero=0.0, sigma_er_frac=0.15) + list(extra)
    types = [classify_block(b.name) for b in blocks]

    params0 = make_default_params(model.config)
    sa = np.array(_build_sa_diag(model.config, params0, opt_fields))
    sa_inv = 1.0 / (sa + 1e-30)
    n_pools = len(model.pool_index)
    cue = float(getattr(model.config.external_inputs, "CUE", 0.47))
    mean_mod, mean_gpp = build_mean_ss_modifier(forcing, params0)
    mean_input = mean_gpp * cue
    target_names = list(model.config.external_inputs.partition.keys())
    target_idx = [model.pool_index[nm] for nm in target_names] or None

    def _fwd(x):
        p = vector_to_params(x, params0, opt_fields)
        c12 = _analytical_c12_ss(p, n_pools, mean_input, mean_mod, target_indices=target_idx)
        out = run_model(model, forcing, state0=state0._replace(C12=c12), params=p)
        return jnp.concatenate([b.predict(out, p) for b in blocks])

    x_opt = params_to_vector(params_opt, opt_fields)
    k = np.array(jax.jacobian(_fwd)(x_opt))
    se = np.array(jnp.concatenate([b.Se for b in blocks]))
    lens = [int(b.y.shape[0]) for b in blocks]
    starts = np.cumsum([0] + lens)
    names = list(data["oe_result"].state_names)
    tau_idx = [names.index(f"log_tau[{model.pool_index[p]}]") for p in _POOLS]
    return dict(k=k, se=se, types=types, starts=starts, sa_inv=sa_inv, tau_idx=tau_idx)


def _dfs_per_pool(ctx: dict, active_types: list[str]) -> tuple[float, np.ndarray]:
    k, se, types = ctx["k"], ctx["se"], ctx["types"]
    starts, sa_inv, tau_idx = ctx["starts"], ctx["sa_inv"], ctx["tau_idx"]
    mask = np.zeros(k.shape[0], dtype=bool)
    for i, t in enumerate(types):
        if t in active_types:
            mask[starts[i]:starts[i + 1]] = True
    if not mask.any():
        return 0.0, np.zeros(len(tau_idx))
    ks, ss = k[mask], se[mask]
    ktsek = (ks.T / ss) @ ks
    a = np.linalg.inv(ktsek + np.diag(sa_inv)) @ ktsek
    return float(np.trace(a)), np.diag(a)[tau_idx]


def analyse(data: dict) -> dict:
    ctx = _build_ctx(data)
    P, R = OBS_POOL_D14C, OBS_RESP_D14C
    out = {}
    for key, types in [("frac", [P]), ("bulk", [P]), ("resp", [R]), ("bulk+resp", [P, R])]:
        tot, per = _dfs_per_pool(ctx, types)
        out[key] = dict(total=tot, per_pool=per)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Figure
# ════════════════════════════════════════════════════════════════════════════

def _panel(ax, per_pool: dict, series: list[tuple[str, str, str]], title: str) -> None:
    x = np.arange(len(_POOLS))
    n = len(series)
    w = 0.8 / n
    for j, (key, label, color) in enumerate(series):
        vals = per_pool[key]["per_pool"]
        off = (j - (n - 1) / 2) * w
        edge = "black" if key == "bulk+resp" else "none"
        ax.bar(x + off, vals, w, color=color, alpha=0.92, label=label,
               edgecolor=edge, linewidth=1.1 if key == "bulk+resp" else 0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"τ$_{{\\mathrm{{{s}}}}}$" for s in _POOL_SHORT], fontsize=12)
    ax.set_ylabel("Information resolved  (DFS per pool)", fontsize=10)
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold", color=_C_SUM)
    ax.set_ylim(0, 1.12)
    ax.axhline(1.0, lw=0.8, ls=":", color="0.55")
    ax.text(-0.42, 1.01, "fully resolved", fontsize=8, color="0.5",
            ha="left", va="bottom")
    ax.legend(fontsize=9, framealpha=0.95, loc="upper right", ncol=1)
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _panel_operators(ax, ops: dict, frac_alone: float) -> None:
    """Panel (c): total DFS by bulk operator — bulk-alone vs bulk+respired."""
    names = list(ops.keys())
    x = np.arange(len(names))
    w = 0.38
    alone = [ops[n][0] for n in names]
    withr = [ops[n][1] for n in names]
    ax.bar(x - w / 2, alone, w, color=_C_BULK, alpha=0.92, label="Bulk alone")
    ax.bar(x + w / 2, withr, w, color=_C_SUM, alpha=0.92, label="Bulk + Respired",
           edgecolor="black", linewidth=1.0)
    for xi, va in zip(x - w / 2, alone):
        ax.text(xi, va + 0.04, f"{va:.2f}", ha="center", va="bottom", fontsize=8.5, color="0.3")
    for xi, va in zip(x + w / 2, withr):
        ax.text(xi, va + 0.04, f"{va:.2f}", ha="center", va="bottom", fontsize=8.5, color="0.3")
    ax.axhline(frac_alone, lw=1.4, ls="--", color=_C_FRAC, alpha=0.95)
    ax.text(1.0, frac_alone - 0.06, f"Fractions alone ({frac_alone:.2f})",
            ha="center", va="top", fontsize=9, color=_C_FRAC, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Total DFS", fontsize=10)
    ax.set_title("(c) Bulk operator matters for bulk-alone,\n      not for bulk + respired",
                 fontsize=12, loc="left", fontweight="bold", color=_C_SUM)
    ax.set_ylim(0, 3.6)
    ax.legend(fontsize=9, framealpha=0.95, loc="upper left")
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def make_figure(hf: dict, br: dict, ops: dict, frac_alone: float, out_path: str) -> None:
    fig, (ax_hf, ax_br, ax_op) = plt.subplots(
        1, 3, figsize=(19.5, 5.4), gridspec_kw={"width_ratios": [1.05, 0.9, 1.0]},
    )

    _panel(
        ax_hf, hf,
        series=[
            ("frac",      f"Fractions alone   (Σ={hf['frac']['total']:.2f})", _C_FRAC),
            ("bulk",      f"Bulk depth-proxy   (Σ={hf['bulk']['total']:.2f})", _C_BULK),
            ("resp",      f"Respired alone   (Σ={hf['resp']['total']:.2f})",  _C_RESP),
            ("bulk+resp", f"Bulk + Respired   (Σ={hf['bulk+resp']['total']:.2f})", _C_SUM),
        ],
        title="(a) Harvard Forest — fractions available",
    )
    _panel(
        ax_br, br,
        series=[
            ("bulk",      f"Bulk depth-proxy   (Σ={br['bulk']['total']:.2f})", _C_BULK),
            ("resp",      f"Respired alone   (Σ={br['resp']['total']:.2f})",  _C_RESP),
            ("bulk+resp", f"Bulk + Respired   (Σ={br['bulk+resp']['total']:.2f})", _C_SUM),
        ],
        title="(b) Barrow — no density fractions exist in ISRaD",
    )
    _panel_operators(ax_op, ops, frac_alone)

    # Caption strip under the per-pool panels (kept out of the plotting area).
    ax_hf.text(0.5, -0.17,
               "Fractions (green) ≈ Bulk + Respired (dark) on every pool",
               transform=ax_hf.transAxes, fontsize=8.5, style="italic",
               color="0.35", ha="center", va="top")
    ax_br.text(0.5, -0.17,
               "No fractions here — bulk + respired is the only option, still nearly saturates",
               transform=ax_br.transAxes, fontsize=8.5, style="italic",
               color="0.35", ha="center", va="top")
    ax_op.text(0.5, -0.17,
               "Bulk-alone is rank-limited (≤ pool count); how you weight layers only moves it 1.0→2.0",
               transform=ax_op.transAxes, fontsize=8.5, style="italic",
               color="0.35", ha="center", va="top")

    fig.suptitle(
        "Bulk + respired ¹⁴C is on par with density-fraction ¹⁴C\n"
        "Bulk = stock-weighted (slow + passive) · Respired = flux-weighted (active) "
        "· together they span the pool space fractions resolve directly",
        fontsize=13.5, color=_C_SUM, y=1.03,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")


if __name__ == "__main__":
    print("\n══ Harvard Forest (fraction) ══")
    hf_frac = analyse(run_hf_canonical(pool_14c_source="fraction"))
    print("\n══ Harvard Forest (bulk depth-proxy) ══")
    hf_bulk = analyse(run_hf_canonical(pool_14c_source="bulk"))
    print("\n══ Harvard Forest (single mixture) ══")
    hf_mix = analyse(run_hf_canonical(pool_14c_source="mixture"))
    print("\n══ Harvard Forest (per-layer mixture) ══")
    hf_per = analyse(run_hf_canonical(pool_14c_source="perlayer"))

    # Panels (a)/(b): per-pool complementarity (depth-proxy bulk for continuity).
    hf = {
        "frac":      hf_frac["frac"],
        "bulk":      hf_bulk["bulk"],
        "resp":      hf_bulk["resp"],
        "bulk+resp": hf_bulk["bulk+resp"],
    }

    # Panel (c): total DFS by bulk operator — bulk-alone vs bulk+respired.
    ops = {
        "Depth-proxy":    (hf_bulk["bulk"]["total"], hf_bulk["bulk+resp"]["total"]),
        "Single\nmixture": (hf_mix["bulk"]["total"],  hf_mix["bulk+resp"]["total"]),
        "Per-layer\nmixture": (hf_per["bulk"]["total"], hf_per["bulk+resp"]["total"]),
    }
    frac_alone = hf_frac["frac"]["total"]

    print("\n══ Barrow (bulk depth-proxy — no fractions available) ══")
    br = analyse(run_barrow_canonical())

    print("\n── Per-pool DFS summary ──")
    for site, res, keys in [("HF", hf, ["frac", "bulk", "resp", "bulk+resp"]),
                            ("Barrow", br, ["bulk", "resp", "bulk+resp"])]:
        print(f"\n {site}")
        for kkey in keys:
            pp = res[kkey]["per_pool"]
            print(f"   {kkey:<10s} Σ={res[kkey]['total']:.3f}  "
                  f"[active {pp[0]:.2f}  slow {pp[1]:.2f}  passive {pp[2]:.2f}]")

    print("\n── HF bulk-operator ladder (total DFS) ──")
    print(f"   {'operator':<18s} {'bulk-alone':>11s} {'bulk+resp':>11s}")
    for name, (a, b) in ops.items():
        print(f"   {name.replace(chr(10),' '):<18s} {a:>11.3f} {b:>11.3f}")
    print(f"   {'fractions-alone':<18s} {frac_alone:>11.3f}")

    out_path = os.path.join(_NB_ROOT, "bulk_respired_on_par_with_fraction_14c.png")
    make_figure(hf, br, ops, frac_alone, out_path)
