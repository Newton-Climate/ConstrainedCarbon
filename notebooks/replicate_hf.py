"""
replicate_hf.py — Forward simulation replicating Sierra et al. (2012) /
Gaudinski et al. (2000) at Harvard Forest.

Approach
--------
  1. Long C12+C14 spinup (1400–1900) to bring both pools to consistent
     steady-state values.  Uses constant synthetic GPP with the Gaudinski
     input budget:
       leaf litter   150 gC m⁻² yr⁻¹  → Oi
       dead roots     65 gC m⁻² yr⁻¹  → dead_roots aboveground pool
     GPP_syn = 215 / CUE / 365 = 1.254 gC m⁻² day⁻¹ (soil_input_fraction=0.698)

  2. Main run (1900–2010): continue from spinup final state; compare Δ¹⁴C
     to hf212-01 NWN (respired CO₂) and hf212-03 (density fractions).

  * Q10=1.0, constant climate (T=10°C, θ=0.35), constant GPP.
  * Atmospheric Δ¹⁴C: IntCal20 (pre-1950) + Hua 2021 / Graven 2017 (post-1950).

Run
---
  python notebooks/replicate_hf.py
"""
from __future__ import annotations

import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Path resolution ───────────────────────────────────────────────────────────
_SCRIPT_ROOT   = os.path.dirname(os.path.abspath(__file__))
_WORKTREE_ROOT = os.path.dirname(_SCRIPT_ROOT)
_SRC_ROOT      = os.path.join(_WORKTREE_ROOT, "src")


def _find_data_root(start: str) -> str:
    candidate = start
    for _ in range(4):
        if os.path.isdir(os.path.join(candidate, "data")):
            return candidate
        candidate = os.path.dirname(candidate)
    return _WORKTREE_ROOT


_REPO_ROOT = os.environ.get("ECOSYSTEM_REPO_ROOT") or _find_data_root(_WORKTREE_ROOT)

if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from ecosystem_complexity.api import build_model, run_model
from ecosystem_complexity.config import load_config
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.schemas import ForcingData
from ecosystem_complexity.state import make_default_params, make_initial_state

# ── File paths ────────────────────────────────────────────────────────────────
def _wt(rel):
    return os.path.join(_WORKTREE_ROOT, rel)


def _data(rel):
    return os.path.join(_REPO_ROOT, rel)


HF_SOIL_14C_PATH = _data("data/harvard_forest/hf212-03-14c-org.csv")
HF_RESP_14C_PATH = _data("data/harvard_forest/hf212-01-14c-no-treat.csv")
HUA_PATH         = _data("data/shared/atm_14C/Hua_2021.csv")
GRAVEN_PATH      = _data("data/shared/atm_14C/Graven_2017.csv")
INTCAL_PATH      = _data("data/shared/atm_14C/intcal20.14c")
SIERRA_CONFIG    = _wt("configs/harvard_sierra2012_config.yaml")

_R_STD = 1.176e-12

# Input budget (Gaudinski 2000 Fig. 1):
#   leaf litter  150 gC m⁻² yr⁻¹  → Oi  (direct soil input, soil_frac=0.698)
#   dead roots    65 gC m⁻² yr⁻¹  → dead_roots (ag pool, ag_frac=0.302)
#   total model  215 gC m⁻² yr⁻¹
# GPP_syn = 215 / CUE / 365  (CUE=0.47 from config)
# +2.7 gC/yr DOC input to MinAss (Gaudinski Fig. 1 leachate pathway)
_GPP_SYN = (215.0 + 2.7) / 0.47 / 365.0   # 1.2697 gC m⁻² day⁻¹


# ════════════════════════════════════════════════════════════════════════════
# Forcing builder
# ════════════════════════════════════════════════════════════════════════════

def _build_forcing(
    start_year: float,
    end_year: float,
    atm_years: np.ndarray,
    atm_d14C: np.ndarray,
) -> ForcingData:
    """Daily synthetic forcing: constant GPP + historical Δ¹⁴C_atm."""
    n_days = int(round((end_year - start_year) * 365.25))
    t0_days  = (start_year - 1970.0) * 365.25
    time_arr = np.arange(n_days, dtype=np.float32) + t0_days
    years_arr = 1970.0 + time_arr / 365.25
    d14c_arr  = np.interp(years_arr, atm_years, atm_d14C).astype(np.float32)

    n_layers = 2
    return ForcingData(
        time          = jnp.array(time_arr),
        air_temp      = jnp.full(n_days, 10.0, dtype=jnp.float32),
        sw_radiation  = jnp.full(n_days, 12.0, dtype=jnp.float32),
        precip        = jnp.full(n_days, 3.0,  dtype=jnp.float32),
        vpd           = jnp.full(n_days, 0.5,  dtype=jnp.float32),
        soil_temp     = jnp.full((n_days, n_layers), 10.0, dtype=jnp.float32),
        soil_moisture = jnp.full((n_days, n_layers), 0.35, dtype=jnp.float32),
        snow_depth    = jnp.zeros(n_days, dtype=jnp.float32),
        active_layer  = jnp.ones(n_days,  dtype=jnp.float32),
        delta14C_atm  = jnp.array(d14c_arr),
        GPP_obs       = jnp.full(n_days, _GPP_SYN, dtype=jnp.float32),
        NPP_obs       = jnp.full(n_days, float("nan"), dtype=jnp.float32),
    )


# ════════════════════════════════════════════════════════════════════════════
# Observation loaders
# ════════════════════════════════════════════════════════════════════════════

def _load_obs(time_years: np.ndarray) -> tuple[dict, np.ndarray]:
    """
    Load fraction Δ¹⁴C observations from hf212-03.

    Sierra et al. (2012) Fig. 3 caption: "The Oe/a fraction combines the
    Oe/a L and Oe/a H fractions of the original model.  Similarly, A-LF
    (< 80 µm) and A-LF (> 80 µm) were combined in a single fraction."

    So the four observable combined fractions are:
        "Oi"    → organic_Oi              (single pool)
        "Oe/a"  → organic_Oe_a_L + Oe_a_H (mass-weighted)
        "A-LF"  → mineral_ALF_gt80 + ALF_lt80 (mass-weighted)
        "A-min" → mineral_MinAss          (single pool)

    Returns
    -------
    frac_obs : dict  horizon_label → (T,) array of observed Δ¹⁴C (NaN where unobserved)
    resp_obs : (T,) array of respired CO₂ Δ¹⁴C (NaN where unobserved)
    """
    T = len(time_years)
    # Maps CSV horizon name → label used in frac_obs dict
    _horizon_label = {
        "Oi":    "Oi",
        "Oe":    "Oe/a",
        "A-lf":  "A-LF",
        "A-min": "A-min",
    }
    d14c_df = pd.read_csv(HF_SOIL_14C_PATH)
    d14c_df["horizon"] = d14c_df["horizon"].str.strip()

    frac_obs: dict[str, np.ndarray] = {}
    for year in d14c_df["year"].unique():
        yr_df = d14c_df[d14c_df["year"] == year]
        t_idx = int(np.argmin(np.abs(time_years - (float(year) + 0.5))))
        for horizon, label in _horizon_label.items():
            vals = yr_df.loc[yr_df["horizon"] == horizon, "of.14c.12c"].dropna()
            if len(vals):
                if label not in frac_obs:
                    frac_obs[label] = np.full(T, np.nan, dtype=np.float32)
                frac_obs[label][t_idx] = float(vals.mean())

    resp_df  = pd.read_csv(HF_RESP_14C_PATH)
    nwn      = resp_df[resp_df["site"] == "NWN"]
    by_date  = nwn.groupby("year.date")["delta.14c"].mean()
    resp_obs = np.full(T, np.nan, dtype=np.float32)
    for obs_year, val in by_date.items():
        t_idx = int(np.argmin(np.abs(time_years - float(obs_year))))
        resp_obs[t_idx] = float(val)

    return frac_obs, resp_obs


# Sierra Fig. 3 combined fractions: horizon label → list of pool names to mass-average
_COMBINED_FRACS = {
    "Oi":   ["organic_Oi"],
    "Oe/a": ["organic_Oe_a_L", "organic_Oe_a_H"],
    "A-LF": ["mineral_ALF_gt80", "mineral_ALF_lt80"],
    "A-min":["mineral_MinAss"],
}


def _combined_d14c(
    label: str,
    pool_index,
    d14c_sim: np.ndarray,
    c12_sim: np.ndarray,
) -> np.ndarray:
    """
    Mass-weighted Δ¹⁴C time series for a combined fraction.

    For single-pool fractions this is just d14c_sim[:, i].
    For multi-pool fractions: Σ(C12_i × Δ¹⁴C_i) / Σ(C12_i).
    """
    pools = _COMBINED_FRACS[label]
    indices = [pool_index[p] for p in pools]
    c12_total = c12_sim[:, indices].sum(axis=1)                    # (T,)
    d14c_avg  = (c12_sim[:, indices] * d14c_sim[:, indices]).sum(axis=1) / (
        c12_total + 1e-30
    )
    return d14c_avg


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("Sierra et al. (2012) replication — Harvard Forest")
    print("=" * 55)

    config = load_config(SIERRA_CONFIG)
    model  = build_model(SIERRA_CONFIG)
    idx    = model.pool_index
    params = make_default_params(config)
    print(f"Pools ({len(idx)}): {idx.pool_names}")
    print(f"τ (yr): { {n: round(float(np.exp(params.log_tau[i]))/365, 1) for i, n in enumerate(idx.pool_names)} }")
    print(f"GPP_syn = {_GPP_SYN:.4f} gC m⁻² day⁻¹ "
          f"(150 leaf + 65 roots = 215 gC m⁻² yr⁻¹)")

    print("\nLoading atmospheric ¹⁴C record…")
    atm_years, atm_d14C = load_full_14C_record(
        HUA_PATH, GRAVEN_PATH, INTCAL_PATH,
        hemisphere="NH", start_year=1400.0, end_year=2025.0,
    )

    # ── Initialise from Gaudinski 2000 Fig. 1 C stocks (paper's approach) ────
    # Sierra 2012 Sec 2.2: "the carbon content for each fraction reported in
    # Fig. 1 was used as the initial values of the simulation; simulations
    # started in the year 1900."  Do NOT spinup to model SS — the large
    # Oe/a H stock (1366 gC m⁻²) is a non-SS initial condition that is key
    # to propagating bomb ¹⁴C into the mineral pools over subsequent decades.
    _gaud_stocks = {
        "dead_roots":        390.0,
        "organic_Oi":        220.0,
        "organic_Oe_a_L":    388.0,
        "organic_Oe_a_H":   1366.0,
        "mineral_ALF_gt80":   90.0,
        "mineral_ALF_lt80": 1800.0,
        "mineral_MinAss":    560.0,
    }
    d14c_1900 = float(np.interp(1900.0, atm_years, atm_d14C))
    R_1900    = _R_STD * (1.0 + d14c_1900 / 1000.0)
    n_pools   = len(idx)
    C12_arr   = np.zeros(n_pools, dtype=np.float32)
    C14_arr   = np.zeros(n_pools, dtype=np.float32)
    for name, val in _gaud_stocks.items():
        i = idx[name]
        C12_arr[i] = float(val)
        C14_arr[i] = float(val) * R_1900
    state_1900 = make_initial_state(config, {})._replace(
        C12=jnp.array(C12_arr), C14=jnp.array(C14_arr)
    )
    print(f"\nInitial state (Gaudinski 2000 Fig. 1, Δ¹⁴C_atm(1900)={d14c_1900:.1f}‰):")
    print(f"  {'Pool':<22}  {'C12 (gC/m²)':>12}  {'Δ¹⁴C_init':>10}")
    for i, name in enumerate(idx.pool_names):
        print(f"  {name:<22}  {C12_arr[i]:>12.0f}  {d14c_1900:>+10.1f}")

    # ── Main run 1900–2010 ────────────────────────────────────────────────────
    print("\nMain run 1900–2010 (dynamic C12 + C14)…")
    forcing_main = _build_forcing(1900.0, 2011.0, atm_years, atm_d14C)
    t0 = time.perf_counter()
    out = run_model(model, forcing_main, state0=state_1900, params=params)
    jax.block_until_ready(out.delta14C)
    print(f"  Done in {time.perf_counter()-t0:.1f} s")

    t0_days2    = (1900.0 - 1970.0) * 365.25
    n_days2     = len(forcing_main.time)
    time_years  = 1970.0 + (np.arange(n_days2, dtype=np.float64) + t0_days2) / 365.25
    d14c_sim    = np.array(out.delta14C)   # (T, n_pools)
    c12_sim     = np.array(out.C12)

    # Respired CO₂ flux-weighted Δ¹⁴C
    tau_arr       = np.exp(np.array(params.log_tau))
    flux_w        = c12_sim / (tau_arr[None, :] + 1e-30)
    d14c_resp_sim = (d14c_sim * flux_w).sum(-1) / (flux_w.sum(-1) + 1e-30)

    # Observations
    frac_obs, resp_obs = _load_obs(time_years)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print(f"\n{'Pool':<22}  {'mean Δ¹⁴C':>10}  {'mean C12':>10}")
    print("  " + "─" * 46)
    for i, name in enumerate(idx.pool_names):
        print(f"  {name:<22}  {np.nanmean(d14c_sim[:, i]):>+10.1f}  "
              f"{np.nanmean(c12_sim[:, i]):>10.0f}")

    # Pre-compute combined-fraction Δ¹⁴C time series
    combined_d14c_ts: dict[str, np.ndarray] = {
        label: _combined_d14c(label, idx, d14c_sim, c12_sim)
        for label in _COMBINED_FRACS
    }

    print("\nFraction Δ¹⁴C comparison — combined fractions [cf. Sierra Fig. 3]:")
    print("  (Oe/a = Oe_a_L+Oe_a_H mass-weighted;  A-LF = ALF_gt80+ALF_lt80 mass-weighted)")
    print(f"  {'Fraction':<10}  {'1996 sim':>9}  {'1996 obs':>9}  {'bias':>6}  "
          f"{'2007 sim':>9}  {'2007 obs':>9}  {'bias':>6}")
    print("  " + "─" * 72)
    for label in ["Oi", "Oe/a", "A-LF", "A-min"]:
        ts  = combined_d14c_ts[label]
        obs_ts = frac_obs.get(label)
        row = []
        for yr in [1996.0, 2007.0]:
            t_idx = int(np.argmin(np.abs(time_years - (yr + 0.5))))
            s = float(ts[t_idx])
            o = float(obs_ts[t_idx]) if obs_ts is not None and not np.isnan(obs_ts[t_idx]) else float("nan")
            row.append((s, o))
        s96, o96 = row[0]; s07, o07 = row[1]
        b96 = f"{s96-o96:>+6.0f}" if not np.isnan(o96) else "   n/a"
        b07 = f"{s07-o07:>+6.0f}" if not np.isnan(o07) else "   n/a"
        o96s = f"{o96:>+9.0f}" if not np.isnan(o96) else "      n/a"
        o07s = f"{o07:>+9.0f}" if not np.isnan(o07) else "      n/a"
        print(f"  {label:<10}  {s96:>+9.0f}  {o96s}  {b96}  "
              f"{s07:>+9.0f}  {o07s}  {b07}")

    mask = ~np.isnan(resp_obs)
    if mask.sum() > 0:
        rmse = float(np.sqrt(np.mean((d14c_resp_sim[mask] - resp_obs[mask]) ** 2)))
        bias = float(np.mean(d14c_resp_sim[mask] - resp_obs[mask]))
        print(f"\nRespired CO₂ Δ¹⁴C ({mask.sum()} obs): RMSE={rmse:.1f}‰  bias={bias:+.1f}‰")

    # ════════════════════════════════════════════════════════════════════════
    # Figure  [mirrors Sierra et al. 2012 Figs. 2 & 3]
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Harvard Forest — Sierra et al. (2012) replication\n"
        "Obs: hf212-03 fractions (combined per Sierra Fig. 3) + hf212-01 resp. CO₂",
        fontsize=11,
    )
    mask_atm  = (atm_years >= 1950) & (atm_years <= 2011)
    mask_plot = time_years >= 1950

    # ── Panel (a): Organic layer — sub-fractions + combined Oe/a ─────────────
    ax = axes[0]
    ax.plot(atm_years[mask_atm], atm_d14C[mask_atm],
            color="gray", lw=1, ls="--", label="Atmosphere", zorder=0)
    # Individual sub-fractions (thin, for reference)
    sub_colors = {"organic_Oi": "#e07b39", "organic_Oe_a_L": "#b05a20",
                  "organic_Oe_a_H": "#6b3417"}
    for pn, col in sub_colors.items():
        pi = idx[pn]
        lbl = pn.replace("organic_", "")
        ax.plot(time_years[mask_plot], d14c_sim[mask_plot, pi],
                color=col, lw=0.9, ls=":", alpha=0.7, label=f"{lbl} (sub)")
    # Combined fractions compared to obs (thick lines + scatter)
    # Oi: single pool, orange
    oi_ts = combined_d14c_ts["Oi"]
    ax.plot(time_years[mask_plot], oi_ts[mask_plot],
            color="#e07b39", lw=2.2, label="Oi (sim)")
    if "Oi" in frac_obs:
        om = ~np.isnan(frac_obs["Oi"])
        ax.scatter(time_years[om], frac_obs["Oi"][om],
                   color="#e07b39", s=60, zorder=5, edgecolors="k", lw=0.7)
    # Oe/a: combined Oe_a_L+Oe_a_H, brown
    oea_ts = combined_d14c_ts["Oe/a"]
    ax.plot(time_years[mask_plot], oea_ts[mask_plot],
            color="#7c3f00", lw=2.2, label="Oe/a (sim, combined)")
    if "Oe/a" in frac_obs:
        om = ~np.isnan(frac_obs["Oe/a"])
        ax.scatter(time_years[om], frac_obs["Oe/a"][om],
                   color="#7c3f00", s=60, zorder=5, marker="^",
                   edgecolors="k", lw=0.7)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set(xlabel="Year", ylabel="Δ¹⁴C (‰)",
           title="(a) Organic layer [cf. Sierra Fig. 3]")
    ax.legend(fontsize=7); ax.set_xlim(1950, 2011)

    # ── Panel (b): Mineral layer — sub-fractions + combined A-LF ─────────────
    ax = axes[1]
    ax.plot(atm_years[mask_atm], atm_d14C[mask_atm],
            color="gray", lw=1, ls="--", label="Atmosphere", zorder=0)
    sub_colors_min = {"mineral_ALF_gt80": "#a8d8ea", "mineral_ALF_lt80": "#2c7fb8",
                      "mineral_MinAss": "#1a4060"}
    for pn, col in sub_colors_min.items():
        pi = idx[pn]
        lbl = pn.replace("mineral_", "")
        ax.plot(time_years[mask_plot], d14c_sim[mask_plot, pi],
                color=col, lw=0.9, ls=":", alpha=0.7, label=f"{lbl} (sub)")
    # A-LF: combined ALF_gt80+ALF_lt80, blue
    alf_ts = combined_d14c_ts["A-LF"]
    ax.plot(time_years[mask_plot], alf_ts[mask_plot],
            color="#2c7fb8", lw=2.2, label="A-LF (sim, combined)")
    if "A-LF" in frac_obs:
        om = ~np.isnan(frac_obs["A-LF"])
        ax.scatter(time_years[om], frac_obs["A-LF"][om],
                   color="#2c7fb8", s=60, zorder=5, edgecolors="k", lw=0.7)
    # MinAss: single pool, dark blue
    min_ts = combined_d14c_ts["A-min"]
    ax.plot(time_years[mask_plot], min_ts[mask_plot],
            color="#1a4060", lw=2.2, label="MinAss (sim)")
    if "A-min" in frac_obs:
        om = ~np.isnan(frac_obs["A-min"])
        ax.scatter(time_years[om], frac_obs["A-min"][om],
                   color="#1a4060", s=60, zorder=5, marker="^",
                   edgecolors="k", lw=0.7)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set(xlabel="Year", ylabel="Δ¹⁴C (‰)",
           title="(b) Mineral layer [cf. Sierra Fig. 3]")
    ax.legend(fontsize=7); ax.set_xlim(1950, 2011)

    # ── Panel (c): Respired CO₂ [cf. Sierra Fig. 2] ───────────────────────────
    ax = axes[2]
    ax.plot(atm_years[mask_atm], atm_d14C[mask_atm],
            color="gray", lw=1, ls="--", label="Atmosphere", zorder=0)
    ax.plot(time_years[mask_plot], d14c_resp_sim[mask_plot],
            color="k", lw=1.5, label="Simulated resp. Δ¹⁴C")
    obs_mask = ~np.isnan(resp_obs)
    ax.scatter(time_years[obs_mask], resp_obs[obs_mask],
               color="crimson", s=40, zorder=5, label="hf212-01 NWN obs",
               edgecolors="k", lw=0.5)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set(xlabel="Year", ylabel="Δ¹⁴C (‰)",
           title="(c) Respired CO₂ [cf. Sierra Fig. 2]")
    ax.legend(fontsize=8); ax.set_xlim(1950, 2011)

    fig.tight_layout()
    out_path = os.path.join(_SCRIPT_ROOT, "replicate_hf.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
