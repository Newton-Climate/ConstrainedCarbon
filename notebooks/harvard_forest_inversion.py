"""
harvard_forest_inversion.py — Optimize Harvard Forest soil-only model τ values
against pool-level Δ¹⁴C observations from hf212-03-14c-org.csv.

Run from the repository root:
    python notebooks/harvard_forest_inversion.py

Observations used
-----------------
  hf212-03-14c-org.csv : 4 soil pools × 2 years (1996, 2007) = 8 Δ¹⁴C values
    Oi   → organic_litter
    Oe   → organic_fast
    A-lf → mineral_A_fast
    A-min→ mineral_A_slow

Parameters optimized
--------------------
  log_tau                      (6 pools)  — turnover times
  log_external_input_partition (4 logits) — carbon input partition

All other parameters (Q10, moisture, transfer fractions) are held fixed at
their analytically-derived prior values (from hf271 C-stock constraint).

Output
------
  notebooks/harvard_forest_inversion.png  — 3-panel figure:
    (a) Loss convergence
    (b) τ before vs. after (bar chart)
    (c) Δ¹⁴C before vs. after vs. obs
"""
from __future__ import annotations

import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yaml

# ── Make src importable when run as a script ────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from ecosystem_complexity.api import (
    build_model, run_model, optimize,
    _get_opt_fields, _params_to_vector, _vector_to_params,
)
from ecosystem_complexity.config import PoolIndex, load_config
from ecosystem_complexity.data.parsers import attach_atm14C, load_harvard_forest
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.schemas import ForcingData, ObservationData
from ecosystem_complexity.state import make_default_params

# ── File paths (relative to repo root) ──────────────────────────────────────
HF_HR_PATH = (
    "data/harvard_forest/AMF_US-Ha1_FLUXNET_FULLSET_1991-2020_3-5/"
    "AMF_US-Ha1_FLUXNET_FULLSET_HR_1991-2020_3-5.csv"
)
HF_SOIL_CONFIG  = "configs/harvard_forest_soil_only.yaml"
HF_SOIL_C_PATH  = "data/harvard_forest/hf271-07-soils.csv"
HF_SOIL_14C_PATH = "data/harvard_forest/hf212-03-14c-org.csv"
HUA_PATH    = "data/shared/atm_14C/Hua_2021.csv"
GRAVEN_PATH = "data/shared/atm_14C/Graven_2017.csv"
INTCAL_PATH = "data/shared/atm_14C/intcal20.14c"

_R_STD = 1.176e-12   # IAEA modern standard (same as tracer_14C.py)

# Horizon → pool name mapping (hf212-03 → harvard_forest_soil_only.yaml)
_HORIZON_TO_POOL = {
    "Oi":    "organic_litter",
    "Oe":    "organic_fast",
    "A-lf":  "mineral_A_fast",
    "A-min": "mineral_A_slow",
}

# Pool display styles (colour, marker) for Δ¹⁴C panel
_POOL_STYLES = {
    "organic_litter":  ("Org litter (Oi)",   "tab:green",  "o"),
    "organic_fast":    ("Org fast (Oe)",      "tab:olive",  "s"),
    "mineral_A_fast":  ("Min-A fast (A-lf)",  "tab:orange", "^"),
    "mineral_A_slow":  ("Min-A slow (A-min)", "tab:brown",  "D"),
}


# ════════════════════════════════════════════════════════════════════════════
# Data helpers
# ════════════════════════════════════════════════════════════════════════════

def _slice_forcing(forcing, start, end):
    return ForcingData(
        time=forcing.time[start:end],
        air_temp=forcing.air_temp[start:end],
        sw_radiation=forcing.sw_radiation[start:end],
        precip=forcing.precip[start:end],
        vpd=forcing.vpd[start:end],
        soil_temp=forcing.soil_temp[start:end],
        soil_moisture=forcing.soil_moisture[start:end],
        snow_depth=forcing.snow_depth[start:end],
        active_layer=forcing.active_layer[start:end],
        delta14C_atm=forcing.delta14C_atm[start:end],
        GPP_obs=forcing.GPP_obs[start:end],
        NPP_obs=forcing.NPP_obs[start:end],
    )


def build_hf_soil_14C_obs(
    soil_14c_path: str,
    forcing_time: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """
    Build ``ObservationData.delta14C_obs`` from hf212-03-14c-org.csv.

    Returns ``{pool_name: array(T,)}`` where each array is NaN everywhere
    except at the timestep(s) nearest to the observation year (1996, 2007).
    The loss function masks NaN entries so sparse obs are handled correctly.
    """
    d14c_df = pd.read_csv(soil_14c_path)
    d14c_df["horizon"] = d14c_df["horizon"].str.strip()

    time_np = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T = len(time_np)

    obs_dict: dict[str, np.ndarray] = {}

    for _, row in d14c_df.iterrows():
        horizon = row["horizon"]
        if horizon not in _HORIZON_TO_POOL:
            continue
        pool_name = _HORIZON_TO_POOL[horizon]
        obs_year  = float(row["year"]) + 0.5   # mid-year (July)
        d14c_val  = float(row["of.14c.12c"])

        # Find the timestep closest to this observation year
        t_idx = int(np.argmin(np.abs(years_np - obs_year)))

        if pool_name not in obs_dict:
            obs_dict[pool_name] = np.full(T, np.nan, dtype=np.float32)
        obs_dict[pool_name][t_idx] = d14c_val

    return {k: jnp.array(v) for k, v in obs_dict.items()}


def build_obs_state0(config, pool_index, site_cfg):
    """Build EcosystemState initialised from 1996 hf271 C stocks + hf212-03 Δ¹⁴C."""
    from ecosystem_complexity.state import make_initial_state

    soils = pd.read_csv(HF_SOIL_C_PATH)
    org_mean = soils.loc[soils["horizon"] == "O", "carbon.gm2"].dropna().mean()
    min_mean = soils.loc[soils["horizon"] == "M", "carbon.gm2"].dropna().mean()
    min_B_mean = 1.5 * min_mean

    C12_by_pool = {
        "organic_litter":    0.40 * org_mean,
        "organic_fast":      0.60 * org_mean,
        "mineral_A_fast":    0.30 * min_mean,
        "mineral_A_slow":    0.70 * min_mean,
        "mineral_B_slow":    0.20 * min_B_mean,
        "mineral_B_passive": 0.80 * min_B_mean,
    }

    d14c_df = pd.read_csv(HF_SOIL_14C_PATH)
    d14c_df["horizon"] = d14c_df["horizon"].str.strip()
    d14c_1996 = d14c_df[d14c_df["year"] == 1996]

    d14C_by_pool = {
        "organic_litter":    float(d14c_1996.loc[d14c_1996["horizon"] == "Oi",   "of.14c.12c"].iloc[0]),
        "organic_fast":      float(d14c_1996.loc[d14c_1996["horizon"] == "Oe",   "of.14c.12c"].iloc[0]),
        "mineral_A_fast":    float(d14c_1996.loc[d14c_1996["horizon"] == "A-lf", "of.14c.12c"].iloc[0]),
        "mineral_A_slow":    float(d14c_1996.loc[d14c_1996["horizon"] == "A-min","of.14c.12c"].iloc[0]),
        "mineral_B_slow":    -50.0,
        "mineral_B_passive": -300.0,
    }

    base    = make_initial_state(config, site_cfg)
    n_pools = len(pool_index)
    pool_names_set = set(pool_index.pool_names)

    C12_arr = np.zeros(n_pools, dtype=np.float32)
    C14_arr = np.zeros(n_pools, dtype=np.float32)
    for pool_name, c12_val in C12_by_pool.items():
        if pool_name not in pool_names_set:
            continue
        i = pool_index[pool_name]
        d14c_val   = d14C_by_pool.get(pool_name, 0.0)
        C12_arr[i] = max(c12_val, 0.0)
        C14_arr[i] = C12_arr[i] * _R_STD * (1.0 + d14c_val / 1000.0)

    return base._replace(
        C12=jnp.array(C12_arr),
        C14=jnp.array(C14_arr),
    )


# ════════════════════════════════════════════════════════════════════════════
# Main inversion workflow
# ════════════════════════════════════════════════════════════════════════════

def run_inversion():
    os.chdir(_REPO_ROOT)

    # ── Load atmospheric ¹⁴C record ──────────────────────────────────────────
    print("Loading atmospheric ¹⁴C record…")
    atm14C = load_full_14C_record(
        hua_path=HUA_PATH, graven_path=GRAVEN_PATH, intcal_path=INTCAL_PATH,
        hemisphere="NH", start_year=1500.0, end_year=2025.0,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    print("Building model…")
    model  = build_model(HF_SOIL_CONFIG)
    config = model.config
    idx    = model.pool_index

    with open(HF_SOIL_CONFIG) as fh:
        site_cfg = yaml.safe_load(fh).get("site", {})

    # ── Load forcing (1996+, GPP_obs filled) ─────────────────────────────────
    print("Loading flux forcing…")
    forcing_full, _ = load_harvard_forest(
        hr_path=HF_HR_PATH, config=config, qc_threshold=2,
        include_gpp_forcing=True,
    )
    years_daily, d14c_daily = atm14C
    forcing_full = attach_atm14C(forcing_full, d14c_daily, years_daily)

    time_np   = np.array(forcing_full.time)
    years_all = 1970.0 + time_np / 365.25
    start_idx = int(np.searchsorted(years_all, 1996.0))
    forcing   = _slice_forcing(forcing_full, start_idx, len(time_np))
    T         = int(forcing.time.shape[0])
    time_years = 1970.0 + np.array(forcing.time) / 365.25
    print(f"  Forcing: {T} days  ({time_years[0]:.1f}–{time_years[-1]:.1f})")

    # ── Build observations (8 Δ¹⁴C points) ───────────────────────────────────
    print("Building observations…")
    delta14C_obs = build_hf_soil_14C_obs(HF_SOIL_14C_PATH, forcing.time)
    print(f"  Pools with obs: {sorted(delta14C_obs.keys())}")
    for pool_name, arr in delta14C_obs.items():
        n_valid = int(jnp.sum(~jnp.isnan(arr)))
        print(f"    {pool_name:<25s}  {n_valid} observations")

    obs = ObservationData(
        time=forcing.time,
        NEE=jnp.full(T, jnp.nan),
        GPP=jnp.full(T, jnp.nan),
        ER=jnp.full(T, jnp.nan),
        NEE_unc=jnp.full(T, jnp.nan),
        delta14C_obs=delta14C_obs,
        deltaD14C_obs={},
        C_pools_obs={},
    )

    # ── Build initial state from 1996 observations ────────────────────────────
    state0 = build_obs_state0(config, idx, site_cfg)
    print(f"  State0 total C12: {float(jnp.sum(state0.C12)):.0f} gC m⁻²")

    # ── Forward run with PRIOR params ────────────────────────────────────────
    print("\nRunning prior forward simulation…")
    t0 = time.perf_counter()
    params_prior = make_default_params(config)
    out_prior = run_model(model, forcing, state0=state0, params=params_prior)
    jax.block_until_ready(out_prior.delta14C)
    print(f"  Done  [{time.perf_counter()-t0:.1f}s]")

    # ── Optimization ─────────────────────────────────────────────────────────
    # Restrict to 10 parameters (log_tau × 6 + log_external_input_partition × 4)
    # rather than the full 30-parameter vector, keeping the inversion tractable
    # against only 8 Δ¹⁴C observations.
    _opt_fields = ("log_tau", "log_external_input_partition")
    n_opt_params = sum(
        int(np.prod(getattr(make_default_params(config), f).shape))
        for f in _opt_fields
    )
    print(f"\nRunning optimization (Adam, 800 iterations)…")
    print(f"  Optimizing: {_opt_fields}  ({n_opt_params} params vs 8 obs)")
    t0 = time.perf_counter()
    result = optimize(model, forcing, obs, state0=state0, fields=_opt_fields)
    print(f"  Done  [{time.perf_counter()-t0:.1f}s]")
    print(f"  Converged: {result.converged}  after {result.n_iter} iterations")
    loss_hist = np.array(result.loss_history)
    valid_loss = loss_hist[np.isfinite(loss_hist)]
    if len(valid_loss) > 0:
        print(f"  Initial loss: {float(valid_loss[0]):.4f}")
        print(f"  Final loss:   {float(valid_loss[-1]):.4f}")
        if valid_loss[-1] > 0:
            print(f"  Reduction:    {float(valid_loss[0]/valid_loss[-1]):.1f}×")
    else:
        print("  No finite loss values recorded")

    # ── Forward run with OPTIMIZED params ────────────────────────────────────
    print("\nRunning optimized forward simulation…")
    out_opt = run_model(model, forcing, state0=state0, params=result.params_opt)
    jax.block_until_ready(out_opt.delta14C)

    # ── Print τ comparison ────────────────────────────────────────────────────
    tau_prior = np.exp(np.array(params_prior.log_tau))
    tau_opt   = np.exp(np.array(result.params_opt.log_tau))
    print("\nTurnover times:")
    print(f"  {'Pool':<25s}  {'τ prior (yr)':>12}  {'τ opt (yr)':>10}  {'ratio':>7}")
    print("  " + "─" * 60)
    for i, name in enumerate(idx.pool_names):
        print(f"  {name:<25s}  {tau_prior[i]/365:>12.1f}  {tau_opt[i]/365:>10.1f}  {tau_opt[i]/tau_prior[i]:>7.2f}×")

    # ── Print partition comparison ────────────────────────────────────────────
    import scipy.special
    def _softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    lp_prior = np.array(params_prior.log_external_input_partition)
    lp_opt   = np.array(result.params_opt.log_external_input_partition)
    part_prior = _softmax(lp_prior)
    part_opt   = _softmax(lp_opt)

    # Pool order from config
    from ecosystem_complexity.config import load_config as _lc
    _cfg_raw = _lc(HF_SOIL_CONFIG)
    _ext = _cfg_raw.external_inputs
    part_names = list(_ext.partition.keys()) if _ext is not None else []

    print("\nInput partition (softmax):")
    print(f"  {'Pool':<25s}  {'prior':>8}  {'opt':>8}  {'Δ':>8}")
    print("  " + "─" * 54)
    for i, pname in enumerate(part_names):
        print(f"  {pname:<25s}  {part_prior[i]:>8.3f}  {part_opt[i]:>8.3f}  {part_opt[i]-part_prior[i]:>+8.3f}")

    # ── Print model Δ¹⁴C at obs timesteps vs. observations ───────────────────
    time_np_local = 1970.0 + np.array(forcing.time) / 365.25
    print("\nModel vs. Obs Δ¹⁴C at observation timesteps:")
    print(f"  {'Pool':<22s}  {'Year':>6}  {'Obs':>8}  {'Prior':>8}  {'Opt':>8}  {'Δ(opt-obs)':>10}")
    print("  " + "─" * 68)
    for pool_name in sorted(delta14C_obs.keys()):
        pool_idx_i = idx[pool_name]
        obs_arr = np.array(delta14C_obs[pool_name])
        valid_t = np.where(np.isfinite(obs_arr))[0]
        sim_prior = np.array(out_prior.delta14C)
        sim_opt   = np.array(out_opt.delta14C)
        for t_i in valid_t:
            obs_val    = obs_arr[t_i]
            prior_val  = sim_prior[t_i, pool_idx_i]
            opt_val    = sim_opt[t_i, pool_idx_i]
            yr         = time_np_local[t_i]
            print(f"  {pool_name:<22s}  {yr:>6.1f}  {obs_val:>8.1f}  {prior_val:>8.1f}  {opt_val:>8.1f}  {opt_val-obs_val:>+10.1f}")

    return (time_years, out_prior, out_opt, result,
            delta14C_obs, params_prior, result.params_opt, idx)


# ════════════════════════════════════════════════════════════════════════════
# Figure
# ════════════════════════════════════════════════════════════════════════════

def make_inversion_figure(time_years, out_prior, out_opt, result,
                          delta14C_obs, params_prior, params_opt, pool_idx):
    """3-panel figure: (a) loss, (b) τ bar chart, (c) Δ¹⁴C before/after/obs."""

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.38, wspace=0.28,
        left=0.07, right=0.97, top=0.92, bottom=0.08,
    )
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_tau  = fig.add_subplot(gs[0, 1])
    ax_14c  = fig.add_subplot(gs[1, :])   # full-width bottom panel

    # ── Panel (a): Loss convergence ───────────────────────────────────────────
    iters = np.arange(1, len(result.loss_history) + 1)
    ax_loss.semilogy(iters, np.array(result.loss_history),
                     lw=1.5, color="black", label="total loss")
    ax_loss.semilogy(iters, np.array(result.loss_14C_history),
                     lw=1.0, color="steelblue", linestyle="--", label="Δ¹⁴C loss")
    if result.converged:
        ax_loss.axvline(result.n_iter, lw=0.8, color="gray", linestyle=":",
                        alpha=0.7, label=f"converged @ iter {result.n_iter}")
    ax_loss.set_xlabel("Iteration", fontsize=9)
    ax_loss.set_ylabel("Loss (MSE, ‰²)", fontsize=9)
    ax_loss.set_title("(a) Optimization convergence", fontsize=9, loc="left")
    ax_loss.legend(fontsize=8, framealpha=0.8)
    ax_loss.tick_params(labelsize=8)
    ax_loss.grid(axis="both", lw=0.4, alpha=0.4)

    # Annotate reduction
    _lh = np.array(result.loss_history)
    _finite = _lh[np.isfinite(_lh)]
    if len(_finite) >= 2:
        loss_0 = float(_finite[0])
        loss_f = float(_finite[-1])
        reduction_str = f"{loss_0/loss_f:.1f}×" if loss_f > 0 else "N/A"
        ax_loss.text(0.98, 0.97,
                     f"Initial: {loss_0:.2f}\nFinal:   {loss_f:.2f}\n"
                     f"Reduction: {reduction_str}",
                     transform=ax_loss.transAxes, va="top", ha="right",
                     fontsize=8, color="0.3",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

    # ── Panel (b): τ comparison bar chart ─────────────────────────────────────
    tau_prior = np.exp(np.array(params_prior.log_tau)) / 365.0   # → years
    tau_opt   = np.exp(np.array(params_opt.log_tau)) / 365.0
    pool_names = list(pool_idx.pool_names)
    n_pools = len(pool_names)
    x = np.arange(n_pools)
    w = 0.35

    bars_p = ax_tau.bar(x - w/2, tau_prior, w, label="prior τ", color="steelblue", alpha=0.7)
    bars_o = ax_tau.bar(x + w/2, tau_opt,   w, label="optimized τ", color="tomato", alpha=0.7)

    # Annotate ratio on each pair
    for i in range(n_pools):
        ratio = tau_opt[i] / tau_prior[i]
        if abs(ratio - 1.0) > 0.05:
            ax_tau.text(x[i], max(tau_prior[i], tau_opt[i]) * 1.04,
                        f"{ratio:.2f}×", ha="center", fontsize=7,
                        color="tomato" if ratio > 1.0 else "steelblue")

    short_names = [n.replace("_", "\n") for n in pool_names]
    ax_tau.set_xticks(x)
    ax_tau.set_xticklabels(short_names, fontsize=7)
    ax_tau.set_ylabel("Turnover time (years)", fontsize=9)
    ax_tau.set_title("(b) Turnover times: prior vs. optimized", fontsize=9, loc="left")
    ax_tau.legend(fontsize=8, framealpha=0.8)
    ax_tau.tick_params(axis="y", labelsize=8)
    ax_tau.grid(axis="y", lw=0.4, alpha=0.4)

    # ── Panel (c): Δ¹⁴C trajectories ─────────────────────────────────────────
    d14C_prior = np.array(out_prior.delta14C)   # (T, n_pools)
    d14C_opt   = np.array(out_opt.delta14C)

    for pool_name, (label, color, marker) in _POOL_STYLES.items():
        if pool_name not in set(pool_idx.pool_names):
            continue
        i = pool_idx[pool_name]

        # Prior (gray, dashed)
        ax_14c.plot(time_years, d14C_prior[:, i],
                    lw=1.0, color="gray", linestyle="--", alpha=0.6)
        # Optimized (colour, solid)
        ax_14c.plot(time_years, d14C_opt[:, i],
                    lw=1.5, color=color, alpha=0.9, label=f"{label}")

        # Obs scatter (1996 and 2007)
        if pool_name in delta14C_obs:
            obs_arr = np.array(delta14C_obs[pool_name])
            valid   = np.where(np.isfinite(obs_arr))[0]
            for t_idx in valid:
                ax_14c.scatter(
                    time_years[t_idx], obs_arr[t_idx],
                    s=80, color=color, marker=marker,
                    edgecolors="black", linewidths=0.8, zorder=6,
                )

    # Legend entries for prior/opt line style
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="gray", lw=1.0, linestyle="--", label="Prior model"),
        Line2D([0], [0], color="black", lw=1.5, label="Optimized model"),
    ] + [
        plt.scatter([], [], s=60, color=_POOL_STYLES[p][1],
                    marker=_POOL_STYLES[p][2], edgecolors="black", linewidths=0.6,
                    label=f"{_POOL_STYLES[p][0]} obs")
        for p in _POOL_STYLES if p in delta14C_obs
    ]
    # Add pool line handles from ax
    for pool_name, (label, color, marker) in _POOL_STYLES.items():
        if pool_name not in set(pool_idx.pool_names):
            continue
        legend_handles.append(
            Line2D([0], [0], color=color, lw=1.5, label=label)
        )

    ax_14c.axhline(0, lw=0.5, color="gray", linestyle=":")
    ax_14c.set_ylabel("Δ¹⁴C (‰)", fontsize=9)
    ax_14c.set_xlabel("Year", fontsize=9)
    ax_14c.set_title(
        "(c) Δ¹⁴C trajectories — prior (gray dashed) vs. optimized (coloured)  "
        "·  obs scatter = hf212-03 (1996, 2007)",
        fontsize=9, loc="left"
    )
    ax_14c.tick_params(labelsize=8)
    ax_14c.grid(axis="y", lw=0.4, alpha=0.4)
    ax_14c.legend(handles=legend_handles, fontsize=7.5, ncol=4,
                  framealpha=0.85, loc="upper right")

    fig.suptitle(
        "Harvard Forest — Soil-Only Δ¹⁴C Inversion\n"
        "Optimizing τ against 8 pool-level observations (hf212-03: 4 pools × 1996 + 2007)",
        fontsize=10,
    )

    out_path = os.path.join("notebooks", "harvard_forest_inversion.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    try:
        plt.show()
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import yaml as _yaml

    results = run_inversion()
    make_inversion_figure(*results)
