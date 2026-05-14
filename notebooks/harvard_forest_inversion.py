"""
harvard_forest_inversion.py — Optimize Harvard Forest soil-only model τ values
against pool-level Δ¹⁴C (hf212-03) and respired CO₂ Δ¹⁴C (hf212-01) observations.

Run from the repository root:
    python notebooks/harvard_forest_inversion.py

Observations used
-----------------
  hf212-03-14c-org.csv  : 4 soil pools × 2 years (1996, 2007) = 8 Δ¹⁴C values
    Oi    → organic_litter
    Oe    → organic_fast
    A-lf  → mineral_A_fast
    A-min → mineral_A_slow

  hf212-01-14c-no-treat.csv  : NWN site, 1996–2010, 41 sampling dates
    Multiple reps per date → daily mean used.
    Compared against flux-weighted model respired CO₂ Δ¹⁴C:
      d14C_resp = Σ(C12_i/τ_i × Δ¹⁴C_i) / Σ(C12_i/τ_i)

Parameters optimized
--------------------
  log_tau                      (6 pools)  — turnover times
  log_external_input_partition (4 logits) — carbon input partition

All other parameters (Q10, moisture, transfer fractions) held fixed at their
analytically-derived prior values (from hf271 C-stock constraint).

Output
------
  notebooks/harvard_forest_inversion.png  — 4-panel figure:
    (a) Loss convergence (total + pool Δ¹⁴C + respired CO₂ components)
    (b) τ before vs. after (bar chart with ratio annotation)
    (c) Pool Δ¹⁴C trajectories — prior / optimized / hf212-03 scatter
    (d) Respired CO₂ Δ¹⁴C — model vs. hf212-01 NWN observations
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
)
from ecosystem_complexity.config import load_config
from ecosystem_complexity.data.parsers import attach_atm14C, load_harvard_forest
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.schemas import ForcingData, ObservationData
from ecosystem_complexity.state import make_default_params

# ── File paths (relative to repo root) ──────────────────────────────────────
HF_HR_PATH = (
    "data/harvard_forest/AMF_US-Ha1_FLUXNET_FULLSET_1991-2020_3-5/"
    "AMF_US-Ha1_FLUXNET_FULLSET_HR_1991-2020_3-5.csv"
)
HF_SOIL_CONFIG   = "configs/harvard_forest_soil_only.yaml"
HF_SOIL_C_PATH   = "data/harvard_forest/hf271-07-soils.csv"
HF_SOIL_14C_PATH = "data/harvard_forest/hf212-03-14c-org.csv"
HF_RESP_14C_PATH = "data/harvard_forest/hf212-01-14c-no-treat.csv"
HUA_PATH     = "data/shared/atm_14C/Hua_2021.csv"
GRAVEN_PATH  = "data/shared/atm_14C/Graven_2017.csv"
INTCAL_PATH  = "data/shared/atm_14C/intcal20.14c"

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
    """
    d14c_df = pd.read_csv(soil_14c_path)
    d14c_df["horizon"] = d14c_df["horizon"].str.strip()

    time_np  = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T        = len(time_np)

    obs_dict: dict[str, np.ndarray] = {}

    for _, row in d14c_df.iterrows():
        horizon = row["horizon"]
        if horizon not in _HORIZON_TO_POOL:
            continue
        pool_name = _HORIZON_TO_POOL[horizon]
        obs_year  = float(row["year"]) + 0.5   # mid-year (July)
        d14c_val  = float(row["of.14c.12c"])

        t_idx = int(np.argmin(np.abs(years_np - obs_year)))

        if pool_name not in obs_dict:
            obs_dict[pool_name] = np.full(T, np.nan, dtype=np.float32)
        obs_dict[pool_name][t_idx] = d14c_val

    return {k: jnp.array(v) for k, v in obs_dict.items()}


def build_hf_resp_14C_obs(
    resp_14c_path: str,
    forcing_time: jnp.ndarray,
) -> jnp.ndarray:
    """
    Build ``ObservationData.delta14C_resp`` from hf212-01-14c-no-treat.csv.

    Uses NWN site only.  Multiple reps per date are averaged to a daily mean.
    Returns a (T,) array with NaN at all unmeasured timesteps; non-NaN at the
    41 sampling dates between 1996 and 2010.
    """
    df  = pd.read_csv(resp_14c_path)
    nwn = df[df["site"] == "NWN"]

    # Aggregate replicate measurements to a daily mean per fractional year.
    by_date = nwn.groupby("year.date")["delta.14c"].mean()

    time_np  = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T        = len(time_np)

    arr = np.full(T, np.nan, dtype=np.float32)
    for obs_year, d14c_val in by_date.items():
        t_idx = int(np.argmin(np.abs(years_np - float(obs_year))))
        arr[t_idx] = float(d14c_val)

    return jnp.array(arr)


def build_obs_state0(config, pool_index, site_cfg):
    """Build EcosystemState initialised from 1996 hf271 C stocks + hf212-03 Δ¹⁴C."""
    from ecosystem_complexity.state import make_initial_state

    soils    = pd.read_csv(HF_SOIL_C_PATH)
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

    d14c_df  = pd.read_csv(HF_SOIL_14C_PATH)
    d14c_df["horizon"] = d14c_df["horizon"].str.strip()
    d14c_1996 = d14c_df[d14c_df["year"] == 1996]

    d14C_by_pool = {
        "organic_litter":  float(d14c_1996.loc[d14c_1996["horizon"] == "Oi",    "of.14c.12c"].iloc[0]),
        "organic_fast":    float(d14c_1996.loc[d14c_1996["horizon"] == "Oe",    "of.14c.12c"].iloc[0]),
        "mineral_A_fast":  float(d14c_1996.loc[d14c_1996["horizon"] == "A-lf",  "of.14c.12c"].iloc[0]),
        "mineral_A_slow":  float(d14c_1996.loc[d14c_1996["horizon"] == "A-min", "of.14c.12c"].iloc[0]),
        "mineral_B_slow":    -50.0,
        "mineral_B_passive": -300.0,
    }

    base        = make_initial_state(config, site_cfg)
    n_pools     = len(pool_index)
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

    return base._replace(C12=jnp.array(C12_arr), C14=jnp.array(C14_arr))


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

    # ── Build observations ────────────────────────────────────────────────────
    print("Building observations…")

    # (1) Pool-level Δ¹⁴C — 8 points (4 pools × 1996 + 2007)
    delta14C_obs = build_hf_soil_14C_obs(HF_SOIL_14C_PATH, forcing.time)
    print(f"  Pool Δ¹⁴C obs (hf212-03):")
    for pool_name, arr in sorted(delta14C_obs.items()):
        n_valid = int(jnp.sum(~jnp.isnan(arr)))
        print(f"    {pool_name:<25s}  {n_valid} obs")

    # (2) Respired CO₂ Δ¹⁴C — 41 sampling dates, NWN site
    delta14C_resp = build_hf_resp_14C_obs(HF_RESP_14C_PATH, forcing.time)
    n_resp_obs = int(jnp.sum(~jnp.isnan(delta14C_resp)))
    print(f"  Respired CO₂ Δ¹⁴C obs (hf212-01 NWN): {n_resp_obs} dates")

    obs = ObservationData(
        time=forcing.time,
        NEE=jnp.full(T, jnp.nan),
        GPP=jnp.full(T, jnp.nan),
        ER=jnp.full(T, jnp.nan),
        NEE_unc=jnp.full(T, jnp.nan),
        delta14C_obs=delta14C_obs,
        deltaD14C_obs={},
        C_pools_obs={},
        delta14C_resp=delta14C_resp,   # new field — 41 NWN dates
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

    # Prior respired Δ¹⁴C — flux-weighted mean
    tau_p   = np.exp(np.array(params_prior.log_tau))
    w_prior = np.array(out_prior.C12) / (tau_p[None, :] + 1e-30)
    d14C_resp_prior = (np.array(out_prior.delta14C) * w_prior).sum(-1) / (w_prior.sum(-1) + 1e-30)

    # ── Optimization ─────────────────────────────────────────────────────────
    # 10 parameters (log_tau × 6 + log_external_input_partition × 4)
    # vs 8 pool obs + 41 respired CO₂ obs = 49 total observations
    _opt_fields = ("log_tau", "log_external_input_partition")
    n_opt_params = sum(
        int(np.prod(getattr(make_default_params(config), f).shape))
        for f in _opt_fields
    )
    print(f"\nRunning optimization (Adam, 800 iterations)…")
    print(f"  Fields: {_opt_fields}  ({n_opt_params} params)")
    print(f"  Obs: {int(jnp.sum(~jnp.isnan(jnp.concatenate([a for a in delta14C_obs.values()]))))} pool-Δ¹⁴C"
          f" + {n_resp_obs} resp-Δ¹⁴C = "
          f"{int(jnp.sum(~jnp.isnan(jnp.concatenate([a for a in delta14C_obs.values()]))))+n_resp_obs} total")
    t0 = time.perf_counter()
    result = optimize(model, forcing, obs, state0=state0, fields=_opt_fields)
    print(f"  Done  [{time.perf_counter()-t0:.1f}s]")
    print(f"  Converged: {result.converged}  after {result.n_iter} iterations")
    loss_hist = np.array(result.loss_history)
    valid_loss = loss_hist[np.isfinite(loss_hist)]
    if len(valid_loss) > 0:
        print(f"  Initial loss: {float(valid_loss[0]):.5f}")
        print(f"  Final loss:   {float(valid_loss[-1]):.5f}")
        if valid_loss[-1] > 0:
            print(f"  Reduction:    {float(valid_loss[0]/valid_loss[-1]):.1f}×")

    # ── Forward run with OPTIMIZED params ────────────────────────────────────
    print("\nRunning optimized forward simulation…")
    out_opt = run_model(model, forcing, state0=state0, params=result.params_opt)
    jax.block_until_ready(out_opt.delta14C)

    # Optimized respired Δ¹⁴C
    tau_o  = np.exp(np.array(result.params_opt.log_tau))
    w_opt  = np.array(out_opt.C12) / (tau_o[None, :] + 1e-30)
    d14C_resp_opt = (np.array(out_opt.delta14C) * w_opt).sum(-1) / (w_opt.sum(-1) + 1e-30)

    # ── Print τ comparison ────────────────────────────────────────────────────
    tau_prior_yr = tau_p / 365.0
    tau_opt_yr   = tau_o / 365.0
    print("\nTurnover times:")
    print(f"  {'Pool':<25s}  {'τ prior (yr)':>12}  {'τ opt (yr)':>10}  {'ratio':>7}")
    print("  " + "─" * 60)
    for i, name in enumerate(idx.pool_names):
        print(f"  {name:<25s}  {tau_prior_yr[i]:>12.1f}  {tau_opt_yr[i]:>10.1f}  {tau_opt_yr[i]/tau_prior_yr[i]:>7.2f}×")

    # ── Print partition comparison ────────────────────────────────────────────
    def _softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    lp_prior   = np.array(params_prior.log_external_input_partition)
    lp_opt     = np.array(result.params_opt.log_external_input_partition)
    part_prior = _softmax(lp_prior)
    part_opt   = _softmax(lp_opt)
    _ext       = load_config(HF_SOIL_CONFIG).external_inputs
    part_names = list(_ext.partition.keys()) if _ext is not None else []

    print("\nInput partition (softmax):")
    print(f"  {'Pool':<25s}  {'prior':>8}  {'opt':>8}  {'Δ':>8}")
    print("  " + "─" * 54)
    for i, pname in enumerate(part_names):
        print(f"  {pname:<25s}  {part_prior[i]:>8.3f}  {part_opt[i]:>8.3f}  {part_opt[i]-part_prior[i]:>+8.3f}")

    # ── Print model Δ¹⁴C at pool obs timesteps ───────────────────────────────
    print("\nModel vs. Obs — pool Δ¹⁴C at observation timesteps:")
    print(f"  {'Pool':<22s}  {'Year':>6}  {'Obs':>8}  {'Prior':>8}  {'Opt':>8}  {'Δ(opt-obs)':>10}")
    print("  " + "─" * 68)
    for pool_name in sorted(delta14C_obs.keys()):
        pool_i    = idx[pool_name]
        obs_arr   = np.array(delta14C_obs[pool_name])
        valid_t   = np.where(np.isfinite(obs_arr))[0]
        sim_prior = np.array(out_prior.delta14C)
        sim_opt   = np.array(out_opt.delta14C)
        for t_i in valid_t:
            yr = 1970.0 + float(np.array(forcing.time)[t_i]) / 365.25
            print(f"  {pool_name:<22s}  {yr:>6.1f}  {obs_arr[t_i]:>8.1f}"
                  f"  {sim_prior[t_i, pool_i]:>8.1f}  {sim_opt[t_i, pool_i]:>8.1f}"
                  f"  {sim_opt[t_i, pool_i]-obs_arr[t_i]:>+10.1f}")

    return (time_years, out_prior, out_opt, result,
            delta14C_obs, delta14C_resp,
            d14C_resp_prior, d14C_resp_opt,
            params_prior, result.params_opt, idx)


# ════════════════════════════════════════════════════════════════════════════
# Figure
# ════════════════════════════════════════════════════════════════════════════

def make_inversion_figure(time_years, out_prior, out_opt, result,
                          delta14C_obs, delta14C_resp,
                          d14C_resp_prior, d14C_resp_opt,
                          params_prior, params_opt, pool_idx):
    """4-panel figure: (a) loss, (b) τ bar chart, (c) pool Δ¹⁴C, (d) respired CO₂ Δ¹⁴C."""

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.38, wspace=0.28,
        left=0.07, right=0.97, top=0.91, bottom=0.07,
    )
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_tau  = fig.add_subplot(gs[0, 1])
    ax_14c  = fig.add_subplot(gs[1, 0])
    ax_resp = fig.add_subplot(gs[1, 1])

    iters = np.arange(1, result.n_iter + 1)

    # ── Panel (a): Loss convergence ───────────────────────────────────────────
    lh      = np.array(result.loss_history)
    l14c_h  = np.array(result.loss_14C_history)
    lresp_h = np.array(result.loss_resp_history)

    ax_loss.semilogy(iters, lh,      lw=1.5, color="black",     label="total")
    ax_loss.semilogy(iters, l14c_h,  lw=1.0, color="steelblue",
                     linestyle="--", label="pool Δ¹⁴C")
    ax_loss.semilogy(iters, lresp_h, lw=1.0, color="tomato",
                     linestyle=":",  label="resp CO₂ Δ¹⁴C")
    if result.converged:
        ax_loss.axvline(result.n_iter, lw=0.8, color="gray", linestyle=":",
                        alpha=0.7, label=f"converged @ {result.n_iter}")
    ax_loss.set_xlabel("Iteration", fontsize=9)
    ax_loss.set_ylabel("Loss (MSE, ‰²/timestep)", fontsize=9)
    ax_loss.set_title("(a) Optimization convergence", fontsize=9, loc="left")
    ax_loss.legend(fontsize=8, framealpha=0.8)
    ax_loss.tick_params(labelsize=8)
    ax_loss.grid(axis="both", lw=0.4, alpha=0.4)

    _finite = lh[np.isfinite(lh)]
    if len(_finite) >= 2:
        l0, lf = float(_finite[0]), float(_finite[-1])
        red = f"{l0/lf:.1f}×" if lf > 0 else "N/A"
        ax_loss.text(0.98, 0.97,
                     f"Initial: {l0:.4f}\nFinal:   {lf:.4f}\nReduction: {red}",
                     transform=ax_loss.transAxes, va="top", ha="right",
                     fontsize=8, color="0.3",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

    # ── Panel (b): τ comparison bar chart ─────────────────────────────────────
    tau_prior = np.exp(np.array(params_prior.log_tau)) / 365.0
    tau_opt   = np.exp(np.array(params_opt.log_tau))   / 365.0
    pool_names = list(pool_idx.pool_names)
    n_pools = len(pool_names)
    x = np.arange(n_pools)
    w = 0.35

    ax_tau.bar(x - w/2, tau_prior, w, label="prior τ",     color="steelblue", alpha=0.7)
    ax_tau.bar(x + w/2, tau_opt,   w, label="optimized τ", color="tomato",    alpha=0.7)

    for i in range(n_pools):
        ratio = tau_opt[i] / tau_prior[i]
        if abs(ratio - 1.0) > 0.02:
            ax_tau.text(x[i], max(tau_prior[i], tau_opt[i]) * 1.05,
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

    # ── Panel (c): Pool Δ¹⁴C trajectories ────────────────────────────────────
    d14C_prior = np.array(out_prior.delta14C)
    d14C_opt   = np.array(out_opt.delta14C)

    pool_names_set = set(pool_idx.pool_names)
    for pool_name, (label, color, marker) in _POOL_STYLES.items():
        if pool_name not in pool_names_set:
            continue
        i = pool_idx[pool_name]
        ax_14c.plot(time_years, d14C_prior[:, i],
                    lw=0.8, color="gray", linestyle="--", alpha=0.5)
        ax_14c.plot(time_years, d14C_opt[:, i],
                    lw=1.5, color=color, alpha=0.9, label=label)
        if pool_name in delta14C_obs:
            obs_arr = np.array(delta14C_obs[pool_name])
            valid   = np.where(np.isfinite(obs_arr))[0]
            for t_i in valid:
                ax_14c.scatter(time_years[t_i], obs_arr[t_i],
                               s=70, color=color, marker=marker,
                               edgecolors="black", linewidths=0.8, zorder=6)

    from matplotlib.lines import Line2D
    handles_c = [
        Line2D([0], [0], color="gray", lw=0.8, linestyle="--", alpha=0.5, label="Prior"),
        Line2D([0], [0], color="black", lw=1.5,                              label="Optimized"),
    ] + [
        Line2D([0], [0], color=_POOL_STYLES[p][1], lw=1.5,
               marker=_POOL_STYLES[p][2], ms=6,
               label=f"{_POOL_STYLES[p][0]}")
        for p in _POOL_STYLES if p in pool_names_set
    ]
    ax_14c.axhline(0, lw=0.5, color="gray", linestyle=":")
    ax_14c.set_ylabel("Δ¹⁴C (‰)", fontsize=9)
    ax_14c.set_xlabel("Year", fontsize=9)
    ax_14c.set_title("(c) Pool Δ¹⁴C — prior (gray) vs. optimized  ·  obs = hf212-03",
                     fontsize=9, loc="left")
    ax_14c.tick_params(labelsize=8)
    ax_14c.grid(axis="y", lw=0.4, alpha=0.4)
    ax_14c.legend(handles=handles_c, fontsize=7.5, ncol=2, framealpha=0.85)

    # ── Panel (d): Respired CO₂ Δ¹⁴C ────────────────────────────────────────
    # Obs scatter (NWN means)
    resp_obs_arr  = np.array(delta14C_resp)
    valid_resp    = np.where(np.isfinite(resp_obs_arr))[0]
    obs_yrs_resp  = time_years[valid_resp]
    obs_vals_resp = resp_obs_arr[valid_resp]

    ax_resp.plot(time_years, d14C_resp_prior,
                 lw=0.8, color="gray", linestyle="--", alpha=0.5, label="Prior model")
    ax_resp.plot(time_years, d14C_resp_opt,
                 lw=1.5, color="steelblue", alpha=0.9,            label="Optimized model")
    ax_resp.scatter(obs_yrs_resp, obs_vals_resp,
                    s=40, color="tomato", marker="o",
                    edgecolors="black", linewidths=0.6, zorder=6,
                    label="hf212-01 NWN obs")

    # RMSE annotation
    if len(valid_resp) > 0:
        rmse_prior = np.sqrt(np.mean((d14C_resp_prior[valid_resp] - obs_vals_resp) ** 2))
        rmse_opt   = np.sqrt(np.mean((d14C_resp_opt[valid_resp]   - obs_vals_resp) ** 2))
        ax_resp.text(0.02, 0.97,
                     f"Prior RMSE:  {rmse_prior:.1f} ‰\nOpt RMSE:    {rmse_opt:.1f} ‰",
                     transform=ax_resp.transAxes, va="top", ha="left",
                     fontsize=8, color="0.3",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

    ax_resp.axhline(0, lw=0.5, color="gray", linestyle=":")
    ax_resp.set_ylabel("Δ¹⁴C of respired CO₂ (‰)", fontsize=9)
    ax_resp.set_xlabel("Year", fontsize=9)
    ax_resp.set_title("(d) Respired CO₂ Δ¹⁴C — flux-weighted model vs. hf212-01 NWN",
                      fontsize=9, loc="left")
    ax_resp.tick_params(labelsize=8)
    ax_resp.grid(axis="y", lw=0.4, alpha=0.4)
    ax_resp.legend(fontsize=8, framealpha=0.85)

    fig.suptitle(
        "Harvard Forest — Soil-Only Δ¹⁴C Inversion\n"
        "Pool obs: hf212-03 (4 pools × 1996+2007 = 8 pts)  ·  "
        "Respired CO₂ obs: hf212-01 NWN (41 dates, 1996–2010)",
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
    results = run_inversion()
    make_inversion_figure(*results)
