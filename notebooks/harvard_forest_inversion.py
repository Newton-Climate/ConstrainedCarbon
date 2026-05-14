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

    _opt_fields  = ("log_tau", "log_external_input_partition")
    n_opt_params = sum(
        int(np.prod(getattr(make_default_params(config), f).shape))
        for f in _opt_fields
    )
    n_pool_obs = int(jnp.sum(~jnp.isnan(
        jnp.concatenate([a for a in delta14C_obs.values()]))))

    # ── Optimization 1: pool Δ¹⁴C only (no respiration) ─────────────────────
    obs_pool_only = ObservationData(
        time=forcing.time,
        NEE=jnp.full(T, jnp.nan), GPP=jnp.full(T, jnp.nan),
        ER=jnp.full(T, jnp.nan),  NEE_unc=jnp.full(T, jnp.nan),
        delta14C_obs=delta14C_obs, deltaD14C_obs={}, C_pools_obs={},
        delta14C_resp=None,          # respiration constraint OFF
    )
    print(f"\nRun 1 — pool Δ¹⁴C only  ({n_pool_obs} obs, Adam 800 iter)…")
    t0 = time.perf_counter()
    result_pool = optimize(model, forcing, obs_pool_only, state0=state0,
                           fields=_opt_fields)
    dt1 = time.perf_counter() - t0
    lh1 = np.array(result_pool.loss_history)
    vl1 = lh1[np.isfinite(lh1)]
    print(f"  Done [{dt1:.0f}s]  loss {vl1[0]:.4f} → {vl1[-1]:.4f}"
          f"  ({vl1[0]/vl1[-1]:.1f}× reduction)")

    out_pool = run_model(model, forcing, state0=state0, params=result_pool.params_opt)
    jax.block_until_ready(out_pool.delta14C)
    tau_pool = np.exp(np.array(result_pool.params_opt.log_tau))
    w_pool   = np.array(out_pool.C12) / (tau_pool[None, :] + 1e-30)
    d14C_resp_pool = (np.array(out_pool.delta14C) * w_pool).sum(-1) / (w_pool.sum(-1) + 1e-30)

    # ── Optimization 2: pool Δ¹⁴C + respired CO₂ ────────────────────────────
    obs_both = ObservationData(
        time=forcing.time,
        NEE=jnp.full(T, jnp.nan), GPP=jnp.full(T, jnp.nan),
        ER=jnp.full(T, jnp.nan),  NEE_unc=jnp.full(T, jnp.nan),
        delta14C_obs=delta14C_obs, deltaD14C_obs={}, C_pools_obs={},
        delta14C_resp=delta14C_resp,   # respiration constraint ON
    )
    print(f"\nRun 2 — pool Δ¹⁴C + resp Δ¹⁴C  ({n_pool_obs} + {n_resp_obs} obs, Adam 800 iter)…")
    t0 = time.perf_counter()
    result_both = optimize(model, forcing, obs_both, state0=state0,
                           fields=_opt_fields)
    dt2 = time.perf_counter() - t0
    lh2 = np.array(result_both.loss_history)
    vl2 = lh2[np.isfinite(lh2)]
    print(f"  Done [{dt2:.0f}s]  loss {vl2[0]:.4f} → {vl2[-1]:.4f}"
          f"  ({vl2[0]/vl2[-1]:.1f}× reduction)")

    out_both = run_model(model, forcing, state0=state0, params=result_both.params_opt)
    jax.block_until_ready(out_both.delta14C)
    tau_both = np.exp(np.array(result_both.params_opt.log_tau))
    w_both   = np.array(out_both.C12) / (tau_both[None, :] + 1e-30)
    d14C_resp_both = (np.array(out_both.delta14C) * w_both).sum(-1) / (w_both.sum(-1) + 1e-30)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def _softmax(x):
        e = np.exp(x - x.max()); return e / e.sum()

    _ext       = load_config(HF_SOIL_CONFIG).external_inputs
    part_names = list(_ext.partition.keys()) if _ext is not None else []

    print(f"\n{'Pool':<25s}  {'τ prior':>10}  {'τ pool':>10}  {'τ both':>10}")
    print("  " + "─" * 58)
    for i, name in enumerate(idx.pool_names):
        print(f"  {name:<25s}  {tau_p[i]/365:>10.1f}  "
              f"{tau_pool[i]/365:>10.1f}  {tau_both[i]/365:>10.1f}")

    print(f"\n{'Pool':<25s}  {'part prior':>10}  {'part pool':>10}  {'part both':>10}")
    print("  " + "─" * 58)
    pp = _softmax(np.array(params_prior.log_external_input_partition))
    pp1 = _softmax(np.array(result_pool.params_opt.log_external_input_partition))
    pp2 = _softmax(np.array(result_both.params_opt.log_external_input_partition))
    for i, pname in enumerate(part_names):
        print(f"  {pname:<25s}  {pp[i]:>10.3f}  {pp1[i]:>10.3f}  {pp2[i]:>10.3f}")

    return (time_years, out_prior, out_pool, out_both,
            result_pool, result_both,
            delta14C_obs, delta14C_resp,
            d14C_resp_prior, d14C_resp_pool, d14C_resp_both,
            params_prior, result_pool.params_opt, result_both.params_opt, idx)


# ════════════════════════════════════════════════════════════════════════════
# Figure
# ════════════════════════════════════════════════════════════════════════════

def make_inversion_figure(time_years, out_prior, out_pool, out_both,
                          result_pool, result_both,
                          delta14C_obs, delta14C_resp,
                          d14C_resp_prior, d14C_resp_pool, d14C_resp_both,
                          params_prior, params_pool, params_both, pool_idx):
    """
    4-panel comparison figure showing 3 runs:
      Prior  ·  Opt-pool (pool Δ¹⁴C only)  ·  Opt-both (pool Δ¹⁴C + resp Δ¹⁴C)

    (a) Normalized loss convergence for both optimizations
    (b) τ bar chart — 3 groups per pool
    (c) Pool Δ¹⁴C trajectories — 3 lines + hf212-03 obs
    (d) Respired CO₂ Δ¹⁴C — 3 lines + hf212-01 NWN obs
    """
    from matplotlib.lines import Line2D

    # Consistent run colours
    C_PRIOR = "0.55"          # medium gray
    C_POOL  = "steelblue"     # pool-only opt
    C_BOTH  = "tomato"        # pool + resp opt

    pool_names_set = set(pool_idx.pool_names)

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.40, wspace=0.30,
        left=0.07, right=0.97, top=0.91, bottom=0.07,
    )
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_tau  = fig.add_subplot(gs[0, 1])
    ax_14c  = fig.add_subplot(gs[1, 0])
    ax_resp = fig.add_subplot(gs[1, 1])

    # ── Panel (a): Normalized loss convergence ────────────────────────────────
    # Normalise each run to its own initial loss so both fit on the same scale.
    lh1 = np.array(result_pool.loss_history);  lh1 = lh1[np.isfinite(lh1)]
    lh2 = np.array(result_both.loss_history);  lh2 = lh2[np.isfinite(lh2)]
    iters1 = np.arange(1, len(lh1) + 1)
    iters2 = np.arange(1, len(lh2) + 1)
    ax_loss.plot(iters1, lh1 / lh1[0], lw=1.5, color=C_POOL,
                 label=f"pool Δ¹⁴C only  ({lh1[0]:.3f} → {lh1[-1]:.3f})")
    ax_loss.plot(iters2, lh2 / lh2[0], lw=1.5, color=C_BOTH, linestyle="--",
                 label=f"pool + resp Δ¹⁴C  ({lh2[0]:.3f} → {lh2[-1]:.3f})")
    ax_loss.set_xlabel("Iteration", fontsize=9)
    ax_loss.set_ylabel("Loss / initial loss", fontsize=9)
    ax_loss.set_title("(a) Optimization convergence (normalized)", fontsize=9, loc="left")
    ax_loss.legend(fontsize=8, framealpha=0.85)
    ax_loss.tick_params(labelsize=8)
    ax_loss.grid(axis="both", lw=0.4, alpha=0.4)
    ax_loss.set_ylim(bottom=0)

    # ── Panel (b): τ bar chart — 3 groups per pool ───────────────────────────
    tau_prior = np.exp(np.array(params_prior.log_tau)) / 365.0
    tau_pool  = np.exp(np.array(params_pool.log_tau))  / 365.0
    tau_both  = np.exp(np.array(params_both.log_tau))  / 365.0
    pool_names = list(pool_idx.pool_names)
    n_pools = len(pool_names)
    x = np.arange(n_pools)
    w = 0.25

    ax_tau.bar(x - w,   tau_prior, w, label="Prior",             color="0.75",  alpha=0.9)
    ax_tau.bar(x,       tau_pool,  w, label="Pool Δ¹⁴C opt",    color=C_POOL,  alpha=0.75)
    ax_tau.bar(x + w,   tau_both,  w, label="Pool+resp Δ¹⁴C opt", color=C_BOTH, alpha=0.75)

    short_names = [n.replace("_", "\n") for n in pool_names]
    ax_tau.set_xticks(x)
    ax_tau.set_xticklabels(short_names, fontsize=7)
    ax_tau.set_ylabel("Turnover time (years)", fontsize=9)
    ax_tau.set_title("(b) Turnover times — prior vs. constrained", fontsize=9, loc="left")
    ax_tau.legend(fontsize=8, framealpha=0.85)
    ax_tau.tick_params(axis="y", labelsize=8)
    ax_tau.grid(axis="y", lw=0.4, alpha=0.4)

    # ── Panel (c): Pool Δ¹⁴C trajectories — 3 lines ─────────────────────────
    d14C_prior_arr = np.array(out_prior.delta14C)
    d14C_pool_arr  = np.array(out_pool.delta14C)
    d14C_both_arr  = np.array(out_both.delta14C)

    for pool_name, (label, color, marker) in _POOL_STYLES.items():
        if pool_name not in pool_names_set:
            continue
        i = pool_idx[pool_name]
        ax_14c.plot(time_years, d14C_prior_arr[:, i],
                    lw=0.9, color=color, linestyle=":", alpha=0.45)
        ax_14c.plot(time_years, d14C_pool_arr[:, i],
                    lw=1.2, color=color, linestyle="--", alpha=0.75)
        ax_14c.plot(time_years, d14C_both_arr[:, i],
                    lw=1.8, color=color, alpha=1.0, label=label)
        # Obs scatter
        if pool_name in delta14C_obs:
            obs_arr = np.array(delta14C_obs[pool_name])
            valid   = np.where(np.isfinite(obs_arr))[0]
            for t_i in valid:
                ax_14c.scatter(time_years[t_i], obs_arr[t_i],
                               s=80, color=color, marker=marker,
                               edgecolors="black", linewidths=0.9, zorder=7)

    # Legend: line style = run; colour = pool
    handles_c = [
        Line2D([0], [0], color="0.4", lw=0.9, ls=":",  alpha=0.55, label="Prior"),
        Line2D([0], [0], color="0.4", lw=1.2, ls="--", alpha=0.75, label="Pool Δ¹⁴C opt"),
        Line2D([0], [0], color="0.4", lw=1.8,                       label="Pool+resp opt"),
    ] + [
        Line2D([0], [0], color=_POOL_STYLES[p][1], lw=1.8,
               marker=_POOL_STYLES[p][2], ms=6,
               label=_POOL_STYLES[p][0])
        for p in _POOL_STYLES if p in pool_names_set
    ]
    ax_14c.axhline(0, lw=0.5, color="gray", linestyle=":")
    ax_14c.set_ylabel("Δ¹⁴C (‰)", fontsize=9)
    ax_14c.set_xlabel("Year", fontsize=9)
    ax_14c.set_title(
        "(c) Pool Δ¹⁴C  ·  dotted=prior  dashed=pool opt  solid=pool+resp opt"
        "  ·  markers=hf212-03 obs", fontsize=8.5, loc="left")
    ax_14c.tick_params(labelsize=8)
    ax_14c.grid(axis="y", lw=0.4, alpha=0.4)
    ax_14c.legend(handles=handles_c, fontsize=7.5, ncol=2, framealpha=0.85)

    # ── Panel (d): Respired CO₂ Δ¹⁴C — 3 lines ──────────────────────────────
    resp_obs_arr  = np.array(delta14C_resp)
    valid_resp    = np.where(np.isfinite(resp_obs_arr))[0]
    obs_yrs_resp  = time_years[valid_resp]
    obs_vals_resp = resp_obs_arr[valid_resp]

    ax_resp.plot(time_years, d14C_resp_prior,
                 lw=0.9, color=C_PRIOR, linestyle=":", alpha=0.7, label="Prior")
    ax_resp.plot(time_years, d14C_resp_pool,
                 lw=1.4, color=C_POOL,  linestyle="--", alpha=0.85,
                 label="Pool Δ¹⁴C opt")
    ax_resp.plot(time_years, d14C_resp_both,
                 lw=1.8, color=C_BOTH,  alpha=1.0,
                 label="Pool+resp Δ¹⁴C opt")
    ax_resp.scatter(obs_yrs_resp, obs_vals_resp,
                    s=40, color="black", marker="o", alpha=0.7,
                    edgecolors="none", zorder=6, label="hf212-01 NWN obs")

    # RMSE annotation for all 3
    if len(valid_resp) > 0:
        rmse_p = np.sqrt(np.mean((d14C_resp_prior[valid_resp] - obs_vals_resp) ** 2))
        rmse_1 = np.sqrt(np.mean((d14C_resp_pool[valid_resp]  - obs_vals_resp) ** 2))
        rmse_2 = np.sqrt(np.mean((d14C_resp_both[valid_resp]  - obs_vals_resp) ** 2))
        ax_resp.text(0.02, 0.97,
                     f"Prior RMSE:          {rmse_p:.1f} ‰\n"
                     f"Pool Δ¹⁴C RMSE:     {rmse_1:.1f} ‰\n"
                     f"Pool+resp RMSE:  {rmse_2:.1f} ‰",
                     transform=ax_resp.transAxes, va="top", ha="left",
                     fontsize=8, color="0.2",
                     bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.85))

    ax_resp.axhline(0, lw=0.5, color="gray", linestyle=":")
    ax_resp.set_ylabel("Δ¹⁴C of respired CO₂ (‰)", fontsize=9)
    ax_resp.set_xlabel("Year", fontsize=9)
    ax_resp.set_title(
        "(d) Respired CO₂ Δ¹⁴C (flux-weighted) vs. hf212-01 NWN",
        fontsize=9, loc="left")
    ax_resp.tick_params(labelsize=8)
    ax_resp.grid(axis="y", lw=0.4, alpha=0.4)
    ax_resp.legend(fontsize=8, framealpha=0.85)

    fig.suptitle(
        "Harvard Forest Soil-Only Δ¹⁴C Inversion — effect of adding constraints\n"
        "Prior  →  +pool Δ¹⁴C (hf212-03, 8 obs)  →  +resp CO₂ Δ¹⁴C (hf212-01 NWN, 41 obs)",
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
    (time_years, out_prior, out_pool, out_both,
     result_pool, result_both,
     delta14C_obs, delta14C_resp,
     d14C_resp_prior, d14C_resp_pool, d14C_resp_both,
     params_prior, params_pool, params_both, idx) = run_inversion()

    make_inversion_figure(
        time_years, out_prior, out_pool, out_both,
        result_pool, result_both,
        delta14C_obs, delta14C_resp,
        d14C_resp_prior, d14C_resp_pool, d14C_resp_both,
        params_prior, params_pool, params_both, idx,
    )
