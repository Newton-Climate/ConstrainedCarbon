"""
harvard_forest_inversion.py — Harvard Forest soil Δ¹⁴C inversion via OE.

Run from the repository root:
    python notebooks/harvard_forest_inversion.py

Algorithm
---------
  Optimal Estimation (Levenberg-Marquardt) — Rodgers (2000) framework.
  Minimises:  J(x) = (y−F(x))ᵀ Sₑ⁻¹ (y−F(x)) + (x−xₐ)ᵀ Sₐ⁻¹ (x−xₐ)
  Returns posterior covariance Sₓ and averaging kernel A in OEResult.

Observations
------------
  hf212-03-14c-org.csv  : 4 soil pools × 2 years (1996, 2007) = 8 Δ¹⁴C points
  hf212-01-14c-no-treat.csv : NWN site, 1996–2010, 41 resp CO₂ Δ¹⁴C dates

State vector (52 params)
------------------------
  log_tau                      (6)  turnover times
  log_external_input_partition (4)  carbon input partition logits
  log_f_transfer               (42) transfer fraction logits
    — prior σ = 0.5 for 8 real rules, 0.02 for structural zeros

Figure output (4 panels)
------------------------
  (a) OE cost vs. LM iteration for both runs
  (b) τ bar chart — prior / OE pool-only / OE pool+resp
  (c) Pool Δ¹⁴C — 3 lines + hf212-03 scatter
  (d) Respired CO₂ Δ¹⁴C — 3 lines + hf212-01 NWN obs + posterior σ envelope
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
    build_model, run_model, optimize_oe,
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

    _oe_fields = ("log_tau", "log_external_input_partition", "log_f_transfer")
    n_pool_obs = int(jnp.sum(~jnp.isnan(
        jnp.concatenate([a for a in delta14C_obs.values()]))))

    # ── OE Run 1: pool Δ¹⁴C only ─────────────────────────────────────────────
    obs_pool_only = ObservationData(
        time=forcing.time,
        NEE=jnp.full(T, jnp.nan), GPP=jnp.full(T, jnp.nan),
        ER=jnp.full(T, jnp.nan),  NEE_unc=jnp.full(T, jnp.nan),
        delta14C_obs=delta14C_obs, deltaD14C_obs={}, C_pools_obs={},
        delta14C_resp=None,
    )
    print(f"\nOE Run 1 — pool Δ¹⁴C only  ({n_pool_obs} obs, LM ≤20 iter)…")
    t0 = time.perf_counter()
    result_pool = optimize_oe(model, forcing, obs_pool_only, state0=state0,
                              fields=_oe_fields)
    dt1 = time.perf_counter() - t0
    ch1 = np.array(result_pool.cost_history)
    print(f"  Done [{dt1:.0f}s]  cost {ch1[0]:.2f} → {ch1[-1]:.2f}"
          f"  ({result_pool.n_iter} iter, converged={result_pool.converged})")

    out_pool = run_model(model, forcing, state0=state0, params=result_pool.params_opt)
    jax.block_until_ready(out_pool.delta14C)
    tau_pool = np.exp(np.array(result_pool.params_opt.log_tau))
    w_pool   = np.array(out_pool.C12) / (tau_pool[None, :] + 1e-30)
    d14C_resp_pool = (np.array(out_pool.delta14C) * w_pool).sum(-1) / (w_pool.sum(-1) + 1e-30)

    # ── OE Run 2: pool Δ¹⁴C + respired CO₂ ──────────────────────────────────
    obs_both = ObservationData(
        time=forcing.time,
        NEE=jnp.full(T, jnp.nan), GPP=jnp.full(T, jnp.nan),
        ER=jnp.full(T, jnp.nan),  NEE_unc=jnp.full(T, jnp.nan),
        delta14C_obs=delta14C_obs, deltaD14C_obs={}, C_pools_obs={},
        delta14C_resp=delta14C_resp,
    )
    print(f"\nOE Run 2 — pool + resp Δ¹⁴C  ({n_pool_obs} + {n_resp_obs} obs, LM ≤20 iter)…")
    t0 = time.perf_counter()
    result_both = optimize_oe(model, forcing, obs_both, state0=state0,
                              fields=_oe_fields)
    dt2 = time.perf_counter() - t0
    ch2 = np.array(result_both.cost_history)
    print(f"  Done [{dt2:.0f}s]  cost {ch2[0]:.2f} → {ch2[-1]:.2f}"
          f"  ({result_both.n_iter} iter, converged={result_both.converged})")

    out_both = run_model(model, forcing, state0=state0, params=result_both.params_opt)
    jax.block_until_ready(out_both.delta14C)
    tau_both = np.exp(np.array(result_both.params_opt.log_tau))
    w_both   = np.array(out_both.C12) / (tau_both[None, :] + 1e-30)
    d14C_resp_both = (np.array(out_both.delta14C) * w_both).sum(-1) / (w_both.sum(-1) + 1e-30)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print(f"\n{'Pool':<25s}  {'τ prior (yr)':>12}  {'τ OE pool (yr)':>14}  {'τ OE both (yr)':>14}")
    print("  " + "─" * 68)
    for i, name in enumerate(idx.pool_names):
        print(f"  {name:<25s}  {tau_p[i]/365:>12.1f}  "
              f"{tau_pool[i]/365:>14.1f}  {tau_both[i]/365:>14.1f}")

    # Posterior 1-σ on τ (from diagonal of Sₓ in log-space → δτ = τ σ_log_tau)
    sx_diag = np.array(jnp.diag(result_both.Sx))
    n_tau   = len(idx.pool_names)
    tau_sigma_both = tau_both * np.sqrt(sx_diag[:n_tau])
    print(f"\n  Posterior 1-σ on τ (pool+resp):")
    for i, name in enumerate(idx.pool_names):
        print(f"    {name:<25s}  ±{tau_sigma_both[i]/365:.1f} yr")

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
    4-panel OE comparison figure (3 runs: Prior / OE pool / OE pool+resp).

    (a) OE cost vs. LM iteration
    (b) τ bar chart — prior / OE pool-only / OE pool+resp
    (c) Pool Δ¹⁴C trajectories — 3 lines + hf212-03 obs
    (d) Respired CO₂ Δ¹⁴C — 3 lines + posterior σ envelope + hf212-01 NWN obs
    """
    from matplotlib.lines import Line2D

    C_PRIOR = "0.55"
    C_POOL  = "steelblue"
    C_BOTH  = "tomato"

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

    # ── Panel (a): OE cost vs. LM iteration ──────────────────────────────────
    ch1 = np.array(result_pool.cost_history)
    ch2 = np.array(result_both.cost_history)
    ax_loss.semilogy(np.arange(1, len(ch1) + 1), ch1, lw=2.0, color=C_POOL,
                     label=f"pool Δ¹⁴C only  "
                           f"({ch1[0]:.1f} → {ch1[-1]:.1f}, {len(ch1)} iter)")
    ax_loss.semilogy(np.arange(1, len(ch2) + 1), ch2, lw=2.0, color=C_BOTH,
                     linestyle="--",
                     label=f"pool + resp Δ¹⁴C  "
                           f"({ch2[0]:.1f} → {ch2[-1]:.1f}, {len(ch2)} iter)")
    ax_loss.set_xlabel("LM iteration", fontsize=9)
    ax_loss.set_ylabel("OE cost  J(x)", fontsize=9)
    ax_loss.set_title("(a) OE convergence (Levenberg-Marquardt)", fontsize=9, loc="left")
    ax_loss.legend(fontsize=8, framealpha=0.85)
    ax_loss.tick_params(labelsize=8)
    ax_loss.grid(axis="both", lw=0.4, alpha=0.4)

    # ── Panel (b): τ bar chart — 3 groups per pool ───────────────────────────
    tau_prior = np.exp(np.array(params_prior.log_tau)) / 365.0
    tau_pool  = np.exp(np.array(params_pool.log_tau))  / 365.0
    tau_both  = np.exp(np.array(params_both.log_tau))  / 365.0
    pool_names = list(pool_idx.pool_names)
    n_pools = len(pool_names)
    x = np.arange(n_pools)
    w = 0.25

    ax_tau.bar(x - w,   tau_prior, w, label="Prior",             color="0.75",  alpha=0.9)
    ax_tau.bar(x,       tau_pool,  w, label="OE pool Δ¹⁴C",    color=C_POOL,  alpha=0.75)
    ax_tau.bar(x + w,   tau_both,  w, label="OE pool+resp Δ¹⁴C", color=C_BOTH, alpha=0.75)

    short_names = [n.replace("_", "\n") for n in pool_names]
    ax_tau.set_xticks(x)
    ax_tau.set_xticklabels(short_names, fontsize=7)
    ax_tau.set_ylabel("Turnover time (years)", fontsize=9)
    ax_tau.set_title("(b) Turnover times — prior vs. OE constrained", fontsize=9, loc="left")
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
        Line2D([0], [0], color="0.4", lw=1.2, ls="--", alpha=0.75, label="OE pool Δ¹⁴C"),
        Line2D([0], [0], color="0.4", lw=1.8,                       label="OE pool+resp"),
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
        "(c) Pool Δ¹⁴C  ·  dotted=prior  dashed=OE pool  solid=OE pool+resp"
        "  ·  markers=hf212-03 obs", fontsize=8.5, loc="left")
    ax_14c.tick_params(labelsize=8)
    ax_14c.grid(axis="y", lw=0.4, alpha=0.4)
    ax_14c.legend(handles=handles_c, fontsize=7.5, ncol=2, framealpha=0.85)

    # ── Panel (d): Respired CO₂ Δ¹⁴C — 3 lines + posterior σ envelope ────────
    resp_obs_arr  = np.array(delta14C_resp)
    valid_resp    = np.where(np.isfinite(resp_obs_arr))[0]
    obs_yrs_resp  = time_years[valid_resp]
    obs_vals_resp = resp_obs_arr[valid_resp]

    # Posterior σ on resp Δ¹⁴C from result_both.Sx — propagate through
    # d14c_resp = Σ(C12_i/τ_i × Δ¹⁴C_i) / Σ(C12_i/τ_i) w.r.t. log_tau uncertainty.
    # Approximate: ∂d14c_resp/∂log_tau_i ≈ (d14C_both[:,i] - d14c_resp_both) × w_i/Σw
    tau_b    = np.exp(np.array(result_both.params_opt.log_tau))
    C12_b    = np.array(out_both.C12)
    d14c_b   = np.array(out_both.delta14C)
    wb       = C12_b / (tau_b[None, :] + 1e-30)
    wsum_b   = wb.sum(-1, keepdims=True) + 1e-30
    # ∂d14c_resp/∂log_tau_i = -(d14c_resp[:,i] - d14c_resp_both) * w_i/wsum
    jac_resp_logtau = -(d14c_b - d14C_resp_both[:, None]) * (wb / wsum_b)  # (T, n_pools)
    n_tau = len(pool_idx.pool_names)
    Sx_logtau = np.array(result_both.Sx)[:n_tau, :n_tau]
    # Var(d14c_resp) ≈ J Sₓ Jᵀ (diagonal) for each timestep
    resp_var = np.einsum("ti,ij,tj->t", jac_resp_logtau, Sx_logtau, jac_resp_logtau)
    resp_sigma = np.sqrt(np.maximum(resp_var, 0.0))

    ax_resp.fill_between(time_years,
                         d14C_resp_both - resp_sigma,
                         d14C_resp_both + resp_sigma,
                         color=C_BOTH, alpha=0.15, label="OE pool+resp ±1σ")
    ax_resp.plot(time_years, d14C_resp_prior,
                 lw=0.9, color=C_PRIOR, linestyle=":", alpha=0.7, label="Prior")
    ax_resp.plot(time_years, d14C_resp_pool,
                 lw=1.4, color=C_POOL,  linestyle="--", alpha=0.85,
                 label="OE pool Δ¹⁴C opt")
    ax_resp.plot(time_years, d14C_resp_both,
                 lw=1.8, color=C_BOTH,  alpha=1.0,
                 label="OE pool+resp opt")
    ax_resp.scatter(obs_yrs_resp, obs_vals_resp,
                    s=40, color="black", marker="o", alpha=0.7,
                    edgecolors="none", zorder=6, label="hf212-01 NWN obs")

    if len(valid_resp) > 0:
        rmse_p = np.sqrt(np.mean((d14C_resp_prior[valid_resp] - obs_vals_resp) ** 2))
        rmse_1 = np.sqrt(np.mean((d14C_resp_pool[valid_resp]  - obs_vals_resp) ** 2))
        rmse_2 = np.sqrt(np.mean((d14C_resp_both[valid_resp]  - obs_vals_resp) ** 2))
        ax_resp.text(0.02, 0.97,
                     f"Prior RMSE:          {rmse_p:.1f} ‰\n"
                     f"OE pool Δ¹⁴C RMSE: {rmse_1:.1f} ‰\n"
                     f"OE pool+resp RMSE: {rmse_2:.1f} ‰",
                     transform=ax_resp.transAxes, va="top", ha="left",
                     fontsize=8, color="0.2",
                     bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.85))

    ax_resp.axhline(0, lw=0.5, color="gray", linestyle=":")
    ax_resp.set_ylabel("Δ¹⁴C of respired CO₂ (‰)", fontsize=9)
    ax_resp.set_xlabel("Year", fontsize=9)
    ax_resp.set_title(
        "(d) Respired CO₂ Δ¹⁴C (flux-weighted) vs. hf212-01 NWN  ·  shading = OE posterior ±1σ",
        fontsize=9, loc="left")
    ax_resp.tick_params(labelsize=8)
    ax_resp.grid(axis="y", lw=0.4, alpha=0.4)
    ax_resp.legend(fontsize=8, framealpha=0.85)

    fig.suptitle(
        "Harvard Forest Soil-Only Δ¹⁴C Inversion — Optimal Estimation (Levenberg-Marquardt)\n"
        "State vector: log_τ (6) + log_partition (4) + log_f_transfer (42)  ·  "
        "Prior  →  +pool Δ¹⁴C (8 obs)  →  +resp CO₂ Δ¹⁴C (41 obs)",
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
