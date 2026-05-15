"""
harvard_optimal_model.py — 2-pool (active + passive) inversion for Harvard Forest.

Motivation
----------
The full 6-pool model has ~10 free parameters but only ~1 effective degree of
freedom for signal (DFS ≈ 1.0). A 2-pool model with 5 free parameters (2 τ values,
2 partition logits, 1 transfer fraction) better matches the information content of the
observational dataset: pool-level Δ¹⁴C (8 points across 4 horizons in 1996 and 2007)
and respired CO₂ Δ¹⁴C (41 NWN dates, 1996–2010).

Model structure
---------------
  soil_active   : τ ~  2 yr  — fast-cycling, bomb-¹⁴C enriched
  soil_passive  : τ ~100 yr  — slowly cycling, pre-bomb depleted
  transfer rule : soil_active → soil_passive (fraction ~0.25)

Optimised parameters
--------------------
  log_tau                      (2 values) — turnover times in log-space
  log_external_input_partition (2 logits) — carbon input fractions (softmax)
  log_f_transfer               (1 value)  — active→passive transfer in log-space

Run
---
  python notebooks/harvard_optimal_model.py

Output
------
  notebooks/harvard_optimal_model.png — 6-panel figure:
    (a) Loss convergence (pool Δ¹⁴C only  ·  pool + resp Δ¹⁴C)
    (b) τ before vs. after (bar chart with log-scale)
    (c) Pool Δ¹⁴C trajectories — prior / opt / obs
    (d) Respired CO₂ Δ¹⁴C — prior / opt / obs
    (e) Information content: DFS vs. n_params (2-pool vs. 6-pool)
    (f) Age diagnostics: stored bulk Δ¹⁴C vs. respired Rh Δ¹⁴C
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

# ── Path resolution: src always comes from the worktree; data from the repo ──
_SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
_WORKTREE_ROOT = os.path.dirname(_SCRIPT_ROOT)
_SRC_ROOT = os.path.join(_WORKTREE_ROOT, "src")

# Data files live in the main repo, not in the worktree.  Heuristic: walk up
# from the worktree looking for a directory that actually contains data/.
def _find_data_root(start: str) -> str:
    candidate = start
    for _ in range(4):
        if os.path.isdir(os.path.join(candidate, "data")):
            return candidate
        candidate = os.path.dirname(candidate)
    return _WORKTREE_ROOT  # fallback

_REPO_ROOT = (
    os.environ.get("ECOSYSTEM_REPO_ROOT")
    or _find_data_root(_WORKTREE_ROOT)
)

if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from ecosystem_complexity.api import build_model, run_model, optimize
from ecosystem_complexity.config import load_config
from ecosystem_complexity.data.parsers import attach_atm14C, load_harvard_forest
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.schemas import ForcingData, ObservationData
from ecosystem_complexity.state import make_default_params
from ecosystem_complexity.analysis import (
    compute_age_diagnostics,
    run_ablation_study,
)
from ecosystem_complexity.information import (
    compute_fisher,
    compute_dof,
    make_prior_covariance,
    get_param_groups,
    _default_fields,
)

# ── File paths ───────────────────────────────────────────────────────────────
# Config files are in the worktree; data files are in the main repo (_REPO_ROOT).
def _wt(rel):  # worktree-relative path → absolute
    return os.path.join(_WORKTREE_ROOT, rel)

def _data(rel):  # data-relative path → absolute (resolved after os.chdir)
    return os.path.join(_REPO_ROOT, rel)

HF_HR_PATH = _data(
    "data/harvard_forest/AMF_US-Ha1_FLUXNET_FULLSET_1991-2020_3-5/"
    "AMF_US-Ha1_FLUXNET_FULLSET_HR_1991-2020_3-5.csv"
)
_OPT_CONFIG    = _wt("configs/harvard_optimal_config.yaml")
_SOIL_CONFIG   = _wt("configs/harvard_forest_soil_only.yaml")   # 6-pool reference
HF_SOIL_14C_PATH = _data("data/harvard_forest/hf212-03-14c-org.csv")
HF_RESP_14C_PATH = _data("data/harvard_forest/hf212-01-14c-no-treat.csv")
HF_SOIL_C_PATH   = _data("data/harvard_forest/hf271-07-soils.csv")
HUA_PATH     = _data("data/shared/atm_14C/Hua_2021.csv")
GRAVEN_PATH  = _data("data/shared/atm_14C/Graven_2017.csv")
INTCAL_PATH  = _data("data/shared/atm_14C/intcal20.14c")

_R_STD = 1.176e-12

# Map hf212-03 horizon names to both model pools (each obs shared by both for
# a bulk comparison; the 2-pool model does not have horizon-resolved outputs).
_HORIZON_TO_HORIZON_LABEL = {
    "Oi":    "Oi (litter)",
    "Oe":    "Oe (fast org)",
    "A-lf":  "A-lf (min fast)",
    "A-min": "A-min (min slow)",
}

_OPT_FIELDS = ("log_tau", "log_external_input_partition", "log_f_transfer")


# ════════════════════════════════════════════════════════════════════════════
# Data helpers (reused from harvard_forest_inversion.py)
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


def _build_pool_14C_obs_bulk(soil_14c_path, forcing_time, pool_names):
    """
    Build ``delta14C_obs`` for the 2-pool model by mass-averaging the 4
    hf212-03 horizons into a single bulk ``soil_active`` and ``soil_passive``
    observation.

    Strategy:
      soil_active  ← mean(Oi, Oe)          [organic-rich; fast cycling]
      soil_passive ← mean(A-lf, A-min)     [mineral-protected; slow cycling]
    """
    d14c_df = pd.read_csv(soil_14c_path)
    d14c_df["horizon"] = d14c_df["horizon"].str.strip()

    time_np  = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T = len(time_np)

    _active_horizons  = ["Oi", "Oe"]
    _passive_horizons = ["A-lf", "A-min"]

    pool_name_set = set(pool_names)
    obs_dict = {}

    for year in d14c_df["year"].unique():
        yr_df = d14c_df[d14c_df["year"] == year]
        t_idx = int(np.argmin(np.abs(years_np - (float(year) + 0.5))))

        if "soil_active" in pool_name_set:
            vals = yr_df.loc[yr_df["horizon"].isin(_active_horizons), "of.14c.12c"].dropna()
            if len(vals):
                if "soil_active" not in obs_dict:
                    obs_dict["soil_active"] = np.full(T, np.nan, dtype=np.float32)
                obs_dict["soil_active"][t_idx] = float(vals.mean())

        if "soil_passive" in pool_name_set:
            vals = yr_df.loc[yr_df["horizon"].isin(_passive_horizons), "of.14c.12c"].dropna()
            if len(vals):
                if "soil_passive" not in obs_dict:
                    obs_dict["soil_passive"] = np.full(T, np.nan, dtype=np.float32)
                obs_dict["soil_passive"][t_idx] = float(vals.mean())

    return {k: jnp.array(v) for k, v in obs_dict.items()}


def _build_resp_14C_obs(resp_14c_path, forcing_time):
    """NWN daily-mean respired Δ¹⁴C, 1996–2010."""
    df  = pd.read_csv(resp_14c_path)
    nwn = df[df["site"] == "NWN"]
    by_date = nwn.groupby("year.date")["delta.14c"].mean()

    time_np  = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T = len(time_np)

    arr = np.full(T, np.nan, dtype=np.float32)
    for obs_year, d14c_val in by_date.items():
        t_idx = int(np.argmin(np.abs(years_np - float(obs_year))))
        arr[t_idx] = float(d14c_val)
    return jnp.array(arr)


def _build_state0_2pool(config, pool_index):
    """
    Initialise 2-pool state from hf271 C stocks and hf212-03 Δ¹⁴C data.

    Active pool ← sum(Oi + Oe) C stocks, mean Δ¹⁴C from 1996
    Passive pool ← sum(A-lf + A-min) C stocks, mean Δ¹⁴C from 1996
    """
    from ecosystem_complexity.state import make_initial_state

    soils    = pd.read_csv(HF_SOIL_C_PATH)
    org_mean = soils.loc[soils["horizon"] == "O", "carbon.gm2"].dropna().mean()
    min_mean = soils.loc[soils["horizon"] == "M", "carbon.gm2"].dropna().mean()

    d14c_df  = pd.read_csv(HF_SOIL_14C_PATH)
    d14c_df["horizon"] = d14c_df["horizon"].str.strip()
    d14c_1996 = d14c_df[d14c_df["year"] == 1996]

    def _mean_d14c(horizons):
        vals = d14c_1996.loc[d14c_1996["horizon"].isin(horizons), "of.14c.12c"].dropna()
        return float(vals.mean()) if len(vals) else 0.0

    C12_by_pool = {
        "soil_active":  org_mean,
        "soil_passive": min_mean,
    }
    d14C_by_pool = {
        "soil_active":  _mean_d14c(["Oi", "Oe"]),
        "soil_passive": _mean_d14c(["A-lf", "A-min"]),
    }

    base    = make_initial_state(config, {})
    n_pools = len(pool_index)
    C12_arr = np.zeros(n_pools, dtype=np.float32)
    C14_arr = np.zeros(n_pools, dtype=np.float32)

    for pool_name, c12_val in C12_by_pool.items():
        if pool_name not in pool_index.pool_names:
            continue
        i = pool_index[pool_name]
        d14c_val   = d14C_by_pool.get(pool_name, 0.0)
        C12_arr[i] = max(c12_val, 0.0)
        C14_arr[i] = C12_arr[i] * _R_STD * (1.0 + d14c_val / 1000.0)

    return base._replace(C12=jnp.array(C12_arr), C14=jnp.array(C14_arr))


# ════════════════════════════════════════════════════════════════════════════
# Core workflow
# ════════════════════════════════════════════════════════════════════════════

def run_optimal_inversion():

    # ── Atmospheric ¹⁴C record ───────────────────────────────────────────────
    print("Loading atmospheric ¹⁴C record…")
    atm14C = load_full_14C_record(
        hua_path=HUA_PATH, graven_path=GRAVEN_PATH, intcal_path=INTCAL_PATH,
        hemisphere="NH", start_year=1500.0, end_year=2025.0,
    )

    # ── Build 2-pool model ───────────────────────────────────────────────────
    print("Building 2-pool model…")
    model  = build_model(_OPT_CONFIG)
    config = model.config
    idx    = model.pool_index
    print(f"  Pools: {idx.pool_names}")

    # ── Forcing ──────────────────────────────────────────────────────────────
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

    # ── Observations ─────────────────────────────────────────────────────────
    print("Building observations…")
    delta14C_obs  = _build_pool_14C_obs_bulk(HF_SOIL_14C_PATH, forcing.time, idx.pool_names)
    delta14C_resp = _build_resp_14C_obs(HF_RESP_14C_PATH, forcing.time)

    n_pool_obs = sum(int(jnp.sum(~jnp.isnan(a))) for a in delta14C_obs.values())
    n_resp_obs = int(jnp.sum(~jnp.isnan(delta14C_resp)))
    print(f"  Pool Δ¹⁴C obs: {n_pool_obs}  |  Resp Δ¹⁴C obs: {n_resp_obs}")

    _nan_T = jnp.full(T, jnp.nan)
    obs_all = ObservationData(
        time=forcing.time,
        NEE=_nan_T, GPP=_nan_T, ER=_nan_T, NEE_unc=_nan_T,
        delta14C_obs=delta14C_obs,
        deltaD14C_obs={},
        C_pools_obs={},
        delta14C_resp=delta14C_resp,
    )
    obs_pool_only = ObservationData(
        time=forcing.time,
        NEE=_nan_T, GPP=_nan_T, ER=_nan_T, NEE_unc=_nan_T,
        delta14C_obs=delta14C_obs,
        deltaD14C_obs={},
        C_pools_obs={},
        delta14C_resp=None,
    )

    # ── Initial state ────────────────────────────────────────────────────────
    state0 = _build_state0_2pool(config, idx)
    print(f"  State0 total C12: {float(jnp.sum(state0.C12)):.0f} gC m⁻²")

    # ── Prior forward run ─────────────────────────────────────────────────────
    print("\nPrior forward simulation…")
    params_prior = make_default_params(config)
    t0 = time.perf_counter()
    out_prior = run_model(model, forcing, state0=state0, params=params_prior)
    jax.block_until_ready(out_prior.delta14C)
    print(f"  Done  [{time.perf_counter()-t0:.1f}s]")

    tau_p   = np.exp(np.array(params_prior.log_tau))
    w_prior = np.array(out_prior.C12) / (tau_p[None, :] + 1e-30)
    d14C_resp_prior = (np.array(out_prior.delta14C) * w_prior).sum(-1) / (w_prior.sum(-1) + 1e-30)

    # ── Optimisation 1 — pool Δ¹⁴C only ─────────────────────────────────────
    print(f"\nOpt 1 — pool Δ¹⁴C only  ({n_pool_obs} obs)…")
    t0 = time.perf_counter()
    result_pool = optimize(model, forcing, obs_pool_only, state0=state0,
                           fields=_OPT_FIELDS)
    dt1 = time.perf_counter() - t0
    lh1 = np.array(result_pool.loss_history)
    vl1 = lh1[np.isfinite(lh1)]
    print(f"  Done [{dt1:.0f}s]  loss {vl1[0]:.4f} → {vl1[-1]:.4f}")

    out_pool = run_model(model, forcing, state0=state0, params=result_pool.params_opt)
    jax.block_until_ready(out_pool.delta14C)

    # ── Optimisation 2 — pool + resp Δ¹⁴C ───────────────────────────────────
    print(f"\nOpt 2 — pool + resp Δ¹⁴C  ({n_pool_obs} + {n_resp_obs} obs)…")
    t0 = time.perf_counter()
    result_both = optimize(model, forcing, obs_all, state0=state0,
                           fields=_OPT_FIELDS)
    dt2 = time.perf_counter() - t0
    lh2 = np.array(result_both.loss_history)
    vl2 = lh2[np.isfinite(lh2)]
    print(f"  Done [{dt2:.0f}s]  loss {vl2[0]:.4f} → {vl2[-1]:.4f}")

    out_both = run_model(model, forcing, state0=state0, params=result_both.params_opt)
    jax.block_until_ready(out_both.delta14C)

    params_opt = result_both.params_opt

    # ── Respired Δ¹⁴C for all runs ───────────────────────────────────────────
    def _resp_d14c(out, params):
        tau = np.exp(np.array(params.log_tau))
        w   = np.array(out.C12) / (tau[None, :] + 1e-30)
        return (np.array(out.delta14C) * w).sum(-1) / (w.sum(-1) + 1e-30)

    d14C_resp_pool = _resp_d14c(out_pool, result_pool.params_opt)
    d14C_resp_both = _resp_d14c(out_both, params_opt)

    # ── Parameter summary ─────────────────────────────────────────────────────
    tau_opt = np.exp(np.array(params_opt.log_tau))

    def _softmax(x):
        e = np.exp(x - x.max()); return e / e.sum()

    part_opt   = _softmax(np.array(params_opt.log_external_input_partition))
    part_prior = _softmax(np.array(params_prior.log_external_input_partition))

    print(f"\n{'Pool':<16}  {'τ prior (yr)':>13}  {'τ opt (yr)':>12}")
    print("  " + "─" * 43)
    for i, name in enumerate(idx.pool_names):
        print(f"  {name:<16}  {tau_p[i]/365:>13.1f}  {tau_opt[i]/365:>12.1f}")

    print(f"\n{'Pool':<16}  {'part prior':>10}  {'part opt':>10}")
    print("  " + "─" * 38)
    for i, name in enumerate(idx.pool_names):
        print(f"  {name:<16}  {part_prior[i]:>10.3f}  {part_opt[i]:>10.3f}")

    # ── Information content analysis ──────────────────────────────────────────
    print("\nInformation content analysis (2-pool model)…")
    fields_2pool = _OPT_FIELDS
    prior_sigma = make_prior_covariance(params_opt, fields_2pool, model)
    param_groups = get_param_groups(params_opt, fields_2pool, model)

    fisher_2pool = compute_fisher(
        model, forcing, state0, params_opt, obs_all,
        fields=fields_2pool,
    )
    dof_2pool = compute_dof(fisher_2pool, prior_sigma, param_groups=param_groups)

    n_params_2pool = fisher_2pool.FIM_total.shape[0]
    print(f"  2-pool: n_params={n_params_2pool}  DFS={dof_2pool.dfs_total:.3f}"
          f"  DFS/n={dof_2pool.dfs_total/n_params_2pool:.3f}")
    if dof_2pool.dfs_by_group:
        for grp, dfs in dof_2pool.dfs_by_group.items():
            print(f"    {grp}: {dfs:.3f}")

    # Reference: 6-pool DFS (approximate — re-run if needed)
    # Reported from harvard_forest_analysis.py: DFS≈1.029 with 10 params
    dfs_6pool_ref = 1.029
    n_params_6pool_ref = 10

    # ── Age diagnostics ───────────────────────────────────────────────────────
    print("\nAge diagnostics (optimised 2-pool)…")
    age_diag = compute_age_diagnostics(out_both, params_opt, model)
    bulk_d14C_mean = float(np.nanmean(age_diag.bulk_delta14C))
    resp_d14C_mean = float(np.nanmean(age_diag.respired_delta14C))
    print(f"  Stored bulk Δ¹⁴C:  {bulk_d14C_mean:+.1f} ‰")
    print(f"  Respired Rh Δ¹⁴C:  {resp_d14C_mean:+.1f} ‰")
    print(f"  Age gap:            {bulk_d14C_mean - resp_d14C_mean:+.1f} ‰")

    return dict(
        time_years=time_years,
        out_prior=out_prior,
        out_pool=out_pool,
        out_both=out_both,
        result_pool=result_pool,
        result_both=result_both,
        delta14C_obs=delta14C_obs,
        delta14C_resp=delta14C_resp,
        d14C_resp_prior=d14C_resp_prior,
        d14C_resp_pool=d14C_resp_pool,
        d14C_resp_both=d14C_resp_both,
        params_prior=params_prior,
        params_opt=params_opt,
        pool_idx=idx,
        dof_2pool=dof_2pool,
        n_params_2pool=n_params_2pool,
        dfs_6pool_ref=dfs_6pool_ref,
        n_params_6pool_ref=n_params_6pool_ref,
        age_diag=age_diag,
        tau_p=tau_p,
        tau_opt=tau_opt,
        lh1=lh1,
        lh2=lh2,
    )


# ════════════════════════════════════════════════════════════════════════════
# Figure
# ════════════════════════════════════════════════════════════════════════════

def make_figure(r: dict, out_path: str | None = None):
    if out_path is None:
        out_path = _wt("notebooks/harvard_optimal_model.png")
    from matplotlib.lines import Line2D

    C_PRIOR = "0.55"
    C_POOL  = "steelblue"
    C_BOTH  = "tomato"

    time_years   = r["time_years"]
    out_prior    = r["out_prior"]
    out_pool     = r["out_pool"]
    out_both     = r["out_both"]
    delta14C_obs = r["delta14C_obs"]
    d14C_resp    = r["delta14C_resp"]
    pool_idx     = r["pool_idx"]
    dof_2pool    = r["dof_2pool"]
    age_diag     = r["age_diag"]
    tau_p        = r["tau_p"]
    tau_opt      = r["tau_opt"]
    lh1, lh2     = r["lh1"], r["lh2"]

    pool_names = pool_idx.pool_names
    pool_colors = ["tab:green", "tab:brown"]
    pool_markers = ["o", "s"]

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)
    axes = [fig.add_subplot(gs[r_, c]) for r_ in range(3) for c in range(2)]
    ax_loss, ax_tau, ax_pool14C, ax_resp14C, ax_info, ax_age = axes

    # ── (a) Loss convergence ─────────────────────────────────────────────────
    for lh, label, color in [
        (lh1, "pool Δ¹⁴C only", C_POOL),
        (lh2, "pool + resp Δ¹⁴C", C_BOTH),
    ]:
        valid = lh[np.isfinite(lh)]
        if valid[0] > 0:
            ax_loss.plot(np.arange(len(valid)), valid / valid[0],
                         color=color, label=label, lw=1.5)
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Normalised loss")
    ax_loss.set_title("(a) Loss convergence")
    ax_loss.legend(fontsize=9)
    ax_loss.set_yscale("log")

    # ── (b) τ before / after ─────────────────────────────────────────────────
    x = np.arange(len(pool_names))
    w = 0.35
    ax_tau.bar(x - w/2, tau_p / 365,   width=w, color=C_PRIOR, label="prior", alpha=0.85)
    ax_tau.bar(x + w/2, tau_opt / 365, width=w, color=C_BOTH,  label="optimised", alpha=0.85)
    for i, (tp, to) in enumerate(zip(tau_p, tau_opt)):
        ratio = to / tp
        ax_tau.text(i + w/2, to / 365 * 1.05, f"×{ratio:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax_tau.set_xticks(x)
    ax_tau.set_xticklabels([n.replace("soil_", "") for n in pool_names])
    ax_tau.set_ylabel("τ (years)")
    ax_tau.set_yscale("log")
    ax_tau.set_title("(b) Turnover times before / after")
    ax_tau.legend(fontsize=9)

    # ── (c) Pool Δ¹⁴C trajectories ───────────────────────────────────────────
    for i, (pool_name, color, marker) in enumerate(zip(pool_names, pool_colors, pool_markers)):
        if pool_name not in pool_idx.pool_names:
            continue
        pi = pool_idx[pool_name]

        prior_line = np.array(out_prior.delta14C)[:, pi]
        pool_line  = np.array(out_pool.delta14C)[:, pi]
        both_line  = np.array(out_both.delta14C)[:, pi]

        ax_pool14C.plot(time_years, prior_line, color=C_PRIOR, lw=1.0, alpha=0.6,
                        linestyle="--", label=f"prior ({pool_name})" if i == 0 else None)
        ax_pool14C.plot(time_years, pool_line,  color=color,   lw=1.2, alpha=0.7,
                        linestyle=":")
        ax_pool14C.plot(time_years, both_line,  color=color,   lw=1.8,
                        label=f"{pool_name} (opt)")

        if pool_name in delta14C_obs:
            obs_arr  = np.array(delta14C_obs[pool_name])
            obs_mask = ~np.isnan(obs_arr)
            ax_pool14C.scatter(time_years[obs_mask], obs_arr[obs_mask],
                               color=color, marker=marker, s=60, zorder=5,
                               label=f"{pool_name} obs")

    ax_pool14C.set_xlabel("Year")
    ax_pool14C.set_ylabel("Δ¹⁴C (‰)")
    ax_pool14C.set_title("(c) Pool Δ¹⁴C  (prior – opt-pool -- opt-both)")
    ax_pool14C.legend(fontsize=8, ncol=2)

    # ── (d) Respired CO₂ Δ¹⁴C ───────────────────────────────────────────────
    resp_arr  = np.array(d14C_resp)
    resp_mask = ~np.isnan(resp_arr)

    ax_resp14C.plot(time_years, r["d14C_resp_prior"], color=C_PRIOR, lw=1.0,
                    linestyle="--", label="prior", alpha=0.7)
    ax_resp14C.plot(time_years, r["d14C_resp_pool"],  color=C_POOL,  lw=1.2,
                    linestyle=":", label="opt pool Δ¹⁴C only", alpha=0.8)
    ax_resp14C.plot(time_years, r["d14C_resp_both"],  color=C_BOTH,  lw=1.8,
                    label="opt pool + resp Δ¹⁴C")
    ax_resp14C.scatter(time_years[resp_mask], resp_arr[resp_mask],
                       color="k", marker="x", s=30, zorder=5,
                       label="obs (hf212-01 NWN)")
    ax_resp14C.set_xlabel("Year")
    ax_resp14C.set_ylabel("Δ¹⁴C (‰)")
    ax_resp14C.set_title("(d) Respired CO₂ Δ¹⁴C")
    ax_resp14C.legend(fontsize=8)

    # ── (e) Information content: DFS vs. n_params ────────────────────────────
    n_2pool  = r["n_params_2pool"]
    dfs_2pool = dof_2pool.dfs_total
    n_6pool  = r["n_params_6pool_ref"]
    dfs_6pool = r["dfs_6pool_ref"]

    ax_info.scatter([n_6pool], [dfs_6pool], color="gray",   s=120, zorder=5,
                    label=f"6-pool (n={n_6pool}, DFS={dfs_6pool:.2f})", marker="^")
    ax_info.scatter([n_2pool], [dfs_2pool], color=C_BOTH,   s=120, zorder=5,
                    label=f"2-pool (n={n_2pool}, DFS={dfs_2pool:.2f})", marker="o")

    # Ideal line: DFS = n (all params constrained)
    n_max = max(n_6pool, n_2pool) + 2
    ax_info.plot([0, n_max], [0, n_max], color="0.7", lw=1.0,
                 linestyle="--", label="DFS = n (ideal)")
    ax_info.plot([0, n_max], [0, n_max * 0.5], color="0.85", lw=1.0,
                 linestyle=":", label="DFS = 0.5n")

    ax_info.set_xlabel("Number of optimised parameters (n)")
    ax_info.set_ylabel("Degrees of freedom for signal (DFS)")
    ax_info.set_title("(e) Information content vs. model complexity")
    ax_info.legend(fontsize=9)
    ax_info.set_xlim(0, n_max)
    ax_info.set_ylim(0, min(n_max, dfs_2pool + 2))

    # Annotation: DFS/n ratios
    for n, dfs, label in [(n_6pool, dfs_6pool, "6-pool"), (n_2pool, dfs_2pool, "2-pool")]:
        ax_info.annotate(f"DFS/n={dfs/n:.2f}",
                         xy=(n, dfs), xytext=(n + 0.4, dfs - 0.05),
                         fontsize=8, color="0.3")

    # ── (f) Age diagnostics ───────────────────────────────────────────────────
    ax_age.plot(time_years, age_diag.bulk_delta14C,      color="tab:blue",  lw=1.8,
                label="Stored bulk Δ¹⁴C (mass-weighted)")
    ax_age.plot(time_years, age_diag.respired_delta14C,  color="tab:red",   lw=1.8,
                label="Respired Rh Δ¹⁴C (flux-weighted)")

    # Shade the gap
    ax_age.fill_between(time_years,
                         age_diag.respired_delta14C,
                         age_diag.bulk_delta14C,
                         alpha=0.12, color="tab:purple",
                         label="Stored−respired gap")

    if resp_mask.any():
        ax_age.scatter(time_years[resp_mask], resp_arr[resp_mask],
                       color="k", marker="x", s=25, zorder=5, label="resp obs")

    ax_age.set_xlabel("Year")
    ax_age.set_ylabel("Δ¹⁴C (‰)")
    ax_age.set_title("(f) Stored vs. respired carbon age")
    ax_age.legend(fontsize=8)
    ax_age.axhline(0, color="0.7", lw=0.8)

    fig.suptitle(
        "Harvard Forest — 2-Pool Optimal Model  (soil_active + soil_passive)",
        fontsize=13, y=0.99,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_optimal_inversion()
    make_figure(results)
