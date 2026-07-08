"""
invert_hf_sierra.py — Optimal Estimation inversion of the Sierra et al. (2012)
7-pool Harvard Forest model.

Same physical setup as replicate_hf.py (Gaudinski 1900 initial conditions,
constant synthetic GPP, Q10=1, no spinup), but turnover times and inter-pool
transfer fractions are jointly optimized against:

  hf212-03 — density-fraction Δ¹⁴C (4 horizons × 2 years = 8 pool-level obs)
  hf212-01 — respired CO₂ Δ¹⁴C from NWN collar (41 dates, 1996–2010)

Method
------
Levenberg-Marquardt Optimal Estimation (optimize_oe) minimises:

    J(x) = (y − F(x))ᵀ Sₑ⁻¹ (y − F(x))  +  (x − xₐ)ᵀ Sₐ⁻¹ (x − xₐ)

State vector x:
    log_tau          (7 pools)
    log_f_transfer   (7 × 8 logits; structural zeros kept tight via Sₐ)

Prior means xₐ: Gaudinski/Sierra Table 1 τ values; Gaudinski Fig. 1 f values.
Prior σ:  tau_prior_std from config for τ; 0.5 (logit) for active transfers.

Observation mapping (hf212-03 horizon → model pool):
    "Oi"    → organic_Oi              (single pool)
    "Oe"    → organic_Oe_a_L          (dominant light fraction of Oe horizon)
    "A-lf"  → mineral_ALF_lt80        (dominant fraction by mass; 95% of A-LF C)
    "A-min" → mineral_MinAss          (single pool)

NOTE on combined fractions: for the FORWARD comparison in replicate_hf.py,
Sierra Fig. 3 combines Oe_a_L+Oe_a_H and ALF_gt80+ALF_lt80.  For the
INVERSION, observations are mapped to individual pools so that the optimizer
can constrain each pool's τ separately.  The "Oe" obs (+199 ‰ at 1996) acts
as an upper-bound target for organic_Oe_a_L; the posterior τ(Oe_a_L) may
shorten to bring the model closer to this value.

Run
---
    python notebooks/invert_hf_sierra.py
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

from ecosystem_complexity.api import build_model, optimize_oe, run_model
from ecosystem_complexity._oe_helpers import _build_sa_diag
from ecosystem_complexity.config import load_config
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.schemas import ForcingData, ObservationData
from ecosystem_complexity.state import make_default_params, make_initial_state
from ecosystem_complexity.transfer import get_transfer_matrix

# ── File paths ────────────────────────────────────────────────────────────────
def _wt(rel):  return os.path.join(_WORKTREE_ROOT, rel)
def _data(rel): return os.path.join(_REPO_ROOT, rel)

HF_SOIL_14C_PATH = _data("data/harvard_forest/hf212-03-14c-org.csv")
HF_RESP_14C_PATH = _data("data/harvard_forest/hf212-01-14c-no-treat.csv")
HUA_PATH         = _data("data/shared/atm_14C/Hua_2021.csv")
GRAVEN_PATH      = _data("data/shared/atm_14C/Graven_2017.csv")
INTCAL_PATH      = _data("data/shared/atm_14C/intcal20.14c")
SIERRA_CONFIG    = _wt("configs/harvard_sierra2012_config.yaml")

_R_STD   = 1.176e-12
_GPP_SYN = (215.0 + 2.7) / 0.47 / 365.0   # 1.2690 gC m⁻² day⁻¹

# Gaudinski 2000 Fig. 1 initial C stocks (gC m⁻²)
_GAUD_STOCKS = {
    "dead_roots":        390.0,
    "organic_Oi":        220.0,
    "organic_Oe_a_L":    388.0,
    "organic_Oe_a_H":   1366.0,
    "mineral_ALF_gt80":   90.0,
    "mineral_ALF_lt80": 1800.0,
    "mineral_MinAss":    560.0,
}

# Horizon name in hf212-03 → pool name in model (for OE obs mapping)
_HORIZON_TO_POOL = {
    "Oi":    "organic_Oi",
    "Oe":    "organic_Oe_a_L",
    "A-lf":  "mineral_ALF_lt80",
    "A-min": "mineral_MinAss",
}

# Combined fractions for DISPLAY (Sierra Fig. 3 style)
_COMBINED_FRACS = {
    "Oi":   ["organic_Oi"],
    "Oe/a": ["organic_Oe_a_L", "organic_Oe_a_H"],
    "A-LF": ["mineral_ALF_gt80", "mineral_ALF_lt80"],
    "A-min":["mineral_MinAss"],
}


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _build_forcing(
    start_year: float,
    end_year: float,
    atm_years: np.ndarray,
    atm_d14C: np.ndarray,
) -> ForcingData:
    """Identical to replicate_hf.py: constant synthetic GPP + historical Δ¹⁴C_atm."""
    n_days   = int(round((end_year - start_year) * 365.25))
    t0_days  = (start_year - 1970.0) * 365.25
    time_arr = np.arange(n_days, dtype=np.float32) + t0_days
    years_arr = 1970.0 + time_arr / 365.25
    d14c_arr  = np.interp(years_arr, atm_years, atm_d14C).astype(np.float32)
    n_layers  = 2
    return ForcingData(
        time          = jnp.array(time_arr),
        air_temp      = jnp.full(n_days, 10.0,  dtype=jnp.float32),
        sw_radiation  = jnp.full(n_days, 12.0,  dtype=jnp.float32),
        precip        = jnp.full(n_days, 3.0,   dtype=jnp.float32),
        vpd           = jnp.full(n_days, 0.5,   dtype=jnp.float32),
        soil_temp     = jnp.full((n_days, n_layers), 10.0, dtype=jnp.float32),
        soil_moisture = jnp.full((n_days, n_layers), 0.35, dtype=jnp.float32),
        snow_depth    = jnp.zeros(n_days, dtype=jnp.float32),
        active_layer  = jnp.ones(n_days,  dtype=jnp.float32),
        delta14C_atm  = jnp.array(d14c_arr),
        GPP_obs       = jnp.full(n_days, _GPP_SYN, dtype=jnp.float32),
        NPP_obs       = jnp.full(n_days, float("nan"), dtype=jnp.float32),
    )


def _build_obs(
    time_years: np.ndarray,
    T: int,
    forcing_time: jnp.ndarray,
) -> ObservationData:
    """
    Build ObservationData for the OE inversion from hf212-03 + hf212-01.

    Pool-level Δ¹⁴C: 4 horizons × 2 years = 8 observations.
    Respired CO₂ Δ¹⁴C: NWN site, 41 dates 1996–2010.
    """
    # ── Pool Δ¹⁴C (hf212-03) ──────────────────────────────────────────────
    d14c_df = pd.read_csv(HF_SOIL_14C_PATH)
    d14c_df["horizon"] = d14c_df["horizon"].str.strip()

    pool_obs: dict[str, np.ndarray] = {}
    for year in d14c_df["year"].unique():
        yr_df = d14c_df[d14c_df["year"] == year]
        t_idx = int(np.argmin(np.abs(time_years - (float(year) + 0.5))))
        for horizon, pool in _HORIZON_TO_POOL.items():
            vals = yr_df.loc[yr_df["horizon"] == horizon, "of.14c.12c"].dropna()
            if len(vals):
                if pool not in pool_obs:
                    pool_obs[pool] = np.full(T, np.nan, dtype=np.float32)
                pool_obs[pool][t_idx] = float(vals.mean())

    # ── Respired CO₂ Δ¹⁴C (hf212-01 NWN) ────────────────────────────────
    resp_df  = pd.read_csv(HF_RESP_14C_PATH)
    nwn      = resp_df[resp_df["site"] == "NWN"]
    by_date  = nwn.groupby("year.date")["delta.14c"].mean()
    resp_obs = np.full(T, np.nan, dtype=np.float32)
    for obs_year, val in by_date.items():
        t_idx = int(np.argmin(np.abs(time_years - float(obs_year))))
        resp_obs[t_idx] = float(val)

    return ObservationData(
        time          = forcing_time,
        NEE           = jnp.full(T, jnp.nan),
        GPP           = jnp.full(T, jnp.nan),
        ER            = jnp.full(T, jnp.nan),
        NEE_unc       = jnp.full(T, jnp.nan),
        delta14C_obs  = {k: jnp.array(v) for k, v in pool_obs.items()},
        deltaD14C_obs = {},
        C_pools_obs   = {},
        delta14C_resp = jnp.array(resp_obs),
    )


def _combined_d14c(label: str, pool_index, d14c_arr: np.ndarray, c12_arr: np.ndarray) -> np.ndarray:
    """Mass-weighted combined Δ¹⁴C time series for Sierra Fig. 3 fractions."""
    pools = _COMBINED_FRACS[label]
    indices = [pool_index[p] for p in pools]
    c12_total = c12_arr[:, indices].sum(axis=1)
    return (c12_arr[:, indices] * d14c_arr[:, indices]).sum(axis=1) / (c12_total + 1e-30)


def _resp_d14c(d14c_arr: np.ndarray, c12_arr: np.ndarray, tau_arr: np.ndarray) -> np.ndarray:
    """Flux-weighted respired Δ¹⁴C."""
    w = c12_arr / (tau_arr[None, :] + 1e-30)
    return (d14c_arr * w).sum(-1) / (w.sum(-1) + 1e-30)


def _transfer_fracs(params, n_pools: int) -> np.ndarray:
    """Return (n_pools, n_pools) transfer fraction matrix from params."""
    return np.array(get_transfer_matrix(params.log_f_transfer, n_pools))


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("Sierra (2012) OE inversion — Harvard Forest")
    print("=" * 55)

    config = load_config(SIERRA_CONFIG)
    model  = build_model(SIERRA_CONFIG)
    idx    = model.pool_index
    params_prior = make_default_params(config)
    pool_names   = list(idx.pool_names)
    n_pools      = len(idx)

    print(f"Pools ({n_pools}): {pool_names}")
    tau_prior = np.exp(np.array(params_prior.log_tau))
    print(f"τ prior (yr): { {n: round(float(tau_prior[i])/365,1) for i, n in enumerate(pool_names)} }")

    # ── Atmospheric ¹⁴C ──────────────────────────────────────────────────────
    print("\nLoading atmospheric ¹⁴C record…")
    atm_years, atm_d14C = load_full_14C_record(
        HUA_PATH, GRAVEN_PATH, INTCAL_PATH,
        hemisphere="NH", start_year=1400.0, end_year=2025.0,
    )

    # ── Initial state from Gaudinski 1900 stocks ──────────────────────────────
    d14c_1900 = float(np.interp(1900.0, atm_years, atm_d14C))
    R_1900    = _R_STD * (1.0 + d14c_1900 / 1000.0)
    C12_arr   = np.zeros(n_pools, dtype=np.float32)
    C14_arr   = np.zeros(n_pools, dtype=np.float32)
    for name, val in _GAUD_STOCKS.items():
        i = idx[name]; C12_arr[i] = val; C14_arr[i] = val * R_1900
    state_1900 = make_initial_state(config, {})._replace(
        C12=jnp.array(C12_arr), C14=jnp.array(C14_arr)
    )
    print(f"Initial state (Gaudinski 1900, Δ¹⁴C_atm={d14c_1900:.1f}‰)  "
          f"total C = {C12_arr.sum():.0f} gC m⁻²")

    # ── Forcing 1900–2010 (same as replicate_hf.py) ───────────────────────────
    forcing = _build_forcing(1900.0, 2011.0, atm_years, atm_d14C)
    n_days  = int(forcing.time.shape[0])
    t0_days = (1900.0 - 1970.0) * 365.25
    time_years = 1970.0 + (np.arange(n_days, dtype=np.float64) + t0_days) / 365.25
    print(f"Forcing: {n_days} days  (1900.0 – 2010.9)")

    # ── Observations ──────────────────────────────────────────────────────────
    obs = _build_obs(time_years, n_days, forcing.time)
    n_pool_obs = int(sum(np.sum(np.isfinite(np.array(v))) for v in obs.delta14C_obs.values()))
    n_resp_obs = int(np.sum(np.isfinite(np.array(obs.delta14C_resp))))
    print(f"Observations: {n_pool_obs} pool Δ¹⁴C (hf212-03)  +  "
          f"{n_resp_obs} resp Δ¹⁴C (hf212-01 NWN)")
    for pool_name, arr in sorted(obs.delta14C_obs.items()):
        valid = np.where(np.isfinite(np.array(arr)))[0]
        vals  = [f"{time_years[t]:.1f}:{float(arr[t]):+.0f}‰" for t in valid]
        print(f"  {pool_name:<22}  {', '.join(vals)}")

    # ── Prior forward run ──────────────────────────────────────────────────────
    print("\nPrior forward run 1900–2010…")
    t0  = time.perf_counter()
    out_prior = run_model(model, forcing, state0=state_1900, params=params_prior)
    jax.block_until_ready(out_prior.delta14C)
    print(f"  Done in {time.perf_counter()-t0:.1f} s")

    d14c_prior = np.array(out_prior.delta14C)
    c12_prior  = np.array(out_prior.C12)

    # ── OE inversion: log_tau + log_f_transfer ────────────────────────────────
    # Pin dead_roots τ near Gaudinski's 6 yr by overriding its prior σ to be
    # tight in log-space.  Without this, the optimizer compresses dead_roots
    # to <1 yr to chase respired Δ¹⁴C bias — an unphysical posterior.
    opt_fields = ("log_tau", "log_f_transfer")
    sa_diag = np.array(_build_sa_diag(model.config, params_prior, opt_fields))
    _i_dead = idx["dead_roots"]
    sa_diag[_i_dead] = 0.05 ** 2          # σ_log = 0.05 → τ stays within ±5 %
    print(f"  Pinning dead_roots τ at {tau_prior[_i_dead]/365:.1f} yr "
          f"(prior σ_log = 0.05; ≈ ±{tau_prior[_i_dead]/365 * 0.05:.2f} yr)")
    print("\nOptimal Estimation (Levenberg-Marquardt)…")
    print("  Optimizing: log_tau (7) + log_f_transfer (7×8=56, structural zeros frozen)")
    t0 = time.perf_counter()
    oe_result = optimize_oe(
        model, forcing, obs,
        state0=state_1900,
        fields=opt_fields,
        sa_override_diag=jnp.array(sa_diag, dtype=jnp.float32),
    )
    jax.block_until_ready(oe_result.x_opt)
    print(f"  Done in {time.perf_counter()-t0:.1f} s")

    params_post = oe_result.params_opt
    tau_post    = np.exp(np.array(params_post.log_tau))

    # ── Posterior forward run ─────────────────────────────────────────────────
    print("\nPosterior forward run 1900–2010…")
    out_post = run_model(model, forcing, state0=state_1900, params=params_post)
    jax.block_until_ready(out_post.delta14C)
    d14c_post = np.array(out_post.delta14C)
    c12_post  = np.array(out_post.C12)

    # ── Posterior uncertainty (1σ from Sₓ diagonal) ───────────────────────────
    Sx_diag  = np.array(jnp.diag(oe_result.Sx))
    n_tau    = n_pools
    tau_post_sigma_log = np.sqrt(np.abs(Sx_diag[:n_tau]))
    tau_post_lo = tau_post * np.exp(-tau_post_sigma_log)
    tau_post_hi = tau_post * np.exp(+tau_post_sigma_log)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print(f"\n{'Pool':<22}  {'τ prior (yr)':>12}  {'τ post (yr)':>12}  "
          f"{'±1σ (yr)':>10}  {'ratio':>7}")
    print("  " + "─" * 72)
    for i, name in enumerate(pool_names):
        tp = tau_prior[i] / 365.0
        tq = tau_post[i]  / 365.0
        lo = tau_post_lo[i] / 365.0
        hi = tau_post_hi[i] / 365.0
        ratio = tq / tp
        print(f"  {name:<22}  {tp:>12.1f}  {tq:>12.1f}  "
              f"[{lo:.1f}–{hi:.1f}]  {ratio:>7.2f}×")

    print("\nTransfer fractions — prior vs. posterior:")
    F_prior = _transfer_fracs(params_prior, n_pools)
    F_post  = _transfer_fracs(params_post,  n_pools)
    for src, dst, f_p in config.transfer_rules:
        i = idx[src]; j = idx[dst]
        print(f"  {src:<22} → {dst:<22}  "
              f"prior={F_prior[i,j]:.3f}  post={F_post[i,j]:.3f}  "
              f"Δ={F_post[i,j]-F_prior[i,j]:+.3f}")

    # Δ¹⁴C comparison at 1996 and 2007 (combined fractions)
    print("\nCombined-fraction Δ¹⁴C — prior vs. posterior vs. obs [cf. Sierra Fig. 3]:")
    obs_vals = {
        "Oi":   {1996: 132.4, 2007: 63.7},
        "Oe/a": {1996: 199.0, 2007: 140.5},
        "A-LF": {1996: 121.0, 2007: 118.5},
        "A-min":{1996: 67.5,  2007: 69.0},
    }
    print(f"  {'Fraction':<10}  {'yr':>5}  {'prior':>8}  {'post':>8}  {'obs':>8}  {'Δ(post-obs)':>12}")
    print("  " + "─" * 62)
    for label in ["Oi", "Oe/a", "A-LF", "A-min"]:
        for yr in [1996, 2007]:
            t_idx = int(np.argmin(np.abs(time_years - (yr + 0.5))))
            s_prior = float(_combined_d14c(label, idx, d14c_prior, c12_prior)[t_idx])
            s_post  = float(_combined_d14c(label, idx, d14c_post,  c12_post )[t_idx])
            o_val   = obs_vals[label][yr]
            print(f"  {label:<10}  {yr:>5}  {s_prior:>+8.1f}  {s_post:>+8.1f}  "
                  f"{o_val:>+8.1f}  {s_post-o_val:>+12.1f}")

    # Respired CO₂ RMSE
    resp_prior = _resp_d14c(d14c_prior, c12_prior, tau_prior)
    resp_post  = _resp_d14c(d14c_post,  c12_post,  tau_post)
    resp_obs_arr = np.array(obs.delta14C_resp)
    mask = np.isfinite(resp_obs_arr)
    if mask.sum() > 0:
        rmse_prior = float(np.sqrt(np.mean((resp_prior[mask] - resp_obs_arr[mask])**2)))
        rmse_post  = float(np.sqrt(np.mean((resp_post[mask]  - resp_obs_arr[mask])**2)))
        bias_prior = float(np.mean(resp_prior[mask] - resp_obs_arr[mask]))
        bias_post  = float(np.mean(resp_post[mask]  - resp_obs_arr[mask]))
        print(f"\nRespired CO₂ Δ¹⁴C RMSE:  prior={rmse_prior:.1f}‰  "
              f"post={rmse_post:.1f}‰")
        print(f"                   bias:  prior={bias_prior:+.1f}‰  "
              f"post={bias_post:+.1f}‰")

    # ── Figure ────────────────────────────────────────────────────────────────
    _make_figure(
        time_years, atm_years, atm_d14C,
        idx, pool_names, n_pools, config,
        params_prior, params_post,
        tau_prior, tau_post, tau_post_lo, tau_post_hi,
        d14c_prior, c12_prior, d14c_post, c12_post,
        resp_prior, resp_post, resp_obs_arr,
        oe_result,
    )


# ════════════════════════════════════════════════════════════════════════════
# Figure
# ════════════════════════════════════════════════════════════════════════════

def _make_figure(
    time_years, atm_years, atm_d14C,
    idx, pool_names, n_pools, config,
    params_prior, params_post,
    tau_prior, tau_post, tau_post_lo, tau_post_hi,
    d14c_prior, c12_prior, d14c_post, c12_post,
    resp_prior, resp_post, resp_obs_arr,
    oe_result,
):
    C_PRIOR = "#888888"
    C_POST  = "#d62728"   # red

    obs_vals_96 = {"Oi":132.4, "Oe/a":199.0, "A-LF":121.0, "A-min":67.5}
    obs_vals_07 = {"Oi":63.7,  "Oe/a":140.5, "A-LF":118.5, "A-min":69.0}

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    fig.suptitle(
        "Harvard Forest Sierra (2012) — OE Inversion: τ + transfer fractions\n"
        "Prior = Gaudinski/Sierra Table 1 values  ·  Posterior = constrained by"
        " hf212-03 Δ¹⁴C + hf212-01 NWN resp. CO₂",
        fontsize=11,
    )

    # ── (a) OE cost convergence ───────────────────────────────────────────────
    ax = axes[0, 0]
    costs = np.array(oe_result.cost_history)
    ax.plot(np.arange(1, len(costs)+1), costs / costs[0], color=C_POST, lw=2)
    ax.set_xlabel("LM iteration"); ax.set_ylabel("Cost / initial cost")
    ax.set_title("(a) OE convergence", loc="left", fontsize=9)
    ax.set_ylim(bottom=0); ax.grid(lw=0.4, alpha=0.4)
    ax.text(0.98, 0.95, f"{costs[0]:.2f} → {costs[-1]:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)

    # ── (b) τ prior vs posterior with ±1σ ────────────────────────────────────
    ax = axes[0, 1]
    x = np.arange(n_pools)
    w = 0.3
    bars_p = ax.bar(x - w/2, tau_prior/365, w, label="Prior",     color="0.75", alpha=0.9)
    bars_q = ax.bar(x + w/2, tau_post/365,  w, label="Posterior", color=C_POST, alpha=0.8)
    # ±1σ error bars on posterior
    yerr_lo = (tau_post - tau_post_lo) / 365
    yerr_hi = (tau_post_hi - tau_post)  / 365
    ax.errorbar(x + w/2, tau_post/365,
                yerr=[yerr_lo, yerr_hi],
                fmt="none", color="k", capsize=3, lw=1.2)
    short = [n.replace("organic_","org.").replace("mineral_","min.").replace("dead_","") for n in pool_names]
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("Turnover time (yr)"); ax.set_yscale("log")
    ax.set_title("(b) Turnover times — prior vs posterior (±1σ)", loc="left", fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis="y", lw=0.4, alpha=0.4)
    # Annotate ratio
    for i in range(n_pools):
        ratio = tau_post[i] / tau_prior[i]
        ax.text(i + w/2, tau_post[i]/365 * 1.05, f"{ratio:.2f}×",
                ha="center", va="bottom", fontsize=6)

    # ── (c) Transfer fractions prior vs posterior ─────────────────────────────
    ax = axes[0, 2]
    transfer_labels, f_prior_vals, f_post_vals = [], [], []
    F_prior = _transfer_fracs(params_prior, n_pools)
    F_post  = _transfer_fracs(params_post,  n_pools)
    for src, dst, _ in config.transfer_rules:
        i = idx[src]; j = idx[dst]
        lbl = f"{src.split('_',1)[-1][:4]}\n→{dst.split('_',1)[-1][:4]}"
        transfer_labels.append(lbl)
        f_prior_vals.append(F_prior[i, j])
        f_post_vals.append(F_post[i, j])
    xt = np.arange(len(transfer_labels))
    ax.bar(xt - w/2, f_prior_vals, w, label="Prior",     color="0.75", alpha=0.9)
    ax.bar(xt + w/2, f_post_vals,  w, label="Posterior", color=C_POST, alpha=0.8)
    ax.set_xticks(xt); ax.set_xticklabels(transfer_labels, fontsize=7)
    ax.set_ylabel("Transfer fraction"); ax.set_ylim(0, 1)
    ax.set_title("(c) Transfer fractions — prior vs posterior", loc="left", fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis="y", lw=0.4, alpha=0.4)

    # ── (d) Organic layer Δ¹⁴C ───────────────────────────────────────────────
    ax = axes[1, 0]
    mask_atm  = (atm_years >= 1950) & (atm_years <= 2011)
    mask_plot = time_years >= 1950
    ax.plot(atm_years[mask_atm], atm_d14C[mask_atm],
            color="gray", lw=1, ls="--", label="Atmosphere", zorder=0)
    colors_org = {"Oi": "#e07b39", "Oe/a": "#7c3f00"}
    obs_96 = {"Oi": 132.4, "Oe/a": 199.0}
    obs_07 = {"Oi": 63.7,  "Oe/a": 140.5}
    for label, col in colors_org.items():
        ts_p = _combined_d14c(label, idx, d14c_prior, c12_prior)
        ts_q = _combined_d14c(label, idx, d14c_post,  c12_post)
        ax.plot(time_years[mask_plot], ts_p[mask_plot], color=col, lw=1.2, ls=":", alpha=0.7)
        ax.plot(time_years[mask_plot], ts_q[mask_plot], color=col, lw=2.0, label=f"{label} (post)")
        # Obs points
        for yr, oval in [(1996.5, obs_96[label]), (2007.5, obs_07[label])]:
            t_idx = int(np.argmin(np.abs(time_years - yr)))
            ax.scatter(time_years[t_idx], oval, color=col, s=70, zorder=5,
                       edgecolors="k", lw=0.7, marker="^")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set(xlabel="Year", ylabel="Δ¹⁴C (‰)", title="(d) Organic layer (prior=dotted)")
    ax.legend(fontsize=8); ax.set_xlim(1950, 2011)

    # ── (e) Mineral layer Δ¹⁴C ───────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(atm_years[mask_atm], atm_d14C[mask_atm],
            color="gray", lw=1, ls="--", label="Atmosphere", zorder=0)
    colors_min = {"A-LF": "#2c7fb8", "A-min": "#1a4060"}
    obs_96m = {"A-LF": 121.0, "A-min": 67.5}
    obs_07m = {"A-LF": 118.5, "A-min": 69.0}
    for label, col in colors_min.items():
        ts_p = _combined_d14c(label, idx, d14c_prior, c12_prior)
        ts_q = _combined_d14c(label, idx, d14c_post,  c12_post)
        ax.plot(time_years[mask_plot], ts_p[mask_plot], color=col, lw=1.2, ls=":", alpha=0.7)
        ax.plot(time_years[mask_plot], ts_q[mask_plot], color=col, lw=2.0, label=f"{label} (post)")
        for yr, oval in [(1996.5, obs_96m[label]), (2007.5, obs_07m[label])]:
            t_idx = int(np.argmin(np.abs(time_years - yr)))
            ax.scatter(time_years[t_idx], oval, color=col, s=70, zorder=5,
                       edgecolors="k", lw=0.7, marker="^")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set(xlabel="Year", ylabel="Δ¹⁴C (‰)", title="(e) Mineral layer (prior=dotted)")
    ax.legend(fontsize=8); ax.set_xlim(1950, 2011)

    # ── (f) Respired CO₂ Δ¹⁴C ───────────────────────────────────────────────
    ax = axes[1, 2]
    ax.plot(atm_years[mask_atm], atm_d14C[mask_atm],
            color="gray", lw=1, ls="--", label="Atmosphere", zorder=0)
    ax.plot(time_years[mask_plot], resp_prior[mask_plot],
            color=C_PRIOR, lw=1.2, ls=":", alpha=0.7, label="Prior")
    ax.plot(time_years[mask_plot], resp_post[mask_plot],
            color=C_POST,  lw=2.0, label="Posterior")
    obs_mask = np.isfinite(resp_obs_arr)
    ax.scatter(time_years[obs_mask], resp_obs_arr[obs_mask],
               color="k", s=30, zorder=5, label="hf212-01 NWN", edgecolors="none")
    mask = obs_mask
    if mask.sum() > 0:
        rmse_p = float(np.sqrt(np.mean((resp_prior[mask] - resp_obs_arr[mask])**2)))
        rmse_q = float(np.sqrt(np.mean((resp_post[mask]  - resp_obs_arr[mask])**2)))
        ax.text(0.02, 0.97, f"Prior RMSE = {rmse_p:.1f}‰\nPost  RMSE = {rmse_q:.1f}‰",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set(xlabel="Year", ylabel="Δ¹⁴C (‰)", title="(f) Respired CO₂ [cf. Sierra Fig. 2]")
    ax.legend(fontsize=8); ax.set_xlim(1950, 2011)

    fig.tight_layout()
    out_path = os.path.join(_SCRIPT_ROOT, "harvard_forest_radiocarbon_inversion_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
