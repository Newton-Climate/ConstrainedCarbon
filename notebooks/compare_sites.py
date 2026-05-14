"""
compare_sites.py — Forward-run the prior model against real observations
for Harvard Forest (US-Ha1) and Barrow, Alaska (US-A10), then produce a
two-site comparison figure.  A third row shows the Harvard Forest soil-only
model (soil pools driven by observed GPP, without aboveground vegetation).

Run from the repository root:
    python notebooks/compare_sites.py

Output:  notebooks/compare_sites.png  (also displayed if a display is available)
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yaml

# ── Make sure the src layout is importable when run as a script ─────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from ecosystem_complexity.config import PoolIndex, load_config
from ecosystem_complexity.data.parsers import (
    attach_atm14C,
    load_barrow_alaska,
    load_harvard_forest,
    validate_forcing,
)
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.fluxes import thawed_frac as compute_thawed_frac
from ecosystem_complexity.model import EcosystemModel
from ecosystem_complexity.state import (
    EcosystemState,
    make_default_params,
    make_initial_state,
)
from ecosystem_complexity.tracer_14C import compute_delta14C

# ── File paths (relative to repo root) ──────────────────────────────────────
HF_HR_PATH = (
    "data/harvard_forest/AMF_US-Ha1_FLUXNET_FULLSET_1991-2020_3-5/"
    "AMF_US-Ha1_FLUXNET_FULLSET_HR_1991-2020_3-5.csv"
)
BR_ERA5_PATH = (
    "data/barrow_alaska/AMF_US-A10_FLUXNET_2011-2022_v1.3_r1/"
    "AMF_US-A10_FLUXNET_ERA5_DD_1981-2025_v1.3_r1.csv"
)
BR_FLUXMET_PATH = (
    "data/barrow_alaska/AMF_US-A10_FLUXNET_2011-2022_v1.3_r1/"
    "AMF_US-A10_FLUXNET_FLUXMET_DD_2011-2022_v1.3_r1.csv"
)
HUA_PATH    = "data/shared/atm_14C/Hua_2021.csv"
GRAVEN_PATH = "data/shared/atm_14C/Graven_2017.csv"
INTCAL_PATH = "data/shared/atm_14C/intcal20.14c"
HF_CONFIG   = "configs/harvard_forest.yaml"
BR_CONFIG   = "configs/barrow_alaska.yaml"

# ── Harvard Forest soil-only paths ──────────────────────────────────────────
HF_SOIL_CONFIG  = "configs/harvard_forest_soil_only.yaml"
HF_SOIL_C_PATH  = "data/harvard_forest/hf271-07-soils.csv"
HF_SOIL_14C_PATH = "data/harvard_forest/hf212-03-14c-org.csv"
HF_RESP_14C_PATH = "data/harvard_forest/hf212-01-14c-no-treat.csv"

# Standard ¹⁴C/¹²C ratio (IAEA modern standard, same value as tracer_14C.py)
_R_STD = 1.176e-12

SPINUP_YEARS   = 30    # ¹²C spinup iterations (fast; full convergence ~200)
CONVERGENCE_TOL = 1e-4


# ════════════════════════════════════════════════════════════════════════════
# Core helpers (mini-api — api.py is not yet merged on this branch)
# ════════════════════════════════════════════════════════════════════════════

def _build_forcing_dict(forcing):
    """
    Convert ForcingData → plain dict of JAX arrays expected by model steps.
    Fills NaN values so they never propagate through the model:
      - sw_radiation  NaN → 0 W m⁻²  (night / missing)
      - air_temp      NaN → 5 °C      (mild fallback; pre-obs ERA5 period)
      - soil_temp     NaN → air_temp  (after air_temp fill)
      - soil_moisture NaN → 0.3 m³/m³
      - delta14C_atm  NaN → 0.0 ‰
    """
    sw_rad = jnp.nan_to_num(forcing.sw_radiation, nan=0.0)
    air_t  = jnp.nan_to_num(forcing.air_temp,     nan=5.0)
    soil_temp = jnp.where(
        jnp.isnan(forcing.soil_temp),
        air_t[:, None],          # broadcast (T,) → (T, n_layers)
        forcing.soil_temp,
    )
    soil_moisture = jnp.where(
        jnp.isnan(forcing.soil_moisture),
        0.3,
        forcing.soil_moisture,
    )
    delta14C_atm = jnp.where(
        jnp.isnan(forcing.delta14C_atm),
        0.0,
        forcing.delta14C_atm,
    )
    return {
        "sw_radiation":  sw_rad,
        "soil_temp":     soil_temp,
        "soil_moisture": soil_moisture,
        "delta14C_atm":  delta14C_atm,
        "GPP_obs":       forcing.GPP_obs,
        "NPP_obs":       forcing.NPP_obs,
    }


def run_forward(model, forcing, state0, params=None):
    """
    Forward simulation via jax.lax.scan.
    Returns (outputs_dict, final_state).  Outputs arrays are shape (T, ...).
    """
    if params is None:
        params = model.params

    forcing_dict = _build_forcing_dict(forcing)

    def _body(carry, t):
        state, p = carry
        ft = jax.tree_util.tree_map(lambda x: x[t], forcing_dict)
        # Re-compute thawed_frac from the current soil temperature so that
        # permafrost-eligible layers thaw and refreeze seasonally rather than
        # staying locked at their initialisation value (0=frozen) forever.
        state = state._replace(thawed_frac=compute_thawed_frac(ft["soil_temp"]))
        state = model.step_12C(state, p, ft)
        state = model.step_14C(state, p, ft)
        diags = model.diagnose(state, p, ft)
        d14C  = compute_delta14C(state.C14, state.C12)
        outs  = {
            "C12":      state.C12,
            "C14":      state.C14,
            "delta14C": d14C,
            "NEE":      diags["NEE"],
            "GPP":      diags["GPP"],
            "ER":       diags["ER"],
            "Rh":       diags["Rh"],
            "Ra":       diags["Ra"],
        }
        return (state, p), outs

    T = int(forcing.time.shape[0])
    (final_state, _), outputs = jax.lax.scan(
        _body, (state0, params), jnp.arange(T)
    )
    return outputs, final_state


def spinup_12C(model, annual_forcing, state0, n_years=SPINUP_YEARS,
               tol=CONVERGENCE_TOL):
    """Cycle one year of forcing until ¹²C pools reach near-equilibrium."""
    state = state0
    for yr in range(n_years):
        C12_before = np.array(state.C12)
        _, state = run_forward(model, annual_forcing, state)
        C12_after  = np.array(state.C12)
        delta = np.max(np.abs(C12_after - C12_before))
        total = C12_before.sum()
        if total > 0 and delta < tol * total:
            print(f"    ¹²C converged at year {yr + 1}  (Δmax={delta:.2e})")
            return state
    print(f"    ¹²C spinup reached {n_years} years  (Δmax={delta:.2e})")
    return state


def _slice_forcing(forcing, start, end):
    """Return a ForcingData slice [start:end] along the time axis."""
    from ecosystem_complexity.data.schemas import ForcingData
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


# ════════════════════════════════════════════════════════════════════════════
# Harvard Forest soil-only helpers
# ════════════════════════════════════════════════════════════════════════════

def load_hf_soil_initial_conditions(soil_c_path: str, soil_14c_path: str):
    """
    Parse observed soil C stocks and Δ¹⁴C values for Harvard Forest.

    Returns
    -------
    C12_by_pool : dict[str, float]
        Estimated gC m⁻² for each soil pool in harvard_forest_soil_only.yaml.
        Derived from hf271-07-soils.csv (2013 measurements).

    d14C_by_pool : dict[str, float]
        Δ¹⁴C (‰) per pool, using 1996 hf212-03 values as initial condition.
        Pools without direct measurements get physically-motivated priors.

    Notes
    -----
    Horizon mapping (hf271 "M" and "O"):
      O horizon (~0–5 cm):  Oi → organic_litter (~40%), Oe → organic_fast (~60%)
      M horizon (~5–20 cm): A-lf → mineral_A_fast (~30%), A-min → mineral_A_slow (~70%)
      mineral_B (20–60 cm): estimated as 1.5× the M-horizon stock,
                            split 20% slow / 80% passive.

    Δ¹⁴C horizon mapping (hf212-03):
      Oi → organic_litter,  Oe → organic_fast,
      A-lf → mineral_A_fast, A-min → mineral_A_slow.
      Deeper pools: mineral_B_slow ~ −50 ‰, mineral_B_passive ~ −300 ‰ (prior).
    """
    # ── C12 stocks from hf271 ────────────────────────────────────────────────
    soils = pd.read_csv(soil_c_path)

    org_mean = soils.loc[soils["horizon"] == "O", "carbon.gm2"].dropna().mean()
    min_mean = soils.loc[soils["horizon"] == "M", "carbon.gm2"].dropna().mean()

    # mineral_B estimated from field profiles; depth is ~3× deeper than M
    min_B_mean = 1.5 * min_mean

    C12_by_pool = {
        "organic_litter":    0.40 * org_mean,
        "organic_fast":      0.60 * org_mean,
        "mineral_A_fast":    0.30 * min_mean,
        "mineral_A_slow":    0.70 * min_mean,
        "mineral_B_slow":    0.20 * min_B_mean,
        "mineral_B_passive": 0.80 * min_B_mean,
    }

    # ── Δ¹⁴C initial conditions from hf212-03 (1996 values) ─────────────────
    d14c = pd.read_csv(soil_14c_path)
    d14c_1996 = d14c[d14c["year"] == 1996].copy()

    # Strip trailing whitespace from horizon names
    d14c_1996 = d14c_1996.copy()
    d14c_1996["horizon"] = d14c_1996["horizon"].str.strip()

    def _get_d14c(horizon):
        row = d14c_1996[d14c_1996["horizon"] == horizon]
        if len(row) == 0:
            return None
        return float(row["of.14c.12c"].iloc[0])

    d14C_by_pool = {
        "organic_litter":    _get_d14c("Oi")   or 132.4,
        "organic_fast":      _get_d14c("Oe")   or 199.0,
        "mineral_A_fast":    _get_d14c("A-lf") or 121.0,
        "mineral_A_slow":    _get_d14c("A-min") or 67.5,
        "mineral_B_slow":    -50.0,    # no obs; typical 20-60 cm mineral
        "mineral_B_passive": -300.0,   # no obs; old passive fraction prior
    }

    return C12_by_pool, d14C_by_pool


def load_hf_resp_14C_obs(resp_14c_path: str):
    """
    Load Δ¹⁴C of soil-respired CO₂ from hf212-01 (NWN control site).

    Returns
    -------
    years : np.ndarray  (decimal year, e.g. 1996.37)
    d14C  : np.ndarray  (‰)
    """
    df = pd.read_csv(resp_14c_path)
    nwn = df[df["site"] == "NWN"].copy()
    years = np.array(nwn["year.date"], dtype=float)
    d14c  = np.array(nwn["delta.14c"], dtype=float)
    # Keep finite values only
    mask = np.isfinite(years) & np.isfinite(d14c)
    return years[mask], d14c[mask]


def build_obs_state0(
    config,
    pool_index: PoolIndex,
    C12_by_pool: dict[str, float],
    d14C_by_pool: dict[str, float],
    site_cfg: dict,
) -> EcosystemState:
    """
    Build an EcosystemState initialised from observed C stocks and Δ¹⁴C.

    C14 is inferred from  C14 = C12 × R_std × (1 + Δ14C / 1000).

    Pools not listed in ``C12_by_pool`` remain at zero.
    """
    base = make_initial_state(config, site_cfg)
    n_pools = len(pool_index)

    C12_arr = np.zeros(n_pools)
    C14_arr = np.zeros(n_pools)

    pool_names_set = set(pool_index.pool_names)
    for pool_name, c12_val in C12_by_pool.items():
        if pool_name not in pool_names_set:
            continue
        i = pool_index[pool_name]
        d14c_val = d14C_by_pool.get(pool_name, 0.0)
        C12_arr[i] = max(c12_val, 0.0)
        C14_arr[i] = C12_arr[i] * _R_STD * (1.0 + d14c_val / 1000.0)

    return base._replace(
        C12=jnp.array(C12_arr, dtype=jnp.float32),
        C14=jnp.array(C14_arr, dtype=jnp.float32),
    )


def run_soil_only_harvard(atm14C):
    """
    Load the soil-only Harvard Forest model, initialise from 1996 observations,
    and run the forward model over 1996–2020.

    Returns
    -------
    time_years  : np.ndarray          decimal years (1996–2020)
    output      : dict[str, ndarray]  model outputs (C12, delta14C, Rh, …)
    obs_resp    : tuple               (years_arr, d14C_resp_arr) from hf212-01
    obs_pool_14C: dict                {year: {pool: Δ14C}} from hf212-03
    pool_index  : PoolIndex
    """
    print(f"\n{'━'*60}")
    print("  Harvard Forest — Soil Only (external GPP inputs)")
    print(f"{'━'*60}")

    # ── Config & model ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    config = load_config(HF_SOIL_CONFIG)
    with open(HF_SOIL_CONFIG) as fh:
        raw_yaml = yaml.safe_load(fh)
    site_cfg = raw_yaml.get("site", {})

    pool_index = PoolIndex(config)
    params     = make_default_params(config)
    model      = EcosystemModel(config, params, pool_index)
    print(f"  Model built  ({len(pool_index)} pools, "
          f"{len(config.soil_layers)} layers)  [{time.perf_counter()-t0:.1f}s]")

    # ── Load Harvard Forest flux forcing with GPP_obs ────────────────────────
    t0 = time.perf_counter()
    forcing_full, _ = load_harvard_forest(
        hr_path=HF_HR_PATH,
        config=config,
        qc_threshold=2,
        include_gpp_forcing=True,
    )
    print(f"  Flux forcing loaded  ({int(forcing_full.time.shape[0])} days)  "
          f"[{time.perf_counter()-t0:.1f}s]")

    # Attach atmospheric ¹⁴C record
    years_daily, d14c_daily = atm14C
    forcing_full = attach_atm14C(forcing_full, d14c_daily, years_daily)

    # ── Slice to 1996 onward ─────────────────────────────────────────────────
    time_np = np.array(forcing_full.time)
    # time axis is days since 1970-01-01; 1996-01-01 ≈ day 9497
    years_full = 1970.0 + time_np / 365.25
    start_idx = int(np.searchsorted(years_full, 1996.0))
    forcing = _slice_forcing(forcing_full, start_idx, len(time_np))
    print(f"  Sliced to 1996+ ({int(forcing.time.shape[0])} days)")

    # ── Load soil observations for initial conditions ────────────────────────
    t0 = time.perf_counter()
    C12_by_pool, d14C_by_pool = load_hf_soil_initial_conditions(
        HF_SOIL_C_PATH, HF_SOIL_14C_PATH
    )
    state0 = build_obs_state0(config, pool_index, C12_by_pool, d14C_by_pool, site_cfg)
    print(
        f"  State0 from obs  "
        f"(total C12 = {float(jnp.sum(state0.C12)):.0f} gC m⁻²)  "
        f"[{time.perf_counter()-t0:.1f}s]"
    )
    print("  Pool Δ¹⁴C at t=0:")
    for pool_name, d14c_val in d14C_by_pool.items():
        i = pool_index[pool_name]
        c12_val = float(state0.C12[i])
        print(f"    {pool_name:<25s}  C12={c12_val:8.1f} gC m⁻²  Δ¹⁴C={d14c_val:+.1f} ‰")

    # ── Forward run from 1996 ────────────────────────────────────────────────
    t0 = time.perf_counter()
    print("  Running forward simulation…")
    output, _ = run_forward(model, forcing, state0)
    jax.block_until_ready(output["Rh"])
    print(f"  Forward run done  [{time.perf_counter()-t0:.1f}s]")

    # ── Load observational data for plotting ─────────────────────────────────
    obs_resp = load_hf_resp_14C_obs(HF_RESP_14C_PATH)

    # Pool Δ¹⁴C obs from hf212-03 (both years)
    d14c_df = pd.read_csv(HF_SOIL_14C_PATH)
    d14c_df["horizon"] = d14c_df["horizon"].str.strip()
    horizon_to_pool = {
        "Oi":    "organic_litter",
        "Oe":    "organic_fast",
        "A-lf":  "mineral_A_fast",
        "A-min": "mineral_A_slow",
    }
    obs_pool_14C = {}  # {year: {pool_name: delta14C}}
    for _, row in d14c_df.iterrows():
        yr = int(row["year"])
        hz = row["horizon"]
        if hz in horizon_to_pool:
            obs_pool_14C.setdefault(yr, {})[horizon_to_pool[hz]] = float(row["of.14c.12c"])

    time_years = 1970.0 + np.array(forcing.time) / 365.25
    return time_years, output, obs_resp, obs_pool_14C, pool_index


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════

def rmse(sim, obs):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((sim[mask] - obs[mask]) ** 2)))


def bias(sim, obs):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(sim[mask] - obs[mask]))


def r2(sim, obs):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return np.nan
    ss_res = np.sum((sim[mask] - obs[mask]) ** 2)
    ss_tot = np.sum((obs[mask] - obs[mask].mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


# ════════════════════════════════════════════════════════════════════════════
# Per-site runner
# ════════════════════════════════════════════════════════════════════════════

def run_site(site_name, config_path, load_fn, load_kwargs, atm14C):
    """
    Load data, build model, spin up, run forward, return (time_years, output, obs).
    """
    print(f"\n{'━'*60}")
    print(f"  {site_name}")
    print(f"{'━'*60}")

    # ── Config & model ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    config = load_config(config_path)
    with open(config_path) as fh:
        site_cfg = yaml.safe_load(fh).get("site", {})

    pool_index = PoolIndex(config)
    params     = make_default_params(config)
    model      = EcosystemModel(config, params, pool_index)
    print(f"  Model built  ({len(pool_index)} pools, "
          f"{len(config.soil_layers)} layers)  [{time.perf_counter()-t0:.1f}s]")

    # ── Data loading ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    forcing, obs = load_fn(**load_kwargs, config=config)
    print(f"  Data loaded  ({int(forcing.time.shape[0])} days)  "
          f"[{time.perf_counter()-t0:.1f}s]")

    # Attach atmospheric ¹⁴C record
    years_daily, d14c_daily = atm14C
    forcing = attach_atm14C(forcing, d14c_daily, years_daily)

    # Validation warnings
    warns = validate_forcing(forcing, config)
    for w in warns:
        print(f"  ⚠  {w}")

    # ── Spinup ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    state0  = make_initial_state(config, site_cfg)
    year_f  = _slice_forcing(forcing, 0, min(365, int(forcing.time.shape[0])))

    print(f"  Running ¹²C spinup ({SPINUP_YEARS} yr max)…")
    state0 = spinup_12C(model, year_f, state0)
    print(f"  Spinup done  [{time.perf_counter()-t0:.1f}s]")

    # ── Forward run ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print("  Running forward simulation…")
    output, _ = run_forward(model, forcing, state0)
    jax.block_until_ready(output["NEE"])   # ensure compile+run complete
    print(f"  Forward run done  [{time.perf_counter()-t0:.1f}s]")

    # ── Convert time axis ────────────────────────────────────────────────────
    time_years = 1970.0 + np.array(forcing.time) / 365.25

    return time_years, output, obs


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════

VARS = [
    ("NEE", "NEE (gC m⁻² day⁻¹)", "steelblue"),
    ("GPP", "GPP (gC m⁻² day⁻¹)", "forestgreen"),
    ("ER",  "ER  (gC m⁻² day⁻¹)", "tomato"),
]

# Pool display names and colours for the Δ¹⁴C panel
_POOL_STYLES = {
    "organic_litter":    ("Org litter (Oi)",  "tab:green",  "o"),
    "organic_fast":      ("Org fast (Oe)",    "tab:olive",  "s"),
    "mineral_A_fast":    ("Min-A fast (A-lf)","tab:orange", "^"),
    "mineral_A_slow":    ("Min-A slow (A-min)","tab:brown", "D"),
    "mineral_B_slow":    ("Min-B slow",        "tab:gray",  "v"),
    "mineral_B_passive": ("Min-B passive",     "tab:purple","P"),
}


def plot_site_row(axes, time_years, output, obs, site_label):
    """Fill one row of axes with time-series panels for one site."""
    for ax, (var, ylabel, color) in zip(axes, VARS):
        sim_arr = np.array(output[var])
        obs_arr = np.array(getattr(obs, var))

        # Observations as semi-transparent scatter
        ax.scatter(
            time_years, obs_arr,
            s=1, alpha=0.25, color=color, linewidths=0, label="obs",
        )
        # Model as solid line
        ax.plot(
            time_years, sim_arr,
            lw=0.9, color="black", alpha=0.85, label="model (prior)",
        )

        # Skill text
        _rmse = rmse(sim_arr, obs_arr)
        _bias = bias(sim_arr, obs_arr)
        _r2   = r2(sim_arr, obs_arr)
        ax.text(
            0.02, 0.97,
            f"RMSE={_rmse:.2f}  bias={_bias:+.2f}  R²={_r2:.2f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=7, color="0.3",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", lw=0.4, alpha=0.5)

        if ax is axes[0]:
            ax.set_title(site_label, fontsize=9, fontweight="bold", loc="left")
        if ax is axes[-1]:
            ax.legend(
                loc="upper right", fontsize=7,
                markerscale=4, framealpha=0.8,
            )


def _flux_weighted_delta14C(output, pool_index, params):
    """
    Compute flux-weighted Δ¹⁴C of respired CO₂ from pool C12/τ and Δ¹⁴C.

    Returns np.ndarray of shape (T,).
    """
    tau_vals = np.exp(np.array(params.log_tau))   # (n_pools,)
    C12_arr  = np.array(output["C12"])             # (T, n_pools)
    d14C_arr = np.array(output["delta14C"])        # (T, n_pools)
    weights  = C12_arr / (tau_vals[None, :] + 1e-30)   # (T, n_pools)
    w_total  = weights.sum(axis=-1, keepdims=True) + 1e-30
    d14C_resp = (d14C_arr * weights).sum(axis=-1) / w_total[:, 0]
    return d14C_resp


def plot_soil_only_row(
    axes,
    time_years,
    output,
    obs_resp,
    obs_pool_14C,
    pool_index,
    params,
    site_label,
):
    """
    Fill one row of axes for the Harvard Forest soil-only model.

    axes[0]: Rh (heterotrophic respiration) time series
    axes[1]: Per-pool Δ¹⁴C — model lines + 1996 and 2007 obs scatter
    axes[2]: Total soil C12 stock (kgC m⁻²) + 2013 obs band from hf271
    """
    ax_rh, ax_14c, ax_c12 = axes

    # ── Panel a: Rh ──────────────────────────────────────────────────────────
    rh_arr = np.array(output["Rh"])
    ax_rh.plot(time_years, rh_arr, lw=0.9, color="saddlebrown", alpha=0.85,
               label="model Rh")

    # Overlay flux-weighted Δ¹⁴C of respiration vs. NWN observations
    # (shown in panel b instead — keep Rh clean)

    ax_rh.set_ylabel("Rh (gC m⁻² day⁻¹)", fontsize=8)
    ax_rh.tick_params(labelsize=7)
    ax_rh.grid(axis="y", lw=0.4, alpha=0.5)
    ax_rh.set_title(site_label, fontsize=9, fontweight="bold", loc="left")
    ax_rh.legend(loc="upper left", fontsize=7, framealpha=0.8)

    # ── Panel b: Per-pool Δ¹⁴C ───────────────────────────────────────────────
    d14C_arr = np.array(output["delta14C"])   # (T, n_pools)
    pool_names = list(pool_index.pool_names)

    # Model lines for each pool
    pool_names_set = set(pool_index.pool_names)
    for pool_name, (label, color, marker) in _POOL_STYLES.items():
        if pool_name not in pool_names_set:
            continue
        i = pool_index[pool_name]
        ax_14c.plot(time_years, d14C_arr[:, i], lw=1.0, color=color,
                    alpha=0.8, label=label)

    # Obs scatter (1996 and 2007)
    obs_years_map = {1996: 1996.0, 2007: 2007.0}
    pool_to_style = {k: v for k, v in _POOL_STYLES.items()
                     if k in obs_pool_14C.get(1996, {}) or k in obs_pool_14C.get(2007, {})}
    for yr, yr_float in obs_years_map.items():
        if yr not in obs_pool_14C:
            continue
        for pool_name, d14c_val in obs_pool_14C[yr].items():
            if pool_name not in _POOL_STYLES:
                continue
            _, color, marker = _POOL_STYLES[pool_name]
            ax_14c.scatter(
                yr_float, d14c_val,
                s=40, color=color, marker=marker,
                edgecolors="black", linewidths=0.5, zorder=5,
            )

    # Flux-weighted respired Δ¹⁴C (model) vs. NWN obs
    d14C_resp_model = _flux_weighted_delta14C(output, pool_index, params)
    ax_14c.plot(time_years, d14C_resp_model, lw=1.2, color="black",
                linestyle="--", alpha=0.7, label="resp (flux-weighted)")

    resp_years, resp_d14c = obs_resp
    ax_14c.scatter(resp_years, resp_d14c, s=10, color="black", alpha=0.5,
                   zorder=4, label="resp obs (NWN)")

    ax_14c.axhline(0, lw=0.5, color="gray", linestyle=":")
    ax_14c.set_ylabel("Δ¹⁴C (‰)", fontsize=8)
    ax_14c.tick_params(labelsize=7)
    ax_14c.grid(axis="y", lw=0.4, alpha=0.5)
    ax_14c.legend(loc="upper right", fontsize=6, framealpha=0.8, ncol=2)

    # ── Panel c: Obs-comparable soil C12 stock (0–20 cm) ─────────────────────
    # hf271 covers O horizon (0-7 cm) and M horizon (0-15 cm mineral core).
    # These map to model pools:  organic_litter + organic_fast  (0-5 cm)
    #                         +  mineral_A_fast + mineral_A_slow (5-20 cm)
    # mineral_B (20-60 cm) is in the model but NOT in hf271, so we exclude it
    # from the comparison rather than fabricating an obs reference.
    _obs_comparable_pools = [
        "organic_litter", "organic_fast",
        "mineral_A_fast",  "mineral_A_slow",
    ]
    C12_arr_all = np.array(output["C12"])   # (T, n_pools)
    obs_pool_indices = [
        pool_index[p] for p in _obs_comparable_pools if p in pool_names_set
    ]
    C12_obs_comparable = C12_arr_all[:, obs_pool_indices].sum(axis=-1) / 1000.0

    # Also plot total for context (dashed)
    C12_total = C12_arr_all.sum(axis=-1) / 1000.0
    ax_c12.plot(time_years, C12_total, lw=0.8, color="sienna", alpha=0.4,
                linestyle="--", label="model total (all depths)")
    ax_c12.plot(time_years, C12_obs_comparable, lw=1.1, color="sienna", alpha=0.85,
                label="model O+M (0–20 cm, comparable to obs)")

    # 2013 observed O+M stock from hf271 (horizontal band)
    try:
        soils = pd.read_csv(HF_SOIL_C_PATH)
        o_vals = soils.loc[soils["horizon"] == "O", "carbon.gm2"].dropna().values
        m_vals = soils.loc[soils["horizon"] == "M", "carbon.gm2"].dropna().values
        # Sum O+M for total obs-covered stock; propagate independent std
        obs_mean = (o_vals.mean() + m_vals.mean()) / 1000.0  # kgC m⁻²
        obs_std  = np.sqrt(o_vals.std() ** 2 + m_vals.std() ** 2) / 1000.0
        ax_c12.axhspan(
            obs_mean - obs_std, obs_mean + obs_std,
            alpha=0.18, color="sienna",
            label=f"2013 obs O+M ±1σ ({obs_mean:.1f} kgC m⁻²)",
        )
        ax_c12.axhline(obs_mean, lw=1.0, color="sienna", linestyle=":", alpha=0.7)
    except Exception:
        pass

    ax_c12.set_ylabel("Soil C12 stock (kgC m⁻²)", fontsize=8)
    ax_c12.tick_params(labelsize=7)
    ax_c12.grid(axis="y", lw=0.4, alpha=0.5)
    ax_c12.legend(loc="upper left", fontsize=7, framealpha=0.8)


def _equilibrium_annotation(config, params, GPP_mean=3.5):
    """
    Compute the model's prior steady-state C12 for each directly-forced pool
    and return a formatted annotation string for the C12 stock panel.

    At steady state (no transfers):  C_eq = input_rate × τ
    where input_rate = GPP × CUE × partition_fraction.

    The ratio obs/C_eq tells us how much longer τ must be to match observations.
    """
    import numpy as np
    ext = config.external_inputs
    tau = np.exp(np.array(params.log_tau))
    partition = np.array(list(ext.partition.values()))
    partition_normed = partition / partition.sum()
    target_names = list(ext.partition.keys())
    idx = PoolIndex(config)

    NPP = GPP_mean * ext.CUE * ext.soil_input_fraction
    lines = ["Prior τ equilibrium vs. obs:"]
    for i, name in enumerate(target_names):
        pi = idx[name]
        inp = NPP * partition_normed[i]
        C_eq = inp * tau[pi]
        lines.append(f"  {name.split('_', 1)[-1]}: C_eq={C_eq/1000:.2f} kgC m⁻²  "
                     f"(τ={tau[pi]/365:.1f} yr)")
    return "\n".join(lines)


def make_soil_only_figure(hf_soil_data, out_path):
    """
    Standalone 3-panel figure for the Harvard Forest soil-only model.

    Layout: 3 rows × 1 column, all sharing the same time x-axis (1996–2020).
      Row 0 — Rh (heterotrophic respiration)
      Row 1 — Per-pool Δ¹⁴C with obs scatter + flux-weighted respired Δ¹⁴C
      Row 2 — Obs-comparable soil C12 stock (0–20 cm) vs. 2013 hf271 obs band,
               annotated with the prior-equilibrium values to show the τ mismatch.
    """
    soil_time, soil_out, obs_resp, obs_pool_14C, pool_idx = hf_soil_data
    soil_config = load_config(HF_SOIL_CONFIG)
    soil_params = make_default_params(soil_config)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(10, 12),
        sharex=True,
        gridspec_kw=dict(hspace=0.12, top=0.93, bottom=0.07, left=0.11, right=0.97),
    )
    ax_rh, ax_14c, ax_c12 = axes

    # ── Panel 0: Rh ──────────────────────────────────────────────────────────
    rh_arr = np.array(soil_out["Rh"])
    ax_rh.plot(soil_time, rh_arr, lw=0.9, color="saddlebrown", alpha=0.85,
               label="model Rh (prior τ)")
    ax_rh.set_ylabel("Rh  (gC m⁻² day⁻¹)", fontsize=9)
    ax_rh.tick_params(labelsize=8)
    ax_rh.grid(axis="y", lw=0.4, alpha=0.5)
    ax_rh.legend(fontsize=8, framealpha=0.8)
    ax_rh.set_title(
        "Harvard Forest — Soil-Only Model  (prior parameters, obs-initialized 1996)",
        fontsize=10, fontweight="bold", loc="left",
    )

    # ── Panel 1: Per-pool Δ¹⁴C ───────────────────────────────────────────────
    d14C_arr = np.array(soil_out["delta14C"])
    pool_names_set = set(pool_idx.pool_names)
    for pool_name, (label, color, marker) in _POOL_STYLES.items():
        if pool_name not in pool_names_set:
            continue
        i = pool_idx[pool_name]
        ax_14c.plot(soil_time, d14C_arr[:, i], lw=1.1, color=color,
                    alpha=0.85, label=label)

    # Obs scatter at 1996 and 2007 with year labels
    for yr, yr_float in [(1996, 1996.3), (2007, 2007.3)]:
        if yr not in obs_pool_14C:
            continue
        for pool_name, d14c_val in obs_pool_14C[yr].items():
            if pool_name not in _POOL_STYLES:
                continue
            _, color, marker = _POOL_STYLES[pool_name]
            ax_14c.scatter(yr_float, d14c_val, s=60, color=color, marker=marker,
                           edgecolors="black", linewidths=0.7, zorder=5)

    # Flux-weighted respired Δ¹⁴C
    d14C_resp_model = _flux_weighted_delta14C(soil_out, pool_idx, soil_params)
    ax_14c.plot(soil_time, d14C_resp_model, lw=1.3, color="black",
                linestyle="--", alpha=0.75, label="resp Δ¹⁴C (flux-weighted)")
    resp_years, resp_d14c = obs_resp
    ax_14c.scatter(resp_years, resp_d14c, s=12, color="black", alpha=0.55,
                   zorder=4, label="resp obs — NWN control")

    ax_14c.axhline(0, lw=0.6, color="gray", linestyle=":")
    ax_14c.set_ylabel("Δ¹⁴C  (‰)", fontsize=9)
    ax_14c.tick_params(labelsize=8)
    ax_14c.grid(axis="y", lw=0.4, alpha=0.5)
    ax_14c.legend(fontsize=7, framealpha=0.8, ncol=2, loc="upper right")

    # ── Panel 2: Obs-comparable C12 stock + equilibrium annotation ───────────
    _obs_comparable_pools = [
        "organic_litter", "organic_fast",
        "mineral_A_fast",  "mineral_A_slow",
    ]
    C12_arr_all = np.array(soil_out["C12"])
    obs_pool_indices = [
        pool_idx[p] for p in _obs_comparable_pools if p in pool_names_set
    ]
    C12_obs_comparable = C12_arr_all[:, obs_pool_indices].sum(axis=-1) / 1000.0
    C12_total = C12_arr_all.sum(axis=-1) / 1000.0

    ax_c12.plot(soil_time, C12_total, lw=0.9, color="sienna", alpha=0.35,
                linestyle="--", label="model: all depths (0–60 cm, incl. estimated mineral_B)")
    ax_c12.plot(soil_time, C12_obs_comparable, lw=1.4, color="sienna", alpha=0.9,
                label="model: O + M equivalent (0–20 cm)")

    # 2013 obs band
    try:
        soils = pd.read_csv(HF_SOIL_C_PATH)
        o_vals = soils.loc[soils["horizon"] == "O", "carbon.gm2"].dropna().values
        m_vals = soils.loc[soils["horizon"] == "M", "carbon.gm2"].dropna().values
        obs_mean = (o_vals.mean() + m_vals.mean()) / 1000.0
        obs_std  = np.sqrt(o_vals.std() ** 2 + m_vals.std() ** 2) / 1000.0
        ax_c12.axhspan(obs_mean - obs_std, obs_mean + obs_std,
                       alpha=0.20, color="sienna",
                       label=f"2013 obs O+M ±1σ  ({obs_mean:.1f} kgC m⁻²)")
        ax_c12.axhline(obs_mean, lw=1.1, color="sienna", linestyle=":", alpha=0.75)
    except Exception:
        obs_mean = None

    # Mark prior equilibrium: horizontal dashed line for obs-comparable pools.
    # Solve the full cascade analytically (not just directly-forced pools) so
    # that organic_fast (which is transfer-fed, not externally forced) is included.
    ext = soil_config.external_inputs
    tau_vals = np.exp(np.array(soil_params.log_tau))
    idx_cfg = PoolIndex(soil_config)

    # Use mean observed GPP_obs from the forcing as NPP driver
    GPP_obs_arr = np.array(soil_out["GPP"])   # diagnose GPP (LUE)
    # Prefer the mean of the raw GPP_obs forcing if available; fallback to 3.5
    # (the model's internal GPP may differ from prescribed GPP_obs)
    GPP_mean = 3.5   # annual mean Harvard Forest AmeriFlux
    NPP_mean = GPP_mean * ext.CUE * ext.soil_input_fraction
    partition_raw = np.array(list(ext.partition.values()))
    partition_normed = partition_raw / partition_raw.sum()
    target_names = list(ext.partition.keys())

    # Full cascade equilibrium (same algebra as analytical inversion)
    pn = {n: partition_normed[i] for i, n in enumerate(target_names)}
    def _tau(name):
        return float(tau_vals[idx_cfg[name]])

    inp_lit = NPP_mean * pn.get("organic_litter", 0)
    C_lit   = inp_lit * _tau("organic_litter")
    r_lit   = C_lit / _tau("organic_litter")
    C_fast  = 0.35 * r_lit * _tau("organic_fast")
    r_fast  = C_fast / _tau("organic_fast")
    inp_Af  = NPP_mean * pn.get("mineral_A_fast", 0)
    C_Af    = (inp_Af + 0.15 * r_lit + 0.20 * r_fast) * _tau("mineral_A_fast")
    r_Af    = C_Af / _tau("mineral_A_fast")
    inp_As  = NPP_mean * pn.get("mineral_A_slow", 0)
    C_As    = (inp_As + 0.05 * r_fast + 0.12 * r_Af) * _tau("mineral_A_slow")

    C_eq_comparable = (C_lit + C_fast + C_Af + C_As) / 1000.0   # kgC m⁻²

    ax_c12.axhline(C_eq_comparable, lw=1.5, color="firebrick", linestyle="-.",
                   alpha=0.85,
                   label=f"prior equilibrium (0–20 cm)  {C_eq_comparable:.1f} kgC m⁻²")

    # Annotation box
    ratio = (obs_mean / C_eq_comparable) if obs_mean else float("nan")
    if obs_mean:
        if abs(ratio - 1.0) < 0.15:
            status = f"Equilibrium matches obs within {abs(ratio-1)*100:.0f}%  ✓"
        else:
            status = (f"Ratio obs / eq: {ratio:.2f}×\n"
                      f"→ τ calibration still needed")
        note = (
            f"Analytical equil. (0–20 cm): {C_eq_comparable:.2f} kgC m⁻²\n"
            f"Observed mean (2013):         {obs_mean:.2f} kgC m⁻²\n"
            f"{status}"
        )
    else:
        note = ""
    if note:
        ax_c12.text(
            0.99, 0.97, note,
            transform=ax_c12.transAxes, va="top", ha="right",
            fontsize=7, color="firebrick",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="firebrick",
                      alpha=0.85, lw=0.8),
        )

    ax_c12.set_ylabel("Soil C¹² stock  (kgC m⁻²)", fontsize=9)
    ax_c12.set_xlabel("Year", fontsize=9)
    ax_c12.tick_params(labelsize=8)
    ax_c12.grid(axis="y", lw=0.4, alpha=0.5)
    ax_c12.legend(fontsize=7.5, framealpha=0.8, loc="lower left")

    fig.suptitle(
        "Harvard Forest  US-Ha1 — Soil-Only Model  (prior parameters, not yet calibrated)\n"
        "Initialized from 1996 field observations  ·  Forced by observed GPP (AmeriFlux)",
        fontsize=10,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSoil-only figure saved → {out_path}")
    try:
        plt.show()
    except Exception:
        pass


def make_figure(hf_data, br_data, hf_soil_data, out_path):
    """Assemble the 3-site comparison figure."""
    fig = plt.figure(figsize=(15, 12))
    gs  = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.40, wspace=0.28,
        left=0.06, right=0.98, top=0.93, bottom=0.07,
    )

    hf_axes   = [fig.add_subplot(gs[0, c]) for c in range(3)]
    br_axes   = [fig.add_subplot(gs[1, c], sharex=hf_axes[c]) for c in range(3)]
    soil_axes = [fig.add_subplot(gs[2, c]) for c in range(3)]

    hf_time, hf_out, hf_obs = hf_data
    br_time, br_out, br_obs = br_data
    soil_time, soil_out, obs_resp, obs_pool_14C, pool_idx = hf_soil_data

    # Need params for flux-weighted Δ¹⁴C
    soil_config = load_config(HF_SOIL_CONFIG)
    soil_params = make_default_params(soil_config)

    plot_site_row(hf_axes, hf_time, hf_out, hf_obs,
                  "Harvard Forest  US-Ha1 (42.5°N) — temperate deciduous")
    plot_site_row(br_axes, br_time, br_out, br_obs,
                  "Barrow, Alaska  US-A10 (71.3°N) — arctic tundra")
    plot_soil_only_row(
        soil_axes, soil_time, soil_out, obs_resp, obs_pool_14C,
        pool_idx, soil_params,
        "Harvard Forest  US-Ha1 — Soil Only (prescribed GPP input)",
    )

    for ax in hf_axes:
        ax.tick_params(labelbottom=False)
    for ax in br_axes:
        ax.tick_params(labelbottom=False)
    for ax in soil_axes:
        ax.set_xlabel("Year", fontsize=8)

    fig.suptitle(
        "Prior model vs. AmeriFlux observations — NEE, GPP, Ecosystem Respiration\n"
        "(model not yet calibrated; runs from default YAML parameters)",
        fontsize=10,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    try:
        plt.show()
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
# Metrics summary table
# ════════════════════════════════════════════════════════════════════════════

def print_metrics(site, time_years, output, obs):
    print(f"\n  {site} — skill scores")
    print(f"  {'Variable':<8}  {'RMSE':>7}  {'Bias':>8}  {'R²':>6}  {'N_obs':>6}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*6}  {'─'*6}")
    for var, _, _ in VARS:
        sim_arr = np.array(output[var])
        obs_arr = np.array(getattr(obs, var))
        mask    = np.isfinite(obs_arr)
        print(
            f"  {var:<8}  {rmse(sim_arr, obs_arr):7.3f}  "
            f"{bias(sim_arr, obs_arr):+8.3f}  "
            f"{r2(sim_arr, obs_arr):6.3f}  "
            f"{mask.sum():6d}"
        )


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    os.chdir(_REPO_ROOT)   # all paths are relative to repo root

    # ── Atmospheric ¹⁴C record (loaded once, shared by all sites) ───────────
    print("Loading atmospheric ¹⁴C record…")
    t0 = time.perf_counter()
    atm14C = load_full_14C_record(
        hua_path=HUA_PATH,
        graven_path=GRAVEN_PATH,
        intcal_path=INTCAL_PATH,
        hemisphere="NH",
        start_year=1500.0,
        end_year=2025.0,
    )
    print(f"  ¹⁴C record ready  [{time.perf_counter()-t0:.1f}s]")

    # ── Harvard Forest ───────────────────────────────────────────────────────
    hf_data = run_site(
        site_name="Harvard Forest  (US-Ha1)",
        config_path=HF_CONFIG,
        load_fn=load_harvard_forest,
        load_kwargs={"hr_path": HF_HR_PATH, "qc_threshold": 2},
        atm14C=atm14C,
    )

    # ── Barrow, Alaska ───────────────────────────────────────────────────────
    br_data = run_site(
        site_name="Barrow, Alaska  (US-A10)",
        config_path=BR_CONFIG,
        load_fn=load_barrow_alaska,
        load_kwargs={
            "era5_path":    BR_ERA5_PATH,
            "fluxmet_path": BR_FLUXMET_PATH,
            "qc_threshold": 0.5,
        },
        atm14C=atm14C,
    )

    # ── Harvard Forest — Soil Only ───────────────────────────────────────────
    hf_soil_data = run_soil_only_harvard(atm14C)

    # ── Metrics ──────────────────────────────────────────────────────────────
    print_metrics("Harvard Forest", *hf_data)
    print_metrics("Barrow, Alaska", *br_data)

    # ── Figures ──────────────────────────────────────────────────────────────
    out_path = os.path.join("notebooks", "compare_sites.png")
    make_figure(hf_data, br_data, hf_soil_data, out_path)

    soil_out_path = os.path.join("notebooks", "harvard_forest_soil_only.png")
    make_soil_only_figure(hf_soil_data, soil_out_path)


if __name__ == "__main__":
    main()
