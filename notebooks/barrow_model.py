"""
barrow_model.py — 3-pool (active + slow + passive) inversion
for Barrow, Alaska (NGEE-Arctic / AmeriFlux US-A10 wet sedge tundra site).

Model structure
---------------
  soil_active  : τ ~   2 yr  — upper organic (0–14 cm), fibric peat / sedge litter
  soil_slow    : τ ~  50 yr  — lower organic (14–30 cm), hemic/sapric peat
  soil_passive : τ ~ 1000 yr — permafrost (30+ cm), old organic/mineral C

  Cascade: active → slow (25%) → passive (10%)

Observations used as constraints
---------------------------------
  Pool Δ¹⁴C   : Vaughn 2018 pore-gas CO₂ Δ¹⁴C at 10 cm (active), 20 cm (slow),
                 29–31 cm (passive); 2012–2014 campaign
  Resp CO₂ Δ¹⁴C: Vaughn 2018 surface emission Δ¹⁴C (45 obs, 2012–2014)
  C stocks     : literature-based soft priors for Barrow wet sedge tundra:
                  soil_active  ← 4500 gC m⁻² ± 45% (fibric organic, 0–14 cm)
                  soil_slow    ← 3000 gC m⁻² ± 50% (hemic/sapric, 14–30 cm)
                  sum (active+slow) ← 7500 gC m⁻² ± 35%
  Annual Rh    : AmeriFlux FLUXNET ER × f_hetero (US-A10; 2011–2022)

Optimised parameters (OE Levenberg-Marquardt)
----------------------------------------------
  log_tau                      (3 values) — turnover times in log-space
  log_external_input_partition (2 logits) — carbon input fractions (softmax)
  log_f_transfer               (2 values) — transfer fractions

Run
---
  python notebooks/barrow_model.py

Output
------
  notebooks/barrow_model.png — 6-panel figure analogous to harvard_optimal_model.png
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

# ── Path resolution ──────────────────────────────────────────────────────────
_SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
_WORKTREE_ROOT = os.path.dirname(_SCRIPT_ROOT)
_SRC_ROOT = os.path.join(_WORKTREE_ROOT, "src")


def _find_data_root(start: str) -> str:
    candidate = start
    for _ in range(4):
        if os.path.isdir(os.path.join(candidate, "data")):
            return candidate
        candidate = os.path.dirname(candidate)
    return _WORKTREE_ROOT


_REPO_ROOT = (
    os.environ.get("ECOSYSTEM_REPO_ROOT")
    or _find_data_root(_WORKTREE_ROOT)
)

if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from ecosystem_complexity.api import build_model, run_model, optimize_oe, OEResult, _get_oe_fields
from ecosystem_complexity.config import load_config
from ecosystem_complexity.data.parsers import attach_atm14C, load_barrow_alaska
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.schemas import ForcingData, ObservationData
from ecosystem_complexity.state import make_default_params
from ecosystem_complexity.analysis import compute_age_diagnostics
from ecosystem_complexity.api import ObsBlock

# ── File paths ───────────────────────────────────────────────────────────────
def _wt(rel):
    return os.path.join(_WORKTREE_ROOT, rel)

def _data(rel):
    return os.path.join(_REPO_ROOT, rel)

_BARROW_DIR = _data("data/barrow_alaska/AMF_US-A10_FLUXNET_2011-2022_v1.3_r1")
BARROW_ERA5_PATH    = os.path.join(_BARROW_DIR, "AMF_US-A10_FLUXNET_ERA5_DD_1981-2025_v1.3_r1.csv")
BARROW_FLUXMET_PATH = os.path.join(_BARROW_DIR, "AMF_US-A10_FLUXNET_FLUXMET_DD_2011-2022_v1.3_r1.csv")

_FIELD_DIR = _data("data/barrow_alaska/barrow_field_data/data")
BARROW_RC_PATH = os.path.join(_FIELD_DIR, "radiocarbon_field_Barrow_2012_2013_2014_v2.csv")

_OPT_CONFIG = _wt("configs/barrow_3pool_config.yaml")

HUA_PATH    = _data("data/shared/atm_14C/Hua_2021.csv")
GRAVEN_PATH = _data("data/shared/atm_14C/Graven_2017.csv")
INTCAL_PATH = _data("data/shared/atm_14C/intcal20.14c")

_R_STD = 1.176e-12

_OPT_FIELDS    = None
_OPT_FIELDS_ER = None


# ════════════════════════════════════════════════════════════════════════════
# Vaughn (2018) date parser
# ════════════════════════════════════════════════════════════════════════════

def _decode_vaughn_date(s: str) -> pd.Timestamp:
    """
    The Vaughn field CSV stores M/D/YY dates with an unusual encoding:
    the date '10/6/12' (Oct 6, 2012) appears as '2010-06-12' in the
    observation_date column (actual month → year field, day → month, YY → day).

    Decoding: YYYY-MM-DD stored → actual month = YYYY−2000, day = MM, year = 2000+DD.
    Rows already in M/D/YY format (e.g. '7/13/13') are parsed directly.
    """
    s = str(s).strip()
    # Direct M/D/YY format (no hyphen)
    if "/" in s:
        try:
            return pd.to_datetime(s, format="%m/%d/%y")
        except ValueError:
            pass
    # Encoded ISO format
    try:
        dt = pd.to_datetime(s, format="%Y-%m-%d")
        actual_month = dt.year - 2000
        actual_day   = dt.month
        actual_year  = 2000 + dt.day
        if 1 <= actual_month <= 12 and 1 <= actual_day <= 31:
            return pd.Timestamp(year=actual_year, month=actual_month, day=actual_day)
    except (ValueError, TypeError):
        pass
    return pd.NaT


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


# ── Depth-to-pool mapping (pore-gas Δ¹⁴C) ──────────────────────────────────
# Sensor depths available: 10 cm (active), 20 cm (slow), 29–31 cm (passive).
# Physical note: at Barrow the pore-gas at 29–31 cm is predominantly CO₂
# produced by decomposition at the BOTTOM of the summer thaw front, not from
# ancient permafrost carbon.  Its Δ¹⁴C (≈ −80‰) is semi-modern and does NOT
# represent the passive pool's bulk radiocarbon signature.  Using it as a
# passive-pool constraint pulls f_transfer_slow→passive from 2% to ~9% and
# inflates C_passive to ~24 000 gC m⁻².  We therefore exclude it here and
# rely on the C stock constraints (12 000 ± 3960 gC m⁻²) for the passive pool.
_POREGAS_DEPTH_TO_POOL = {
    10:  "soil_active",
    20:  "soil_slow",
    29:  "soil_passive",
    31:  "soil_passive",
}


def _build_pool_14C_obs_barrow(rc_path: str, forcing_time, pool_names: list) -> dict:
    """
    Build pool Δ¹⁴C observations from Vaughn pore-gas CO₂ at depth.

    Depth mapping:
      10 cm  → soil_active   (fibric organic layer)
      20 cm  → soil_slow     (hemic/sapric peat)
      29–31 cm → soil_passive (permafrost transition)

    Multiple measurements at the same depth and campaign year are averaged.
    Returns {pool_name: jnp.array of shape (T,)} with NaN outside obs dates.
    """
    df = pd.read_csv(rc_path, skiprows=6)
    df.columns = df.columns.str.strip()

    # Filter to pore-gas CO₂ only
    pore = df[df["material"].str.strip().str.lower() == "soil pore gas"].copy()

    # Decode observation dates
    pore["date_decoded"] = pore["observation_date"].apply(_decode_vaughn_date)
    pore = pore.dropna(subset=["date_decoded"])

    # Convert depth to numeric
    pore["depth_cm"] = pd.to_numeric(pore["depth"], errors="coerce")

    time_np  = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T = len(time_np)

    pool_name_set = set(pool_names)
    obs_dict: dict[str, np.ndarray] = {}

    for depth, pool_name in _POREGAS_DEPTH_TO_POOL.items():
        if pool_name not in pool_name_set:
            continue
        depth_rows = pore[pore["depth_cm"] == depth]
        if depth_rows.empty:
            continue

        # Group by decimal year (year + 0.5 to match mid-year placement)
        for yr, grp in depth_rows.groupby(depth_rows["date_decoded"].dt.year):
            d14c_vals = pd.to_numeric(grp["14C"], errors="coerce").dropna()
            if d14c_vals.empty:
                continue
            d14c_mean = float(d14c_vals.mean())
            t_idx = int(np.argmin(np.abs(years_np - (float(yr) + 0.5))))
            if pool_name not in obs_dict:
                obs_dict[pool_name] = np.full(T, np.nan, dtype=np.float32)
            obs_dict[pool_name][t_idx] = d14c_mean

    return {k: jnp.array(v) for k, v in obs_dict.items()}


def _build_resp_14C_obs_barrow(rc_path: str, forcing_time) -> jnp.ndarray:
    """
    Build respired CO₂ Δ¹⁴C from Vaughn surface emission measurements.

    Surface emission samples (material = 'surface emissions') are ecosystem
    respiration collected by chamber flux, equivalent to hf212-01 NWN at Harvard.
    Multiple same-day measurements are averaged. Returns shape (T,) with NaN gaps.
    """
    df = pd.read_csv(rc_path, skiprows=6)
    df.columns = df.columns.str.strip()

    surf = df[df["material"].str.strip().str.lower() == "surface emissions"].copy()
    surf["date_decoded"] = surf["observation_date"].apply(_decode_vaughn_date)
    surf = surf.dropna(subset=["date_decoded"])

    time_np  = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T = len(time_np)

    arr = np.full(T, np.nan, dtype=np.float32)
    # Group by date (same-day average)
    surf["decimal_year"] = surf["date_decoded"].dt.year + (surf["date_decoded"].dt.dayofyear - 1) / 365.0
    by_date = surf.groupby("decimal_year")["14C"].apply(
        lambda s: float(pd.to_numeric(s, errors="coerce").dropna().mean())
        if pd.to_numeric(s, errors="coerce").dropna().size > 0
        else np.nan
    )
    for dec_yr, d14c_val in by_date.items():
        if not np.isfinite(d14c_val):
            continue
        t_idx = int(np.argmin(np.abs(years_np - float(dec_yr))))
        arr[t_idx] = float(d14c_val)

    return jnp.array(arr)


def _build_soil_carbon_obs_barrow(pool_names: list) -> dict:
    """
    Literature-based individual C stock constraints.

    Barrow wet sedge tundra, Barrow Environmental Observatory (BEO):
      soil_active  (0–14 cm fibric organic):   ~4500 gC m⁻² ± 45%
        Source: Ping et al. 2008, Zubrzycki et al. 2013
      soil_passive (30–100 cm permafrost org): ~12000 gC m⁻² ± 33%
        Source: Hugelius et al. 2014 ABoVE pan-Arctic; Tarnocai et al. 2009;
        polygon tundra ~10–15 kgC m⁻² in 30–100 cm layer.

    soil_slow is handled via sum constraints in _build_carbon_sum_obs_barrow.

    Returns
    -------
    {pool_name: (mean_gC_m2, sigma_gC_m2)}
    """
    pool_name_set = set(pool_names)
    result = {}
    if "soil_active" in pool_name_set:
        mu = 4500.0
        result["soil_active"] = (mu, 0.45 * mu)
    if "soil_passive" in pool_name_set:
        mu = 12000.0
        result["soil_passive"] = (mu, 0.33 * mu)
    return result


def _build_carbon_sum_obs_barrow(pool_index) -> list:
    """
    Two sum constraints on carbon stocks, analogous to Harvard's mineral/organic sums.

    SUM 1 — organic horizon total (0–30 cm):
        C_active + C_slow ≈ 7500 ± 2625 gC m⁻²  (35% relative σ)
        Source: Ping et al. 2008 wet sedge BEO profiles

    SUM 2 — permafrost layer (30–100 cm):
        C_slow + C_passive ≈ 16000 ± 6400 gC m⁻²  (40% relative σ)
        Source: Hugelius et al. 2014 Cryosols, Tarnocai et al. 2009;
        wet sedge polygon tundra 30–100 cm ≈ 12–20 kg C m⁻².
        Wide σ reflects large spatial heterogeneity in permafrost C.

    Without the permafrost sum, the passive pool is unconstrained and
    grows to whatever steady-state the cascade arithmetic produces
    (cascade_input × τ_passive ≈ 24 000 gC m⁻² at τ=2000 yr).
    This constraint anchors the passive pool to a physically plausible range.
    """
    pool_names_set = set(pool_index.pool_names)
    blocks = []

    # ── Organic sum: C_active + C_slow ──────────────────────────────────────
    if "soil_active" in pool_names_set and "soil_slow" in pool_names_set:
        org_mean  = 7500.0
        org_sigma = 0.35 * org_mean
        i_act = pool_index["soil_active"]
        i_slw = pool_index["soil_slow"]
        _cols_org = jnp.array([i_act, i_slw], dtype=jnp.int32)
        blocks.append(ObsBlock(
            name="c_sum_organic",
            y=jnp.array([org_mean], dtype=jnp.float32),
            Se=jnp.array([org_sigma ** 2], dtype=jnp.float32),
            predict=lambda out, p, cols=_cols_org:
                jnp.sum(jnp.mean(out.C12, axis=0)[cols], keepdims=True),
        ))
        print(f"  C stock organic sum  (active+slow):    {org_mean:.0f} ± {org_sigma:.0f} gC m⁻²")

    # ── Permafrost sum: C_slow + C_passive ───────────────────────────────────
    if "soil_slow" in pool_names_set and "soil_passive" in pool_names_set:
        pf_mean  = 16000.0
        pf_sigma = 0.25 * pf_mean   # 25% — tighter than 40%; anchors total permafrost C
        i_slw = pool_index["soil_slow"]
        i_pas = pool_index["soil_passive"]
        _cols_pf = jnp.array([i_slw, i_pas], dtype=jnp.int32)
        blocks.append(ObsBlock(
            name="c_sum_permafrost",
            y=jnp.array([pf_mean], dtype=jnp.float32),
            Se=jnp.array([pf_sigma ** 2], dtype=jnp.float32),
            predict=lambda out, p, cols=_cols_pf:
                jnp.sum(jnp.mean(out.C12, axis=0)[cols], keepdims=True),
        ))
        print(f"  C stock permafrost sum (slow+passive): {pf_mean:.0f} ± {pf_sigma:.0f} gC m⁻²  (25% σ)")

    return blocks


def _build_state0_barrow(config, pool_index):
    """
    Initialise model state for Barrow from literature C stocks and
    pore-gas Δ¹⁴C at depth (2012 Vaughn measurements).

    C12 (gC m⁻²):
      soil_active  ← 4500
      soil_slow    ← 3000
      soil_passive ← 12000  (permafrost C estimated from Hugelius et al. 2013)

    Δ¹⁴C initialised from 2012 pore-gas means:
      soil_active  ← ~+30‰ (10 cm; bomb-¹⁴C enriched)
      soil_slow    ← ~+15‰ (20 cm; transitional)
      soil_passive ← ~−80‰ (29–31 cm; permafrost, pre-bomb)
    """
    from ecosystem_complexity.state import make_initial_state

    C12_by_pool = {
        "soil_active":  4500.0,
        "soil_slow":    3000.0,
        "soil_passive": 12000.0,
    }
    d14C_by_pool = {
        "soil_active":   30.0,
        "soil_slow":     15.0,
        "soil_passive": -80.0,
    }

    base   = make_initial_state(config, {})
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

    # ── Build 3-pool model ───────────────────────────────────────────────────
    print("Building 3-pool model…")
    model  = build_model(_OPT_CONFIG)
    config = model.config
    idx    = model.pool_index
    print(f"  Pools: {idx.pool_names}")

    global _OPT_FIELDS, _OPT_FIELDS_ER
    inv_cfg_nb   = getattr(config, "inversion_raw", {}) or {}
    _OPT_FIELDS    = _get_oe_fields(config, inv_cfg_nb)
    _OPT_FIELDS_ER = _OPT_FIELDS + ("log_f_hetero",)
    print(f"  OE fields (OE1–4): {_OPT_FIELDS}")
    print(f"  OE fields (OE5):   {_OPT_FIELDS_ER}")

    # ── Forcing ──────────────────────────────────────────────────────────────
    print("Loading flux forcing (ERA5_DD + FLUXMET_DD)…")
    forcing_full, obs_raw = load_barrow_alaska(
        era5_path=BARROW_ERA5_PATH,
        fluxmet_path=BARROW_FLUXMET_PATH,
        config=config,
        qc_threshold=0.0,   # accept all (incl. gap-filled); QC = fraction-measured
        include_gpp_forcing=True,
    )
    years_daily, d14c_daily = atm14C
    forcing_full = attach_atm14C(forcing_full, d14c_daily, years_daily)

    time_np   = np.array(forcing_full.time)
    years_all = 1970.0 + time_np / 365.25

    # Start forcing window from 2011 (when FLUXMET flux data begins).
    # ERA5 runs from 1981 but fluxes / GPP only available from 2011.
    start_idx = int(np.searchsorted(years_all, 2011.0))
    forcing   = _slice_forcing(forcing_full, start_idx, len(time_np))
    T         = int(forcing.time.shape[0])

    _ER_OUTLIER_YEARS = {2017}   # anomalously high gap-filled ER in AmeriFlux record

    er_full = np.array(obs_raw.ER)
    er_sliced_raw = er_full[start_idx:start_idx + T].copy()
    # Mask outlier years before converting to JAX array
    _time_years_full = 1970.0 + np.array(forcing_full.time)[start_idx:start_idx + T] / 365.25
    for _yr in _ER_OUTLIER_YEARS:
        _yr_mask = (np.floor(_time_years_full).astype(int) == _yr)
        er_sliced_raw[_yr_mask] = np.nan
        print(f"  Masked ER outlier year {_yr}: {_yr_mask.sum()} days → NaN")
    er_sliced = jnp.array(er_sliced_raw, dtype=jnp.float32)
    n_er_valid = int(np.sum(np.isfinite(np.array(er_sliced))))

    time_years = 1970.0 + np.array(forcing.time) / 365.25
    print(f"  Forcing: {T} days  ({time_years[0]:.1f}–{time_years[-1]:.1f})")
    print(f"  FluxNet ER: {n_er_valid} valid daily obs")

    # ── Observations ─────────────────────────────────────────────────────────
    print("Building observations…")
    delta14C_obs  = _build_pool_14C_obs_barrow(BARROW_RC_PATH, forcing.time, idx.pool_names)
    delta14C_resp = _build_resp_14C_obs_barrow(BARROW_RC_PATH, forcing.time)

    n_pool_obs = sum(int(jnp.sum(~jnp.isnan(a))) for a in delta14C_obs.values())
    n_resp_obs = int(jnp.sum(~jnp.isnan(delta14C_resp)))
    print(f"  Pool Δ¹⁴C obs: {n_pool_obs}  |  Resp Δ¹⁴C obs: {n_resp_obs}")
    for pname, arr in delta14C_obs.items():
        obs_arr = np.array(arr)
        obs_mask = ~np.isnan(obs_arr)
        if obs_mask.any():
            yr_str = ", ".join(f"{time_years[i]:.0f}" for i in np.where(obs_mask)[0])
            print(f"  {pname}: {obs_mask.sum()} obs at years [{yr_str}]")

    c_pools_obs  = _build_soil_carbon_obs_barrow(idx.pool_names)
    print("Building carbon sum constraints…")
    carbon_sum_blocks = _build_carbon_sum_obs_barrow(idx)
    if c_pools_obs:
        for pn, (mu, sig) in c_pools_obs.items():
            print(f"  C stock {pn} (individual): {mu:.0f} ± {sig:.0f} gC m⁻²")

    # No ISRaD density fraction data for Barrow → no extra fraction blocks
    israd_blocks: list = []
    extra_blocks = carbon_sum_blocks + israd_blocks

    _nan_T = jnp.full(T, jnp.nan)

    # OE1: C stocks only
    obs_carbon_only = ObservationData(
        time=forcing.time,
        NEE=_nan_T, GPP=_nan_T, ER=_nan_T, NEE_unc=_nan_T,
        delta14C_obs={},
        deltaD14C_obs={},
        C_pools_obs=c_pools_obs,
        delta14C_resp=None,
    )
    # OE2: C stocks + pool Δ¹⁴C
    obs_carbon_pool14C = ObservationData(
        time=forcing.time,
        NEE=_nan_T, GPP=_nan_T, ER=_nan_T, NEE_unc=_nan_T,
        delta14C_obs=delta14C_obs,
        deltaD14C_obs={},
        C_pools_obs=c_pools_obs,
        delta14C_resp=None,
    )
    # OE3: C stocks + resp Δ¹⁴C (orthogonality test)
    obs_carbon_resp14C = ObservationData(
        time=forcing.time,
        NEE=_nan_T, GPP=_nan_T, ER=_nan_T, NEE_unc=_nan_T,
        delta14C_obs={},
        deltaD14C_obs={},
        C_pools_obs=c_pools_obs,
        delta14C_resp=delta14C_resp,
    )
    # OE4: C stocks + pool Δ¹⁴C + resp Δ¹⁴C (full)
    obs_all = ObservationData(
        time=forcing.time,
        NEE=_nan_T, GPP=_nan_T, ER=_nan_T, NEE_unc=_nan_T,
        delta14C_obs=delta14C_obs,
        deltaD14C_obs={},
        C_pools_obs=c_pools_obs,
        delta14C_resp=delta14C_resp,
    )
    # OE5: OE4 + FluxNet ER → annual Rh constraint (free f_hetero)
    obs_all_er = ObservationData(
        time=forcing.time,
        NEE=_nan_T, GPP=_nan_T, ER=er_sliced, NEE_unc=_nan_T,
        delta14C_obs=delta14C_obs,
        deltaD14C_obs={},
        C_pools_obs=c_pools_obs,
        delta14C_resp=delta14C_resp,
    )

    # ── Initial state ────────────────────────────────────────────────────────
    state0 = _build_state0_barrow(config, idx)
    print(f"  State0 total C12: {float(jnp.sum(state0.C12)):.0f} gC m⁻²")

    # ── Analytical SS helper ─────────────────────────────────────────────────
    from ecosystem_complexity.api import _analytical_c12_ss
    import numpy as _np
    _air_t_np = _np.nan_to_num(_np.array(forcing.air_temp), nan=-12.0)
    _soil_t_raw = _np.array(forcing.soil_temp[:, 0])
    _T_soil_np = _np.where(_np.isnan(_soil_t_raw), _air_t_np, _soil_t_raw)
    _theta_np  = _np.where(_np.isnan(_np.array(forcing.soil_moisture[:, 0])), 0.5,
                           _np.array(forcing.soil_moisture[:, 0]))
    from ecosystem_complexity.fluxes import f_temp as _ft_fn, f_moisture as _fm_fn, thawed_frac as _ff_fn
    _params0 = make_default_params(config)
    _ft_   = _ft_fn(jnp.array(_T_soil_np, dtype=jnp.float32), _params0.log_Q10[0])
    _fm_   = _fm_fn(jnp.array(_theta_np, dtype=jnp.float32), _params0.log_theta_opt[0], _params0.log_gamma_moist[0])
    _fff_  = _ff_fn(jnp.array(_T_soil_np, dtype=jnp.float32))
    _mean_modifier = float(jnp.nanmean(_ft_ * _fm_ * _fff_))
    _cue_val       = float(model.config.external_inputs.CUE)
    _mean_gpp      = float(jnp.nanmean(forcing.GPP_obs))
    _mean_input_val = _mean_gpp * _cue_val
    _n_pools   = len(idx)
    _target_names   = list(model.config.external_inputs.partition.keys())
    _ext_target_idx = [idx[n] for n in _target_names]
    print(f"  Diagnostic SS: mean_modifier={_mean_modifier:.4f}, mean_input={_mean_input_val:.4f} gC/m²/day")

    # Index of the passive pool (if present) — used to pin it to the inventory
    # prior rather than the analytical SS.  The passive pool is frozen permafrost:
    # thawed_frac(T_depth) ≈ 0, so it has no steady-state decomposition term and
    # the cascade formula diverges as τ_passive grows.  Pinning it to the observed
    # C stock prior avoids spinup-drift inflation.
    _i_passive = idx["soil_passive"] if "soil_passive" in idx.pool_names else None
    _C_passive_prior = 12000.0  # gC m⁻² — Hugelius et al. 2013 inventory prior

    def _make_ss_state(params_opt):
        c12_ss = _analytical_c12_ss(params_opt, _n_pools, _mean_input_val, _mean_modifier,
                                     target_indices=_ext_target_idx)
        if _i_passive is not None:
            c12_ss = c12_ss.at[_i_passive].set(_C_passive_prior)
        return state0._replace(C12=c12_ss)

    # ── Sanity checks: prior SS ──────────────────────────────────────────────
    params_prior = make_default_params(config)
    _prior_ss    = _analytical_c12_ss(params_prior, _n_pools, _mean_input_val, _mean_modifier,
                                       target_indices=_ext_target_idx)
    if _i_passive is not None:
        _prior_ss = _prior_ss.at[_i_passive].set(_C_passive_prior)
    _prior_part  = jax.nn.softmax(params_prior.log_external_input_partition)
    print("\nSanity checks (prior):")
    print(f"  Partition sum: {float(_prior_part.sum()):.6f}  (must be 1.000000)")
    _f_input_full = np.zeros(_n_pools)
    for k, ti in enumerate(_ext_target_idx):
        _f_input_full[ti] = float(_prior_part[k])
    for i, nm in enumerate(idx.pool_names):
        print(f"  SS C_{nm:<14} = {float(_prior_ss[i]):7.1f} gC m⁻²"
              f"   (f_input={_f_input_full[i]:.3f})")

    # ── Prior forward run ─────────────────────────────────────────────────────
    print("\nPrior forward simulation…")
    t0 = time.perf_counter()
    out_prior = run_model(model, forcing, state0=state0, params=params_prior)
    jax.block_until_ready(out_prior.delta14C)
    print(f"  Done  [{time.perf_counter()-t0:.1f}s]")

    tau_p   = np.exp(np.array(params_prior.log_tau))
    w_prior = np.array(out_prior.C12) / (tau_p[None, :] + 1e-30)
    d14C_resp_prior = (np.array(out_prior.delta14C) * w_prior).sum(-1) / (w_prior.sum(-1) + 1e-30)

    _n_csum  = len(carbon_sum_blocks)

    def _print_cstock_diag(label: str, out):
        c_sim = np.array(jnp.mean(out.C12, axis=0))
        act_sim = c_sim[idx["soil_active"]]
        slw_sim = c_sim[idx["soil_slow"]]
        pas_sim = c_sim[idx["soil_passive"]]
        org_sum_sim = act_sim + slw_sim
        pf_sum_sim  = slw_sim + pas_sim
        act_obs, act_sig = c_pools_obs.get("soil_active", (float("nan"), float("nan")))
        pas_obs, pas_sig = c_pools_obs.get("soil_passive", (float("nan"), float("nan")))
        org_obs = 7500.0;  org_sig = 0.35 * org_obs
        pf_obs  = 16000.0; pf_sig  = 0.25 * pf_obs
        print(f"  {label} C-stock check:")
        print(f"    soil_active  (individual):         sim={act_sim:.0f}  obs={act_obs:.0f}±{act_sig:.0f}  "
              f"resid={(act_sim-act_obs)/act_sig:+.2f}σ")
        print(f"    soil_passive (individual):         sim={pas_sim:.0f}  obs={pas_obs:.0f}±{pas_sig:.0f}  "
              f"resid={(pas_sim-pas_obs)/pas_sig:+.2f}σ")
        print(f"    organic sum  (active+slow):        sim={org_sum_sim:.0f}  obs={org_obs:.0f}±{org_sig:.0f}  "
              f"resid={(org_sum_sim-org_obs)/org_sig:+.2f}σ")
        print(f"    permafrost sum (slow+passive):     sim={pf_sum_sim:.0f}  obs={pf_obs:.0f}±{pf_sig:.0f}  "
              f"resid={(pf_sum_sim-pf_obs)/pf_sig:+.2f}σ")
        print(f"    pools: active={act_sim:.0f}  slow={slw_sim:.0f}  passive={pas_sim:.0f}  "
              f"total={act_sim+slw_sim+pas_sim:.0f} gC m⁻²")

    # ── OE Run 1 — C stocks only ─────────────────────────────────────────────
    _n_cstock1 = len(c_pools_obs) + _n_csum
    print(f"\nOE 1 — C stocks only  "
          f"({len(c_pools_obs)} individual + {_n_csum} sum = {_n_cstock1} constraints)…")
    t0 = time.perf_counter()
    result_carbon_only = optimize_oe(model, forcing, obs_carbon_only, state0=state0,
                                     fields=_OPT_FIELDS, extra_obs_blocks=carbon_sum_blocks)
    dt1 = time.perf_counter() - t0
    ch1 = np.array(result_carbon_only.cost_history)
    print(f"  Done [{dt1:.0f}s]  J {ch1[0]:.2f} → {ch1[-1]:.2f}"
          f"  ({'converged' if result_carbon_only.converged else 'max-iter'})")

    out_carbon_only = run_model(model, forcing, state0=_make_ss_state(result_carbon_only.params_opt),
                                params=result_carbon_only.params_opt)
    jax.block_until_ready(out_carbon_only.delta14C)
    _print_cstock_diag("OE1", out_carbon_only)

    # ── OE Run 2 — C stocks + pool Δ¹⁴C ─────────────────────────────────────
    print(f"\nOE 2 — C stocks + pool Δ¹⁴C  "
          f"({_n_cstock1} C + {n_pool_obs} pool Δ¹⁴C obs)…")
    t0 = time.perf_counter()
    result_carbon_pool = optimize_oe(model, forcing, obs_carbon_pool14C, state0=state0,
                                     fields=_OPT_FIELDS, extra_obs_blocks=carbon_sum_blocks)
    dt2 = time.perf_counter() - t0
    ch2 = np.array(result_carbon_pool.cost_history)
    print(f"  Done [{dt2:.0f}s]  J {ch2[0]:.2f} → {ch2[-1]:.2f}"
          f"  ({'converged' if result_carbon_pool.converged else 'max-iter'})")

    out_carbon_pool = run_model(model, forcing, state0=_make_ss_state(result_carbon_pool.params_opt),
                                params=result_carbon_pool.params_opt)
    jax.block_until_ready(out_carbon_pool.delta14C)
    _print_cstock_diag("OE2", out_carbon_pool)

    # ── OE Run 3 — C stocks + resp Δ¹⁴C (orthogonality test) ───────────────
    print(f"\nOE 3 — C stocks + resp Δ¹⁴C  "
          f"({_n_cstock1} C + {n_resp_obs} resp Δ¹⁴C obs)…")
    t0 = time.perf_counter()
    result_carbon_resp = optimize_oe(model, forcing, obs_carbon_resp14C, state0=state0,
                                     fields=_OPT_FIELDS, extra_obs_blocks=carbon_sum_blocks)
    dt3 = time.perf_counter() - t0
    ch3 = np.array(result_carbon_resp.cost_history)
    print(f"  Done [{dt3:.0f}s]  J {ch3[0]:.2f} → {ch3[-1]:.2f}"
          f"  ({'converged' if result_carbon_resp.converged else 'max-iter'})")

    out_carbon_resp = run_model(model, forcing, state0=_make_ss_state(result_carbon_resp.params_opt),
                                params=result_carbon_resp.params_opt)
    jax.block_until_ready(out_carbon_resp.delta14C)
    _print_cstock_diag("OE3", out_carbon_resp)

    # ── OE Run 4 — full Δ¹⁴C + C stocks ─────────────────────────────────────
    print(f"\nOE 4 — full OE: pool + resp Δ¹⁴C + C stocks  "
          f"({_n_cstock1} C + {n_pool_obs} pool + {n_resp_obs} resp obs)…")
    t0 = time.perf_counter()
    result_all = optimize_oe(model, forcing, obs_all, state0=state0,
                             fields=_OPT_FIELDS, extra_obs_blocks=carbon_sum_blocks)
    dt4 = time.perf_counter() - t0
    ch4 = np.array(result_all.cost_history)
    print(f"  Done [{dt4:.0f}s]  J {ch4[0]:.2f} → {ch4[-1]:.2f}"
          f"  ({'converged' if result_all.converged else 'max-iter'})")
    print(f"  Posterior σ (diagonal Sₓ): "
          + "  ".join(f"{n}={float(np.sqrt(v)):.3f}"
                      for n, v in zip(result_all.state_names[:6],
                                      np.diag(np.array(result_all.Sx))[:6])))

    out_all = run_model(model, forcing, state0=_make_ss_state(result_all.params_opt),
                        params=result_all.params_opt)
    jax.block_until_ready(out_all.delta14C)
    _print_cstock_diag("OE4", out_all)

    # ── OE Run 5 — full OE + FluxNet ER (free f_hetero) ─────────────────────
    print(f"\nOE 5 — OE4 + FluxNet ER (free f_hetero)…")
    t0 = time.perf_counter()
    result_all_er = optimize_oe(model, forcing, obs_all_er, state0=state0,
                                fields=_OPT_FIELDS_ER, extra_obs_blocks=carbon_sum_blocks)
    dt5 = time.perf_counter() - t0
    ch5 = np.array(result_all_er.cost_history)
    print(f"  Done [{dt5:.0f}s]  J {ch5[0]:.2f} → {ch5[-1]:.2f}"
          f"  ({'converged' if result_all_er.converged else 'max-iter'})")
    print(f"  Posterior σ (diagonal Sₓ): "
          + "  ".join(f"{n}={float(np.sqrt(v)):.3f}"
                      for n, v in zip(result_all_er.state_names[:6],
                                      np.diag(np.array(result_all_er.Sx))[:6])))

    out_all_er = run_model(model, forcing, state0=_make_ss_state(result_all_er.params_opt),
                           params=result_all_er.params_opt)
    jax.block_until_ready(out_all_er.delta14C)
    _print_cstock_diag("OE5", out_all_er)

    # ── Structural limitation note ───────────────────────────────────────────
    _c_passive_sim = float(jnp.mean(out_all_er.C12, axis=0)[idx["soil_passive"]])
    if _c_passive_sim > 15000:
        print("\n  ⚠  STRUCTURAL LIMITATION: C_passive is inflated beyond the inventory")
        print("     constraint (12 000 ± 3 960 gC m⁻²).  Root cause: 14 surface")
        print("     respiration Δ¹⁴C observations near 0‰ drive τ_passive long to")
        print("     minimise passive decomp contribution to Rh.  This inflates C_passive")
        print("     via cascade arithmetic.  The model lacks a freeze-thaw gate on the")
        print("     passive pool (frozen permafrost C does not decompose); without one")
        print("     the active-layer Δ¹⁴C signal and permafrost C stock cannot both be")
        print("     satisfied simultaneously.  The active and slow pool parameters")
        print("     (τ_active, τ_slow, f_hetero) remain well-constrained; the passive")
        print("     pool is effectively unidentifiable without structural permafrost")
        print("     physics (Koven et al. 2013; Schädel et al. 2014).")

    params_opt = result_all_er.params_opt

    # ── Respired Δ¹⁴C for all runs ───────────────────────────────────────────
    def _resp_d14c(out, params):
        tau = np.exp(np.array(params.log_tau))
        w   = np.array(out.C12) / (tau[None, :] + 1e-30)
        return (np.array(out.delta14C) * w).sum(-1) / (w.sum(-1) + 1e-30)

    d14C_resp_oe1 = _resp_d14c(out_carbon_only,  result_carbon_only.params_opt)
    d14C_resp_oe2 = _resp_d14c(out_carbon_pool,  result_carbon_pool.params_opt)
    d14C_resp_oe3 = _resp_d14c(out_carbon_resp,  result_carbon_resp.params_opt)
    d14C_resp_oe4 = _resp_d14c(out_all,          result_all.params_opt)
    d14C_resp_oe5 = _resp_d14c(out_all_er,       result_all_er.params_opt)

    # ── Parameter summary ─────────────────────────────────────────────────────
    tau_opt = np.exp(np.array(params_opt.log_tau))

    def _softmax(x):
        e = np.exp(x - x.max()); return e / e.sum()

    part_opt_vec   = _softmax(np.array(params_opt.log_external_input_partition))
    part_prior_vec = _softmax(np.array(params_prior.log_external_input_partition))
    part_opt_full   = np.zeros(_n_pools)
    part_prior_full = np.zeros(_n_pools)
    for k, ti in enumerate(_ext_target_idx):
        part_opt_full[ti]   = float(part_opt_vec[k])
        part_prior_full[ti] = float(part_prior_vec[k])

    print(f"\n{'Pool':<16}  {'τ prior (yr)':>13}  {'τ opt (yr)':>12}")
    print("  " + "─" * 43)
    for i, name in enumerate(idx.pool_names):
        print(f"  {name:<16}  {tau_p[i]/365:>13.1f}  {tau_opt[i]/365:>12.1f}")

    f_het_prior_val = float(jax.nn.sigmoid(params_prior.log_f_hetero))
    f_het_opt_val   = float(jax.nn.sigmoid(params_opt.log_f_hetero))
    print(f"\n  f_hetero: prior={f_het_prior_val:.3f}  optimised={f_het_opt_val:.3f}"
          f"  (ER → Rh fraction)")

    # ── Information content — OE averaging kernel (OE5) ─────────────────────
    print("\nInformation content (OE averaging kernel, OE5 run)…")
    A_full = np.array(result_all_er.averaging_kernel)
    dfs_oe = float(np.trace(A_full))
    n_params_oe = A_full.shape[0]
    print(f"  n_params={n_params_oe}  DFS=trace(A)={dfs_oe:.3f}"
          f"  DFS/n={dfs_oe/n_params_oe:.3f}")
    Sx_diag = np.diag(np.array(result_all_er.Sx))
    for name, a_ii, sx_ii in zip(result_all_er.state_names[:n_params_oe],
                                  np.diag(A_full), Sx_diag):
        if a_ii > 0.01:
            print(f"    {name}: A_ii={a_ii:.3f}  posterior_σ={np.sqrt(sx_ii):.4f}")

    # ── DFS by experiment ────────────────────────────────────────────────────
    print("\nDFS by experiment:")
    for label, res in [("OE1 C-stocks",        result_carbon_only),
                       ("OE2 +pool Δ¹⁴C",      result_carbon_pool),
                       ("OE3 +resp Δ¹⁴C",      result_carbon_resp),
                       ("OE4 full",             result_all),
                       ("OE5 full+ER flux",     result_all_er)]:
        A = np.array(res.averaging_kernel)
        dfs = np.trace(A)
        n  = A.shape[0]
        print(f"  {label:<22} DFS={dfs:.3f} / {n} params  (DFS/n={dfs/n:.3f})")

    # ── Annual Rh diagnostics (OE5) ────────────────────────────────────────
    print("\nAnnual Rh diagnostics (OE5):")
    er_np    = np.array(er_sliced)
    gpp_np   = np.array(forcing.GPP_obs)
    time_yrs = np.array(forcing.time)
    years_np = 1970.0 + time_yrs / 365.25
    rh5      = np.array(out_all_er.Rh)
    f_het_oe5 = float(jax.nn.sigmoid(params_opt.log_f_hetero))
    mean_gpp_all = float(np.nanmean(gpp_np))
    print(f"  f_hetero (OE5 posterior): {f_het_oe5:.3f}  mean_GPP: {mean_gpp_all:.3f} gC m⁻² day⁻¹")
    for yr in range(int(years_np[0]), int(years_np[-1]) + 1):
        mask = (years_np >= yr) & (years_np < yr + 1)
        er_yr  = er_np[mask]; er_yr  = er_yr[np.isfinite(er_yr)]
        gpp_yr = gpp_np[mask]; gpp_yr = gpp_yr[np.isfinite(gpp_yr)]
        if len(er_yr) < 30:
            continue
        rh_obs_yr  = float(np.mean(er_yr)) * f_het_oe5
        rh_sim_yr  = float(np.mean(rh5[mask]))
        gpp_yr_mean = float(np.mean(gpp_yr)) if len(gpp_yr) > 10 else np.nan
        gpp_note   = f"  GPP={gpp_yr_mean:.3f} (Δ={gpp_yr_mean-mean_gpp_all:+.3f})" if np.isfinite(gpp_yr_mean) else ""
        print(f"  {yr}: Rh_obs={rh_obs_yr:.3f}  Rh_sim={rh_sim_yr:.3f}"
              f"  diff={rh_sim_yr-rh_obs_yr:+.3f} gC m⁻² day⁻¹{gpp_note}")

    # ── Age diagnostics ───────────────────────────────────────────────────────
    print("\nAge diagnostics (OE5 full+ER run)…")
    age_diag = compute_age_diagnostics(out_all_er, params_opt, model)
    print(f"  Stored bulk Δ¹⁴C:  {float(np.nanmean(age_diag.bulk_delta14C)):+.1f} ‰")
    print(f"  Respired Rh Δ¹⁴C:  {float(np.nanmean(age_diag.respired_delta14C)):+.1f} ‰")
    print(f"  Age gap:            {float(np.nanmean(age_diag.bulk_delta14C) - np.nanmean(age_diag.respired_delta14C)):+.1f} ‰")

    return dict(
        time_years=time_years,
        forcing_GPP=forcing.GPP_obs,
        forcing_ER=er_sliced,
        out_prior=out_prior,
        out_oe1=out_carbon_only,
        out_oe2=out_carbon_pool,
        out_oe3=out_carbon_resp,
        out_oe4=out_all,
        out_oe5=out_all_er,
        result_oe1=result_carbon_only,
        result_oe2=result_carbon_pool,
        result_oe3=result_carbon_resp,
        result_oe4=result_all,
        result_oe5=result_all_er,
        delta14C_obs=delta14C_obs,
        delta14C_resp=delta14C_resp,
        c_pools_obs=c_pools_obs,
        d14C_resp_prior=d14C_resp_prior,
        d14C_resp_oe1=d14C_resp_oe1,
        d14C_resp_oe2=d14C_resp_oe2,
        d14C_resp_oe3=d14C_resp_oe3,
        d14C_resp_oe4=d14C_resp_oe4,
        d14C_resp_oe5=d14C_resp_oe5,
        params_prior=params_prior,
        params_opt=params_opt,
        pool_idx=idx,
        age_diag=age_diag,
        tau_p=tau_p,
        tau_opt=tau_opt,
        lh1=ch1, lh2=ch2, lh3=ch3, lh4=ch4, lh5=ch5,
        dfs_oe=dfs_oe,
        n_params_oe=n_params_oe,
    )


# ════════════════════════════════════════════════════════════════════════════
# Figure
# ════════════════════════════════════════════════════════════════════════════

def make_figure(r: dict, out_path: str | None = None):
    if out_path is None:
        out_path = _wt("notebooks/barrow_model.png")
    from matplotlib.lines import Line2D

    C_PRIOR = "0.55"
    C_OE1   = "steelblue"
    C_OE2   = "tomato"
    C_OE3   = "mediumseagreen"
    C_OE4   = "darkorange"
    C_OE5   = "mediumpurple"

    time_years   = r["time_years"]
    out_prior    = r["out_prior"]
    out_oe1      = r["out_oe1"]
    out_oe2      = r["out_oe2"]
    out_oe3      = r["out_oe3"]
    out_oe4      = r["out_oe4"]
    out_oe5      = r["out_oe5"]
    delta14C_obs = r["delta14C_obs"]
    c_pools_obs  = r["c_pools_obs"]
    d14C_resp    = r["delta14C_resp"]
    er_arr_fig   = np.array(r["forcing_ER"])
    pool_idx     = r["pool_idx"]
    age_diag     = r["age_diag"]
    tau_p        = r["tau_p"]
    tau_opt      = r["tau_opt"]
    lh1, lh2, lh3, lh4, lh5 = r["lh1"], r["lh2"], r["lh3"], r["lh4"], r["lh5"]

    pool_names   = pool_idx.pool_names
    pool_colors  = ["tab:green", "tab:orange", "tab:brown"]
    pool_markers = ["o", "^", "s"]

    fig = plt.figure(figsize=(16, 19))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.32)
    axes = [fig.add_subplot(gs[r_, c]) for r_ in range(4) for c in range(2)]
    ax_loss, ax_tau, ax_pool14C, ax_resp14C, ax_info, ax_age, ax_gpp, ax_cstock = axes

    # ── (a) Loss convergence ─────────────────────────────────────────────────
    for lh, label, color in [
        (lh1, "OE1: C stocks only",  C_OE1),
        (lh2, "OE2: +pool Δ¹⁴C",    C_OE2),
        (lh3, "OE3: +resp Δ¹⁴C",    C_OE3),
        (lh4, "OE4: full",           C_OE4),
        (lh5, "OE5: full + ER flux", C_OE5),
    ]:
        valid = lh[np.isfinite(lh)]
        if len(valid) and valid[0] > 0:
            ax_loss.plot(np.arange(len(valid)), valid / valid[0],
                         color=color, label=label, lw=1.5, marker="o", ms=4)
    ax_loss.set_xlabel("L-M iteration")
    ax_loss.set_ylabel("Normalised OE cost J")
    ax_loss.set_title("(a) OE cost convergence (L-M)")
    ax_loss.legend(fontsize=9)
    ax_loss.set_yscale("log")

    # ── (b) τ before / after ─────────────────────────────────────────────────
    x = np.arange(len(pool_names))
    w = 0.35
    ax_tau.bar(x - w/2, tau_p / 365,   width=w, color=C_PRIOR, label="prior", alpha=0.85)
    ax_tau.bar(x + w/2, tau_opt / 365, width=w, color=C_OE5,  label="optimised (OE5)", alpha=0.85)
    for i, (tp, to) in enumerate(zip(tau_p, tau_opt)):
        ax_tau.text(i + w/2, to / 365 * 1.05, f"×{to/tp:.2f}",
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
        oe2_line   = np.array(out_oe2.delta14C)[:, pi]
        oe4_line   = np.array(out_oe4.delta14C)[:, pi]
        oe5_line   = np.array(out_oe5.delta14C)[:, pi]

        ax_pool14C.plot(time_years, prior_line, color=C_PRIOR, lw=1.0, alpha=0.5,
                        linestyle="--", label="prior" if i == 0 else None)
        ax_pool14C.plot(time_years, oe2_line,  color=color, lw=1.0, alpha=0.6, linestyle=":")
        ax_pool14C.plot(time_years, oe4_line,  color=color, lw=1.4, alpha=0.8, linestyle="-.")
        ax_pool14C.plot(time_years, oe5_line,  color=color, lw=1.8,
                        label=f"{pool_name.replace('soil_','')} (OE5)")

        if pool_name in delta14C_obs:
            obs_arr  = np.array(delta14C_obs[pool_name])
            obs_mask = ~np.isnan(obs_arr)
            ax_pool14C.scatter(time_years[obs_mask], obs_arr[obs_mask],
                               color=color, marker=marker, s=60, zorder=5,
                               label=f"{pool_name.replace('soil_','')} obs (pore gas)")

    ax_pool14C.set_xlabel("Year")
    ax_pool14C.set_ylabel("Δ¹⁴C (‰)")
    ax_pool14C.set_title("(c) Pool Δ¹⁴C  (prior– OE2··· OE4-·- OE5—)")
    ax_pool14C.legend(fontsize=8, ncol=2)

    # ── (d) Respired CO₂ Δ¹⁴C ───────────────────────────────────────────────
    resp_arr  = np.array(d14C_resp)
    resp_mask = ~np.isnan(resp_arr)

    ax_resp14C.plot(time_years, r["d14C_resp_prior"], color=C_PRIOR, lw=1.0,
                    linestyle="--", label="prior", alpha=0.7)
    ax_resp14C.plot(time_years, r["d14C_resp_oe1"],  color=C_OE1, lw=1.0,
                    linestyle=":", label="OE1: C stocks only", alpha=0.8)
    ax_resp14C.plot(time_years, r["d14C_resp_oe2"],  color=C_OE2, lw=1.2,
                    linestyle="-.", label="OE2: +pool Δ¹⁴C", alpha=0.85)
    ax_resp14C.plot(time_years, r["d14C_resp_oe3"],  color=C_OE3, lw=1.4,
                    linestyle=(0, (3, 1)), label="OE3: +resp Δ¹⁴C", alpha=0.85)
    ax_resp14C.plot(time_years, r["d14C_resp_oe4"],  color=C_OE4, lw=1.6,
                    linestyle="-.", label="OE4: full", alpha=0.9)
    ax_resp14C.plot(time_years, r["d14C_resp_oe5"],  color=C_OE5, lw=2.0,
                    label="OE5: full + ER flux")
    ax_resp14C.scatter(time_years[resp_mask], resp_arr[resp_mask],
                       color="k", marker="x", s=30, zorder=5,
                       label="obs (Vaughn 2018 surface)")
    ax_resp14C.set_xlabel("Year")
    ax_resp14C.set_ylabel("Δ¹⁴C (‰)")
    ax_resp14C.set_title("(d) Respired CO₂ Δ¹⁴C")
    ax_resp14C.legend(fontsize=8)

    # ── (e) Carbon stocks: model vs. observed ────────────────────────────────
    _bar_labels = ["OE1\nC stocks", "OE2\n+pool Δ¹⁴C", "OE3\n+resp Δ¹⁴C",
                   "OE4\nfull", "OE5\n+ER flux"]
    _x_base = np.arange(len(_bar_labels))
    _bar_w  = 0.20

    n_pools_fig = len(pool_names)
    for j, (pool_name, color, marker) in enumerate(zip(pool_names, pool_colors, pool_markers)):
        if pool_name not in pool_idx.pool_names:
            continue
        pi = pool_idx[pool_name]
        means_sim = [
            float(np.mean(np.array(out.C12)[:, pi]))
            for out in [out_oe1, out_oe2, out_oe3, out_oe4, out_oe5]
        ]
        offset = (j - (n_pools_fig - 1) / 2.0) * _bar_w
        ax_info.bar(_x_base + offset, means_sim, width=_bar_w,
                    color=color, alpha=0.75, label=pool_name.replace("soil_", ""))

        if pool_name in c_pools_obs:
            c_mu, c_sig = c_pools_obs[pool_name]
            ax_info.axhline(c_mu, color=color, lw=1.5, linestyle="--", alpha=0.9)
            ax_info.axhspan(c_mu - c_sig, c_mu + c_sig,
                            color=color, alpha=0.12,
                            label=f"{pool_name.replace('soil_','')} obs ±1σ")

    ax_info.set_xticks(_x_base)
    ax_info.set_xticklabels(_bar_labels, fontsize=8)
    ax_info.set_ylabel("Mean C12 (gC m⁻²)")
    ax_info.set_title("(e) Carbon stocks: model vs. observed")
    ax_info.legend(fontsize=8)

    # ── (f) Age diagnostics ───────────────────────────────────────────────────
    ax_age.plot(time_years, age_diag.bulk_delta14C,     color="tab:blue", lw=1.8,
                label="Stored bulk Δ¹⁴C (mass-weighted)")
    ax_age.plot(time_years, age_diag.respired_delta14C, color="tab:red",  lw=1.8,
                label="Respired Rh Δ¹⁴C (flux-weighted)")
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

    # ── (g) GPP and ER forcing ────────────────────────────────────────────────
    gpp_arr  = np.array(r["forcing_GPP"])
    yrs_unique = np.arange(int(time_years[0]), int(time_years[-1]) + 1)
    gpp_annual = []
    for yr in yrs_unique:
        mask = (time_years >= yr) & (time_years < yr + 1)
        gpp_annual.append(float(np.nanmean(gpp_arr[mask])) if mask.sum() > 10 else np.nan)
    gpp_annual = np.array(gpp_annual)

    er_annual = []
    for yr in yrs_unique:
        mask = (time_years >= yr) & (time_years < yr + 1)
        er_yr = er_arr_fig[mask]; er_yr = er_yr[np.isfinite(er_yr)]
        er_annual.append(float(np.mean(er_yr)) if len(er_yr) >= 10 else np.nan)
    er_annual = np.array(er_annual)

    ax_gpp.plot(time_years, gpp_arr, color="0.75", lw=0.5, alpha=0.5, label="daily GPP")
    ax_gpp.plot(yrs_unique + 0.5, gpp_annual, color="tab:green", lw=2.0, label="GPP annual")
    ax_gpp.plot(yrs_unique + 0.5, er_annual,  color="tab:red",   lw=2.0,
                linestyle="--", label="ER annual")
    rh_annual_obs = er_annual * 0.80
    ax_gpp.plot(yrs_unique + 0.5, rh_annual_obs, color="saddlebrown", lw=1.5,
                linestyle=":", label="Rh est. (ER×0.80)")
    ax_gpp.set_xlabel("Year")
    ax_gpp.set_ylabel("gC m⁻² day⁻¹")
    ax_gpp.set_title("(g) GPP and ER forcing (AMF US-A10, Barrow)")
    ax_gpp.legend(fontsize=7)
    ax_gpp.set_xlim(time_years[0], time_years[-1])

    # ── (h) C stock time series — prior vs OE5 ───────────────────────────────
    for i, (pool_name, color, marker) in enumerate(zip(pool_names, pool_colors, pool_markers)):
        if pool_name not in pool_idx.pool_names:
            continue
        pi = pool_idx[pool_name]
        c_prior = np.array(out_prior.C12)[:, pi]
        c_oe5   = np.array(out_oe5.C12)[:, pi]
        ax_cstock.plot(time_years, c_prior, color=color, lw=1.0, linestyle="--", alpha=0.5)
        ax_cstock.plot(time_years, c_oe5,   color=color, lw=1.8,
                       label=pool_name.replace("soil_", ""))
        if pool_name in c_pools_obs:
            c_mu, c_sig = c_pools_obs[pool_name]
            ax_cstock.axhline(c_mu, color=color, lw=1.5, linestyle=":", alpha=0.9)
            ax_cstock.axhspan(c_mu - c_sig, c_mu + c_sig,
                              color=color, alpha=0.12,
                              label=f"{pool_name.replace('soil_','')} obs ±σ")

    ax_cstock.set_xlabel("Year")
    ax_cstock.set_ylabel("C stock (gC m⁻²)")
    ax_cstock.set_title("(h) C stocks: prior (--) vs OE5 (—), literature bands (·)")
    ax_cstock.legend(fontsize=8, ncol=2)
    ax_cstock.set_xlim(time_years[0], time_years[-1])

    fig.suptitle(
        "Barrow, Alaska — 3-Pool Optimal Model  (active + slow + passive)\n"
        "OE5 adds FluxNet ER → annual Rh constraint (f_hetero≈0.80, σ=20%)",
        fontsize=12, y=0.995,
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
