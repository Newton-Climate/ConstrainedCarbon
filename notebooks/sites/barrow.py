"""
sites/barrow.py — Barrow, Alaska (US-A10) site module.

Contains all Barrow-specific data helpers, the canonical Barrow OE5
inversion workflow, and the site figure.  Imported by
``notebooks/barrow_model.py``.

Public API
----------
  run_optimal_inversion() → dict   run OE1–OE5, return results
  make_figure(r, out_path)         save 8-panel figure
"""
from __future__ import annotations

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# ── Path resolution ───────────────────────────────────────────────────────────
_SITES_ROOT    = os.path.dirname(os.path.abspath(__file__))
_NB_ROOT       = os.path.dirname(_SITES_ROOT)
_WORKTREE_ROOT = os.path.dirname(_NB_ROOT)
_SRC_ROOT      = os.path.join(_WORKTREE_ROOT, "src")

if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from notebook_utils import (
    find_data_root, softmax_np,
    build_mean_ss_modifier,
    build_oe5_obs_sets,
    run_oe5_sequence,
    print_prior_ss_check,
    print_dfs_table,
    print_tau_summary,
    print_rh_diagnostics,
    print_averaging_kernel,
    make_oe5_figure,
)

from ecosystem_complexity.api import (
    build_model, run_model, optimize_oe, OEResult, _get_oe_fields,
    ObsBlock, _analytical_c12_ss,
)
from ecosystem_complexity.config import load_config
from ecosystem_complexity.data.parsers import attach_atm14C, load_barrow_alaska, slice_forcing
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.schemas import ForcingData, ObservationData
from ecosystem_complexity.data.israd_observations import build_perlayer_mixture_obs_blocks
from ecosystem_complexity.state import make_default_params
from ecosystem_complexity.analysis import compute_age_diagnostics, compute_resp_delta14C

_REPO_ROOT = (
    os.environ.get("ECOSYSTEM_REPO_ROOT")
    or find_data_root(_WORKTREE_ROOT)
)


def _wt(rel: str) -> str:
    return os.path.join(_WORKTREE_ROOT, rel)


def _data(rel: str) -> str:
    return os.path.join(_REPO_ROOT, rel)


# ── File paths ────────────────────────────────────────────────────────────────
_BARROW_DIR         = _data("data/barrow_alaska/AMF_US-A10_FLUXNET_2011-2022_v1.3_r1")
BARROW_ERA5_PATH    = os.path.join(_BARROW_DIR, "AMF_US-A10_FLUXNET_ERA5_DD_1981-2025_v1.3_r1.csv")
BARROW_FLUXMET_PATH = os.path.join(_BARROW_DIR, "AMF_US-A10_FLUXNET_FLUXMET_DD_2011-2022_v1.3_r1.csv")
_FIELD_DIR          = _data("data/barrow_alaska/barrow_field_data/data")
BARROW_RC_PATH      = os.path.join(_FIELD_DIR, "radiocarbon_field_Barrow_2012_2013_2014_v2.csv")
OPT_CONFIG          = _wt("configs/barrow_3pool_config.yaml")
HUA_PATH            = _data("data/shared/atm_14C/Hua_2021.csv")
GRAVEN_PATH         = _data("data/shared/atm_14C/Graven_2017.csv")
INTCAL_PATH         = _data("data/shared/atm_14C/intcal20.14c")

_R_STD = 1.176e-12

# Depth-to-pool mapping for legacy Vaughn pore-gas Δ¹⁴C
_POREGAS_DEPTH_TO_POOL = {
    10: "soil_active",
    20: "soil_slow",
    29: "soil_passive",
    31: "soil_passive",
}

# ISRaD path
ISRAD_LAYER_PATH = _data("data/shared/israd/ISRaD_extra_flat_layer_v 2.6.6.2024-01-25.csv")

# Vaughn 2018 profiles (BEO, 0.1–0.4 km from flux tower) plus Nave 2021
# deep permafrost layers for the passive pool.
_BARROW_ISRAD_PROFILES = [
    "A3C", "A4C", "B3C", "C1C", "H3C", "TC",
    "Z210C", "Z415C", "Z53C",
    "BARR_1",
]

# Non-overlapping depth bins that match the Barrow 3-pool YAML boundaries.
_BARROW_ISRAD_DEPTH_TO_POOL = [
    ("soil_active",   (-10.0,  14.0)),
    ("soil_slow",     ( 14.0,  30.0)),
    ("soil_passive",  ( 30.0, 200.0)),
]

_BARROW_ISRAD_OBS_YEAR = 2014.0


# ════════════════════════════════════════════════════════════════════════════
# Barrow-specific date decoder
# ════════════════════════════════════════════════════════════════════════════

def _decode_vaughn_date(s: str) -> pd.Timestamp:
    """
    Decode the idiosyncratic Vaughn field CSV date encoding.

    '10/6/12' (Oct 6, 2012) is stored as '2010-06-12' (month→year, day→month,
    YY→day).  Rows already in M/D/YY format are parsed directly.
    """
    s = str(s).strip()
    if "/" in s:
        try:
            return pd.to_datetime(s, format="%m/%d/%y")
        except ValueError:
            pass
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
# Barrow data helpers
# ════════════════════════════════════════════════════════════════════════════

def build_pool_14C_obs_legacy_poregas(rc_path: str, forcing_time, pool_names: list) -> dict:
    """
    Build legacy pool Δ¹⁴C observations from Vaughn pore-gas CO₂ at depth.

    Depth → pool mapping:  10 cm → active,  20 cm → slow,  29–31 cm → passive.
    Multiple measurements at the same depth and year are averaged.
    """
    df = pd.read_csv(rc_path, skiprows=6)
    df.columns = df.columns.str.strip()

    pore = df[df["material"].str.strip().str.lower() == "soil pore gas"].copy()
    pore["date_decoded"] = pore["observation_date"].apply(_decode_vaughn_date)
    pore = pore.dropna(subset=["date_decoded"])
    pore["depth_cm"] = pd.to_numeric(pore["depth"], errors="coerce")

    time_np  = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T        = len(time_np)
    pool_name_set = set(pool_names)
    obs_dict: dict[str, np.ndarray] = {}

    for depth, pool_name in _POREGAS_DEPTH_TO_POOL.items():
        if pool_name not in pool_name_set:
            continue
        depth_rows = pore[pore["depth_cm"] == depth]
        if depth_rows.empty:
            continue
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


def build_resp_14C_obs(rc_path: str, forcing_time) -> jnp.ndarray:
    """
    Build respired CO₂ Δ¹⁴C from Vaughn surface emission measurements.

    Multiple same-day measurements are averaged.  Returns shape (T,) with NaN gaps.
    """
    df = pd.read_csv(rc_path, skiprows=6)
    df.columns = df.columns.str.strip()
    surf = df[df["material"].str.strip().str.lower() == "surface emissions"].copy()
    surf["date_decoded"] = surf["observation_date"].apply(_decode_vaughn_date)
    surf = surf.dropna(subset=["date_decoded"])

    time_np  = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T        = len(time_np)
    arr      = np.full(T, np.nan, dtype=np.float32)

    surf["decimal_year"] = (surf["date_decoded"].dt.year
                            + (surf["date_decoded"].dt.dayofyear - 1) / 365.0)
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


def _build_pool_14C_obs_israd_summary(
    pool_names: list,
    israd_layer_path: str = ISRAD_LAYER_PATH,
    print_summary: bool = True,
) -> dict[str, tuple[float, float, int, int]]:
    """Return canonical Barrow ISRaD pool Δ¹⁴C summaries by model pool."""
    df = pd.read_csv(israd_layer_path, low_memory=False)
    df = df[df["pro_name"].isin(_BARROW_ISRAD_PROFILES)].copy()
    df = df.dropna(subset=["lyr_14c", "lyr_top", "lyr_bot"])
    df["lyr_mid"] = 0.5 * (df["lyr_top"].astype(float) + df["lyr_bot"].astype(float))

    pool_name_set = set(pool_names)
    pool_obs: dict[str, tuple[float, float, int, int]] = {}
    for pool_name, (z_top, z_bot) in _BARROW_ISRAD_DEPTH_TO_POOL:
        if pool_name not in pool_name_set:
            continue
        mask = (df["lyr_mid"] >= z_top) & (df["lyr_mid"] < z_bot)
        vals = df.loc[mask, "lyr_14c"].astype(float).values
        if len(vals) < 2:
            continue
        mean_val = float(np.mean(vals))
        sigma_val = max(float(np.std(vals, ddof=1)), 25.0)
        n_profiles = int(df.loc[mask, "pro_name"].nunique())
        pool_obs[pool_name] = (mean_val, sigma_val, len(vals), n_profiles)
        if print_summary:
            print(f"  Barrow ISRaD {pool_name:<14s} ({z_top:.0f}–{z_bot:.0f} cm): "
                  f"Δ¹⁴C = {mean_val:+7.1f} ± {sigma_val:5.1f} ‰  "
                  f"(n={len(vals)} obs across {n_profiles} profiles)")
    return pool_obs


def build_pool_14C_obs(
    forcing_time,
    pool_names: list,
    israd_layer_path: str = ISRAD_LAYER_PATH,
    print_summary: bool = True,
) -> dict:
    """Build canonical Barrow pool Δ¹⁴C observations from ISRaD bulk layers."""
    pool_obs = _build_pool_14C_obs_israd_summary(pool_names, israd_layer_path, print_summary)
    time_np = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T = len(time_np)

    obs_dict: dict[str, jnp.ndarray] = {}
    for pool_name, (mean_val, _, _, _) in pool_obs.items():
        arr = np.full(T, np.nan, dtype=np.float32)
        t_idx = int(np.argmin(np.abs(years_np - (_BARROW_ISRAD_OBS_YEAR + 0.5))))
        arr[t_idx] = mean_val
        obs_dict[pool_name] = jnp.array(arr)
    return obs_dict


def build_pool_14C_obs_blocks(
    pool_index,
    forcing_time,
    israd_layer_path: str = ISRAD_LAYER_PATH,
    print_summary: bool = True,
) -> tuple[list, dict[str, float]]:
    """Build canonical ObsBlocks for Barrow ISRaD bulk-layer pool Δ¹⁴C."""
    pool_obs = _build_pool_14C_obs_israd_summary(pool_index.pool_names, israd_layer_path, print_summary)
    time_np = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    t_yr = jnp.array(
        np.where((years_np >= _BARROW_ISRAD_OBS_YEAR) & (years_np < _BARROW_ISRAD_OBS_YEAR + 1))[0],
        dtype=jnp.int32,
    )
    if t_yr.shape[0] == 0:
        t_yr = jnp.array(
            [int(np.argmin(np.abs(years_np - (_BARROW_ISRAD_OBS_YEAR + 0.5))))],
            dtype=jnp.int32,
        )

    blocks = []
    sigma_dict: dict[str, float] = {}
    for pool_name, (mean_val, sigma_val, _, _) in pool_obs.items():
        pi = int(pool_index[pool_name])
        sigma_dict[pool_name] = sigma_val
        blocks.append(ObsBlock(
            name=f"israd_layer_{pool_name}",
            y=jnp.array([mean_val], dtype=jnp.float32),
            Se=jnp.array([sigma_val ** 2], dtype=jnp.float32),
            predict=lambda out, p, t=t_yr, pi=pi:
                jnp.mean(out.delta14C[t, pi], keepdims=True),
        ))
    return blocks, sigma_dict


def build_barrow_bulk_perlayer_mixture_14C_obs(
    forcing_time,
    pool_index,
    israd_layer_path: str = ISRAD_LAYER_PATH,
    print_summary: bool = True,
) -> list:
    """Per-layer carbon-mass-weighted mixture bulk Δ¹⁴C for Barrow.

    Thin site adapter around the generalized
    :func:`ecosystem_complexity.data.israd_observations.build_perlayer_mixture_obs_blocks`.
    Barrow has no ISRaD density fractions, so every depth bin uses the model
    C12-partition fallback weighting (``fraction_df=None``) — demonstrating the
    operator works for sites without fraction data, not only Harvard Forest.
    """
    if not os.path.isfile(israd_layer_path):
        print(f"  ISRaD layer file not found: {israd_layer_path} — skipping")
        return []

    lay = pd.read_csv(israd_layer_path, low_memory=False)
    lay = lay[lay["pro_name"].isin(_BARROW_ISRAD_PROFILES)]

    rows = build_perlayer_mixture_obs_blocks(
        lay, pool_index, forcing_time,
        depth_bins=_BARROW_ISRAD_DEPTH_TO_POOL,
        obs_year=_BARROW_ISRAD_OBS_YEAR,
        fraction_df=None,
        pool_order=("soil_active", "soil_slow", "soil_passive"),
        sigma_floor=25.0,
        name_prefix="israd_perlayer",
    )
    if print_summary:
        for r in rows:
            z0, z1 = r["depth_cm"]
            print(
                f"  Barrow per-layer bulk Δ¹⁴C {r['bin_label']:<14s} "
                f"({z0:.0f}–{z1:.0f} cm): {r['mean']:+7.1f} ± {r['sigma']:5.1f}‰  "
                f"weights={r['weight_source']}  (n={r['n']} layers)"
            )
    return [r["block"] for r in rows]


def build_pool_14C_obs_israd(
    israd_layer_path: str,
    forcing_time,
    pool_names: list,
    pro_name: str | None = None,
) -> tuple[dict, dict]:
    """Compatibility wrapper for the canonical all-profile ISRaD builder."""
    if pro_name not in (None, "all plots"):
        print(f"  NOTE: ignoring deprecated Barrow single-plot request ({pro_name}); "
              "using canonical all-profile ISRaD aggregate.")
    obs_dict = build_pool_14C_obs(forcing_time, pool_names, israd_layer_path, print_summary=True)
    pool_obs = _build_pool_14C_obs_israd_summary(pool_names, israd_layer_path, print_summary=False)
    sigma_dict = {pool_name: sigma for pool_name, (_, sigma, _, _) in pool_obs.items()}
    return obs_dict, sigma_dict


def build_israd_14c_blocks(
    obs_dict: dict,
    sigma_dict: dict,
    forcing_time,
    pool_index,
) -> list:
    """Compatibility wrapper around the canonical ISRaD ObsBlock builder."""
    del obs_dict, sigma_dict
    blocks, _ = build_pool_14C_obs_blocks(pool_index, forcing_time)
    return blocks


def build_soil_carbon_obs(pool_names: list) -> dict:
    """
    Literature-based individual C stock constraints for Barrow.

    soil_active  (0–14 cm):    4500 gC m⁻² ± 45%   (Ping et al. 2008)
    soil_passive (30–100 cm): 12000 gC m⁻² ± 33%   (Hugelius et al. 2014)
    soil_slow is handled via sum constraints.
    """
    pool_name_set = set(pool_names)
    result = {}
    if "soil_active"  in pool_name_set:
        result["soil_active"]  = (4500.0,  0.45 * 4500.0)
    if "soil_passive" in pool_name_set:
        result["soil_passive"] = (12000.0, 0.33 * 12000.0)
    return result


def build_carbon_sum_obs(pool_index) -> list:
    """
    Two sum-constraint ObsBlocks for Barrow:
      organic sum  (active + slow)   ≈ 7500 ± 2625 gC m⁻²  (Ping et al. 2008)
      permafrost sum (slow + passive) ≈ 16000 ± 4000 gC m⁻²  (Hugelius et al. 2014)
    """
    pool_names_set = set(pool_index.pool_names)
    blocks = []

    if "soil_active" in pool_names_set and "soil_slow" in pool_names_set:
        org_mean  = 7500.0
        org_sigma = 0.35 * org_mean
        cols = jnp.array([pool_index["soil_active"], pool_index["soil_slow"]],
                         dtype=jnp.int32)
        blocks.append(ObsBlock(
            name="c_sum_organic",
            y=jnp.array([org_mean], dtype=jnp.float32),
            Se=jnp.array([org_sigma ** 2], dtype=jnp.float32),
            predict=lambda out, p, c=cols:
                jnp.sum(jnp.mean(out.C12, axis=0)[c], keepdims=True),
        ))
        print(f"  C stock organic sum  (active+slow):    {org_mean:.0f} ± {org_sigma:.0f} gC m⁻²")

    if "soil_slow" in pool_names_set and "soil_passive" in pool_names_set:
        pf_mean  = 16000.0
        pf_sigma = 0.25 * pf_mean
        cols = jnp.array([pool_index["soil_slow"], pool_index["soil_passive"]],
                         dtype=jnp.int32)
        blocks.append(ObsBlock(
            name="c_sum_permafrost",
            y=jnp.array([pf_mean], dtype=jnp.float32),
            Se=jnp.array([pf_sigma ** 2], dtype=jnp.float32),
            predict=lambda out, p, c=cols:
                jnp.sum(jnp.mean(out.C12, axis=0)[c], keepdims=True),
        ))
        print(f"  C stock permafrost sum (slow+passive): {pf_mean:.0f} ± {pf_sigma:.0f} gC m⁻²")

    return blocks


def build_state0(config, pool_index):
    """
    Initialise model state for Barrow from literature C stocks and
    2012 Vaughn pore-gas Δ¹⁴C.

    C12:   active=4500,  slow=3000,  passive=12000 gC m⁻²
    Δ¹⁴C:  active=+30‰,  slow=+15‰,  passive=−80‰
    """
    from ecosystem_complexity.state import make_initial_state

    C12_by_pool = {"soil_active": 4500.0, "soil_slow": 3000.0, "soil_passive": 12000.0}
    d14C_by_pool = {"soil_active": 30.0,  "soil_slow": 15.0,   "soil_passive": -80.0}

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
# Main inversion workflow
# ════════════════════════════════════════════════════════════════════════════

def run_optimal_inversion() -> dict:
    """
    Run the canonical Barrow OE1–OE5 inversion sequence.

    Returns a results dict compatible with ``make_figure()``.
    """
    # ── Atmospheric ¹⁴C record ───────────────────────────────────────────────
    print("Loading atmospheric ¹⁴C record…")
    atm14C = load_full_14C_record(
        hua_path=HUA_PATH, graven_path=GRAVEN_PATH, intcal_path=INTCAL_PATH,
        hemisphere="NH", start_year=1500.0, end_year=2025.0,
    )

    # ── Build model ──────────────────────────────────────────────────────────
    print("Building 3-pool model…")
    model  = build_model(OPT_CONFIG)
    config = model.config
    idx    = model.pool_index
    print(f"  Pools: {idx.pool_names}")

    inv_cfg      = getattr(config, "inversion_raw", {}) or {}
    oe_fields    = _get_oe_fields(config, inv_cfg)
    oe_fields_er = oe_fields + ("log_f_hetero",)
    print(f"  OE fields (OE1–4): {oe_fields}")
    print(f"  OE fields (OE5):   {oe_fields_er}")

    # ── Forcing ──────────────────────────────────────────────────────────────
    print("Loading flux forcing (ERA5_DD + FLUXMET_DD)…")
    forcing_full, obs_raw = load_barrow_alaska(
        era5_path=BARROW_ERA5_PATH,
        fluxmet_path=BARROW_FLUXMET_PATH,
        config=config,
        qc_threshold=0.0,
        include_gpp_forcing=True,
    )
    years_daily, d14c_daily = atm14C
    forcing_full = attach_atm14C(forcing_full, d14c_daily, years_daily)

    time_np   = np.array(forcing_full.time)
    years_all = 1970.0 + time_np / 365.25
    start_idx = int(np.searchsorted(years_all, 2011.0))
    forcing   = slice_forcing(forcing_full, start_idx, len(time_np))
    T         = int(forcing.time.shape[0])

    # Mask outlier ER years
    _ER_OUTLIER_YEARS = {2017}
    er_sliced_raw    = np.array(obs_raw.ER)[start_idx:start_idx + T].copy()
    _time_years_full = 1970.0 + np.array(forcing_full.time)[start_idx:start_idx + T] / 365.25
    for _yr in _ER_OUTLIER_YEARS:
        _mask = (np.floor(_time_years_full).astype(int) == _yr)
        er_sliced_raw[_mask] = np.nan
        print(f"  Masked ER outlier year {_yr}: {_mask.sum()} days → NaN")
    er_sliced  = jnp.array(er_sliced_raw, dtype=jnp.float32)
    n_er_valid = int(np.sum(np.isfinite(np.array(er_sliced))))

    time_years = 1970.0 + np.array(forcing.time) / 365.25
    print(f"  Forcing: {T} days  ({time_years[0]:.1f}–{time_years[-1]:.1f})")
    print(f"  FluxNet ER: {n_er_valid} valid daily obs")

    # ── Observations ─────────────────────────────────────────────────────────
    print("Building observations…")
    print("  Pool Δ¹⁴C source: ISRaD bulk soil layers (Vaughn 2018 + Nave 2021)")
    delta14C_obs  = build_pool_14C_obs(forcing.time, idx.pool_names, print_summary=True)
    israd_blocks, israd_sigma_dict = build_pool_14C_obs_blocks(idx, forcing.time, print_summary=False)
    delta14C_resp = build_resp_14C_obs(BARROW_RC_PATH, forcing.time)
    c_pools_obs   = build_soil_carbon_obs(idx.pool_names)

    n_pool_obs = len(israd_blocks)
    n_resp_obs = int(jnp.sum(~jnp.isnan(delta14C_resp)))
    print(f"  Pool Δ¹⁴C blocks: {n_pool_obs}  |  Resp Δ¹⁴C obs: {n_resp_obs}")
    for pname, arr in delta14C_obs.items():
        obs_arr  = np.array(arr)
        obs_mask = ~np.isnan(obs_arr)
        if obs_mask.any():
            yr_str = ", ".join(f"{time_years[i]:.0f}" for i in np.where(obs_mask)[0])
            print(f"  {pname}: {obs_mask.sum()} obs at years [{yr_str}]")

    print("  Omitting legacy carbon-sum constraints; keeping individual active/passive stocks only.")
    carbon_sum_blocks: list = []
    if c_pools_obs:
        for pn, (mu, sig) in c_pools_obs.items():
            print(f"  C stock {pn} (individual): {mu:.0f} ± {sig:.0f} gC m⁻²")

    extra_blocks: list = list(israd_blocks)

    # ── Initial state ─────────────────────────────────────────────────────────
    state0 = build_state0(config, idx)
    print(f"  State0 total C12: {float(jnp.sum(state0.C12)):.0f} gC m⁻²")

    # ── Analytical SS helper ──────────────────────────────────────────────────
    mean_modifier, mean_input_val = build_mean_ss_modifier(
        forcing, model, t_nan_fill=-12.0, theta_nan_fill=0.5
    )
    print(f"  Diagnostic SS: mean_modifier={mean_modifier:.4f},"
          f" mean_input={mean_input_val:.4f} gC/m²/day")

    params_prior   = make_default_params(config)
    n_pools        = len(idx)
    target_names   = list(config.external_inputs.partition.keys())
    ext_target_idx = [idx[n] for n in target_names]

    # Barrow: passive pool is frozen — pin its C12 to the inventory prior
    # instead of using the analytical SS formula (which diverges for frozen pools).
    _i_passive      = idx["soil_passive"] if "soil_passive" in idx.pool_names else None
    _C_passive_prior = 12000.0  # gC m⁻² — Hugelius et al. 2013

    prior_ss = _analytical_c12_ss(params_prior, n_pools, mean_input_val,
                                   mean_modifier, target_indices=ext_target_idx)
    if _i_passive is not None:
        prior_ss = prior_ss.at[_i_passive].set(_C_passive_prior)
    print_prior_ss_check(prior_ss, params_prior, idx, ext_target_idx)

    def _make_ss_state(params_opt):
        c12_ss = _analytical_c12_ss(params_opt, n_pools, mean_input_val,
                                     mean_modifier, target_indices=ext_target_idx)
        if _i_passive is not None:
            c12_ss = c12_ss.at[_i_passive].set(_C_passive_prior)
        return state0._replace(C12=c12_ss)

    # ── Prior forward run ─────────────────────────────────────────────────────
    print("\nPrior forward simulation…")
    t0 = time.perf_counter()
    out_prior = run_model(model, forcing, state0=state0, params=params_prior)
    jax.block_until_ready(out_prior.delta14C)
    print(f"  Done  [{time.perf_counter()-t0:.1f}s]")

    tau_p = np.exp(np.array(params_prior.log_tau))
    d14C_resp_prior = compute_resp_delta14C(out_prior, params_prior)

    # ── C-stock diagnostic (Barrow-specific constraints) ──────────────────────
    def _print_cstock_diag(label: str, out):
        c_sim       = np.array(jnp.mean(out.C12, axis=0))
        act_sim     = c_sim[idx["soil_active"]]
        slw_sim     = c_sim[idx["soil_slow"]]
        pas_sim     = c_sim[idx["soil_passive"]]
        org_sum_sim = act_sim + slw_sim
        pf_sum_sim  = slw_sim + pas_sim
        act_obs, act_sig = c_pools_obs.get("soil_active",  (float("nan"), float("nan")))
        pas_obs, pas_sig = c_pools_obs.get("soil_passive", (float("nan"), float("nan")))
        org_obs = 7500.0;  org_sig = 0.35 * org_obs
        pf_obs  = 16000.0; pf_sig  = 0.25 * pf_obs
        print(f"  {label} C-stock check:")
        print(f"    soil_active  (individual):         "
              f"sim={act_sim:.0f}  obs={act_obs:.0f}±{act_sig:.0f}  "
              f"resid={(act_sim-act_obs)/act_sig:+.2f}σ")
        print(f"    soil_passive (individual):         "
              f"sim={pas_sim:.0f}  obs={pas_obs:.0f}±{pas_sig:.0f}  "
              f"resid={(pas_sim-pas_obs)/pas_sig:+.2f}σ")
        print(f"    organic sum  (active+slow):        "
              f"sim={org_sum_sim:.0f}  obs={org_obs:.0f}±{org_sig:.0f}  "
              f"resid={(org_sum_sim-org_obs)/org_sig:+.2f}σ")
        print(f"    permafrost sum (slow+passive):     "
              f"sim={pf_sum_sim:.0f}  obs={pf_obs:.0f}±{pf_sig:.0f}  "
              f"resid={(pf_sum_sim-pf_obs)/pf_sig:+.2f}σ")
        print(f"    pools: active={act_sim:.0f}  slow={slw_sim:.0f}  passive={pas_sim:.0f}  "
              f"total={act_sim+slw_sim+pas_sim:.0f} gC m⁻²")

    # ── OE1–OE5 sequence ─────────────────────────────────────────────────────
    obs_sets = build_oe5_obs_sets(
        forcing,
        delta14C_obs={},
        delta14C_resp=delta14C_resp,
        c_pools_obs=c_pools_obs,
        er_sliced=er_sliced,
    )
    seq = run_oe5_sequence(
        model=model, forcing=forcing, state0=state0,
        obs_sets=obs_sets,
        carbon_sum_blocks=carbon_sum_blocks,
        extra_blocks=extra_blocks,
        oe_fields=oe_fields, oe_fields_er=oe_fields_er,
        make_ss_state_fn=_make_ss_state,
        print_cstock_diag_fn=_print_cstock_diag,
        n_pool_obs=n_pool_obs, n_resp_obs=n_resp_obs,
    )

    params_opt = seq["params_opt"]

    # ── Passive pool structural limitation check ──────────────────────────────
    _c_passive_sim = float(jnp.mean(seq["out_oe5"].C12, axis=0)[idx["soil_passive"]])
    if _c_passive_sim > 15000:
        print("\n  ⚠  STRUCTURAL LIMITATION: C_passive is inflated beyond the inventory"
              "\n     constraint (12 000 ± 3 960 gC m⁻²).  Root cause: 14 surface"
              "\n     respiration Δ¹⁴C obs near 0‰ drive τ_passive long to minimise"
              "\n     passive decomp contribution to Rh (cascade inflation).")

    # ── Respired Δ¹⁴C for all runs ───────────────────────────────────────────
    d14C_resp_oe1 = compute_resp_delta14C(seq["out_oe1"], seq["result_oe1"].params_opt)
    d14C_resp_oe2 = compute_resp_delta14C(seq["out_oe2"], seq["result_oe2"].params_opt)
    d14C_resp_oe3 = compute_resp_delta14C(seq["out_oe3"], seq["result_oe3"].params_opt)
    d14C_resp_oe4 = compute_resp_delta14C(seq["out_oe4"], seq["result_oe4"].params_opt)
    d14C_resp_oe5 = compute_resp_delta14C(seq["out_oe5"], seq["result_oe5"].params_opt)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    tau_p, tau_opt = print_tau_summary(params_prior, params_opt, idx.pool_names)
    dfs_oe, n_params_oe = print_averaging_kernel(seq["result_oe5"])
    print_dfs_table(seq)
    print_rh_diagnostics(seq["out_oe5"], forcing, er_sliced, params_opt)

    print("\nAge diagnostics (OE5 full+ER run)…")
    age_diag = compute_age_diagnostics(seq["out_oe5"], params_opt, model)
    bulk_d14C_mean = float(np.nanmean(age_diag.bulk_delta14C))
    resp_d14C_mean = float(np.nanmean(age_diag.respired_delta14C))
    print(f"  Stored bulk Δ¹⁴C:  {bulk_d14C_mean:+.1f} ‰")
    print(f"  Respired Rh Δ¹⁴C:  {resp_d14C_mean:+.1f} ‰")
    print(f"  Age gap:            {bulk_d14C_mean - resp_d14C_mean:+.1f} ‰")

    return dict(
        time_years=time_years,
        forcing_GPP=forcing.GPP_obs,
        forcing_ER=er_sliced,
        out_prior=out_prior,
        **{k: seq[k] for k in ("out_oe1","out_oe2","out_oe3","out_oe4","out_oe5",
                                "result_oe1","result_oe2","result_oe3","result_oe4","result_oe5",
                                "lh1","lh2","lh3","lh4","lh5")},
        delta14C_obs=delta14C_obs,
        israd_sigma_dict=israd_sigma_dict,
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
        dfs_oe=dfs_oe,
        n_params_oe=n_params_oe,
    )


# ════════════════════════════════════════════════════════════════════════════
# Legacy pore-gas workflow
# ════════════════════════════════════════════════════════════════════════════

def run_optimal_inversion_legacy_poregas() -> dict:
    """
    Run the pre-canonical Barrow OE1–OE5 workflow with pore-gas pool Δ¹⁴C.

    This remains only as an explicitly named legacy path.
    """
    # ── Atmospheric ¹⁴C record ───────────────────────────────────────────────
    print("Loading atmospheric ¹⁴C record…")
    atm14C = load_full_14C_record(
        hua_path=HUA_PATH, graven_path=GRAVEN_PATH, intcal_path=INTCAL_PATH,
        hemisphere="NH", start_year=1500.0, end_year=2025.0,
    )

    # ── Build model ──────────────────────────────────────────────────────────
    print("Building 3-pool model…")
    model  = build_model(OPT_CONFIG)
    config = model.config
    idx    = model.pool_index
    print(f"  Pools: {idx.pool_names}")

    inv_cfg      = getattr(config, "inversion_raw", {}) or {}
    oe_fields    = _get_oe_fields(config, inv_cfg)
    oe_fields_er = oe_fields + ("log_f_hetero",)

    # ── Forcing ──────────────────────────────────────────────────────────────
    print("Loading flux forcing…")
    forcing_full, obs_raw = load_barrow_alaska(
        era5_path=BARROW_ERA5_PATH,
        fluxmet_path=BARROW_FLUXMET_PATH,
        config=config,
        qc_threshold=0.0,
        include_gpp_forcing=True,
    )
    years_daily, d14c_daily = atm14C
    forcing_full = attach_atm14C(forcing_full, d14c_daily, years_daily)

    time_np   = np.array(forcing_full.time)
    years_all = 1970.0 + time_np / 365.25
    start_idx = int(np.searchsorted(years_all, 2011.0))
    forcing   = slice_forcing(forcing_full, start_idx, len(time_np))
    T         = int(forcing.time.shape[0])

    _ER_OUTLIER_YEARS = {2017}
    er_sliced_raw    = np.array(obs_raw.ER)[start_idx:start_idx + T].copy()
    _time_years_full = 1970.0 + np.array(forcing_full.time)[start_idx:start_idx + T] / 365.25
    for _yr in _ER_OUTLIER_YEARS:
        _mask = (np.floor(_time_years_full).astype(int) == _yr)
        er_sliced_raw[_mask] = np.nan
        print(f"  Masked ER outlier year {_yr}: {_mask.sum()} days → NaN")
    er_sliced  = jnp.array(er_sliced_raw, dtype=jnp.float32)
    n_er_valid = int(np.sum(np.isfinite(np.array(er_sliced))))

    time_years = 1970.0 + np.array(forcing.time) / 365.25
    print(f"  Forcing: {T} days  ({time_years[0]:.1f}–{time_years[-1]:.1f})")
    print(f"  FluxNet ER: {n_er_valid} valid daily obs")

    print("\nBuilding legacy Vaughn pore-gas pool Δ¹⁴C observations…")
    delta14C_obs = build_pool_14C_obs_legacy_poregas(BARROW_RC_PATH, forcing.time, idx.pool_names)
    n_pool_obs = sum(int(jnp.sum(~jnp.isnan(a))) for a in delta14C_obs.values())

    # Surface respiration Δ¹⁴C (pore-gas file, unchanged)
    delta14C_resp = build_resp_14C_obs(BARROW_RC_PATH, forcing.time)
    n_resp_obs    = int(jnp.sum(~jnp.isnan(delta14C_resp)))
    print(f"  Pool Δ¹⁴C obs: {n_pool_obs}  |  Resp Δ¹⁴C obs: {n_resp_obs}")

    # ── C-stock constraints (same as standard run) ───────────────────────────
    c_pools_obs = build_soil_carbon_obs(idx.pool_names)
    print("Building carbon sum constraints…")
    carbon_sum_blocks = build_carbon_sum_obs(idx)
    for pn, (mu, sig) in c_pools_obs.items():
        print(f"  C stock {pn} (individual): {mu:.0f} ± {sig:.0f} gC m⁻²")

    extra_blocks = carbon_sum_blocks

    # ── Initial state ─────────────────────────────────────────────────────────
    state0 = build_state0(config, idx)
    print(f"  State0 total C12: {float(jnp.sum(state0.C12)):.0f} gC m⁻²")

    # ── Analytical SS helper ──────────────────────────────────────────────────
    mean_modifier, mean_input_val = build_mean_ss_modifier(
        forcing, model, t_nan_fill=-12.0, theta_nan_fill=0.5
    )
    params_prior   = make_default_params(config)
    n_pools        = len(idx)
    target_names   = list(config.external_inputs.partition.keys())
    ext_target_idx = [idx[n] for n in target_names]

    _i_passive      = idx["soil_passive"] if "soil_passive" in idx.pool_names else None
    _C_passive_prior = 12000.0

    prior_ss = _analytical_c12_ss(params_prior, n_pools, mean_input_val,
                                   mean_modifier, target_indices=ext_target_idx)
    if _i_passive is not None:
        prior_ss = prior_ss.at[_i_passive].set(_C_passive_prior)
    print_prior_ss_check(prior_ss, params_prior, idx, ext_target_idx)

    def _make_ss_state(params_opt):
        c12_ss = _analytical_c12_ss(params_opt, n_pools, mean_input_val,
                                     mean_modifier, target_indices=ext_target_idx)
        if _i_passive is not None:
            c12_ss = c12_ss.at[_i_passive].set(_C_passive_prior)
        return state0._replace(C12=c12_ss)

    # ── Prior forward run ─────────────────────────────────────────────────────
    print("\nPrior forward simulation…")
    t0 = time.perf_counter()
    out_prior = run_model(model, forcing, state0=state0, params=params_prior)
    jax.block_until_ready(out_prior.delta14C)
    print(f"  Done  [{time.perf_counter()-t0:.1f}s]")

    tau_p = np.exp(np.array(params_prior.log_tau))
    d14C_resp_prior = compute_resp_delta14C(out_prior, params_prior)

    def _print_cstock_diag(label: str, out):
        c_sim       = np.array(jnp.mean(out.C12, axis=0))
        act_sim     = c_sim[idx["soil_active"]]
        slw_sim     = c_sim[idx["soil_slow"]]
        pas_sim     = c_sim[idx["soil_passive"]]
        org_sum_sim = act_sim + slw_sim
        pf_sum_sim  = slw_sim + pas_sim
        act_obs, act_sig = c_pools_obs.get("soil_active",  (float("nan"), float("nan")))
        pas_obs, pas_sig = c_pools_obs.get("soil_passive", (float("nan"), float("nan")))
        org_obs = 7500.0;  org_sig = 0.35 * org_obs
        pf_obs  = 16000.0; pf_sig  = 0.25 * pf_obs
        print(f"  {label} C-stock check:")
        print(f"    soil_active  (individual):         "
              f"sim={act_sim:.0f}  obs={act_obs:.0f}±{act_sig:.0f}  "
              f"resid={(act_sim-act_obs)/act_sig:+.2f}σ")
        print(f"    soil_passive (individual):         "
              f"sim={pas_sim:.0f}  obs={pas_obs:.0f}±{pas_sig:.0f}  "
              f"resid={(pas_sim-pas_obs)/pas_sig:+.2f}σ")
        print(f"    organic sum  (active+slow):        "
              f"sim={org_sum_sim:.0f}  obs={org_obs:.0f}±{org_sig:.0f}  "
              f"resid={(org_sum_sim-org_obs)/org_sig:+.2f}σ")
        print(f"    permafrost sum (slow+passive):     "
              f"sim={pf_sum_sim:.0f}  obs={pf_obs:.0f}±{pf_sig:.0f}  "
              f"resid={(pf_sum_sim-pf_obs)/pf_sig:+.2f}σ")
        print(f"    pools: active={act_sim:.0f}  slow={slw_sim:.0f}  passive={pas_sim:.0f}  "
              f"total={act_sim+slw_sim+pas_sim:.0f} gC m⁻²")

    obs_sets = build_oe5_obs_sets(
        forcing,
        delta14C_obs=delta14C_obs,
        delta14C_resp=delta14C_resp,
        c_pools_obs=c_pools_obs,
        er_sliced=er_sliced,
    )
    seq = run_oe5_sequence(
        model=model, forcing=forcing, state0=state0,
        obs_sets=obs_sets,
        carbon_sum_blocks=carbon_sum_blocks,
        extra_blocks=extra_blocks,
        oe_fields=oe_fields, oe_fields_er=oe_fields_er,
        make_ss_state_fn=_make_ss_state,
        print_cstock_diag_fn=_print_cstock_diag,
        n_pool_obs=n_pool_obs, n_resp_obs=n_resp_obs,
    )

    params_opt = seq["params_opt"]

    # ── Respired Δ¹⁴C ────────────────────────────────────────────────────────
    d14C_resp_oe1 = compute_resp_delta14C(seq["out_oe1"], seq["result_oe1"].params_opt)
    d14C_resp_oe2 = compute_resp_delta14C(seq["out_oe2"], seq["result_oe2"].params_opt)
    d14C_resp_oe3 = compute_resp_delta14C(seq["out_oe3"], seq["result_oe3"].params_opt)
    d14C_resp_oe4 = compute_resp_delta14C(seq["out_oe4"], seq["result_oe4"].params_opt)
    d14C_resp_oe5 = compute_resp_delta14C(seq["out_oe5"], seq["result_oe5"].params_opt)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    tau_p, tau_opt = print_tau_summary(params_prior, params_opt, idx.pool_names)
    dfs_oe, n_params_oe = print_averaging_kernel(seq["result_oe5"])
    print_dfs_table(seq)
    print_rh_diagnostics(seq["out_oe5"], forcing, er_sliced, params_opt)

    print("\nAge diagnostics (OE5 full+ER run)…")
    age_diag = compute_age_diagnostics(seq["out_oe5"], params_opt, model)
    bulk_d14C_mean = float(np.nanmean(age_diag.bulk_delta14C))
    resp_d14C_mean = float(np.nanmean(age_diag.respired_delta14C))
    print(f"  Stored bulk Δ¹⁴C:  {bulk_d14C_mean:+.1f} ‰")
    print(f"  Respired Rh Δ¹⁴C:  {resp_d14C_mean:+.1f} ‰")
    print(f"  Age gap:            {bulk_d14C_mean - resp_d14C_mean:+.1f} ‰")

    return dict(
        time_years=time_years,
        forcing_GPP=forcing.GPP_obs,
        forcing_ER=er_sliced,
        out_prior=out_prior,
        **{k: seq[k] for k in ("out_oe1","out_oe2","out_oe3","out_oe4","out_oe5",
                                "result_oe1","result_oe2","result_oe3","result_oe4","result_oe5",
                                "lh1","lh2","lh3","lh4","lh5")},
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
        dfs_oe=dfs_oe,
        n_params_oe=n_params_oe,
    )


# ════════════════════════════════════════════════════════════════════════════
# Compatibility wrapper
# ════════════════════════════════════════════════════════════════════════════

def run_optimal_inversion_israd(pro_name: str | None = None) -> dict:
    """
    Compatibility wrapper for the canonical all-profile ISRaD workflow.

    ``pro_name`` is ignored so the Barrow site has one public pool-Δ¹⁴C
    source definition across scripts and analyses.
    """
    if pro_name not in (None, "all plots"):
        print(f"  NOTE: ignoring deprecated Barrow single-plot request ({pro_name}); "
              "using canonical all-profile ISRaD aggregate.")
    return run_optimal_inversion()


# ════════════════════════════════════════════════════════════════════════════
# Figure
# ════════════════════════════════════════════════════════════════════════════

def _format_israd_sigma_summary(r: dict) -> str:
    sigma = r.get("israd_sigma_dict", {})
    if not sigma:
        return "canonical pool Δ¹⁴C: ISRaD bulk layers (Vaughn 2018 + Nave 2021)"
    sigma_str = "  ".join(
        f"{k.replace('soil_', '')} σ={v:.1f}‰"
        for k, v in sigma.items()
    )
    return (
        "canonical pool Δ¹⁴C: ISRaD bulk layers (Vaughn 2018 + Nave 2021)"
        f"  |  {sigma_str}"
    )


def make_figure(r: dict, out_path: str | None = None):
    """Save the 8-panel canonical Barrow OE comparison figure."""
    if out_path is None:
        out_path = _wt("notebooks/barrow_three_pool_canonical_oe_summary.png")
    make_oe5_figure(
        r=r,
        site_title="Barrow, Alaska — 3-Pool Canonical OE  (active + slow + passive)",
        site_subtitle=(
            f"{_format_israd_sigma_summary(r)}"
            "  |  OE5 adds FluxNet ER → annual Rh constraint (f_hetero≈0.80, σ=20%)"
        ),
        resp_obs_label="obs (Vaughn 2018 surface)",
        pool_colors=["tab:green", "tab:orange", "tab:brown"],
        pool_markers=["o", "^", "s"],
        rh_er_frac=0.80,
        out_path=out_path,
    )


def make_figure_israd(r: dict, out_path: str | None = None):
    """Compatibility wrapper that saves the canonical Barrow figure under the legacy ISRaD filename."""
    if out_path is None:
        out_path = _wt("notebooks/barrow_three_pool_canonical_oe_summary_compatibility_alias.png")
    make_oe5_figure(
        r=r,
        site_title="Barrow, Alaska — 3-Pool Canonical OE  (active + slow + passive)",
        site_subtitle=(
            f"{_format_israd_sigma_summary(r)}"
            "  |  OE5 adds FluxNet ER → annual Rh constraint (f_hetero≈0.80, σ=20%)"
        ),
        resp_obs_label="obs (Vaughn 2018 surface)",
        pool_colors=["tab:green", "tab:orange", "tab:brown"],
        pool_markers=["o", "^", "s"],
        rh_er_frac=0.80,
        out_path=out_path,
    )
