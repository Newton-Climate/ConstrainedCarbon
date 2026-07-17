"""
sites/eight_mile_lake.py — Eight-mile Lake (US-EML) canonical OE workflow.
"""
from __future__ import annotations

import os
import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd

_SITES_ROOT = os.path.dirname(os.path.abspath(__file__))
_NB_ROOT = os.path.dirname(_SITES_ROOT)
_WORKTREE_ROOT = os.path.dirname(_NB_ROOT)
_SRC_ROOT = os.path.join(_WORKTREE_ROOT, "src")

for _p in (_SRC_ROOT, _SITES_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from notebook_utils import find_data_root

from canonical import _run_oe_canonical

from ecosystem_complexity.api import (
    build_model, ObsBlock,
)
from ecosystem_complexity.data.loaders import load_eight_mile_lake
from ecosystem_complexity.data.parsers import attach_atm14C, slice_forcing
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.schemas import ObservationData
from ecosystem_complexity.data.israd_observations import (
    FractionMappingRule,
    add_layer_midpoint,
    summarize_by_depth,
    obs_blocks_from_single_year_summary,
    obs_dict_from_single_year_summary,
    build_fraction_obs_blocks,
)
from ecosystem_complexity.state import make_initial_state

_REPO_ROOT = (
    os.environ.get("ECOSYSTEM_REPO_ROOT")
    or find_data_root(_WORKTREE_ROOT)
)


def _wt(rel: str) -> str:
    return os.path.join(_WORKTREE_ROOT, rel)


def _data(rel: str) -> str:
    return os.path.join(_REPO_ROOT, rel)


EML_HH_PATH = _data("data/AMF_US-EML_data/AMF_US-EML_BASE_HH_1-4.csv")
ISRAD_LAYER_PATH = _data("data/shared/israd/ISRaD_extra_flat_layer_v 2.6.6.2024-01-25.csv")
ISRAD_FRAC_PATH = _data("data/shared/israd/ISRaD_extra_flat_fraction_v 2.6.6.2024-01-25.csv")
ISRAD_FLUX_PATH = _data("data/shared/israd/ISRaD_extra_flat_flux_v 2.6.6.2024-01-25.csv")
HUA_PATH = _data("data/shared/atm_14C/Hua_2021.csv")
GRAVEN_PATH = _data("data/shared/atm_14C/Graven_2017.csv")
INTCAL_PATH = _data("data/shared/atm_14C/intcal20.14c")
OPT_CONFIG = _wt("configs/eight_mile_lake_3pool_config.yaml")

_R_STD = 1.176e-12
_EML_LAYER_ENTRY = "Hicks_Pries_2012"
_EML_FLUX_ENTRY = "Hicks_Pries_2013"
_EML_BULK_YEAR = 2004.0
_BULK_DEPTH_TO_POOL = [
    ("soil_active", (-40.0, 0.0)),
    ("soil_slow", (0.0, 20.0)),
    ("soil_passive", (20.0, 60.0)),
]
_MACRO_DEPTH_TO_POOL = [
    ("soil_active", (-40.0, -20.0)),
    ("soil_slow", (-70.0, -40.0)),
]


def _eml_layer_df() -> pd.DataFrame:
    df = pd.read_csv(ISRAD_LAYER_PATH, low_memory=False)
    df = df[
        (df["entry_name"] == _EML_LAYER_ENTRY)
        & (df["site_name"].astype(str).str.upper() == "EML")
        & (df["pro_treatment"].astype(str).str.lower() == "control")
    ].copy()
    return df


def _eml_frac_df() -> pd.DataFrame:
    df = pd.read_csv(ISRAD_FRAC_PATH, low_memory=False)
    df = df[
        (df["entry_name"] == _EML_LAYER_ENTRY)
        & (df["site_name"].astype(str).str.upper() == "EML")
        & (df["pro_treatment"].astype(str).str.lower() == "control")
        & (df["frc_property"].astype(str).str.lower() == "macrofossil")
    ].copy()
    return df


def _eml_flux_df() -> pd.DataFrame:
    df = pd.read_csv(ISRAD_FLUX_PATH, low_memory=False)
    df = df[
        (df["entry_name"] == _EML_FLUX_ENTRY)
        & (df["site_name"].astype(str).str.upper() == "EML")
        & (df["flx_pathway"].astype(str).str.lower() == "soil emission")
        & (df["flx_analyte"].astype(str).str.upper() == "CO2")
    ].copy()
    return df


def _summary_by_depth(
    df: pd.DataFrame,
    value_col: str,
    depth_pairs: list[tuple[str, tuple[float, float]]],
    min_n: int = 2,
) -> dict[str, tuple[float, float, int]]:
    sub = add_layer_midpoint(df.dropna(subset=[value_col, "lyr_top", "lyr_bot"]).copy())
    return summarize_by_depth(sub, value_col, depth_pairs, min_n=min_n)


def build_pool_14C_obs(forcing_time, pool_names: list[str], print_summary: bool = True) -> dict:
    df = _eml_layer_df()
    pool_obs = _summary_by_depth(df, "lyr_14c", _BULK_DEPTH_TO_POOL)
    obs_dict = {
        pool_name: arr
        for pool_name, arr in obs_dict_from_single_year_summary(pool_obs, forcing_time, _EML_BULK_YEAR).items()
        if pool_name in pool_names
    }
    if print_summary:
        for pool_name, (mean_val, sigma_val, n_obs) in pool_obs.items():
            if pool_name in obs_dict:
                print(
                    f"  EML bulk {pool_name:<12s}: Δ¹⁴C = {mean_val:+7.1f} ± {sigma_val:5.1f} ‰"
                    f"  (n={n_obs})"
                )
    return obs_dict


def build_pool_14C_obs_blocks(pool_index, forcing_time) -> list[ObsBlock]:
    df = _eml_layer_df()
    pool_obs = _summary_by_depth(df, "lyr_14c", _BULK_DEPTH_TO_POOL)
    return obs_blocks_from_single_year_summary(
        pool_obs, pool_index, forcing_time, _EML_BULK_YEAR, name_prefix="israd_layer"
    )


def build_macrofossil_14C_blocks(pool_index, forcing_time, print_summary: bool = True) -> list[ObsBlock]:
    df = add_layer_midpoint(_eml_frac_df())
    rows = build_fraction_obs_blocks(
        df,
        forcing_time,
        pool_index,
        rules=[
            FractionMappingRule("soil_active", "macrofossil", (-40.0, -20.0)),
            FractionMappingRule("soil_slow", "macrofossil", (-70.0, -40.0)),
        ],
        year_col=None,
        entry_to_year={_EML_LAYER_ENTRY: int(_EML_BULK_YEAR)},
        min_sigma=20.0,
        singleton_sigma=20.0,
        name_prefix="israd_macrofossil",
    )
    if print_summary:
        for row in rows:
            print(
                f"  EML macrofossil {row['pool_name']:<7s}: Δ¹⁴C = {row['mean']:+7.1f} ± {row['sigma']:5.1f} ‰"
                f"  (n={row['n']})"
            )
    return [row["block"] for row in rows]


def build_resp_14C_obs(forcing_time) -> jnp.ndarray:
    df = _eml_flux_df()
    df = df.dropna(subset=["flx_14c", "flx_obs_date_y", "flx_obs_date_m"]).copy()
    df["decimal_year"] = (
        pd.to_numeric(df["flx_obs_date_y"], errors="coerce")
        + (pd.to_numeric(df["flx_obs_date_m"], errors="coerce") - 0.5) / 12.0
    )
    by_date = df.groupby("decimal_year")["flx_14c"].mean()

    time_np = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T = len(time_np)
    arr = np.full(T, np.nan, dtype=np.float32)
    for dec_yr, d14c_val in by_date.items():
        if not np.isfinite(d14c_val):
            continue
        t_idx = int(np.argmin(np.abs(years_np - float(dec_yr))))
        arr[t_idx] = float(d14c_val)
    return jnp.array(arr)


def build_soil_carbon_obs(pool_names: list[str], print_summary: bool = True) -> dict:
    df = _eml_layer_df()
    df = df.dropna(subset=["lyr_soc", "lyr_top", "lyr_bot"]).copy()
    df["soc_gm2"] = pd.to_numeric(df["lyr_soc"], errors="coerce") * 1.0e4
    df["lyr_mid"] = 0.5 * (
        pd.to_numeric(df["lyr_top"], errors="coerce")
        + pd.to_numeric(df["lyr_bot"], errors="coerce")
    )
    result: dict[str, tuple[float, float]] = {}
    for pool_name, (z_top, z_bot) in _BULK_DEPTH_TO_POOL:
        if pool_name not in pool_names:
            continue
        sub = df.loc[(df["lyr_mid"] >= z_top) & (df["lyr_mid"] < z_bot)]
        per_profile = sub.groupby("pro_name")["soc_gm2"].sum()
        if len(per_profile) < 2:
            continue
        mean_val = float(per_profile.mean())
        sigma_val = max(float(per_profile.std(ddof=1)), 0.20 * mean_val)
        result[pool_name] = (mean_val, sigma_val)
        if print_summary:
            print(
                f"  EML C stock {pool_name:<12s}: {mean_val:7.0f} ± {sigma_val:6.0f} gC m⁻²"
                f"  (n={len(per_profile)})"
            )
    return result


def build_state0(config, pool_index, c_pools_obs: dict, delta14c_obs: dict):
    base = make_initial_state(config, {})
    n_pools = len(pool_index)
    c12_arr = np.zeros(n_pools, dtype=np.float32)
    c14_arr = np.zeros(n_pools, dtype=np.float32)

    fallback_c = {
        "soil_active": 20000.0,
        "soil_slow": 10000.0,
        "soil_passive": 8000.0,
    }
    fallback_d14c = {
        "soil_active": 50.0,
        "soil_slow": -250.0,
        "soil_passive": -500.0,
    }
    for pool_name in pool_index.pool_names:
        c_mean = c_pools_obs.get(pool_name, (fallback_c.get(pool_name, 1000.0),))[0]
        d14c_arr = np.array(delta14c_obs.get(pool_name, jnp.array([])), dtype=float)
        if d14c_arr.size:
            finite = d14c_arr[np.isfinite(d14c_arr)]
            d14c_val = float(finite[0]) if finite.size else fallback_d14c.get(pool_name, 0.0)
        else:
            d14c_val = fallback_d14c.get(pool_name, 0.0)
        i = pool_index[pool_name]
        c12_arr[i] = max(float(c_mean), 0.0)
        c14_arr[i] = c12_arr[i] * _R_STD * (1.0 + d14c_val / 1000.0)
    return base._replace(C12=jnp.array(c12_arr), C14=jnp.array(c14_arr))


def run_eml_canonical() -> dict:
    label = "Eight-mile Lake"
    print(f"\n══ {label} — canonical OE inversion ═══════════════════════════════")
    atm14c = load_full_14C_record(
        hua_path=HUA_PATH, graven_path=GRAVEN_PATH, intcal_path=INTCAL_PATH,
        hemisphere="NH", start_year=1500.0, end_year=2025.0,
    )

    model = build_model(OPT_CONFIG)
    config = model.config
    idx = model.pool_index
    opt_fields = ("log_tau", "log_f_transfer")
    print(f"  Pools: {idx.pool_names}")
    print(f"  opt_fields: {opt_fields}")

    forcing_full, _ = load_eight_mile_lake(
        hh_path=EML_HH_PATH, config=config, include_gpp_forcing=True,
    )
    years_daily, d14c_daily = atm14c
    forcing_full = attach_atm14C(forcing_full, d14c_daily, years_daily)
    time_np = np.array(forcing_full.time)
    years_all = 1970.0 + time_np / 365.25
    start_idx = int(np.searchsorted(years_all, 2008.0))
    forcing = slice_forcing(forcing_full, start_idx, len(time_np))
    T = int(forcing.time.shape[0])
    time_years = 1970.0 + np.array(forcing.time) / 365.25
    print(f"  Forcing: {T} days ({time_years[0]:.1f}–{time_years[-1]:.1f})")

    delta14c_obs = build_pool_14C_obs(forcing.time, idx.pool_names, print_summary=True)
    delta14c_resp = build_resp_14C_obs(forcing.time)
    c_pools_obs = build_soil_carbon_obs(idx.pool_names, print_summary=True)
    bulk_blocks = build_pool_14C_obs_blocks(idx, forcing.time)
    macro_blocks = build_macrofossil_14C_blocks(idx, forcing.time, print_summary=True)
    extra_blocks = list(bulk_blocks) + list(macro_blocks)

    n_resp_obs = int(jnp.sum(~jnp.isnan(delta14c_resp)))
    print(f"  Resp Δ¹⁴C observations: {n_resp_obs}")
    print(f"  Extra Δ¹⁴C blocks: {len(extra_blocks)}")

    obs_full = ObservationData(
        time=forcing.time,
        NEE=jnp.full(T, jnp.nan),
        GPP=jnp.full(T, jnp.nan),
        ER=jnp.full(T, jnp.nan),
        NEE_unc=jnp.full(T, jnp.nan),
        delta14C_obs={},
        deltaD14C_obs={},
        C_pools_obs=c_pools_obs,
        delta14C_resp=delta14c_resp,
    )

    state0 = build_state0(config, idx, c_pools_obs, delta14c_obs)
    print(f"  State0 total C12: {float(jnp.sum(state0.C12)):.0f} gC m⁻²")

    fit = _run_oe_canonical(
        model, forcing, state0, obs_full, extra_blocks, opt_fields, label,
    )

    return dict(
        model=model, config=config, idx=idx, opt_fields=opt_fields,
        forcing=forcing, time_years=time_years,
        obs_full=obs_full, extra_blocks=extra_blocks,
        state0_obs=state0,
        delta14C_obs=delta14c_obs, delta14C_resp=delta14c_resp,
        c_pools_obs=c_pools_obs,
        **fit,
    )


def build_summary(data: dict, analysis: dict) -> dict:
    idx = data["idx"]
    params_opt = data["params_opt"]
    tau_days = np.exp(np.array(params_opt.log_tau))
    transfers = np.exp(np.array(params_opt.log_f_transfer))
    names = idx.pool_names
    transfer_rules = []
    for src_i, src_name in enumerate(names):
        for dst_i, dst_name in enumerate(names):
            frac = float(transfers[src_i, dst_i])
            if frac > 1e-6:
                transfer_rules.append({"src": src_name, "dst": dst_name, "fraction": frac})
    return {
        "site": "US-EML",
        "forcing_start_year": float(data["time_years"][0]),
        "forcing_end_year": float(data["time_years"][-1]),
        "optimized_tau_days": {name: float(val) for name, val in zip(names, tau_days)},
        "c_pools_obs_gCm2": {k: {"mean": float(v[0]), "sigma": float(v[1])} for k, v in data["c_pools_obs"].items()},
        "dfs_total": float(analysis["dof_full"].dfs_total),
        "transfer_matrix_nonzero": transfer_rules,
        "n_resp_14c_obs": int(jnp.sum(~jnp.isnan(data["delta14C_resp"]))),
        "n_extra_obs_blocks": len(data["extra_blocks"]),
        "cost_history": [float(x) for x in np.array(data["oe_result"].cost_history)],
        "converged": bool(data["oe_result"].converged),
    }
