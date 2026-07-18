"""
sites/howland_forest.py — Howland Forest (US-Ho1) canonical OE workflow.
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

from ecosystem_complexity.sites.driver import run_oe_canonical

from ecosystem_complexity.api import build_model, ObsBlock
from ecosystem_complexity.data.loaders import load_howland_forest
from ecosystem_complexity.data.parsers import attach_atm14C, slice_forcing
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.schemas import ObservationData
from ecosystem_complexity.data.israd_observations import (
    FractionMappingRule,
    add_layer_midpoint,
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


HO1_DD_PATH = _data("data/AMF_US-Ho1_FLUXNET_FULLSET_1996-2023_3-6/AMF_US-Ho1_FLUXNET_FULLSET_DD_1996-2023_3-6.csv")
ISRAD_LAYER_PATH = _data("data/shared/israd/ISRaD_extra_flat_layer_v 2.6.6.2024-01-25.csv")
ISRAD_FRAC_PATH = _data("data/shared/israd/ISRaD_extra_flat_fraction_v 2.6.6.2024-01-25.csv")
ISRAD_FLUX_PATH = _data("data/shared/israd/ISRaD_extra_flat_flux_v 2.6.6.2024-01-25.csv")
HUA_PATH = _data("data/shared/atm_14C/Hua_2021.csv")
GRAVEN_PATH = _data("data/shared/atm_14C/Graven_2017.csv")
INTCAL_PATH = _data("data/shared/atm_14C/intcal20.14c")
OPT_CONFIG = _wt("configs/howland_forest_3pool_config.yaml")

_R_STD = 1.176e-12
_UPLAND_PATTERNS = ("Tower Pit", "Nutrient Control Pit")
_BULK_DEPTH_TO_POOL = [
    ("soil_active", (-12.0, 0.0)),
    ("soil_slow", (0.0, 20.0)),
    ("soil_passive", (20.0, 130.0)),
]


def _is_upland_profile(name: str) -> bool:
    s = str(name)
    return any(p in s for p in _UPLAND_PATTERNS) and ("Swamp" not in s)


def _howland_layer_df() -> pd.DataFrame:
    df = pd.read_csv(ISRAD_LAYER_PATH, low_memory=False)
    df = df[
        df["site_name"].astype(str).str.contains("Howland", case=False, na=False)
        & (df["pro_treatment"].astype(str).str.lower() == "control")
        & df["pro_name"].apply(_is_upland_profile)
    ].copy()
    return df


def _howland_fraction_df() -> pd.DataFrame:
    df = pd.read_csv(ISRAD_FRAC_PATH, low_memory=False)
    df = df[
        df["site_name"].astype(str).str.contains("Howland", case=False, na=False)
        & (df["pro_treatment"].astype(str).str.lower() == "control")
        & df["pro_name"].apply(_is_upland_profile)
        & (df["entry_name"] == "Gaudinski_2001")
        & (df["frc_scheme"] == "density")
    ].copy()
    return df


def _howland_flux_df() -> pd.DataFrame:
    df = pd.read_csv(ISRAD_FLUX_PATH, low_memory=False)
    df = df[
        df["site_name"].astype(str).str.contains("Howland", case=False, na=False)
        & (df["pro_treatment"].astype(str).str.lower() == "control")
        & (df["pro_name"].astype(str) == "Tower Pit")
        & (df["flx_pathway"].astype(str).str.lower() == "soil emission")
    ].copy()
    return df


def _dedupe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return df
    return df.drop_duplicates(subset=keep).copy()


def build_pool_14C_obs(forcing_time, pool_names: list[str], print_summary: bool = True) -> dict[str, jnp.ndarray]:
    df = _howland_layer_df()
    df = _dedupe(df, ["entry_name", "pro_name", "lyr_obs_date_y", "lyr_top", "lyr_bot", "lyr_14c"])
    df = df.dropna(subset=["lyr_14c", "lyr_top", "lyr_bot", "lyr_obs_date_y"]).copy()
    df["lyr_mid"] = 0.5 * (
        pd.to_numeric(df["lyr_top"], errors="coerce")
        + pd.to_numeric(df["lyr_bot"], errors="coerce")
    )

    time_np = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T = len(time_np)
    pool_set = set(pool_names)
    obs_dict: dict[str, np.ndarray] = {}

    for pool_name, (z_top, z_bot) in _BULK_DEPTH_TO_POOL:
        if pool_name not in pool_set:
            continue
        mask = (df["lyr_mid"] >= z_top) & (df["lyr_mid"] < z_bot)
        sub = df.loc[mask].copy()
        if sub.empty:
            continue
        for year, grp in sub.groupby("lyr_obs_date_y"):
            vals = pd.to_numeric(grp["lyr_14c"], errors="coerce").dropna()
            if vals.empty:
                continue
            if pool_name not in obs_dict:
                obs_dict[pool_name] = np.full(T, np.nan, dtype=np.float32)
            t_idx = int(np.argmin(np.abs(years_np - (float(year) + 0.5))))
            obs_dict[pool_name][t_idx] = float(vals.mean())
        if print_summary and pool_name in obs_dict:
            finite = np.isfinite(obs_dict[pool_name])
            print(f"  Howland bulk {pool_name:<12s}: {finite.sum()} year(s) of Δ¹⁴C constraints")
    return {k: jnp.array(v) for k, v in obs_dict.items()}


def build_density_fraction_blocks(forcing_time, pool_index, print_summary: bool = True) -> list[ObsBlock]:
    df = _howland_fraction_df()
    df = _dedupe(df, ["pro_name", "lyr_obs_date_y", "lyr_top", "lyr_bot", "frc_property", "frc_14c"])
    df = add_layer_midpoint(df.dropna(subset=["frc_14c", "lyr_top", "lyr_bot", "lyr_obs_date_y"]).copy())

    rows = build_fraction_obs_blocks(
        df,
        forcing_time,
        pool_index,
        rules=[
            FractionMappingRule("soil_active", "free light", (-0.5, 7.5)),
            FractionMappingRule("soil_passive", "heavy", (-0.5, 60.0)),
        ],
        year_col="lyr_obs_date_y",
        min_sigma=15.0,
        singleton_sigma=20.0,
        name_prefix="israd_density",
    )
    if print_summary:
        for row in rows:
            print(
                f"  Howland density {row['pool_name']:<12s} yr={row['obs_year']}  "
                f"Δ¹⁴C={row['mean']:+.1f}‰  σ={row['sigma']:.1f}‰  n={row['n']}"
            )
    return [row["block"] for row in rows]


def build_resp_14C_obs(forcing_time) -> jnp.ndarray:
    df = _howland_flux_df()
    df = _dedupe(df, ["pro_name", "flx_obs_date_y", "flx_obs_date_m", "flx_14c"])
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


def build_soil_carbon_obs(pool_names: list[str], print_summary: bool = True) -> dict[str, tuple[float, float]]:
    df = _howland_layer_df()
    df = _dedupe(df, ["entry_name", "pro_name", "lyr_obs_date_y", "lyr_top", "lyr_bot", "lyr_soc"])
    df = df.dropna(subset=["lyr_soc", "lyr_top", "lyr_bot"]).copy()
    df["soc_gm2"] = pd.to_numeric(df["lyr_soc"], errors="coerce") * 1.0e4
    df["lyr_mid"] = 0.5 * (
        pd.to_numeric(df["lyr_top"], errors="coerce")
        + pd.to_numeric(df["lyr_bot"], errors="coerce")
    )
    df["profile"] = (
        df["entry_name"].astype(str) + "|" + df["pro_name"].astype(str) + "|" + df.get("lyr_obs_date_y", pd.Series(index=df.index)).astype(str)
    )

    pool_set = set(pool_names)
    result: dict[str, tuple[float, float]] = {}
    for pool_name, (z_top, z_bot) in _BULK_DEPTH_TO_POOL:
        if pool_name not in pool_set:
            continue
        sub = df.loc[(df["lyr_mid"] >= z_top) & (df["lyr_mid"] < z_bot)]
        gp = sub.groupby("profile")["soc_gm2"].sum()
        if gp.empty:
            continue
        mean_val = float(gp.mean())
        std_val = float(gp.std(ddof=1)) if len(gp) > 1 else 0.0
        sigma_val = max(std_val, 0.40 * mean_val)
        result[pool_name] = (mean_val, sigma_val)
        if print_summary:
            print(f"  Howland C stock {pool_name:<12s}: {mean_val:6.0f} ± {sigma_val:5.0f} gC m⁻²  (n={len(gp)})")
    return result


def build_state0(config, pool_index, c_pools_obs: dict, delta14c_obs: dict):
    base = make_initial_state(config, {})
    n_pools = len(pool_index)
    c12_arr = np.zeros(n_pools, dtype=np.float32)
    c14_arr = np.zeros(n_pools, dtype=np.float32)
    fallback_d14c = {"soil_active": 120.0, "soil_slow": 80.0, "soil_passive": 20.0}

    for pool_name in pool_index.pool_names:
        c_mean = float(c_pools_obs.get(pool_name, (1000.0,))[0])
        arr = np.array(delta14c_obs.get(pool_name, jnp.array([])), dtype=float)
        finite = arr[np.isfinite(arr)]
        d14c_val = float(finite[0]) if finite.size else fallback_d14c.get(pool_name, 0.0)
        i = pool_index[pool_name]
        c12_arr[i] = max(c_mean, 0.0)
        c14_arr[i] = c12_arr[i] * _R_STD * (1.0 + d14c_val / 1000.0)
    return base._replace(C12=jnp.array(c12_arr), C14=jnp.array(c14_arr))


def run_howland_canonical() -> dict:
    label = "Howland Forest"
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

    forcing_full, _ = load_howland_forest(HO1_DD_PATH, config=config, include_gpp_forcing=True)
    years_daily, d14c_daily = atm14c
    forcing_full = attach_atm14C(forcing_full, d14c_daily, years_daily)
    time_np = np.array(forcing_full.time)
    years_all = 1970.0 + time_np / 365.25
    start_idx = int(np.searchsorted(years_all, 1996.0))
    forcing = slice_forcing(forcing_full, start_idx, len(time_np))
    T = int(forcing.time.shape[0])
    time_years = 1970.0 + np.array(forcing.time) / 365.25
    print(f"  Forcing: {T} days ({time_years[0]:.1f}–{time_years[-1]:.1f})")

    delta14c_obs = build_pool_14C_obs(forcing.time, idx.pool_names, print_summary=True)
    delta14c_resp = build_resp_14C_obs(forcing.time)
    c_pools_obs = build_soil_carbon_obs(idx.pool_names, print_summary=True)
    density_blocks = build_density_fraction_blocks(forcing.time, idx, print_summary=True)
    extra_blocks = list(density_blocks)
    n_resp_obs = int(jnp.sum(~jnp.isnan(delta14c_resp)))
    print(f"  Resp Δ¹⁴C observations: {n_resp_obs}")
    print(f"  Extra Δ¹⁴C blocks: {len(extra_blocks)}")

    obs_full = ObservationData(
        time=forcing.time,
        NEE=jnp.full(T, jnp.nan),
        GPP=jnp.full(T, jnp.nan),
        ER=jnp.full(T, jnp.nan),
        NEE_unc=jnp.full(T, jnp.nan),
        delta14C_obs=delta14c_obs,
        deltaD14C_obs={},
        C_pools_obs=c_pools_obs,
        delta14C_resp=delta14c_resp,
    )

    state0 = build_state0(config, idx, c_pools_obs, delta14c_obs)
    print(f"  State0 total C12: {float(jnp.sum(state0.C12)):.0f} gC m⁻²")

    fit = run_oe_canonical(
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
        "site": "US-Ho1",
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
