"""Soil organic carbon constraints: the steady-state prior and total-column stocks.

Stock-source priority in the combined observation path is measured ISRaD >
SoilGrids prediction > the model's own steady state (self-referential, and at
σ=0.50 a deliberate no-op).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from ecosystem_complexity.api import run_model
from ecosystem_complexity.data.israd_observations import add_layer_midpoint
from ecosystem_complexity.data.schemas import ForcingData
from ecosystem_complexity.oe_utils import ss_state_for_params
from ecosystem_complexity.sites.forcing import build_annual_mean_forcing
from ecosystem_complexity.sites.israd_14c import _pool_depth_bins_cm
from ecosystem_complexity.sites.paths import ISRAD_LAYER, SOILGRIDS_CSV
from ecosystem_complexity.state import make_default_params, make_initial_state

logger = logging.getLogger(__name__)

# Matches the σ=0.5 OE prior on log_tau. build_soc_prior sets obs = C(prior params)
# and C = I·τ/modifier at steady state with I fixed by the GPP forcing, so anything
# tighter restates the τ prior at MORE confidence than the prior itself — prior
# double-counting dressed as an observation. At 0.50 the fallback is a no-op.
_SOC_PRIOR_SIGMA_FRAC = 0.50
_SS_TOL = 1e-5
_SS_MAX_YEARS = 2000

def build_soc_prior(model, forcing: ForcingData) -> tuple:
    """Build a site-specific steady-state SOC prior from annual-mean forcing."""
    params_prior = make_default_params(model.config)
    inversion = getattr(model.config, "inversion_raw", {}) or {}
    sigma_fraction = float(inversion.get("sigma_soc_fraction", _SOC_PRIOR_SIGMA_FRAC))
    forcing_mean = build_annual_mean_forcing(forcing)
    base = make_initial_state(model.config, {})
    state = ss_state_for_params(model, forcing_mean, base, params_prior)

    prev_total = None
    n_years = 0
    for n_years in range(1, _SS_MAX_YEARS + 1):
        out = run_model(model, forcing_mean, state0=state, params=params_prior)
        state = out.final_state
        total = float(np.sum(np.array(state.C12, dtype=float)))
        if prev_total is not None:
            rel = abs(total - prev_total) / (abs(prev_total) + 1e-10)
            if rel < _SS_TOL:
                break
        prev_total = total

    c12 = np.array(state.C12, dtype=float)
    c_pools_obs = {
        name: (
            float(c12[i]),
            float(sigma_fraction * c12[i]),
        )
        for i, name in enumerate(model.pool_index.pool_names)
        if float(c12[i]) > 0.0
    }
    total = float(c12.sum())
    c_total_obs = (total, float(sigma_fraction * total)) if total > 0.0 else None
    return state, c_pools_obs, n_years, c_total_obs


_MEASURED_SOC_SIGMA_FRAC = 0.15
# A measured profile is only comparable to the model's Σ_i C12_i if it samples
# most of the column the model represents. ISRaD profiles stop well short at most
# sites (MO Ozark 30 cm, Bartlett 45, EML 62, UMBS 90 vs a 130 cm model column),
# and a truncated total is a downward BIAS, not extra noise — so it is rejected
# rather than inflated.
_MIN_SOC_DEPTH_COVERAGE = 0.8


def _model_column_depth_cm(model) -> float:
    """Depth (cm) of the soil column the model's Σ_i C12_i represents."""
    return max(float(layer.depth_bot_m) for layer in model.config.soil_layers) * 100.0


def build_measured_soc_total(
    israd_name: str, model, sigma_floor_frac: float = _MEASURED_SOC_SIGMA_FRAC
) -> tuple[float, float, float] | None:
    """Measured TOTAL column SOC (gC m⁻²) → (mean, sigma, depth_coverage).

    Under co-located kinetic pools a measured stock cannot be attributed to one
    pool by depth (that was the same error the bulk Δ¹⁴C operator made), so this
    replaces ``build_measured_soc_stocks``: it sums ``lyr_soc`` over each whole
    profile and takes the across-profile mean ± σ.

    Returns None unless the profiles reach ``_MIN_SOC_DEPTH_COVERAGE`` of the
    model column. Depth coverage is the binding issue in ISRaD: profiles stop at
    different depths both across and within sites (UMBS spans 103–6049 gC m⁻²
    across profiles, cv=1.59, almost entirely because of where they stop), and
    comparing a 0–30 cm measurement against a 0–130 cm model column would bias
    the stock — and therefore ⟨τ⟩ — low.
    """
    df = pd.read_csv(ISRAD_LAYER, low_memory=False)
    df = df[df["site_name"] == israd_name].copy()
    df = df.dropna(subset=["lyr_soc", "lyr_top", "lyr_bot"])
    if df.empty:
        return None
    df["soc_gm2"] = pd.to_numeric(df["lyr_soc"], errors="coerce") * 1.0e4
    df["_profile"] = (
        df.get("entry_name", pd.Series(index=df.index)).astype(str) + "|"
        + df.get("pro_name", pd.Series(index=df.index)).astype(str) + "|"
        + df.get("lyr_obs_date_y", pd.Series(index=df.index)).astype(str)
    )
    column_cm = _model_column_depth_cm(model)
    per_profile = df.groupby("_profile").agg(
        total=("soc_gm2", "sum"), z_bot=("lyr_bot", "max")
    )
    per_profile = per_profile[per_profile["total"] > 0.0]
    if per_profile.empty:
        return None
    # Keep only profiles that actually sample most of the modelled column.
    coverage = per_profile["z_bot"] / column_cm
    keep = per_profile[coverage >= _MIN_SOC_DEPTH_COVERAGE]
    if keep.empty:
        return None
    mean_val = float(keep["total"].mean())
    std_val = float(keep["total"].std(ddof=1)) if len(keep) > 1 else 0.0
    sigma = max(std_val, sigma_floor_frac * mean_val)
    depth_cov = float(keep["z_bot"].max() / column_cm)
    return mean_val, sigma, depth_cov


# SoilGrids is a statistical prediction, not a measurement, and it runs high on
# organic soils. Above this the value is rejected outright rather than trusted with
# a wide σ: CZ_Old_Black_Spruce comes back at 416,969 gC m⁻² and Baram Basin at
# 266,281, which exceed any real 1.3 m profile (deep tropical peat domes top out
# around 100k).
_MAX_PLAUSIBLE_SOC_GCM2 = 100_000.0
# SoilGrids per-depth uncertainties share one model and one location, so they are
# strongly correlated; adding them in quadrature would understate the total. Sum
# them linearly (≈50% of the total) instead.
_SOILGRIDS_MIN_SIGMA_FRAC = 0.50


def build_soilgrids_soc_total(
    israd_name: str, model
) -> tuple[float, float, float] | None:
    """SoilGrids v2.0 total column SOC (gC m⁻²) → (mean, sigma, depth_coverage).

    Read from ``notebooks/exports/soilgrids_soc_pools.csv`` (written by
    ``notebooks/download_soilgrids_soc.py``). Unlike ``build_soc_prior`` this is
    genuinely independent of the model, and unlike the ISRaD profiles it spans the
    full 0–130 cm column — which is why it can serve as a stock constraint where
    ISRaD cannot. The CSV stores per-pool depth bins; under co-located kinetic
    pools only their SUM is meaningful, so the bins are summed back to a column
    total here rather than used per pool.

    Returns None when the site is absent or the total is physically implausible.
    Treat the values with suspicion: at Howland — the only site with both — SoilGrids
    says 35,498±10,511 gC m⁻² against a measured 11,200±1,680, a 3.2× overestimate
    with non-overlapping intervals.
    """
    if not os.path.exists(SOILGRIDS_CSV):
        return None
    df = pd.read_csv(SOILGRIDS_CSV)
    df = df[df["site"] == israd_name]
    if df.empty:
        return None
    column_cm = _model_column_depth_cm(model)
    z_bot = float(pd.to_numeric(df["depth_bot_cm"], errors="coerce").max())
    if not np.isfinite(z_bot) or z_bot < _MIN_SOC_DEPTH_COVERAGE * column_cm:
        return None
    total = float(pd.to_numeric(df["soc_gCm2"], errors="coerce").sum())
    if not np.isfinite(total) or total <= 0.0 or total > _MAX_PLAUSIBLE_SOC_GCM2:
        return None
    sigma = float(pd.to_numeric(df["soc_sigma_gCm2"], errors="coerce").sum())
    sigma = max(sigma, _SOILGRIDS_MIN_SIGMA_FRAC * total)
    return total, sigma, z_bot / column_cm


def build_measured_soc_stocks(
    israd_name: str, model, sigma_floor_frac: float = _MEASURED_SOC_SIGMA_FRAC
) -> dict[str, tuple[float, float]]:
    """SUPERSEDED — measured ISRaD layer SOC → per-pool C-stock, by depth bin.

    Kept only for reference/back-compat; ``run_site_canonical`` now calls
    ``build_measured_soc_total`` instead. This function assigns a measured stock to
    a pool by depth (active←0–10 cm, slow←10–30, passive←30–130), which is the same
    error the bulk Δ¹⁴C operator used to make: with co-located kinetic pools, no
    observation can attribute stock to an individual pool, so only the column total
    Σ_i C12_i is observable. It also silently under-reports deep pools wherever a
    profile stops short of the modelled column (ISRaD profiles reach only 30 cm at
    MO Ozark, 45 at Bartlett, 62 at EML, 90 at UMBS).
    """
    df = pd.read_csv(ISRAD_LAYER, low_memory=False)
    df = df[df["site_name"] == israd_name].copy()
    df = df.dropna(subset=["lyr_soc", "lyr_top", "lyr_bot"])
    if df.empty:
        return {}
    df["soc_gm2"] = pd.to_numeric(df["lyr_soc"], errors="coerce") * 1.0e4
    df = add_layer_midpoint(df)
    df["_profile"] = (
        df.get("entry_name", pd.Series(index=df.index)).astype(str) + "|"
        + df.get("pro_name", pd.Series(index=df.index)).astype(str) + "|"
        + df.get("lyr_obs_date_y", pd.Series(index=df.index)).astype(str)
    )
    result: dict[str, tuple[float, float]] = {}
    for pool_name, (lo, hi) in _pool_depth_bins_cm(model):
        sub = df.loc[(df["lyr_mid"] >= lo) & (df["lyr_mid"] < hi)]
        gp = sub.groupby("_profile")["soc_gm2"].sum()
        gp = gp[gp > 0.0]
        if gp.empty:
            continue
        mean_val = float(gp.mean())
        std_val = float(gp.std(ddof=1)) if len(gp) > 1 else 0.0
        result[pool_name] = (mean_val, max(std_val, sigma_floor_frac * mean_val))
    return result
