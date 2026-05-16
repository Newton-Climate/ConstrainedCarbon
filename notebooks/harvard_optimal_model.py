"""
harvard_optimal_model.py — 3-pool (active + slow + passive) inversion
for Harvard Forest EMS eddy flux tower site.

Model structure
---------------
  soil_active  : τ ~   2 yr  — fast-cycling (Oi litter), bomb-¹⁴C enriched
  soil_slow    : τ ~  20 yr  — intermediate (Oe + A-lf), post-bomb peak
  soil_passive : τ ~ 100 yr  — mineral-protected (A-min), pre-bomb depleted

  Cascade: active → slow (25%) → passive (10%)

Observations used as constraints
---------------------------------
  Pool Δ¹⁴C   : hf212-03 (Oi, Oe, A-lf, A-min horizons; 1996 + 2007)
  Resp CO₂ Δ¹⁴C: hf212-01 NWN dates (41 obs, 1996–2010)
  C stocks (split organic → active + slow using prior τ ratios; wide σ):
    soil_active  ← f_active × hf324-06 Munger organic  (σ = 40% relative)
    soil_slow    ← f_slow   × hf324-06 Munger organic  (σ = 40% relative)
    soil_passive ← hf271-07 M horizon (EMS tower 0–15 cm) (σ = 35% relative)
  Annual Rh    : FluxNet ER × f_hetero (OE5 only; 2005 excluded as outlier)

Optimised parameters (OE Levenberg-Marquardt)
----------------------------------------------
  log_tau                      (4 values) — turnover times in log-space
  log_external_input_partition (4 logits) — carbon input fractions (softmax)
  log_f_transfer               (3 values) — transfer fractions (active→slow,
                                             slow→passive, passive→stable)

Run
---
  python notebooks/harvard_optimal_model.py

Output
------
  notebooks/harvard_optimal_model.png — 6-panel figure:
    (a) OE cost convergence (L-M iterations)
    (b) τ before vs. after
    (c) Pool Δ¹⁴C trajectories — prior / opt / obs
    (d) Respired CO₂ Δ¹⁴C — prior / opt / obs
    (e) Carbon stocks: modelled vs. observed (active + passive)
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

from ecosystem_complexity.api import build_model, run_model, optimize_oe, OEResult, _get_oe_fields
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
_OPT_CONFIG    = _wt("configs/harvard_3pool_config.yaml")
_SOIL_CONFIG   = _wt("configs/harvard_forest_soil_only.yaml")   # 6-pool reference
HF_SOIL_14C_PATH = _data("data/harvard_forest/hf212-03-14c-org.csv")
HF_RESP_14C_PATH = _data("data/harvard_forest/hf212-01-14c-no-treat.csv")
HF_SOIL_C_PATH   = _data("data/harvard_forest/hf271-07-soils.csv")
HF_SOIL_C_PATH2  = _data("data/harvard_forest/hf324-06-soil-carbon.csv")
ISRAD_FRAC_PATH  = _data(
    "data/shared/israd/ISRaD_extra_flat_fraction_v 2.6.6.2024-01-25.csv"
)
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

# OE field sets — derived from config so config flags (optimize_partition, etc.)
# are automatically respected.  _OPT_FIELDS_ER adds log_f_hetero for OE5 only;
# this overrides the default rather than requiring a config flag, since OE1–4
# use the same config but a different state vector.
# (populated after model is loaded, inside run_optimal_inversion)
_OPT_FIELDS    = None   # set to _get_oe_fields(config) after model load
_OPT_FIELDS_ER = None   # set to _OPT_FIELDS + ("log_f_hetero",)


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
    Build ``delta14C_obs`` by mapping hf212-03 horizons to model pools.

    4-pool mapping (active + slow + passive + stable):
      soil_active  ← Oi               [fresh litter; fastest cycling]
      soil_slow    ← mean(Oe, A-lf)   [intermediate; organic + mineral-A fast]
      soil_passive ← A-min            [mineral-protected; pre-bomb depleted]
      soil_stable  ← (no direct Δ14C obs — left unconstrained by Δ14C)

    3-pool mapping (active + slow + passive):
      same as 4-pool minus stable

    2-pool fallback (active + passive):
      soil_active  ← mean(Oi, Oe)
      soil_passive ← mean(A-lf, A-min)
    """
    d14c_df = pd.read_csv(soil_14c_path)
    d14c_df["horizon"] = d14c_df["horizon"].str.strip()

    time_np  = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T = len(time_np)

    pool_name_set = set(pool_names)
    has_slow   = "soil_slow"   in pool_name_set
    has_stable = "soil_stable" in pool_name_set

    if has_slow:
        # 3-pool or 4-pool: stable gets no Δ14C constraint (not in hf212-03)
        _horizon_map = {
            "soil_active":  ["Oi"],
            "soil_slow":    ["Oe", "A-lf"],
            "soil_passive": ["A-min"],
            # soil_stable intentionally omitted — no direct horizon obs
        }
    else:
        _horizon_map = {
            "soil_active":  ["Oi", "Oe"],
            "soil_passive": ["A-lf", "A-min"],
        }

    obs_dict = {}
    for year in d14c_df["year"].unique():
        yr_df = d14c_df[d14c_df["year"] == year]
        t_idx = int(np.argmin(np.abs(years_np - (float(year) + 0.5))))

        for pool_name, horizons in _horizon_map.items():
            if pool_name not in pool_name_set:
                continue
            vals = yr_df.loc[yr_df["horizon"].isin(horizons), "of.14c.12c"].dropna()
            if len(vals):
                if pool_name not in obs_dict:
                    obs_dict[pool_name] = np.full(T, np.nan, dtype=np.float32)
                obs_dict[pool_name][t_idx] = float(vals.mean())

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


def _build_israd_14C_obs(israd_frac_path: str, forcing_time, pool_index) -> list:
    """
    Build ISRaD density-fraction Δ¹⁴C observations as a list of ObsBlocks.

    Each (pool, measurement_year) pair becomes one ObsBlock with a per-obs Se
    derived from the propagated standard deviation across replicate plots.
    Observation-error variance is σ² = σ_mean² where σ_mean is the standard
    error of the (possibly mass-weighted) mean across fractionation replicates.

    Using separate ObsBlocks from the standard hf212-03 pool_14C block is
    deliberate: ISRaD σ values (22–138‰) are much larger than the uniform
    sigma_pool_14C (10‰) used for horizon observations, so they cannot be
    merged into the same Se array without biasing the inversion.

    Fraction-to-pool mapping (density fractionation):
      free light   → soil_active   (labile, unprotected)
      occluded light → soil_slow   (physically protected)
      heavy        → soil_passive  (mineral-associated)

    Entries and measurement years (inferred from entry_name):
      Gaudinski_2000 → 1996  (free light + heavy only; occluded not reported)
      Savage_unpub   → 2007  (free light + heavy only; occluded not reported)
      McFarlane_2013 → 2011  (all three fractions; mass-weighted means used)

    Time index uses year + 0.6 offset to avoid collision with the hf212-03
    horizon obs that are placed at year + 0.5.

    Returns
    -------
    list[ObsBlock]
        One block per (pool, year) pair.  Empty list if the file is missing or
        contains no usable observations.
    """
    from ecosystem_complexity.api import ObsBlock
    import jax.numpy as jnp

    if not os.path.isfile(israd_frac_path):
        print(f"  ISRaD frac file not found: {israd_frac_path} — skipping")
        return []

    df = pd.read_csv(israd_frac_path, low_memory=False)

    # Filter to Harvard Forest site + density fractionation scheme only.
    # ISRaD 'frc_property' holds the fraction name; 'frc_scheme' is the method.
    df = df[
        df["site_name"].str.contains("Harvard", na=False) &
        (df["frc_scheme"] == "density")
    ]

    # Map ISRaD frc_property values to model pool names.
    _FRAC_TO_POOL = {
        "free light":     "soil_active",
        "occluded light": "soil_slow",
        "heavy":          "soil_passive",
    }

    # Entry name → measurement year (field campaign dates from publications).
    _ENTRY_YEAR = {
        "Gaudinski_2000": 1996,
        "Savage_unpub":   2007,
        "McFarlane_2013": 2011,
    }

    pool_names_set = set(pool_index.pool_names)
    time_np  = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25

    # Collect weighted (Δ¹⁴C mean, σ_mean) per (pool, year).
    from collections import defaultdict
    # {(pool_name, year): [(d14c_val, weight), ...]}
    records: dict = defaultdict(list)

    for _, row in df.iterrows():
        entry = str(row.get("entry_name", "")).strip()
        yr = _ENTRY_YEAR.get(entry, None)
        if yr is None:
            continue

        frac = str(row.get("frc_property", "")).strip().lower()
        pool = _FRAC_TO_POOL.get(frac, None)
        if pool is None or pool not in pool_names_set:
            continue

        d14c_raw = row.get("frc_14c", np.nan)
        if d14c_raw is None or (isinstance(d14c_raw, float) and np.isnan(d14c_raw)):
            continue
        d14c = float(d14c_raw)
        if not np.isfinite(d14c):
            continue

        # Use mass fraction as weight if available (McFarlane_2013); else 1.0.
        mass_pct_raw = row.get("frc_mass_perc", np.nan)
        try:
            mass_pct_f = float(mass_pct_raw)
            w = mass_pct_f if np.isfinite(mass_pct_f) and mass_pct_f > 0 else 1.0
        except (TypeError, ValueError):
            w = 1.0

        records[(pool, yr)].append((float(d14c), w))

    # Exclude McFarlane_2013 (2011) soil_passive: the density cutoff used by
    # McFarlane (<1.85 g cm⁻³ vs. Gaudinski/Savage <1.65 g cm⁻³) retains more
    # fine-POM in the heavy fraction, biasing its Δ¹⁴C toward younger values
    # (−16‰ vs. −74‰ from Gaudinski and −86‰ from Savage).  Including it would
    # pull the passive pool toward an unrealistically young age while the 1996
    # and 2007 heavy-fraction obs already bracket the passive pool constraint.
    _EXCLUDE = {("soil_passive", 2011)}

    blocks = []
    for (pool, yr), pts in sorted(records.items()):
        if (pool, yr) in _EXCLUDE:
            print(f"  ISRaD obs: {pool:<14} yr={yr}  EXCLUDED (density-cutoff inconsistency)")
            continue
        vals = np.array([v for v, _ in pts])
        wts  = np.array([w for _, w in pts])
        wts  = wts / wts.sum()  # normalise

        d14c_mean = float(np.dot(wts, vals))

        # σ_mean: weighted standard deviation / sqrt(n_eff)
        n = len(vals)
        if n > 1:
            var_w = float(np.dot(wts, (vals - d14c_mean) ** 2))
            # Effective n for weighted stats: 1 / sum(w²) (uses normalised wts)
            n_eff = 1.0 / float(np.dot(wts, wts))
            sigma_mean = float(np.sqrt(var_w / max(n_eff, 1.0)))
        else:
            # Single obs: fallback σ = 50‰ (conservative)
            sigma_mean = 50.0

        # Clamp σ_mean to a minimum (measurement precision floor)
        sigma_mean = max(sigma_mean, 15.0)

        t_idx = int(np.argmin(np.abs(years_np - (float(yr) + 0.6))))
        pool_col = pool_index[pool]

        _t   = jnp.array([t_idx], dtype=jnp.int32)
        _col = jnp.array([pool_col], dtype=jnp.int32)
        _y   = jnp.array([d14c_mean], dtype=jnp.float32)
        _se  = jnp.array([sigma_mean ** 2], dtype=jnp.float32)

        blocks.append(ObsBlock(
            name=f"israd_{pool}_{yr}",
            y=_y,
            Se=_se,
            predict=lambda out, p, t=_t, col=_col: out.delta14C[t, col],
        ))
        print(f"  ISRaD obs: {pool:<14} yr={yr}  Δ¹⁴C={d14c_mean:+.1f}‰  "
              f"σ={sigma_mean:.1f}‰  n={n}")

    return blocks


def _build_state0(config, pool_index):
    """
    Initialise model state from hf271 C stocks and hf212-03 Δ¹⁴C data.

    4-pool mapping (active + slow + passive + stable):
      soil_active  ← 0.40 × O_mean
      soil_slow    ← 0.60 × O_mean + 0.30 × M_mean
      soil_passive ← 0.50 × M_mean   (upper mineral, A-min)
      soil_stable  ← 0.20 × M_mean + 5000 gC m⁻²   (deep organo-mineral)
        Δ¹⁴C initialised to −250‰ (old, pre-bomb mineral-associated C)

    3-pool mapping (active + slow + passive):
      soil_active  ← 0.40 × O_mean
      soil_slow    ← 0.60 × O_mean + 0.30 × M_mean
      soil_passive ← 0.70 × M_mean

    2-pool fallback (active + passive):
      soil_active  ← O_mean
      soil_passive ← M_mean

    Δ¹⁴C is initialised from 1996 hf212-03 horizon means (except soil_stable).
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

    pool_names_set = set(pool_index.pool_names)
    has_slow   = "soil_slow"   in pool_names_set
    has_stable = "soil_stable" in pool_names_set

    if has_slow and has_stable:
        # 4-pool
        # soil_passive gets ~50% of mineral C; soil_stable absorbs the rest
        # plus a ~5000 gC/m² "deep" reservoir with very old Δ¹⁴C
        stable_extra = 5000.0  # gC m⁻² additional deep mineral C
        C12_by_pool = {
            "soil_active":  0.40 * org_mean,
            "soil_slow":    0.60 * org_mean + 0.30 * min_mean,
            "soil_passive": 0.50 * min_mean,
            "soil_stable":  0.20 * min_mean + stable_extra,
        }
        d14C_by_pool = {
            "soil_active":  _mean_d14c(["Oi"]),
            "soil_slow":    _mean_d14c(["Oe", "A-lf"]),
            "soil_passive": _mean_d14c(["A-min"]),
            "soil_stable":  -250.0,   # old organo-mineral C, no direct obs
        }
    elif has_slow:
        # 3-pool
        C12_by_pool = {
            "soil_active":  0.40 * org_mean,
            "soil_slow":    0.60 * org_mean + 0.30 * min_mean,
            "soil_passive": 0.70 * min_mean,
        }
        d14C_by_pool = {
            "soil_active":  _mean_d14c(["Oi"]),
            "soil_slow":    _mean_d14c(["Oe", "A-lf"]),
            "soil_passive": _mean_d14c(["A-min"]),
        }
    else:
        # 2-pool fallback
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


_HF324_CORE_AREA_CM2 = 10.0  # cm² — back-calculated from bulk density × depth × %C


def _load_horizon_means(hf324_path: str, hf271_path: str) -> dict:
    """
    Load per-horizon carbon stock means and uncertainties from the two EMS datasets.

    Returns a dict with keys:
      "org_mean", "org_sem", "org_n"   — hf324 Munger organic horizon
      "min_mean", "min_sem", "min_n"   — hf271 M (mineral) horizon

    Raw values only — no pool-to-horizon assignment is done here.  The caller
    decides how to combine horizons into pool constraints.
    """
    result = {}

    df324 = pd.read_csv(hf324_path, encoding="latin1")
    munger = df324[
        (df324["site"] == "ems") &
        (df324["use.not"] == 1) &
        (df324["contact"].str.lower().str.strip() == "munger")
    ].copy()
    munger["gC_m2"] = munger["c.mass.rocks"] * 10000.0 / _HF324_CORE_AREA_CM2
    org_vals = munger.loc[munger["horizon"] == "organic", "gC_m2"].dropna()
    if len(org_vals) >= 2:
        result["org_mean"] = float(org_vals.mean())
        result["org_sem"]  = float(org_vals.std(ddof=1) / np.sqrt(len(org_vals)))
        result["org_n"]    = len(org_vals)

    df271 = pd.read_csv(hf271_path)
    m_vals = df271.loc[df271["horizon"] == "M", "carbon.gm2"].dropna()
    if len(m_vals) >= 2:
        result["min_mean"] = float(m_vals.mean())
        result["min_sem"]  = float(m_vals.std(ddof=1) / np.sqrt(len(m_vals)))
        result["min_n"]    = len(m_vals)

    return result


def _build_soil_carbon_obs(hf324_path: str, hf271_path: str, pool_names: list) -> dict:
    """
    Load the *individual* carbon stock constraint for soil_active only.

    The organic horizon (hf324 Munger) contains BOTH active and slow fractions,
    so assigning a fixed fraction to each pool is physically arbitrary and
    contradicted by density-fractionation data.  Instead:

      - soil_active is constrained individually using a prior-τ-ratio split of
        the organic horizon (f_active ≈ 0.286).  σ is set to 40% relative so the
        Δ¹⁴C constraints dominate; this is a soft prior, not a hard assignment.

      - soil_slow and soil_passive are NOT constrained individually here.
        They are constrained by two sum-constraints built in
        ``_build_carbon_sum_obs``:
            organic sum  → C_active + C_slow  ≈ org_mean ± org_sem
            mineral sum  → C_slow  + C_passive ≈ min_mean ± min_σ

        This correctly reflects that the organic horizon total is known but the
        active/slow split is uncertain, and that the mineral horizon contains
        both slow (mineral-associated fast) and passive SOM.

    Data sources
    ------------
    hf324-06-soil-carbon.csv — Munger plots only (EMS site, n≈214, 2014).
    hf271-07-soils.csv — M horizon at EMS tower (n≈135).

    Returns
    -------
    {pool_name: (mean_gC_m2, sigma_gC_m2)}
        Only ``soil_active`` is returned (and only when present in pool_names).
    """
    # Prior-τ fraction for the active pool within the organic horizon.
    # Used only to anchor the soil_active individual constraint.
    # τ_active=730d, τ_slow=7300d, f_as=0.25 → C_active:C_slow = 730:1825
    _TAU_ACTIVE = 730.0
    _TAU_SLOW   = 7300.0
    _F_AS       = 0.25
    _r_act  = _TAU_ACTIVE
    _r_slw  = _F_AS * _TAU_SLOW
    _f_act  = _r_act / (_r_act + _r_slw)   # ≈ 0.286
    _SIGMA_REL = 0.40                        # 40% relative σ on the split

    pool_name_set = set(pool_names)
    result = {}

    hz = _load_horizon_means(hf324_path, hf271_path)

    if "soil_active" in pool_name_set and "org_mean" in hz:
        c_act = _f_act * hz["org_mean"]
        s_act = max(hz["org_sem"] * _f_act, _SIGMA_REL * c_act)
        result["soil_active"] = (c_act, s_act)

    return result


def _build_carbon_sum_obs(hf324_path: str, hf271_path: str, pool_index) -> list:
    """
    Build two sum-constraint ObsBlocks for carbon stocks.

    SUM CONSTRAINT 1 — organic horizon total (hf324 Munger, n≈214):
        C_active_sim + C_slow_sim ≈ org_mean ± org_sem
        Rationale: the organic horizon (Oi/Oe) contains both the active and slow
        SOM fractions; we know the total horizon stock precisely but not how much
        belongs to each pool.  Using the sum avoids the arbitrary prior-τ split.

    SUM CONSTRAINT 2 — mineral horizon total (hf271 M horizon, n≈135):
        C_slow_sim + C_passive_sim ≈ min_mean ± min_σ
        Rationale: the mineral horizon contains both mineral-associated fast SOM
        (slow pool) and older mineral-protected SOM (passive pool).  Assigning it
        entirely to passive over-constrains one pool and leaves slow unconstrained.
        σ = max(SEM, 35% relative) to account for spatial heterogeneity.

    Returns
    -------
    list[ObsBlock]  — two blocks (organic sum, mineral sum); empty if data missing.
    """
    from ecosystem_complexity.api import ObsBlock

    _SIGMA_REL_MIN = 0.35   # 35% relative σ on mineral horizon (spatial noise)

    hz = _load_horizon_means(hf324_path, hf271_path)
    pool_names_set = set(pool_index.pool_names)
    blocks = []

    # ── Organic sum: C_active + C_slow ──────────────────────────────────────
    if ("soil_active" in pool_names_set and "soil_slow" in pool_names_set
            and "org_mean" in hz):
        org_mean = hz["org_mean"]
        org_sem  = hz["org_sem"]
        i_act = pool_index["soil_active"]
        i_slw = pool_index["soil_slow"]
        _cols_org = jnp.array([i_act, i_slw], dtype=jnp.int32)
        blocks.append(ObsBlock(
            name="c_sum_organic",
            y=jnp.array([org_mean], dtype=jnp.float32),
            Se=jnp.array([org_sem ** 2], dtype=jnp.float32),
            predict=lambda out, p, cols=_cols_org:
                jnp.sum(jnp.mean(out.C12, axis=0)[cols], keepdims=True),
        ))
        print(f"  C stock organic sum  (active+slow):  "
              f"{org_mean:.0f} ± {org_sem:.0f} gC m⁻²  "
              f"(n={hz['org_n']})")

    # ── Mineral sum: C_slow + C_passive ──────────────────────────────────────
    if ("soil_slow" in pool_names_set and "soil_passive" in pool_names_set
            and "min_mean" in hz):
        min_mean  = hz["min_mean"]
        min_sigma = max(hz["min_sem"], _SIGMA_REL_MIN * min_mean)
        i_slw = pool_index["soil_slow"]
        i_pas = pool_index["soil_passive"]
        _cols_min = jnp.array([i_slw, i_pas], dtype=jnp.int32)
        blocks.append(ObsBlock(
            name="c_sum_mineral",
            y=jnp.array([min_mean], dtype=jnp.float32),
            Se=jnp.array([min_sigma ** 2], dtype=jnp.float32),
            predict=lambda out, p, cols=_cols_min:
                jnp.sum(jnp.mean(out.C12, axis=0)[cols], keepdims=True),
        ))
        print(f"  C stock mineral sum  (slow+passive): "
              f"{min_mean:.0f} ± {min_sigma:.0f} gC m⁻²  "
              f"(n={hz['min_n']})")

    return blocks


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

    # Derive OE field sets from config so optimize_partition / other flags are
    # automatically respected — no manual sync needed when the YAML changes.
    global _OPT_FIELDS, _OPT_FIELDS_ER
    inv_cfg_nb   = getattr(config, "inversion_raw", {}) or {}
    _OPT_FIELDS    = _get_oe_fields(config, inv_cfg_nb)
    _OPT_FIELDS_ER = _OPT_FIELDS + ("log_f_hetero",)
    print(f"  OE fields (OE1–4): {_OPT_FIELDS}")
    print(f"  OE fields (OE5):   {_OPT_FIELDS_ER}")

    # ── Forcing ──────────────────────────────────────────────────────────────
    print("Loading flux forcing…")
    forcing_full, obs_raw = load_harvard_forest(
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

    # Slice ER from raw observations to the same time window
    er_full = np.array(obs_raw.ER)
    er_sliced_raw = er_full[start_idx:start_idx + T].copy()

    # Mask out known disturbance years where ER is unreliable.
    # 2005: ice storm / hurricane disturbance at Harvard Forest resulted in
    # anomalously low ER (obs=0.54 gC m⁻² day⁻¹, less than half the long-term
    # mean).  Including it biases the ER constraint.
    _ER_OUTLIER_YEARS = {2005}
    _er_time_yrs = 1970.0 + np.array(forcing.time) / 365.25
    for _yr in _ER_OUTLIER_YEARS:
        _mask_yr = (_er_time_yrs >= _yr) & (_er_time_yrs < _yr + 1)
        er_sliced_raw[_mask_yr] = np.nan
        print(f"  ER: masked year {_yr} as disturbance outlier "
              f"({int(_mask_yr.sum())} days set to NaN)")

    er_sliced = jnp.array(er_sliced_raw, dtype=jnp.float32)
    n_er_valid = int(np.sum(np.isfinite(np.array(er_sliced))))
    print(f"  FluxNet ER: {n_er_valid} valid daily obs in window")
    time_years = 1970.0 + np.array(forcing.time) / 365.25
    print(f"  Forcing: {T} days  ({time_years[0]:.1f}–{time_years[-1]:.1f})")

    # ── Observations ─────────────────────────────────────────────────────────
    print("Building observations…")
    delta14C_obs  = _build_pool_14C_obs_bulk(HF_SOIL_14C_PATH, forcing.time, idx.pool_names)
    delta14C_resp = _build_resp_14C_obs(HF_RESP_14C_PATH, forcing.time)

    # Individual C stock constraint (soil_active only).
    # soil_slow and soil_passive are handled by sum constraints below.
    c_pools_obs   = _build_soil_carbon_obs(HF_SOIL_C_PATH2, HF_SOIL_C_PATH, idx.pool_names)

    n_pool_obs = sum(int(jnp.sum(~jnp.isnan(a))) for a in delta14C_obs.values())
    n_resp_obs = int(jnp.sum(~jnp.isnan(delta14C_resp)))
    print(f"  Pool Δ¹⁴C obs: {n_pool_obs}  |  Resp Δ¹⁴C obs: {n_resp_obs}")
    if c_pools_obs:
        pn, (mu, sig) = next(iter(c_pools_obs.items()))
        print(f"  C stock {pn} (individual):       {mu:.0f} ± {sig:.0f} gC m⁻²")

    # Sum constraints: organic total (active+slow) and mineral total (slow+passive).
    print("Building carbon sum constraints…")
    carbon_sum_blocks = _build_carbon_sum_obs(HF_SOIL_C_PATH2, HF_SOIL_C_PATH, idx)

    # ISRaD density-fraction Δ¹⁴C (extra ObsBlocks; per-obs σ 15–138‰)
    print("Building ISRaD fraction Δ¹⁴C obs blocks…")
    israd_blocks = _build_israd_14C_obs(ISRAD_FRAC_PATH, forcing.time, idx)
    print(f"  ISRaD blocks: {len(israd_blocks)} (one per pool×year pair)")

    # All extra blocks: sum constraints first (lower Se → stronger signal),
    # then ISRaD fraction Δ¹⁴C.
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
    # OE3: C stocks + resp Δ¹⁴C  (orthogonality test — no pool Δ¹⁴C)
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
    # OE5: OE4 + FluxNet ER → annual Rh constraint
    obs_all_er = ObservationData(
        time=forcing.time,
        NEE=_nan_T, GPP=_nan_T, ER=er_sliced, NEE_unc=_nan_T,
        delta14C_obs=delta14C_obs,
        deltaD14C_obs={},
        C_pools_obs=c_pools_obs,
        delta14C_resp=delta14C_resp,
    )

    # ── Initial state ────────────────────────────────────────────────────────
    state0 = _build_state0(config, idx)
    print(f"  State0 total C12: {float(jnp.sum(state0.C12)):.0f} gC m⁻²")

    # ── Analytical SS helper (used for posterior diagnostic runs) ────────────
    # Pre-compute mean decomp modifier matching what optimize_oe will use,
    # so diagnostic forward runs start from the same C12 as OE _forward.
    from ecosystem_complexity.api import _analytical_c12_ss
    import numpy as _np
    _air_t_np = _np.nan_to_num(_np.array(forcing.air_temp), nan=5.0)
    _soil_t_raw = _np.array(forcing.soil_temp[:, 0])
    _T_soil_np = _np.where(_np.isnan(_soil_t_raw), _air_t_np, _soil_t_raw)
    _theta_np  = _np.where(_np.isnan(_np.array(forcing.soil_moisture[:, 0])), 0.3,
                           _np.array(forcing.soil_moisture[:, 0]))
    from ecosystem_complexity.fluxes import f_temp as _ft_fn, f_moisture as _fm_fn, thawed_frac as _ff_fn
    _params0 = make_default_params(config)
    _ft_   = _ft_fn(jnp.array(_T_soil_np, dtype=jnp.float32), _params0.log_Q10[0])
    _fm_   = _fm_fn(jnp.array(_theta_np, dtype=jnp.float32), _params0.log_theta_opt[0], _params0.log_gamma_moist[0])
    _fff_  = _ff_fn(jnp.array(_T_soil_np, dtype=jnp.float32))
    _mean_modifier = float(jnp.nanmean(_ft_ * _fm_ * _fff_))
    _cue_val   = float(model.config.external_inputs.CUE)
    _mean_gpp  = float(jnp.nanmean(forcing.GPP_obs))
    _mean_input_val = _mean_gpp * _cue_val
    _n_pools   = len(idx)
    # Target pool indices for 2-way softmax (active+slow only; passive excluded).
    _target_names   = list(model.config.external_inputs.partition.keys())
    _ext_target_idx = [idx[n] for n in _target_names]
    print(f"  Diagnostic SS: mean_modifier={_mean_modifier:.4f}, mean_input={_mean_input_val:.4f} gC/m²/day")
    print(f"  Input partition targets: {_target_names}  (indices {_ext_target_idx})")

    def _make_ss_state(params_opt):
        """Return state0 with C12 replaced by analytical SS for given params."""
        c12_ss = _analytical_c12_ss(params_opt, _n_pools, _mean_input_val, _mean_modifier,
                                     target_indices=_ext_target_idx)
        return state0._replace(C12=c12_ss)

    # ── Sanity checks: prior SS and partition ────────────────────────────────
    params_prior = make_default_params(config)
    _prior_ss    = _analytical_c12_ss(params_prior, _n_pools, _mean_input_val, _mean_modifier,
                                       target_indices=_ext_target_idx)
    _prior_part  = jax.nn.softmax(params_prior.log_external_input_partition)  # (n_targets,)
    print("\nSanity checks (prior):")
    print(f"  Partition sum: {float(_prior_part.sum()):.6f}  (must be 1.000000)")
    # Build full n_pools input fraction vector for display
    _f_input_full = np.zeros(_n_pools)
    for k, ti in enumerate(_ext_target_idx):
        _f_input_full[ti] = float(_prior_part[k])
    for i, nm in enumerate(idx.pool_names):
        print(f"  SS C_{nm:<14} = {float(_prior_ss[i]):7.1f} gC m⁻²"
              f"   (f_input={_f_input_full[i]:.3f})")
    _total_prior_ss = float(jnp.sum(_prior_ss))
    print(f"  SS total C12          = {_total_prior_ss:.1f} gC m⁻²")

    # ── Prior forward run ─────────────────────────────────────────────────────
    print("\nPrior forward simulation…")
    t0 = time.perf_counter()
    out_prior = run_model(model, forcing, state0=state0, params=params_prior)
    jax.block_until_ready(out_prior.delta14C)
    print(f"  Done  [{time.perf_counter()-t0:.1f}s]")

    tau_p   = np.exp(np.array(params_prior.log_tau))
    w_prior = np.array(out_prior.C12) / (tau_p[None, :] + 1e-30)
    d14C_resp_prior = (np.array(out_prior.delta14C) * w_prior).sum(-1) / (w_prior.sum(-1) + 1e-30)

    # Count total extra-block obs for header labels
    _n_extra = len(extra_blocks)
    _n_csum  = len(carbon_sum_blocks)
    _n_israd = len(israd_blocks)

    # Helper: print per-OE C-stock constraint diagnostics
    def _print_cstock_diag(label: str, out):
        c_sim = np.array(jnp.mean(out.C12, axis=0))
        act_sim = c_sim[idx["soil_active"]]
        slw_sim = c_sim[idx["soil_slow"]]
        pas_sim = c_sim[idx["soil_passive"]]
        org_sum_sim = act_sim + slw_sim
        min_sum_sim = slw_sim + pas_sim
        act_obs, act_sig = c_pools_obs.get("soil_active", (float("nan"), float("nan")))
        hz = _load_horizon_means(HF_SOIL_C_PATH2, HF_SOIL_C_PATH)
        org_obs = hz.get("org_mean", float("nan"))
        org_sig = hz.get("org_sem",  float("nan"))
        min_obs = hz.get("min_mean", float("nan"))
        min_sig = max(hz.get("min_sem", float("nan")), 0.35 * min_obs)
        print(f"  {label} C-stock check:")
        print(f"    soil_active  (individual):  sim={act_sim:.0f}  obs={act_obs:.0f}±{act_sig:.0f}  "
              f"resid={(act_sim-act_obs)/act_sig:+.2f}σ")
        print(f"    organic sum  (active+slow):  sim={org_sum_sim:.0f}  obs={org_obs:.0f}±{org_sig:.0f}  "
              f"resid={(org_sum_sim-org_obs)/org_sig:+.2f}σ")
        print(f"    mineral sum  (slow+passive): sim={min_sum_sim:.0f}  obs={min_obs:.0f}±{min_sig:.0f}  "
              f"resid={(min_sum_sim-min_obs)/min_sig:+.2f}σ")
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
    if ch1[-1] > 50:
        print(f"  ⚠ OE1 final cost {ch1[-1]:.1f} > 50 — check SS initialisation")

    out_carbon_only = run_model(model, forcing, state0=_make_ss_state(result_carbon_only.params_opt),
                                params=result_carbon_only.params_opt)
    jax.block_until_ready(out_carbon_only.delta14C)
    _print_cstock_diag("OE1", out_carbon_only)

    # ── OE Run 2 — C stocks + pool Δ¹⁴C + ISRaD frac Δ¹⁴C ─────────────────
    print(f"\nOE 2 — C stocks + pool Δ¹⁴C + ISRaD + sum constraints  "
          f"({_n_cstock1} C + {n_pool_obs} pool Δ¹⁴C + {_n_israd} ISRaD obs)…")
    t0 = time.perf_counter()
    result_carbon_pool = optimize_oe(model, forcing, obs_carbon_pool14C, state0=state0,
                                     fields=_OPT_FIELDS, extra_obs_blocks=extra_blocks)
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

    # ── OE Run 4 — full Δ¹⁴C + C stocks + ISRaD frac ────────────────────────
    print(f"\nOE 4 — full OE: pool + resp Δ¹⁴C + C stocks + ISRaD  "
          f"({_n_cstock1} C + {n_pool_obs} pool + {n_resp_obs} resp + {_n_israd} ISRaD obs)…")
    t0 = time.perf_counter()
    result_all = optimize_oe(model, forcing, obs_all, state0=state0,
                             fields=_OPT_FIELDS, extra_obs_blocks=extra_blocks)
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

    # ── OE Run 5 — full OE + FluxNet ER + ISRaD + free f_hetero ─────────────
    print(f"\nOE 5 — OE4 + FluxNet ER (free f_hetero)  "
          f"(full Δ¹⁴C + C stocks + ER + ISRaD)…")
    t0 = time.perf_counter()
    result_all_er = optimize_oe(model, forcing, obs_all_er, state0=state0,
                                fields=_OPT_FIELDS_ER, extra_obs_blocks=extra_blocks)
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

    params_opt = result_all_er.params_opt   # best run is OE5

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

    # Partition arrays have n_target_pools entries (2 for active+slow only).
    part_opt_vec   = _softmax(np.array(params_opt.log_external_input_partition))
    part_prior_vec = _softmax(np.array(params_prior.log_external_input_partition))
    # Map to full n_pools vector for display
    part_opt_full   = np.zeros(_n_pools)
    part_prior_full = np.zeros(_n_pools)
    for k, ti in enumerate(_ext_target_idx):
        part_opt_full[ti]   = float(part_opt_vec[k])
        part_prior_full[ti] = float(part_prior_vec[k])

    print(f"\n{'Pool':<16}  {'τ prior (yr)':>13}  {'τ opt (yr)':>12}")
    print("  " + "─" * 43)
    for i, name in enumerate(idx.pool_names):
        print(f"  {name:<16}  {tau_p[i]/365:>13.1f}  {tau_opt[i]/365:>12.1f}")

    print(f"\n{'Input fractions':<18}  {'prior':>8}  {'optimised':>10}")
    print("  " + "─" * 41)
    for i, name in enumerate(idx.pool_names):
        note = "" if name in _target_names else "  (cascade only)"
        print(f"  {name:<18}  {part_prior_full[i]:>8.3f}  {part_opt_full[i]:>10.3f}{note}")
    print(f"  {'sum (sanity)':<18}  {part_prior_vec.sum():>8.3f}  {part_opt_vec.sum():>10.3f}"
          f"  ← must be 1.000")

    # f_hetero posterior (OE5 only)
    f_het_prior_val = float(jax.nn.sigmoid(params_prior.log_f_hetero))
    f_het_opt_val   = float(jax.nn.sigmoid(params_opt.log_f_hetero))
    print(f"\n  f_hetero: prior={f_het_prior_val:.3f}  optimised={f_het_opt_val:.3f}"
          f"  (ER → Rh fraction)")

    # ── Information content from OE averaging kernel ─────────────────────────
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

    dfs_6pool_ref   = 1.029
    n_params_6pool_ref = 10
    dof_2pool       = type("_DFS", (), {"dfs_total": dfs_oe, "dfs_by_group": {}})()
    n_params_2pool  = n_params_oe

    # ── DFS per experiment ────────────────────────────────────────────────────
    print("\nDFS by experiment:")
    for label, res in [("OE1 C-stocks",        result_carbon_only),
                       ("OE2 +pool Δ¹⁴C",      result_carbon_pool),
                       ("OE3 +resp Δ¹⁴C",      result_carbon_resp),
                       ("OE4 full",             result_all),
                       ("OE5 full+ER flux",     result_all_er)]:
        A = np.array(res.averaging_kernel)
        print(f"  {label:<22} DFS={np.trace(A):.3f} / {A.shape[0]} params")

    # ── Annual Rh diagnostics (OE5) ───────────────────────────────────────────
    print("\nAnnual Rh diagnostics (OE5):")
    er_np      = np.array(er_sliced)
    gpp_np     = np.array(forcing.GPP_obs)
    time_yrs   = np.array(forcing.time)
    years_np   = 1970.0 + time_yrs / 365.25
    rh5        = np.array(out_all_er.Rh)   # Rh from model (correct formula)
    # f_hetero posterior from OE5
    f_het_oe5  = float(jax.nn.sigmoid(params_opt.log_f_hetero))
    mean_gpp_all = float(np.nanmean(gpp_np))
    print(f"  f_hetero (OE5 posterior): {f_het_oe5:.3f}  mean_GPP: {mean_gpp_all:.3f} gC m⁻² day⁻¹")
    for yr in range(int(years_np[0]), int(years_np[-1]) + 1):
        mask = (years_np >= yr) & (years_np < yr + 1)
        er_yr  = er_np[mask]; er_yr  = er_yr[np.isfinite(er_yr)]
        gpp_yr = gpp_np[mask]; gpp_yr = gpp_yr[np.isfinite(gpp_yr)]
        if len(er_yr) < 30:
            continue
        rh_obs_yr  = float(np.mean(er_yr)) * f_het_oe5   # Rh_obs using posterior f_hetero
        rh_sim_yr  = float(np.mean(rh5[mask]))
        gpp_yr_mean = float(np.mean(gpp_yr)) if len(gpp_yr) > 10 else np.nan
        gpp_note   = f"  GPP={gpp_yr_mean:.3f} (Δ={gpp_yr_mean-mean_gpp_all:+.3f})" if np.isfinite(gpp_yr_mean) else ""
        print(f"  {yr}: Rh_obs={rh_obs_yr:.3f}  Rh_sim={rh_sim_yr:.3f}  diff={rh_sim_yr-rh_obs_yr:+.3f} gC m⁻² day⁻¹{gpp_note}")

    # ── Carbon stock diagnostics ──────────────────────────────────────────────
    print("\nCarbon stock constraint diagnostics:")
    hz_diag = _load_horizon_means(HF_SOIL_C_PATH2, HF_SOIL_C_PATH)
    org_obs_diag = hz_diag.get("org_mean", float("nan"))
    org_sem_diag = hz_diag.get("org_sem",  float("nan"))
    min_obs_diag = hz_diag.get("min_mean", float("nan"))
    min_sig_diag = max(hz_diag.get("min_sem", float("nan")), 0.35 * min_obs_diag)
    act_obs_diag, act_sig_diag = c_pools_obs.get("soil_active", (float("nan"), float("nan")))

    def _pool_means(out):
        c = np.mean(np.array(out.C12), axis=0)
        act = c[idx["soil_active"]]; slw = c[idx["soil_slow"]]; pas = c[idx["soil_passive"]]
        return act, slw, pas

    rows = [("OE1", out_carbon_only), ("OE2", out_carbon_pool),
            ("OE3", out_carbon_resp), ("OE4", out_all), ("OE5", out_all_er)]

    print(f"  {'Constraint':<28}  {'obs':>7}  {'±σ':>6}  "
          + "  ".join(f"{lbl:>6}" for lbl, _ in rows))
    for label, obs_val, sig_val, fn in [
        ("soil_active (individual)",  act_obs_diag, act_sig_diag, lambda a,s,p: a),
        ("organic sum (active+slow)", org_obs_diag, org_sem_diag, lambda a,s,p: a+s),
        ("mineral sum (slow+passive)",min_obs_diag, min_sig_diag, lambda a,s,p: s+p),
        ("  soil_active",             float("nan"),  float("nan"), lambda a,s,p: a),
        ("  soil_slow",               float("nan"),  float("nan"), lambda a,s,p: s),
        ("  soil_passive",            float("nan"),  float("nan"), lambda a,s,p: p),
    ]:
        sims = [fn(*_pool_means(out)) for _, out in rows]
        obs_str = f"{obs_val:.0f}" if np.isfinite(obs_val) else "  —"
        sig_str = f"{sig_val:.0f}" if np.isfinite(sig_val) else "  —"
        print(f"  {label:<28}  {obs_str:>7}  {sig_str:>6}  "
              + "  ".join(f"{s:>6.0f}" for s in sims))

    # ── Age diagnostics ───────────────────────────────────────────────────────
    print("\nAge diagnostics (OE5 full+ER run)…")
    age_diag = compute_age_diagnostics(out_all_er, params_opt, model)
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
        dof_2pool=dof_2pool,
        n_params_2pool=n_params_2pool,
        dfs_6pool_ref=dfs_6pool_ref,
        n_params_6pool_ref=n_params_6pool_ref,
        age_diag=age_diag,
        tau_p=tau_p,
        tau_opt=tau_opt,
        lh1=ch1,
        lh2=ch2,
        lh3=ch3,
        lh4=ch4,
        lh5=ch5,
        dfs_oe=dfs_oe,
        n_params_oe=n_params_oe,
    )


# ════════════════════════════════════════════════════════════════════════════
# Figure
# ════════════════════════════════════════════════════════════════════════════

def make_figure(r: dict, out_path: str | None = None):
    if out_path is None:
        out_path = _wt("notebooks/harvard_optimal_model.png")
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
    dof_2pool    = r["dof_2pool"]
    age_diag     = r["age_diag"]
    tau_p        = r["tau_p"]
    tau_opt      = r["tau_opt"]
    lh1, lh2, lh3, lh4, lh5 = r["lh1"], r["lh2"], r["lh3"], r["lh4"], r["lh5"]

    pool_names = pool_idx.pool_names
    pool_colors  = ["tab:green", "tab:orange", "tab:brown", "tab:purple"]
    pool_markers = ["o", "^", "s", "D"]

    fig = plt.figure(figsize=(16, 19))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.32)
    axes = [fig.add_subplot(gs[r_, c]) for r_ in range(4) for c in range(2)]
    ax_loss, ax_tau, ax_pool14C, ax_resp14C, ax_info, ax_age, ax_gpp, ax_cstock = axes

    # ── (a) Loss convergence ─────────────────────────────────────────────────
    for lh, label, color in [
        (lh1, "OE1: C stocks only",       C_OE1),
        (lh2, "OE2: +pool Δ¹⁴C",         C_OE2),
        (lh3, "OE3: +resp Δ¹⁴C",         C_OE3),
        (lh4, "OE4: full",                C_OE4),
        (lh5, "OE5: full + ER flux",      C_OE5),
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
    ax_tau.bar(x + w/2, tau_opt / 365, width=w, color=C_OE3,  label="optimised", alpha=0.85)
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
        oe2_line   = np.array(out_oe2.delta14C)[:, pi]
        oe4_line   = np.array(out_oe4.delta14C)[:, pi]
        oe5_line   = np.array(out_oe5.delta14C)[:, pi]

        ax_pool14C.plot(time_years, prior_line, color=C_PRIOR, lw=1.0, alpha=0.5,
                        linestyle="--", label="prior" if i == 0 else None)
        ax_pool14C.plot(time_years, oe2_line,  color=color, lw=1.0, alpha=0.6,
                        linestyle=":")
        ax_pool14C.plot(time_years, oe4_line,  color=color, lw=1.4, alpha=0.8,
                        linestyle="-.")
        ax_pool14C.plot(time_years, oe5_line,  color=color, lw=1.8,
                        label=f"{pool_name.replace('soil_','')} (OE5)")

        if pool_name in delta14C_obs:
            obs_arr  = np.array(delta14C_obs[pool_name])
            obs_mask = ~np.isnan(obs_arr)
            ax_pool14C.scatter(time_years[obs_mask], obs_arr[obs_mask],
                               color=color, marker=marker, s=60, zorder=5,
                               label=f"{pool_name.replace('soil_','')} obs")

    ax_pool14C.set_xlabel("Year")
    ax_pool14C.set_ylabel("Δ¹⁴C (‰)")
    ax_pool14C.set_title("(c) Pool Δ¹⁴C  (prior– OE2··· OE4-·- OE5—)")
    ax_pool14C.legend(fontsize=8, ncol=2)

    # ── (d) Respired CO₂ Δ¹⁴C ───────────────────────────────────────────────
    resp_arr  = np.array(d14C_resp)
    resp_mask = ~np.isnan(resp_arr)

    ax_resp14C.plot(time_years, r["d14C_resp_prior"], color=C_PRIOR, lw=1.0,
                    linestyle="--", label="prior", alpha=0.7)
    ax_resp14C.plot(time_years, r["d14C_resp_oe1"],   color=C_OE1,  lw=1.0,
                    linestyle=":", label="OE1: C stocks only", alpha=0.8)
    ax_resp14C.plot(time_years, r["d14C_resp_oe2"],   color=C_OE2,  lw=1.2,
                    linestyle="-.", label="OE2: +pool Δ¹⁴C", alpha=0.85)
    ax_resp14C.plot(time_years, r["d14C_resp_oe3"],   color=C_OE3,  lw=1.4,
                    linestyle=(0, (3, 1)), label="OE3: +resp Δ¹⁴C", alpha=0.85)
    ax_resp14C.plot(time_years, r["d14C_resp_oe4"],   color=C_OE4,  lw=1.6,
                    linestyle="-.", label="OE4: full", alpha=0.9)
    ax_resp14C.plot(time_years, r["d14C_resp_oe5"],   color=C_OE5,  lw=2.0,
                    label="OE5: full + ER flux")
    ax_resp14C.scatter(time_years[resp_mask], resp_arr[resp_mask],
                       color="k", marker="x", s=30, zorder=5,
                       label="obs (hf212-01 NWN)")
    ax_resp14C.set_xlabel("Year")
    ax_resp14C.set_ylabel("Δ¹⁴C (‰)")
    ax_resp14C.set_title("(d) Respired CO₂ Δ¹⁴C")
    ax_resp14C.legend(fontsize=8)

    # ── (e) Carbon stock constraints: obs vs. modelled ───────────────────────
    _bar_labels = ["OE1\nC stocks", "OE2\n+pool Δ¹⁴C", "OE3\n+resp Δ¹⁴C",
                   "OE4\nfull", "OE5\n+ER flux"]
    _x_base = np.arange(len(_bar_labels))
    _bar_w  = 0.18

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
                    color=color, alpha=0.75,
                    label=pool_name.replace("soil_", ""))

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

    # ── (g) GPP forcing ───────────────────────────────────────────────────────
    gpp_arr = np.array(r["forcing_GPP"])
    # Annual mean GPP
    yrs_unique = np.arange(int(time_years[0]), int(time_years[-1]) + 1)
    gpp_annual = []
    for yr in yrs_unique:
        mask = (time_years >= yr) & (time_years < yr + 1)
        if mask.sum() > 10:
            gpp_annual.append(float(np.nanmean(gpp_arr[mask])))
        else:
            gpp_annual.append(np.nan)
    gpp_annual = np.array(gpp_annual)

    # Annual ER
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
    # Annotate implied annual Rh = ER × f_hetero
    rh_annual_obs = er_annual * 0.55
    ax_gpp.plot(yrs_unique + 0.5, rh_annual_obs, color="saddlebrown", lw=1.5,
                linestyle=":", label="Rh est. (ER×0.55)")
    ax_gpp.set_xlabel("Year")
    ax_gpp.set_ylabel("gC m⁻² day⁻¹")
    ax_gpp.set_title("(g) GPP and ER forcing (AMF US-Ha1)")
    ax_gpp.legend(fontsize=7)
    ax_gpp.set_xlim(time_years[0], time_years[-1])

    # ── (h) C stock time series — prior vs OE3 ────────────────────────────────
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
                              label=f"{pool_name.replace('soil_','')} obs ±SEM")

    ax_cstock.set_xlabel("Year")
    ax_cstock.set_ylabel("C stock (gC m⁻²)")
    ax_cstock.set_title("(h) C stocks: prior (--) vs OE5 (—), obs bands (·)")
    ax_cstock.legend(fontsize=8, ncol=2)
    ax_cstock.set_xlim(time_years[0], time_years[-1])

    fig.suptitle(
        "Harvard Forest — 3-Pool Optimal Model  (active + slow + passive)\n"
        "OE5 adds FluxNet ER → annual Rh constraint (f_hetero=0.55, σ=15%)",
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
