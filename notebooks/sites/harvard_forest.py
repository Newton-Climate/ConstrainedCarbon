"""
sites/harvard_forest.py — Harvard Forest (US-Ha1) site module.

Contains all Harvard-Forest-specific data helpers, the full OE5 inversion
workflow, and the site figure.  Imported by ``notebooks/harvard_optimal_model.py``.

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
_SITES_ROOT   = os.path.dirname(os.path.abspath(__file__))
_NB_ROOT      = os.path.dirname(_SITES_ROOT)          # notebooks/
_WORKTREE_ROOT = os.path.dirname(_NB_ROOT)             # repo / worktree root
_SRC_ROOT     = os.path.join(_WORKTREE_ROOT, "src")

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
from ecosystem_complexity.data.parsers import attach_atm14C, load_harvard_forest, slice_forcing
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.schemas import ForcingData, ObservationData
from ecosystem_complexity.data.israd_observations import (
    FractionMappingRule,
    build_fraction_obs_blocks,
    add_layer_midpoint,
    summarize_by_depth,
    obs_blocks_from_single_year_summary,
    build_perlayer_mixture_obs_blocks,
)
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
HF_HR_PATH       = _data("data/harvard_forest/AMF_US-Ha1_FLUXNET_FULLSET_1991-2020_3-5/"
                          "AMF_US-Ha1_FLUXNET_FULLSET_HR_1991-2020_3-5.csv")
HF_SOIL_14C_PATH = _data("data/harvard_forest/hf212-03-14c-org.csv")
HF_RESP_14C_PATH = _data("data/harvard_forest/hf212-01-14c-no-treat.csv")
HF_SOIL_C_PATH   = _data("data/harvard_forest/hf271-07-soils.csv")
HF_SOIL_C_PATH2  = _data("data/harvard_forest/hf324-06-soil-carbon.csv")
ISRAD_FRAC_PATH  = _data("data/shared/israd/ISRaD_extra_flat_fraction_v 2.6.6.2024-01-25.csv")
ISRAD_LAYER_PATH = _data("data/shared/israd/ISRaD_extra_flat_layer_v 2.6.6.2024-01-25.csv")
HUA_PATH         = _data("data/shared/atm_14C/Hua_2021.csv")
GRAVEN_PATH      = _data("data/shared/atm_14C/Graven_2017.csv")
INTCAL_PATH      = _data("data/shared/atm_14C/intcal20.14c")
OPT_CONFIG       = _wt("configs/harvard_3pool_config.yaml")

_R_STD              = 1.176e-12
_HF324_CORE_AREA_CM2 = 10.0


# ════════════════════════════════════════════════════════════════════════════
# Harvard Forest data helpers
# ════════════════════════════════════════════════════════════════════════════

def build_pool_14C_obs(soil_14c_path: str, forcing_time, pool_names: list) -> dict:
    """
    Build ``delta14C_obs`` by mapping hf212-03 horizons to model pools.

    3/4-pool mapping:  soil_active ← Oi,  soil_slow ← mean(Oe, A-lf),
                       soil_passive ← A-min,  soil_stable not constrained.
    2-pool fallback:   soil_active ← mean(Oi, Oe),  soil_passive ← mean(A-lf, A-min).
    """
    d14c_df = pd.read_csv(soil_14c_path)
    d14c_df["horizon"] = d14c_df["horizon"].str.strip()

    time_np  = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T = len(time_np)

    pool_name_set = set(pool_names)
    has_slow = "soil_slow" in pool_name_set

    if has_slow:
        horizon_map = {
            "soil_active":  ["Oi"],
            "soil_slow":    ["Oe", "A-lf"],
            "soil_passive": ["A-min"],
        }
    else:
        horizon_map = {
            "soil_active":  ["Oi", "Oe"],
            "soil_passive": ["A-lf", "A-min"],
        }

    obs_dict: dict[str, np.ndarray] = {}
    for year in d14c_df["year"].unique():
        yr_df = d14c_df[d14c_df["year"] == year]
        t_idx = int(np.argmin(np.abs(years_np - (float(year) + 0.5))))
        for pool_name, horizons in horizon_map.items():
            if pool_name not in pool_name_set:
                continue
            vals = yr_df.loc[yr_df["horizon"].isin(horizons), "of.14c.12c"].dropna()
            if len(vals):
                if pool_name not in obs_dict:
                    obs_dict[pool_name] = np.full(T, np.nan, dtype=np.float32)
                obs_dict[pool_name][t_idx] = float(vals.mean())

    return {k: jnp.array(v) for k, v in obs_dict.items()}


def build_resp_14C_obs(resp_14c_path: str, forcing_time) -> jnp.ndarray:
    """NWN daily-mean respired Δ¹⁴C, 1996–2010."""
    df  = pd.read_csv(resp_14c_path)
    nwn = df[df["site"] == "NWN"]
    by_date = nwn.groupby("year.date")["delta.14c"].mean()

    time_np  = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T        = len(time_np)

    arr = np.full(T, np.nan, dtype=np.float32)
    for obs_year, d14c_val in by_date.items():
        t_idx = int(np.argmin(np.abs(years_np - float(obs_year))))
        arr[t_idx] = float(d14c_val)
    return jnp.array(arr)


def build_israd_14C_obs(israd_frac_path: str, forcing_time, pool_index) -> list:
    """
    Build ISRaD density-fraction Δ¹⁴C observations as a list of ObsBlocks.

    Fraction-to-pool mapping:  free light → soil_active,
    occluded light → soil_slow,  heavy → soil_passive.
    McFarlane_2013 soil_passive excluded (density-cutoff inconsistency).
    """
    if not os.path.isfile(israd_frac_path):
        print(f"  ISRaD frac file not found: {israd_frac_path} — skipping")
        return []

    df = pd.read_csv(israd_frac_path, low_memory=False)
    df = df[
        df["site_name"].str.contains("Harvard", na=False) &
        (df["frc_scheme"] == "density")
    ]

    _ENTRY_YEAR = {
        "Gaudinski_2000": 1996,
        "Savage_unpub":   2007,
        "McFarlane_2013": 2011,
    }
    _EXCLUDE = {("soil_passive", 2011)}
    rows = build_fraction_obs_blocks(
        df,
        forcing_time,
        pool_index,
        rules=[
            FractionMappingRule("soil_active", "free light"),
            FractionMappingRule("soil_slow", "occluded light"),
            FractionMappingRule("soil_passive", "heavy"),
        ],
        entry_to_year=_ENTRY_YEAR,
        exclude_pool_years=_EXCLUDE,
        weight_col="frc_mass_perc",
        name_prefix="israd",
    )
    blocks = []
    for row in rows:
        print(
            f"  ISRaD obs: {row['pool_name']:<14} yr={row['obs_year']}  "
            f"Δ¹⁴C={row['mean']:+.1f}‰  σ={row['sigma']:.1f}‰  n={row['n']}"
        )
        blocks.append(row["block"])
    return blocks


# Depth→pool bins for bulk-layer Δ¹⁴C.  Deliberately identical to the HF
# C-stock depth groups in sites/canonical.py so the bulk pool constraint is
# an apples-to-apples swap for the density-fraction constraint on the same
# three pools:  O (above 0 cm) → active, A (0–15 cm) → slow,
# B (15–100 cm) → passive.
_HF_BULK_DEPTH_TO_POOL = [
    ("soil_active",  (-15.0,   0.0)),
    ("soil_slow",    (  0.0,  15.0)),
    ("soil_passive", ( 15.0, 100.0)),
]
# McFarlane_2013 measured whole-soil (bulk) Δ¹⁴C on the same H1–H5 flux-tower
# cores that provide the density fractions, so the two are NOT independent
# samples — bulk is the mass-weighted mixture of those fractions.
_HF_BULK_ENTRY    = "McFarlane_2013"
_HF_BULK_PROFILES = ["H1", "H2", "H3", "H4", "H5"]
_HF_BULK_OBS_YEAR = 2007


def build_hf_bulk_14C_obs(
    israd_layer_path: str,
    forcing_time,
    pool_index,
    print_summary: bool = True,
) -> list:
    """Build ISRaD bulk (whole-soil) layer Δ¹⁴C observations as ObsBlocks.

    Bins the McFarlane_2013 H1–H5 depth-resolved ``lyr_14c`` profiles into the
    three model pools by depth midpoint (see ``_HF_BULK_DEPTH_TO_POOL``) and
    emits one ObsBlock per pool at the 2007 observation year.  In contrast to
    the density-fraction blocks — which isolate one kinetic pool per fraction —
    each bulk value is the mass-weighted mixture of all pools present at that
    depth, so the depth→pool assignment leans on the (strong) vertical age
    gradient rather than chemical separation.
    """
    if not os.path.isfile(israd_layer_path):
        print(f"  ISRaD layer file not found: {israd_layer_path} — skipping")
        return []

    df = pd.read_csv(israd_layer_path, low_memory=False)
    df = df[
        (df["entry_name"] == _HF_BULK_ENTRY)
        & df["pro_name"].isin(_HF_BULK_PROFILES)
    ].copy()
    df = df.dropna(subset=["lyr_14c", "lyr_top", "lyr_bot"])
    df = add_layer_midpoint(df)

    summary = summarize_by_depth(
        df, "lyr_14c", _HF_BULK_DEPTH_TO_POOL, min_n=2, sigma_floor=25.0
    )
    blocks = obs_blocks_from_single_year_summary(
        summary, pool_index, forcing_time, _HF_BULK_OBS_YEAR,
        name_prefix="israd_bulk",
    )
    if print_summary:
        depth_by_pool = dict((p, dr) for p, dr in _HF_BULK_DEPTH_TO_POOL)
        for pool_name, (mean_val, sigma_val, n_obs) in summary.items():
            z_top, z_bot = depth_by_pool[pool_name]
            print(
                f"  HF bulk Δ¹⁴C {pool_name:<14s} ({z_top:.0f}–{z_bot:.0f} cm): "
                f"{mean_val:+7.1f} ± {sigma_val:5.1f}‰  "
                f"(n={n_obs} layers, {_HF_BULK_ENTRY} {_HF_BULK_OBS_YEAR})"
            )
    return blocks


# Minimum profile depth reach (cm) to count a McFarlane core as a whole-soil
# sample for the mass-weighted mixture.  H1 stops at 15 cm and would bias the
# integral young, so it is excluded; H2–H5 reach 45–60 cm.
_HF_BULK_MIX_MIN_DEPTH_CM = 30.0
_HF_BULK_MIX_SIGMA_FLOOR   = 10.0


def build_hf_bulk_mixture_14C_obs(
    israd_layer_path: str,
    forcing_time,
    pool_index,
    print_summary: bool = True,
) -> list:
    """Build the physically-correct *mass-weighted mixture* bulk Δ¹⁴C constraint.

    Instead of assigning each depth layer to a single pool (the depth-proxy in
    :func:`build_hf_bulk_14C_obs`), this treats whole-soil Δ¹⁴C for what it is —
    the carbon-mass-weighted mixture of every pool present:

        Δ¹⁴C_bulk = Σ_p (C12_p / Σ C12) · Δ¹⁴C_p

    The predictor evaluates exactly that combination on the model state (it is
    the same quantity as ``compute_age_diagnostics().bulk_delta14C``), so the
    model's own relative pool masses weight the pools inside the bulk value —
    no depth→pool assumption.  The observation is the whole-column, SOC-weighted
    mean of the McFarlane H2–H5 profiles (H1 excluded: too shallow), with the
    across-profile spread as σ.  Because bulk is a single mixture, this is one
    scalar constraint, not one-per-pool.
    """
    if not os.path.isfile(israd_layer_path):
        print(f"  ISRaD layer file not found: {israd_layer_path} — skipping")
        return []

    df = pd.read_csv(israd_layer_path, low_memory=False)
    df = df[
        (df["entry_name"] == _HF_BULK_ENTRY)
        & df["pro_name"].isin(_HF_BULK_PROFILES)
    ].copy()
    df["lyr_14c"] = pd.to_numeric(df["lyr_14c"], errors="coerce")
    df["lyr_soc"] = pd.to_numeric(df["lyr_soc"], errors="coerce")
    df["lyr_bot"] = pd.to_numeric(df["lyr_bot"], errors="coerce")

    # Restrict to whole-soil profiles (deep enough for a representative column).
    reach = df.groupby("pro_name")["lyr_bot"].max()
    deep = reach[reach >= _HF_BULK_MIX_MIN_DEPTH_CM].index
    both = df.dropna(subset=["lyr_14c", "lyr_soc"])
    both = both[both["pro_name"].isin(deep)]

    per_profile: dict[str, float] = {}
    for pro, g in both.groupby("pro_name"):
        w = g["lyr_soc"].to_numpy(dtype=float)
        v = g["lyr_14c"].to_numpy(dtype=float)
        if w.sum() > 0:
            per_profile[pro] = float(np.dot(w, v) / w.sum())

    if len(per_profile) < 2:
        print("  HF bulk-mixture: <2 whole-soil profiles available — skipping")
        return []

    vals = np.array(list(per_profile.values()), dtype=float)
    mean_val = float(vals.mean())
    sigma_val = max(float(vals.std(ddof=1)), _HF_BULK_MIX_SIGMA_FLOOR)

    time_np = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    t_yr = jnp.array(
        np.where((years_np >= _HF_BULK_OBS_YEAR) & (years_np < _HF_BULK_OBS_YEAR + 1))[0],
        dtype=jnp.int32,
    )
    if t_yr.shape[0] == 0:
        t_yr = jnp.array(
            [int(np.argmin(np.abs(years_np - (_HF_BULK_OBS_YEAR + 0.5))))],
            dtype=jnp.int32,
        )

    def _predict_mixture(out, p, t=t_yr):
        c12 = out.C12[t, :]           # (nt, n_pools)
        d14 = out.delta14C[t, :]      # (nt, n_pools)
        num = jnp.sum(c12 * d14, axis=1)
        den = jnp.sum(c12, axis=1) + 1e-30
        return jnp.mean(num / den, keepdims=True)

    block = ObsBlock(
        name="israd_bulkmix",
        y=jnp.array([mean_val], dtype=jnp.float32),
        Se=jnp.array([sigma_val ** 2], dtype=jnp.float32),
        predict=_predict_mixture,
    )
    if print_summary:
        pro_str = ", ".join(f"{k}:{v:+.0f}" for k, v in per_profile.items())
        print(
            f"  HF bulk-mixture Δ¹⁴C (mass-weighted whole soil): "
            f"{mean_val:+.1f} ± {sigma_val:.1f}‰  "
            f"(n={len(per_profile)} profiles [{pro_str}], "
            f"{_HF_BULK_ENTRY} {_HF_BULK_OBS_YEAR})"
        )
    return [block]


# Depth bins for the per-layer mixture operator.  Surface bins (0–15 cm) have
# measured density-fraction masses; deep bins (15–60 cm) do not and fall back
# to the model's own C12 partition.
_HF_PERLAYER_BINS = [
    ("0-5",   (0.0,   5.0)),
    ("5-15",  (5.0,  15.0)),
    ("15-30", (15.0, 30.0)),
    ("30-45", (30.0, 45.0)),
    ("45-60", (45.0, 60.0)),
]
# density fraction property → model pool (free light≈fast, heavy≈mineral/passive)
_HF_FRAC_POOL_MAP = {
    "free light":     "soil_active",
    "occluded light": "soil_slow",
    "heavy":          "soil_passive",
}
_HF_PERLAYER_POOL_ORDER = ("soil_active", "soil_slow", "soil_passive")
_HF_PERLAYER_SIGMA_FLOOR = 15.0


def build_hf_bulk_perlayer_mixture_14C_obs(
    israd_layer_path: str,
    forcing_time,
    pool_index,
    israd_frac_path: str = ISRAD_FRAC_PATH,
    print_summary: bool = True,
) -> list:
    """Per-layer bulk Δ¹⁴C, each layer a carbon-mass-weighted mixture of pools.

    This is the physically-honest bulk operator: each depth bin's whole-soil
    Δ¹⁴C is predicted as a mixture ``Σ_p w_{p,z}·Δ¹⁴C_p`` of the model pools,
    with the mixing weights set by the carbon mass of each pool *in that layer*
    (not a depth→pool assignment, and not one collapsed whole-profile number).

    Weight source per bin:

    * **Surface bins (0–5, 5–15 cm)** — *measured* density-fraction masses
      (``frc_mass_perc``, McFarlane 2013): free light→active, occluded→slow,
      heavy→passive, normalised.  Fixed weights ⇒ predictor is linear in the
      pool Δ¹⁴C.
    * **Deep bins (15–60 cm)** — no measured fractions exist, so weight by the
      model's own C12 partition, i.e. ``Σ_p (C12_p/ΣC12)·Δ¹⁴C_p`` (the same
      quantity as ``compute_age_diagnostics().bulk_delta14C``).  Weights vary
      with the parameters during the inversion.

    Note: because the box model is not depth-resolved, every deep bin shares the
    *same* global-partition predictor and differs only in its observed value, so
    the deep bins jointly pull that one mixture quantity toward the deep bulk
    mean.  Such collinear rows add no independent degrees of freedom — which is
    precisely why bulk-alone information is rank-limited (≈1.3 DFS at HF) even
    though many layers are supplied.

    Thin site adapter around the generalized
    :func:`ecosystem_complexity.data.israd_observations.build_perlayer_mixture_obs_blocks`.
    """
    if not os.path.isfile(israd_layer_path):
        print(f"  ISRaD layer file not found: {israd_layer_path} — skipping")
        return []

    lay = pd.read_csv(israd_layer_path, low_memory=False)
    lay = lay[
        (lay["entry_name"] == _HF_BULK_ENTRY)
        & lay["pro_name"].isin(_HF_BULK_PROFILES)
    ]

    frac_df = None
    if os.path.isfile(israd_frac_path):
        frac_all = pd.read_csv(israd_frac_path, low_memory=False)
        frac_df = frac_all[
            (frac_all["entry_name"] == _HF_BULK_ENTRY)
            & (frac_all["frc_scheme"] == "density")
        ]

    rows = build_perlayer_mixture_obs_blocks(
        lay, pool_index, forcing_time,
        depth_bins=_HF_PERLAYER_BINS,
        obs_year=_HF_BULK_OBS_YEAR,
        fraction_df=frac_df,
        fraction_to_pool=_HF_FRAC_POOL_MAP,
        pool_order=_HF_PERLAYER_POOL_ORDER,
        sigma_floor=_HF_PERLAYER_SIGMA_FLOOR,
        name_prefix="israd_perlayer",
    )
    if print_summary:
        for r in rows:
            if r["weight_source"] == "measured":
                w = r["weights"]
                wsrc = f"measured [{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]"
            else:
                wsrc = "model-C12 partition"
            print(
                f"  HF per-layer bulk Δ¹⁴C {r['bin_label']:>6s} cm: "
                f"{r['mean']:+7.1f} ± {r['sigma']:5.1f}‰  weights={wsrc}  "
                f"(n={r['n']} layers)"
            )
    return [r["block"] for r in rows]


def load_horizon_means(hf324_path: str, hf271_path: str) -> dict:
    """
    Load per-horizon mean C stocks from hf324 (Munger organic) and hf271 (mineral M).

    Returns keys: org_mean, org_sem, org_n, min_mean, min_sem, min_n.
    """
    result = {}
    df324   = pd.read_csv(hf324_path, encoding="latin1")
    munger  = df324[
        (df324["site"] == "ems") & (df324["use.not"] == 1) &
        (df324["contact"].str.lower().str.strip() == "munger")
    ].copy()
    munger["gC_m2"] = munger["c.mass.rocks"] * 10000.0 / _HF324_CORE_AREA_CM2
    org_vals = munger.loc[munger["horizon"] == "organic", "gC_m2"].dropna()
    if len(org_vals) >= 2:
        result.update(org_mean=float(org_vals.mean()),
                      org_sem=float(org_vals.std(ddof=1) / np.sqrt(len(org_vals))),
                      org_n=len(org_vals))
    df271  = pd.read_csv(hf271_path)
    m_vals = df271.loc[df271["horizon"] == "M", "carbon.gm2"].dropna()
    if len(m_vals) >= 2:
        result.update(min_mean=float(m_vals.mean()),
                      min_sem=float(m_vals.std(ddof=1) / np.sqrt(len(m_vals))),
                      min_n=len(m_vals))
    return result


def build_soil_carbon_obs(hf324_path: str, hf271_path: str, pool_names: list) -> dict:
    """
    Individual C stock constraint for soil_active only (prior-τ-ratio split).

    soil_slow and soil_passive are constrained by sum constraints; see
    ``build_carbon_sum_obs``.
    """
    _TAU_ACTIVE, _TAU_SLOW, _F_AS = 730.0, 7300.0, 0.25
    _r_act = _TAU_ACTIVE
    _r_slw = _F_AS * _TAU_SLOW
    _f_act = _r_act / (_r_act + _r_slw)   # ≈ 0.286
    _SIGMA_REL = 0.40

    hz     = load_horizon_means(hf324_path, hf271_path)
    result = {}
    if "soil_active" in set(pool_names) and "org_mean" in hz:
        c_act = _f_act * hz["org_mean"]
        s_act = max(hz["org_sem"] * _f_act, _SIGMA_REL * c_act)
        result["soil_active"] = (c_act, s_act)
    return result


def build_carbon_sum_obs(hf324_path: str, hf271_path: str, pool_index) -> list:
    """
    Two sum-constraint ObsBlocks:
      organic sum  (active + slow)   ≈ hf324 Munger organic
      mineral sum  (slow + passive)  ≈ hf271 M horizon
    """
    _SIGMA_REL_MIN = 0.35
    hz = load_horizon_means(hf324_path, hf271_path)
    pool_names_set = set(pool_index.pool_names)
    blocks = []

    if ("soil_active" in pool_names_set and "soil_slow" in pool_names_set
            and "org_mean" in hz):
        org_mean, org_sem = hz["org_mean"], hz["org_sem"]
        i_act = pool_index["soil_active"]
        i_slw = pool_index["soil_slow"]
        cols  = jnp.array([i_act, i_slw], dtype=jnp.int32)
        blocks.append(ObsBlock(
            name="c_sum_organic",
            y=jnp.array([org_mean], dtype=jnp.float32),
            Se=jnp.array([org_sem ** 2], dtype=jnp.float32),
            predict=lambda out, p, c=cols:
                jnp.sum(jnp.mean(out.C12, axis=0)[c], keepdims=True),
        ))
        print(f"  C stock organic sum  (active+slow):  "
              f"{org_mean:.0f} ± {org_sem:.0f} gC m⁻²  (n={hz['org_n']})")

    if ("soil_slow" in pool_names_set and "soil_passive" in pool_names_set
            and "min_mean" in hz):
        min_mean  = hz["min_mean"]
        min_sigma = max(hz["min_sem"], _SIGMA_REL_MIN * min_mean)
        i_slw = pool_index["soil_slow"]
        i_pas = pool_index["soil_passive"]
        cols  = jnp.array([i_slw, i_pas], dtype=jnp.int32)
        blocks.append(ObsBlock(
            name="c_sum_mineral",
            y=jnp.array([min_mean], dtype=jnp.float32),
            Se=jnp.array([min_sigma ** 2], dtype=jnp.float32),
            predict=lambda out, p, c=cols:
                jnp.sum(jnp.mean(out.C12, axis=0)[c], keepdims=True),
        ))
        print(f"  C stock mineral sum  (slow+passive): "
              f"{min_mean:.0f} ± {min_sigma:.0f} gC m⁻²  (n={hz['min_n']})")

    return blocks


def build_state0(config, pool_index):
    """
    Initialise model state from hf271 C stocks and 1996 hf212-03 Δ¹⁴C.

    3-pool:   active ← 0.40×O,  slow ← 0.60×O + 0.30×M,  passive ← 0.70×M
    4-pool:   as above with soil_stable ← 0.20×M + 5000 gC m⁻² (Δ¹⁴C = −250‰)
    2-pool:   active ← O,  passive ← M
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
        C12_by_pool = {
            "soil_active":  0.40 * org_mean,
            "soil_slow":    0.60 * org_mean + 0.30 * min_mean,
            "soil_passive": 0.50 * min_mean,
            "soil_stable":  0.20 * min_mean + 5000.0,
        }
        d14C_by_pool = {
            "soil_active":  _mean_d14c(["Oi"]),
            "soil_slow":    _mean_d14c(["Oe", "A-lf"]),
            "soil_passive": _mean_d14c(["A-min"]),
            "soil_stable":  -250.0,
        }
    elif has_slow:
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
        C12_by_pool = {"soil_active": org_mean, "soil_passive": min_mean}
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
# Main inversion workflow
# ════════════════════════════════════════════════════════════════════════════

def run_optimal_inversion() -> dict:
    """
    Run the Harvard Forest OE1–OE5 inversion sequence.

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

    inv_cfg    = getattr(config, "inversion_raw", {}) or {}
    oe_fields  = _get_oe_fields(config, inv_cfg)
    oe_fields_er = oe_fields + ("log_f_hetero",)
    print(f"  OE fields (OE1–4): {oe_fields}")
    print(f"  OE fields (OE5):   {oe_fields_er}")

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
    forcing   = slice_forcing(forcing_full, start_idx, len(time_np))
    T         = int(forcing.time.shape[0])

    # Mask disturbance years in ER
    er_sliced_raw = np.array(obs_raw.ER)[start_idx:start_idx + T].copy()
    _ER_OUTLIER_YEARS = {2005}
    _er_time_yrs = 1970.0 + np.array(forcing.time) / 365.25
    for _yr in _ER_OUTLIER_YEARS:
        _mask = (_er_time_yrs >= _yr) & (_er_time_yrs < _yr + 1)
        er_sliced_raw[_mask] = np.nan
        print(f"  ER: masked year {_yr} as disturbance outlier ({int(_mask.sum())} days → NaN)")
    er_sliced   = jnp.array(er_sliced_raw, dtype=jnp.float32)
    n_er_valid  = int(np.sum(np.isfinite(np.array(er_sliced))))
    time_years  = 1970.0 + np.array(forcing.time) / 365.25
    print(f"  Forcing: {T} days  ({time_years[0]:.1f}–{time_years[-1]:.1f})")
    print(f"  FluxNet ER: {n_er_valid} valid daily obs in window")

    # ── Observations ─────────────────────────────────────────────────────────
    print("Building observations…")
    delta14C_obs  = build_pool_14C_obs(HF_SOIL_14C_PATH, forcing.time, idx.pool_names)
    delta14C_resp = build_resp_14C_obs(HF_RESP_14C_PATH, forcing.time)
    c_pools_obs   = build_soil_carbon_obs(HF_SOIL_C_PATH2, HF_SOIL_C_PATH, idx.pool_names)

    n_pool_obs = sum(int(jnp.sum(~jnp.isnan(a))) for a in delta14C_obs.values())
    n_resp_obs = int(jnp.sum(~jnp.isnan(delta14C_resp)))
    print(f"  Pool Δ¹⁴C obs: {n_pool_obs}  |  Resp Δ¹⁴C obs: {n_resp_obs}")
    if c_pools_obs:
        pn, (mu, sig) = next(iter(c_pools_obs.items()))
        print(f"  C stock {pn} (individual):  {mu:.0f} ± {sig:.0f} gC m⁻²")

    print("Building carbon sum constraints…")
    carbon_sum_blocks = build_carbon_sum_obs(HF_SOIL_C_PATH2, HF_SOIL_C_PATH, idx)

    print("Building ISRaD fraction Δ¹⁴C obs blocks…")
    israd_blocks = build_israd_14C_obs(ISRAD_FRAC_PATH, forcing.time, idx)
    print(f"  ISRaD blocks: {len(israd_blocks)} (one per pool×year pair)")

    extra_blocks = carbon_sum_blocks + israd_blocks

    # ── Initial state ─────────────────────────────────────────────────────────
    state0 = build_state0(config, idx)
    print(f"  State0 total C12: {float(jnp.sum(state0.C12)):.0f} gC m⁻²")

    # ── Analytical SS helper ──────────────────────────────────────────────────
    mean_modifier, mean_input_val = build_mean_ss_modifier(
        forcing, model, t_nan_fill=5.0, theta_nan_fill=0.3
    )
    print(f"  Diagnostic SS: mean_modifier={mean_modifier:.4f},"
          f" mean_input={mean_input_val:.4f} gC/m²/day")

    params_prior    = make_default_params(config)
    n_pools         = len(idx)
    target_names    = list(config.external_inputs.partition.keys())
    ext_target_idx  = [idx[n] for n in target_names]

    prior_ss = _analytical_c12_ss(params_prior, n_pools, mean_input_val,
                                   mean_modifier, target_indices=ext_target_idx)
    print_prior_ss_check(prior_ss, params_prior, idx, ext_target_idx)

    def _make_ss_state(params_opt):
        c12_ss = _analytical_c12_ss(params_opt, n_pools, mean_input_val,
                                     mean_modifier, target_indices=ext_target_idx)
        return state0._replace(C12=c12_ss)

    # ── Prior forward run ─────────────────────────────────────────────────────
    print("\nPrior forward simulation…")
    t0 = time.perf_counter()
    out_prior = run_model(model, forcing, state0=state0, params=params_prior)
    jax.block_until_ready(out_prior.delta14C)
    print(f"  Done  [{time.perf_counter()-t0:.1f}s]")

    tau_p = np.exp(np.array(params_prior.log_tau))
    d14C_resp_prior = compute_resp_delta14C(out_prior, params_prior)

    # ── C-stock diagnostic helper (Harvard-specific constraints) ─────────────
    hz_cached = load_horizon_means(HF_SOIL_C_PATH2, HF_SOIL_C_PATH)

    def _print_cstock_diag(label: str, out):
        c_sim       = np.array(jnp.mean(out.C12, axis=0))
        act_sim     = c_sim[idx["soil_active"]]
        slw_sim     = c_sim[idx["soil_slow"]]
        pas_sim     = c_sim[idx["soil_passive"]]
        org_sum_sim = act_sim + slw_sim
        min_sum_sim = slw_sim + pas_sim
        act_obs, act_sig = c_pools_obs.get("soil_active", (float("nan"), float("nan")))
        org_obs = hz_cached.get("org_mean", float("nan"))
        org_sig = hz_cached.get("org_sem",  float("nan"))
        min_obs = hz_cached.get("min_mean", float("nan"))
        min_sig = max(hz_cached.get("min_sem", float("nan")), 0.35 * min_obs)
        print(f"  {label} C-stock check:")
        print(f"    soil_active  (individual):  "
              f"sim={act_sim:.0f}  obs={act_obs:.0f}±{act_sig:.0f}  "
              f"resid={(act_sim-act_obs)/act_sig:+.2f}σ")
        print(f"    organic sum  (active+slow):  "
              f"sim={org_sum_sim:.0f}  obs={org_obs:.0f}±{org_sig:.0f}  "
              f"resid={(org_sum_sim-org_obs)/org_sig:+.2f}σ")
        print(f"    mineral sum  (slow+passive): "
              f"sim={min_sum_sim:.0f}  obs={min_obs:.0f}±{min_sig:.0f}  "
              f"resid={(min_sum_sim-min_obs)/min_sig:+.2f}σ")
        print(f"    pools: active={act_sim:.0f}  slow={slw_sim:.0f}  passive={pas_sim:.0f}  "
              f"total={act_sim+slw_sim+pas_sim:.0f} gC m⁻²")

    # ── OE1–OE5 sequence ─────────────────────────────────────────────────────
    obs_sets = build_oe5_obs_sets(forcing, delta14C_obs, delta14C_resp,
                                   c_pools_obs, er_sliced)
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
        # legacy DFS reference fields used by compare_sites
        dof_2pool=type("_DFS", (), {"dfs_total": dfs_oe, "dfs_by_group": {}})(),
        n_params_2pool=n_params_oe,
        dfs_6pool_ref=1.029,
        n_params_6pool_ref=10,
    )


# ════════════════════════════════════════════════════════════════════════════
# Figure
# ════════════════════════════════════════════════════════════════════════════

def make_figure(r: dict, out_path: str | None = None):
    """Save the 8-panel OE comparison figure for Harvard Forest."""
    if out_path is None:
        out_path = _wt("notebooks/harvard_forest_three_pool_oe_summary.png")
    make_oe5_figure(
        r=r,
        site_title="Harvard Forest — 3-Pool Optimal Model  (active + slow + passive)",
        site_subtitle="OE5 adds FluxNet ER → annual Rh constraint (f_hetero=0.55, σ=15%)",
        resp_obs_label="obs (hf212-01 NWN)",
        pool_colors=["tab:green", "tab:orange", "tab:brown", "tab:purple"],
        pool_markers=["o", "^", "s", "D"],
        rh_er_frac=0.55,
        out_path=out_path,
    )
