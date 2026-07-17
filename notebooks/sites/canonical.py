"""
sites/canonical.py — One canonical OE inversion per site, plus shared
information-content helpers.

Public API
----------
  run_hf_canonical()      → dict   Harvard Forest 3-pool OE at MAP
  run_barrow_canonical()  → dict   Barrow Alaska 3-pool OE at MAP
  oe_style_ablation(...)  → dict   DFS-by-obs-type using OE time-resolved Jacobian
  build_hf_c_stocks_israd(...)     ISRaD layer-integrated stock constraints for HF

Each canonical inversion uses:
  • The 3-pool config (matches across sites: active / slow / passive)
  • optimize_oe (Levenberg-Marquardt) — single algorithm at both sites
  • Field set ``(log_tau, log_f_transfer)`` (config-driven; partition fixed)
  • Full obs vector: respired Δ¹⁴C + C-stocks + site-specific radiocarbon blocks
    (HF: ISRaD density-fraction blocks; Barrow: ISRaD bulk-layer blocks)
  • Analytical steady-state substitution at every K evaluation, so the
    operating point matches both the optimisation AND the ablation Jacobian.

Returned dict (canonical):
    model, config, idx, opt_fields,
    forcing, time_years,
    obs_full, extra_blocks,
    params_prior, params_opt, out_prior, out_opt,
    state0_obs    – initial state from observations
    state_at_map  – analytical-SS state at MAP (used for K in ablation)
    oe_result     – OEResult with averaging_kernel, Sx, etc.
"""
from __future__ import annotations

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# ── Path setup so this module is runnable from anywhere ─────────────────────
_SITES_DIR    = os.path.dirname(os.path.abspath(__file__))
_NB_ROOT      = os.path.dirname(_SITES_DIR)
_REPO_ROOT    = os.path.dirname(_NB_ROOT)
_SRC_ROOT     = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ecosystem_complexity.api import (
    build_model, run_model, optimize_oe,
)
from ecosystem_complexity.data.parsers import attach_atm14C, slice_forcing
from ecosystem_complexity.data.loaders import load_harvard_forest, load_barrow_alaska
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.schemas import ObservationData
from ecosystem_complexity.state import make_default_params
from ecosystem_complexity.information import (
    OBS_C_STOCKS, OBS_POOL_D14C, OBS_RESP_D14C,
)
from ecosystem_complexity._oe_helpers import ObsBlock, build_oe_prior_sigma
from ecosystem_complexity.analysis import compute_resp_delta14C, compute_age_diagnostics
from ecosystem_complexity.oe_utils import (
    build_mean_ss_modifier as _shared_mean_ss_modifier,
    ss_state_for_params as _shared_ss_state_for,
)
from ecosystem_complexity.oe_diagnostics import (
    oe_style_ablation as _shared_oe_style_ablation,
    oe_constraint_ladder as _shared_oe_constraint_ladder,
)

# Per-site helpers already implemented in the site modules
from sites.barrow import (
    BARROW_ERA5_PATH, BARROW_FLUXMET_PATH, BARROW_RC_PATH,
    OPT_CONFIG as BARROW_CONFIG,
    HUA_PATH, GRAVEN_PATH, INTCAL_PATH,
    build_pool_14C_obs as barrow_build_pool_14C_obs,
    build_pool_14C_obs_blocks as barrow_build_pool_14C_obs_blocks,
    build_resp_14C_obs as barrow_build_resp_14C_obs,
    build_soil_carbon_obs as barrow_build_c_obs,
    build_state0 as barrow_build_state0,
)
from sites.harvard_forest import (
    HF_HR_PATH, HF_SOIL_14C_PATH, HF_RESP_14C_PATH,
    OPT_CONFIG as HF_CONFIG,
    build_pool_14C_obs as hf_build_pool_14C_obs,
    build_resp_14C_obs as hf_build_resp_14C_obs,
    build_israd_14C_obs as hf_build_israd_blocks,
    build_hf_bulk_14C_obs as hf_build_bulk_blocks,
    build_hf_bulk_mixture_14C_obs as hf_build_bulk_mixture_blocks,
    build_hf_bulk_perlayer_mixture_14C_obs as hf_build_bulk_perlayer_blocks,
    ISRAD_FRAC_PATH,
    ISRAD_LAYER_PATH as HF_ISRAD_LAYER_PATH,
    build_state0 as hf_build_state0,
)


# ════════════════════════════════════════════════════════════════════════════
# HF C-stock constraints — ISRaD H1–H5 layer-integrated
# ════════════════════════════════════════════════════════════════════════════

_ISRAD_LAYER_PATH = os.path.join(
    _REPO_ROOT,
    "data/shared/israd/ISRaD_extra_flat_layer_v 2.6.6.2024-01-25.csv",
)
# Ehleringer / Harvard Forest profiles (lat 42.538, lon −72.172 → 0 km
# from US-Ha1 flux tower) with both lyr_soc and density-fractionated frc_14c.
_HF_ISRAD_PROFILES = ["H1", "H2", "H3", "H4", "H5"]

# Sum constraint mapping: name → (pool tuple, depth window cm).
# Realistic depth ranges for a temperate hardwood forest:
#   O horizon: a few cm above mineral soil  (ISRaD H1–H5 reach −9 cm)
#   A horizon: 0–15 cm
#   B horizon: 15–100 cm
_HF_3POOL_GROUPS = [
    ("O_horizon",  ("soil_active",),               ( -15.0,   0.0)),
    ("A_horizon",  ("soil_slow",),                 (   0.0,  15.0)),
    ("B_horizon",  ("soil_passive",),              (  15.0, 100.0)),
]


def build_hf_c_stocks_israd(model, csv_path: str = _ISRAD_LAYER_PATH) -> dict:
    """Return ``{pool_name: (mean_gC_m2, sigma_gC_m2)}`` for HF 3-pool from ISRaD.

    Uses Ehleringer profiles H1–H5 at the flux tower (0 km).  Layer
    ``lyr_soc`` (gC cm⁻²) is integrated by depth into three pool stocks:

      O horizon (above 0 cm)   → soil_active
      A horizon (0–15 cm)      → soil_slow
      B horizon (15–100 cm)    → soil_passive

    σ is the across-profile std with a 10 % floor.
    """
    import pandas as pd
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["pro_name"].isin(_HF_ISRAD_PROFILES)].copy()
    df = df.dropna(subset=["lyr_soc", "lyr_top", "lyr_bot"])
    df["soc_gm2"] = df["lyr_soc"].astype(float) * 1.0e4  # gC/cm² → gC/m²
    # Assign each layer to ONE depth bin by its midpoint, so stocks are
    # partitioned without double-counting layers that straddle a boundary.
    df["lyr_mid"] = 0.5 * (df["lyr_top"].astype(float) + df["lyr_bot"].astype(float))

    pool_names = set(model.pool_index.pool_names)
    c_pools: dict[str, tuple[float, float]] = {}

    for group_name, pools, (z_top, z_bot) in _HF_3POOL_GROUPS:
        if not (set(pools) <= pool_names):
            continue
        mask = (df["lyr_mid"] >= z_top) & (df["lyr_mid"] < z_bot)
        sub = df.loc[mask]
        per_profile = sub.groupby("pro_name")["soc_gm2"].sum()
        if len(per_profile) < 2:
            continue
        y_mean = float(per_profile.mean())
        y_sigma = max(float(per_profile.std(ddof=1)), 0.10 * y_mean)
        c_pools[pools[0]] = (y_mean, y_sigma)
        # Actual measured depth span (not the sentinel filter bounds)
        z_actual_top = float(sub["lyr_top"].min())
        z_actual_bot = float(sub["lyr_bot"].max())
        print(f"  HF C stock {group_name:<11s} → {pools[0]:<14s}: "
              f"{y_mean:6.0f} ± {y_sigma:5.0f} gC m⁻²  "
              f"(n={len(per_profile)} ISRaD profiles, "
              f"measured depth {z_actual_top:.0f}–{z_actual_bot:.0f} cm)")
    return c_pools


# ════════════════════════════════════════════════════════════════════════════
# OE-style ablation (time-resolved Jacobian at MAP — matches OEResult.A)
# ════════════════════════════════════════════════════════════════════════════

def _mean_ss_modifier(forcing, params0) -> tuple[float, float]:
    """Backward-compatible wrapper around the shared OE steady-state helper."""
    return _shared_mean_ss_modifier(forcing, params0)


def oe_style_ablation(
    model, forcing, state0, params_opt, observations,
    opt_fields: tuple,
    extra_obs_blocks: list | None = None,
) -> dict:
    """Backward-compatible wrapper around the shared OE ablation helper."""
    return _shared_oe_style_ablation(
        model, forcing, state0, params_opt, observations,
        opt_fields=opt_fields, extra_obs_blocks=extra_obs_blocks,
    )


def oe_constraint_ladder(
    model, forcing, state0, params_opt, observations,
    opt_fields: tuple,
    extra_obs_blocks: list | None = None,
) -> list[dict]:
    """Backward-compatible wrapper around the shared OE ladder helper."""
    return _shared_oe_constraint_ladder(
        model, forcing, state0, params_opt, observations,
        opt_fields=opt_fields, extra_obs_blocks=extra_obs_blocks,
    )


# ════════════════════════════════════════════════════════════════════════════
# Shared OE driver — single canonical optimize_oe call
# ════════════════════════════════════════════════════════════════════════════

def _ss_state_for(model, forcing, state0, params):
    """Backward-compatible wrapper around the shared OE steady-state helper."""
    return _shared_ss_state_for(model, forcing, state0, params)


def _run_oe_canonical(
    model, forcing, state0, obs_full,
    extra_blocks: list,
    opt_fields: tuple,
    site_label: str,
) -> dict:
    """Single canonical optimize_oe call with structured return.

    This is THE shared inversion driver for every canonical site (HF, Barrow,
    EML, Howland).  It runs the prior and the MAP from their own analytical
    steady state — the exact operating points ``optimize_oe`` costs against —
    so the returned diagnostics are self-consistent regardless of site.
    """
    params_prior = make_default_params(model.config)
    print(f"\n[{site_label}] prior forward simulation…")
    # Run the prior from its own analytical steady state — the same operating
    # point optimize_oe costs the prior against (y_prior = _forward(xa)).  Using
    # the observed-stock state0 here would plot a "prior" that is not at steady
    # state and does not match the prior the inversion actually sees.
    state_at_prior = _ss_state_for(model, forcing, state0, params_prior)
    out_prior = run_model(model, forcing, state0=state_at_prior, params=params_prior)
    jax.block_until_ready(out_prior.delta14C)

    print(f"[{site_label}] optimize_oe  fields={opt_fields}  "
          f"extras={[b.name for b in extra_blocks]}")
    t0 = time.perf_counter()
    result = optimize_oe(
        model, forcing, obs_full, state0=state0,
        fields=opt_fields, extra_obs_blocks=extra_blocks,
    )
    ch = np.array(result.cost_history)
    print(f"  Done [{time.perf_counter()-t0:.1f}s]  J {ch[0]:.2f} → {ch[-1]:.2f}  "
          f"({result.n_iter} iter, converged={result.converged})")

    params_opt = result.params_opt
    state_at_map = _ss_state_for(model, forcing, state0, params_opt)
    out_opt = run_model(model, forcing, state0=state_at_map, params=params_opt)
    jax.block_until_ready(out_opt.delta14C)

    return dict(
        params_prior=params_prior, params_opt=params_opt,
        out_prior=out_prior, out_opt=out_opt,
        state_at_prior=state_at_prior, state_at_map=state_at_map,
        oe_result=result,
    )


# ════════════════════════════════════════════════════════════════════════════
# Harvard Forest canonical inversion
# ════════════════════════════════════════════════════════════════════════════

def run_hf_canonical(pool_14c_source: str = "fraction") -> dict:
    """Single OE inversion at Harvard Forest (3-pool config).

    Obs:
      pool Δ¹⁴C    ISRaD radiocarbon → 3 pools (source set by ``pool_14c_source``)
      resp Δ¹⁴C    hf212-01 NWN respired CO₂  (41 dates)
      C stocks     ISRaD H1–H5 layer-integrated, one entry per pool

    Parameters
    ----------
    pool_14c_source : {"fraction", "bulk", "mixture", "perlayer", "both"}
        Which ISRaD ¹⁴C constraint carries the pool Δ¹⁴C information:

        ``"fraction"`` (default, canonical)
            Density-fraction Δ¹⁴C (Gaudinski 2000, Savage 2007, McFarlane 2013);
            each fraction isolates one kinetic pool.
        ``"bulk"``
            Whole-soil layer Δ¹⁴C (McFarlane 2013 H1–H5), binned to pools by
            *depth* — a depth→pool proxy, one constraint per pool.
        ``"mixture"``
            Whole-soil layer Δ¹⁴C as a single carbon-mass-weighted mixture
            constraint, ``Σ_p (C12_p/ΣC12)·Δ14C_p`` — uses the model's own pool
            masses, no depth→pool assumption.  One scalar constraint.
        ``"perlayer"``
            Per-layer carbon-mass-weighted mixture, one constraint per depth bin:
            surface bins use *measured* density-fraction masses as weights, deep
            bins fall back to the model C12 partition.  The physically-honest,
            rank-limited operator.
        ``"both"``
            Fraction + depth-proxy bulk blocks together (bulk's marginal
            information on top of fractions).

    Returns the structured dict described in the module docstring.
    """
    if pool_14c_source not in ("fraction", "bulk", "mixture", "perlayer", "both"):
        raise ValueError(
            f"pool_14c_source must be 'fraction', 'bulk', 'mixture', 'perlayer', "
            f"or 'both'; got {pool_14c_source!r}"
        )
    label = f"Harvard Forest [{pool_14c_source} ¹⁴C]"
    print(f"\n══ {label} — canonical OE inversion ══════════════════════════════")
    atm14C = load_full_14C_record(
        hua_path=HUA_PATH, graven_path=GRAVEN_PATH, intcal_path=INTCAL_PATH,
        hemisphere="NH", start_year=1500.0, end_year=2025.0,
    )

    model  = build_model(HF_CONFIG)
    config = model.config
    idx    = model.pool_index
    print(f"  Pools: {idx.pool_names}")

    inv_cfg = getattr(config, "inversion_raw", {}) or {}
    opt_fields = ("log_tau", "log_f_transfer")
    print(f"  opt_fields: {opt_fields}")

    print("  Loading forcing (FluxNet HR, 1996+)…")
    forcing_full, _ = load_harvard_forest(
        hr_path=HF_HR_PATH, config=config, qc_threshold=2,
        include_gpp_forcing=True,
    )
    years_daily, d14c_daily = atm14C
    forcing_full = attach_atm14C(forcing_full, d14c_daily, years_daily)
    time_np = np.array(forcing_full.time)
    years_all = 1970.0 + time_np / 365.25
    start_idx = int(np.searchsorted(years_all, 1996.0))
    forcing = slice_forcing(forcing_full, start_idx, len(time_np))
    T = int(forcing.time.shape[0])
    time_years = 1970.0 + np.array(forcing.time) / 365.25
    print(f"  Forcing: {T} days  ({time_years[0]:.1f}–{time_years[-1]:.1f})")

    # Pool Δ¹⁴C: carried entirely by ISRaD ObsBlocks (built below), so the
    # standard pool-Δ¹⁴C obs vector stays empty.  We deliberately do NOT load
    # hf212-03 horizon-bulk Δ¹⁴C here: those horizon means come from the same
    # physical NWN profiles that feed the ISRaD entries, so including both
    # would treat correlated samples as independent.
    delta14C_obs  = {}
    delta14C_resp = hf_build_resp_14C_obs(HF_RESP_14C_PATH, forcing.time)
    n_resp_obs = int(jnp.sum(~jnp.isnan(delta14C_resp)))
    print(f"  Resp Δ¹⁴C: {n_resp_obs}")

    # C stocks from ISRaD (one entry per pool — goes through standard c_stock block)
    print("  Building HF C-stock obs from ISRaD H1–H5…")
    c_pools_obs = build_hf_c_stocks_israd(model)

    # Pool Δ¹⁴C ISRaD blocks — source selectable for the bulk-vs-fraction test.
    extra_blocks = []
    if pool_14c_source in ("fraction", "both"):
        israd_blocks = hf_build_israd_blocks(ISRAD_FRAC_PATH, forcing.time, idx)
        print(f"  ISRaD fraction Δ¹⁴C blocks: {len(israd_blocks)}")
        extra_blocks += list(israd_blocks)
    if pool_14c_source in ("bulk", "both"):
        bulk_blocks = hf_build_bulk_blocks(HF_ISRAD_LAYER_PATH, forcing.time, idx)
        print(f"  ISRaD bulk-layer Δ¹⁴C blocks: {len(bulk_blocks)}")
        extra_blocks += list(bulk_blocks)
    if pool_14c_source == "mixture":
        mix_blocks = hf_build_bulk_mixture_blocks(HF_ISRAD_LAYER_PATH, forcing.time, idx)
        print(f"  ISRaD bulk-mixture Δ¹⁴C blocks: {len(mix_blocks)}")
        extra_blocks += list(mix_blocks)
    if pool_14c_source == "perlayer":
        perlayer_blocks = hf_build_bulk_perlayer_blocks(HF_ISRAD_LAYER_PATH, forcing.time, idx)
        print(f"  ISRaD per-layer mixture Δ¹⁴C blocks: {len(perlayer_blocks)}")
        extra_blocks += list(perlayer_blocks)

    obs_full = ObservationData(
        time=forcing.time,
        NEE=jnp.full(T, jnp.nan), GPP=jnp.full(T, jnp.nan),
        ER=jnp.full(T, jnp.nan), NEE_unc=jnp.full(T, jnp.nan),
        delta14C_obs=delta14C_obs, deltaD14C_obs={},
        C_pools_obs=c_pools_obs, delta14C_resp=delta14C_resp,
    )

    state0 = hf_build_state0(config, idx)
    print(f"  State0 total C12: {float(jnp.sum(state0.C12)):.0f} gC m⁻²")

    fit = _run_oe_canonical(
        model, forcing, state0, obs_full, extra_blocks, opt_fields, label,
    )

    return dict(
        model=model, config=config, idx=idx, opt_fields=opt_fields,
        forcing=forcing, time_years=time_years,
        obs_full=obs_full, extra_blocks=extra_blocks,
        state0_obs=state0, pool_14c_source=pool_14c_source,
        delta14C_obs=delta14C_obs, delta14C_resp=delta14C_resp,
        c_pools_obs=c_pools_obs,
        **fit,
    )


# ════════════════════════════════════════════════════════════════════════════
# Barrow canonical inversion
# ════════════════════════════════════════════════════════════════════════════

def run_barrow_canonical() -> dict:
    """Single OE inversion at Barrow, Alaska (3-pool permafrost config).

    Obs:
      pool Δ¹⁴C    ISRaD bulk soil layers (Vaughn 2018 + Nave 2021)
      resp Δ¹⁴C    Vaughn surface-emission CO₂
      C stocks     individual active (Ping 2008) and passive (Hugelius 2014)
                   constraints only; correlated sum constraints are omitted
    """
    label = "Barrow, Alaska"
    print(f"\n══ {label} — canonical OE inversion ═══════════════════════════════")
    atm14C = load_full_14C_record(
        hua_path=HUA_PATH, graven_path=GRAVEN_PATH, intcal_path=INTCAL_PATH,
        hemisphere="NH", start_year=1500.0, end_year=2025.0,
    )

    model  = build_model(BARROW_CONFIG)
    config = model.config
    idx    = model.pool_index
    print(f"  Pools: {idx.pool_names}")
    opt_fields = ("log_tau", "log_f_transfer")
    print(f"  opt_fields: {opt_fields}")

    forcing_full, _ = load_barrow_alaska(
        era5_path=BARROW_ERA5_PATH, fluxmet_path=BARROW_FLUXMET_PATH,
        config=config, qc_threshold=0.0, include_gpp_forcing=True,
    )
    years_daily, d14c_daily = atm14C
    forcing_full = attach_atm14C(forcing_full, d14c_daily, years_daily)
    time_np = np.array(forcing_full.time)
    years_all = 1970.0 + time_np / 365.25
    start_idx = int(np.searchsorted(years_all, 2011.0))
    forcing = slice_forcing(forcing_full, start_idx, len(time_np))
    T = int(forcing.time.shape[0])
    time_years = 1970.0 + np.array(forcing.time) / 365.25
    print(f"  Forcing: {T} days ({time_years[0]:.1f}–{time_years[-1]:.1f})")

    # Pool Δ¹⁴C: ISRaD bulk soil layer data at BEO (Vaughn 2018 + Nave 2021)
    # — replaces noisy Vaughn pore-gas constraint.
    print("  Building Barrow pool Δ¹⁴C obs from ISRaD bulk layers (Vaughn 2018 + Nave 2021)…")
    delta14C_obs = barrow_build_pool_14C_obs(forcing.time, idx.pool_names, print_summary=True)
    israd_pool_blocks, _ = barrow_build_pool_14C_obs_blocks(idx, forcing.time, print_summary=False)
    obs_delta14C = {}                                       # ObsBlocks carry the inversion obs
    delta14C_resp = barrow_build_resp_14C_obs(BARROW_RC_PATH, forcing.time)
    c_pools_obs   = barrow_build_c_obs(idx.pool_names)
    n_resp_obs = int(jnp.sum(~jnp.isnan(delta14C_resp)))
    print(f"  Pool Δ¹⁴C: {len(israd_pool_blocks)} ISRaD layer blocks  "
          f"Resp Δ¹⁴C: {n_resp_obs}  C stocks: {len(c_pools_obs)}")

    # NOTE: we omit barrow_build_c_sum_blocks (organic = active+slow and
    # permafrost = slow+passive).  Those two sums share the slow term but
    # come from independent surveys (Ping 2008, Hugelius 2014) and disagree
    # on the implied slow value (3000 vs 4000 gC/m²).  Combined with the
    # individual active/passive constraints already in C_pools_obs, they
    # over-determine three pools with inconsistent evidence.  Keep only
    # the individual pool constraints; let the cascade + ISRaD Δ¹⁴C
    # constrain the slow pool.
    extra_blocks = list(israd_pool_blocks)

    obs_full = ObservationData(
        time=forcing.time,
        NEE=jnp.full(T, jnp.nan), GPP=jnp.full(T, jnp.nan),
        ER=jnp.full(T, jnp.nan), NEE_unc=jnp.full(T, jnp.nan),
        delta14C_obs=obs_delta14C, deltaD14C_obs={},
        C_pools_obs=c_pools_obs, delta14C_resp=delta14C_resp,
    )

    state0 = barrow_build_state0(config, idx)
    print(f"  State0 total C12: {float(jnp.sum(state0.C12)):.0f} gC m⁻²")

    fit = _run_oe_canonical(
        model, forcing, state0, obs_full, extra_blocks, opt_fields, label,
    )

    return dict(
        model=model, config=config, idx=idx, opt_fields=opt_fields,
        forcing=forcing, time_years=time_years,
        obs_full=obs_full, extra_blocks=extra_blocks,
        state0_obs=state0,
        delta14C_obs=delta14C_obs, delta14C_resp=delta14C_resp,
        c_pools_obs=c_pools_obs,
        **fit,
    )


# ════════════════════════════════════════════════════════════════════════════
# Shared 6-panel information-content figure
# ════════════════════════════════════════════════════════════════════════════

from ecosystem_complexity.information import (
    PARAM_GROUP_TAU, PARAM_GROUP_PARTITION,
    PARAM_GROUP_TRANSFER, PARAM_GROUP_ENV,
    compute_fisher, compute_dof, compute_posterior,
    get_param_groups,
)
from ecosystem_complexity.analysis import (
    run_complexity_ladder, age_diagnostics_summary,
)

_GROUP_COLORS = {
    PARAM_GROUP_TAU:       "steelblue",
    PARAM_GROUP_PARTITION: "darkorange",
    PARAM_GROUP_TRANSFER:  "seagreen",
    PARAM_GROUP_ENV:       "mediumpurple",
}


def _short_obs_label(s: str) -> str:
    return (s.replace("C_stocks", "C-stock")
             .replace("pool_delta14C", "pool Δ¹⁴C")
             .replace("resp_delta14C", "resp Δ¹⁴C")
             .replace(" + ", "\n+ "))


def _canonical_prior_sigma(model, opt_fields) -> np.ndarray:
    """Prior sigma used by canonical information analysis and exports."""
    return np.array(
        build_oe_prior_sigma(model.config, make_default_params(model.config), tuple(opt_fields)),
        dtype=float,
    )


def site_information_analysis(data: dict) -> dict:
    """Run the OE ablation, full Fisher, complexity ladder, age diagnostics.

    Uses the canonical inversion's MAP state (analytical-SS at params_opt)
    so FIM evaluations match the optimisation operating point.
    """
    import matplotlib  # noqa: F401  (force import for plotting later)

    model      = data["model"]
    forcing    = data["forcing"]
    state_map  = data["state_at_map"]
    params_opt = data["params_opt"]
    obs_full   = data["obs_full"]
    opt_fields = list(data["opt_fields"])
    extras     = data.get("extra_blocks", [])

    print("\n── OE-style ablation (time-resolved Jacobian) ──")
    ablation_oe = oe_style_ablation(
        model, forcing, state_map, params_opt, obs_full,
        opt_fields=tuple(opt_fields), extra_obs_blocks=extras,
    )
    print(f"\n  {'Scenario':<55s}  {'n_obs':>6}  {'DFS':>6}")
    print("  " + "─" * 75)
    for key, row in ablation_oe.items():
        print(f"  {key:<55s}  {row['n_obs']:6d}  {row['dfs_total']:6.3f}")

    # Full Fisher at MAP (time-mean obs convention, for panel diagnostics).
    inv_cfg = getattr(model.config, "inversion_raw", {}) or {}
    obs_sigmas = dict(
        obs_sigma_d14C=float(inv_cfg.get("sigma_pool_14C", 10.0)),
        obs_sigma_resp=float(inv_cfg.get("sigma_resp_14C", 10.0)),
    )
    prior_sigma  = _canonical_prior_sigma(model, opt_fields)
    param_groups = get_param_groups(params_opt, opt_fields, model)
    fisher_full  = compute_fisher(model, forcing, state_map, params_opt, obs_full,
                                  fields=opt_fields, **obs_sigmas)
    dof_full     = compute_dof(fisher_full, prior_sigma, param_groups=param_groups)
    post_full    = compute_posterior(fisher_full, prior_sigma)

    ladder_rungs = oe_constraint_ladder(
        model, forcing, state_map, params_opt, obs_full,
        opt_fields=tuple(opt_fields), extra_obs_blocks=extras,
    )

    age_prior = compute_age_diagnostics(data["out_prior"], data["params_prior"], model)
    age_opt   = compute_age_diagnostics(data["out_opt"],   params_opt,           model)

    return dict(
        ablation_oe=ablation_oe,
        fisher_full=fisher_full,
        dof_full=dof_full,
        post_full=post_full,
        prior_sigma=prior_sigma,
        param_groups=param_groups,
        ladder=ladder_rungs,
        age_prior=age_prior,
        age_opt=age_opt,
    )


def site_figure(
    data: dict, analysis: dict, out_path: str,
    pool_styles: dict, site_title: str,
) -> None:
    """Render the canonical 6-panel information-content figure."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    time_years = data["time_years"]
    delta14C_obs = data["delta14C_obs"]
    delta14C_resp = data["delta14C_resp"]

    ablation_oe  = analysis["ablation_oe"]
    fisher_full  = analysis["fisher_full"]
    dof_full     = analysis["dof_full"]
    post_full    = analysis["post_full"]
    prior_sigma  = analysis["prior_sigma"]
    param_groups = analysis["param_groups"]
    ladder       = analysis["ladder"]
    age_prior    = analysis["age_prior"]
    age_opt      = analysis["age_opt"]

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(
        2, 3, figure=fig, hspace=0.42, wspace=0.38,
        left=0.06, right=0.97, top=0.92, bottom=0.07,
    )
    ax_abl = fig.add_subplot(gs[0, 0])
    ax_dfs = fig.add_subplot(gs[0, 1])
    ax_ur  = fig.add_subplot(gs[0, 2])
    ax_eig = fig.add_subplot(gs[1, 0])
    ax_age = fig.add_subplot(gs[1, 1])
    ax_lad = fig.add_subplot(gs[1, 2])

    # (a) Ablation — OE-style DFS
    obs_type_colors = {
        OBS_C_STOCKS:   "tab:blue",
        OBS_POOL_D14C:  "tab:green",
        OBS_RESP_D14C:  "tab:red",
    }
    scenario_labels = list(ablation_oe.keys())
    short_labels = [_short_obs_label(s) for s in scenario_labels]
    x = np.arange(len(short_labels))
    dfs_totals = [v["dfs_total"] for v in ablation_oe.values()]
    ax_abl.bar(x, dfs_totals, 0.55, color="0.75", alpha=0.55, label="Total DFS")
    single_dfs = {label: row["dfs_total"]
                  for label, row in ablation_oe.items()
                  if label in obs_type_colors}
    for ot, color in obs_type_colors.items():
        if ot in single_dfs:
            ax_abl.axhline(single_dfs[ot], lw=1.2, linestyle="--",
                           color=color, alpha=0.75,
                           label=f"{_short_obs_label(ot)} alone: {single_dfs[ot]:.2f}")
    ax_abl.set_xticks(x)
    ax_abl.set_xticklabels(short_labels, fontsize=7.5)
    ax_abl.set_ylabel("Degrees of freedom for signal (DFS)", fontsize=9)
    ax_abl.set_title("(a) DFS by observation subset", fontsize=9, loc="left")
    ax_abl.legend(fontsize=7.5, framealpha=0.9)
    ax_abl.tick_params(axis="y", labelsize=8)
    ax_abl.grid(axis="y", lw=0.4, alpha=0.4)
    ax_abl.set_ylim(bottom=0)

    # (b) Per-parameter DFS (time-mean Fisher)
    param_names = fisher_full.param_names or []
    n_params = len(param_names)
    dfs_per = dof_full.dfs_per_param
    colors_per = ["0.6"] * n_params
    for grp, idxs in param_groups.items():
        c = _GROUP_COLORS.get(grp, "0.6")
        for i in idxs:
            if i < n_params:
                colors_per[i] = c
    y = np.arange(n_params)
    ax_dfs.barh(y, dfs_per, 0.65, color=colors_per, alpha=0.82)
    ax_dfs.set_yticks(y)
    ax_dfs.set_yticklabels(param_names, fontsize=6.5)
    ax_dfs.set_xlabel("DFS per parameter (time-mean Fisher)", fontsize=8)
    ax_dfs.set_title("(b) Per-parameter DFS", fontsize=9, loc="left")
    ax_dfs.tick_params(axis="x", labelsize=8)
    ax_dfs.grid(axis="x", lw=0.4, alpha=0.4)
    ax_dfs.axvline(0, lw=0.6, color="black")
    legend_patches = [Patch(facecolor=_GROUP_COLORS[g], label=g, alpha=0.8)
                       for g in param_groups if g in _GROUP_COLORS]
    if legend_patches:
        ax_dfs.legend(handles=legend_patches, fontsize=7.5,
                      framealpha=0.9, loc="lower right")

    # (c) Uncertainty reduction for log_tau
    tau_idxs = param_groups.get(PARAM_GROUP_TAU, [])
    if tau_idxs:
        tau_names = [param_names[i].replace("log_tau[", "").rstrip("]")
                     for i in tau_idxs]
        prior_s = prior_sigma[tau_idxs]
        post_s = post_full.posterior_sigma[tau_idxs]
        n_tau = len(tau_idxs)
        x_tau = np.arange(n_tau)
        w = 0.35
        ax_ur.bar(x_tau - w/2, prior_s, w, label="Prior σ (log τ)",
                  color="0.7", alpha=0.85)
        ax_ur.bar(x_tau + w/2, post_s, w, label="Posterior σ (log τ)",
                  color="steelblue", alpha=0.85)
        for i, (ps, qs) in enumerate(zip(prior_s, post_s)):
            ur = (1 - qs/ps) * 100
            ax_ur.text(i, max(ps, qs) + 0.005, f"{ur:.0f}%",
                       ha="center", va="bottom", fontsize=7, color="0.3")
        ax_ur.set_xticks(x_tau)
        ax_ur.set_xticklabels(tau_names, fontsize=8, rotation=20, ha="right")
        ax_ur.set_ylabel("Standard deviation in log-space", fontsize=9)
        ax_ur.set_title("(c) Uncertainty reduction for log_τ", fontsize=9, loc="left")
        ax_ur.legend(fontsize=8, framealpha=0.9)
        ax_ur.tick_params(axis="y", labelsize=8)
        ax_ur.grid(axis="y", lw=0.4, alpha=0.4)
        ax_ur.set_ylim(bottom=0)

    # (d) FIM eigenspectrum
    eigvals = np.maximum(fisher_full.eigenvalues, 1e-12)
    k = np.arange(1, len(eigvals) + 1)
    ax_eig.semilogy(k, eigvals, "o-", color="steelblue", lw=1.5, ms=5,
                    alpha=0.9, label="FIM eigenvalues (all obs)")
    eig_colors = {OBS_C_STOCKS: "tab:blue", OBS_POOL_D14C: "tab:green",
                  OBS_RESP_D14C: "tab:red"}
    for ot, FIM_k in fisher_full.FIM_per_type.items():
        ev = np.linalg.eigvalsh(FIM_k)[::-1]
        ev = np.maximum(ev, 1e-12)
        ax_eig.semilogy(k, ev, "--", color=eig_colors.get(ot, "0.5"),
                        lw=1.0, alpha=0.7,
                        label=f"{_short_obs_label(ot)} only")
    dfs_sorted = dof_full.dfs_per_param[np.argsort(dof_full.dfs_per_param)[::-1]]
    eff_rank = int(np.searchsorted(np.cumsum(dfs_sorted),
                                    0.9 * dof_full.dfs_total)) + 1
    ax_eig.axvline(eff_rank, lw=1.0, linestyle=":", color="firebrick",
                   alpha=0.7, label=f"90% DFS rank = {eff_rank}")
    ax_eig.set_xlabel("Eigenvalue rank", fontsize=9)
    ax_eig.set_ylabel("FIM eigenvalue (log scale)", fontsize=9)
    ax_eig.set_title("(d) FIM eigenspectrum by obs type", fontsize=9, loc="left")
    ax_eig.legend(fontsize=7.5, framealpha=0.9)
    ax_eig.tick_params(labelsize=8)
    ax_eig.grid(axis="both", lw=0.4, alpha=0.4)
    ax_eig.set_xlim(0.5, len(eigvals) + 0.5)

    # (e) Storage vs respiration age
    ax_age.plot(time_years, age_prior.bulk_delta14C,
                lw=0.9, color="0.5", linestyle=":", alpha=0.7, label="Bulk — prior")
    ax_age.plot(time_years, age_opt.bulk_delta14C,
                lw=1.6, color="sienna", alpha=0.9, label="Bulk — optimised")
    ax_age.plot(time_years, age_prior.respired_delta14C,
                lw=0.9, color="0.4", linestyle="--", alpha=0.6, label="Respired — prior")
    ax_age.plot(time_years, age_opt.respired_delta14C,
                lw=1.6, color="tomato", linestyle="--", alpha=0.9,
                label="Respired — optimised")
    for pool_name, (label, color, marker) in pool_styles.items():
        if pool_name not in delta14C_obs:
            continue
        obs_arr = np.array(delta14C_obs[pool_name])
        valid = np.where(np.isfinite(obs_arr))[0]
        for t_i in valid:
            ax_age.scatter(time_years[t_i], obs_arr[t_i],
                           s=60, color=color, marker=marker,
                           edgecolors="black", linewidths=0.7, zorder=7,
                           label=label)
    resp_arr = np.array(delta14C_resp)
    valid_resp = np.where(np.isfinite(resp_arr))[0]
    if len(valid_resp):
        ax_age.scatter(time_years[valid_resp], resp_arr[valid_resp],
                       s=20, color="black", marker="x", alpha=0.55,
                       zorder=6, label="Resp obs")
    ax_age.axhline(0, lw=0.5, color="gray", linestyle=":")
    ax_age.set_ylabel("Δ¹⁴C (‰)", fontsize=9)
    ax_age.set_xlabel("Year", fontsize=9)
    ax_age.set_title("(e) Storage vs. respiration age", fontsize=9, loc="left")
    ax_age.tick_params(labelsize=8)
    ax_age.grid(axis="y", lw=0.4, alpha=0.4)
    handles, labels = ax_age.get_legend_handles_labels()
    seen = {}
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen[lab] = h
    ax_age.legend(list(seen.values()), list(seen.keys()),
                  fontsize=7, framealpha=0.9, ncol=2)

    # (f) One-constraint-at-a-time ladder
    lad_labels = [r["label"] for r in ladder]
    lad_dfs = [r["dfs"] for r in ladder]
    lad_types = [r["obs_type"] for r in ladder]
    x_lad = np.arange(len(lad_labels))
    lad_colors = {
        OBS_C_STOCKS: "tab:blue",
        OBS_POOL_D14C: "tab:green",
        OBS_RESP_D14C: "tab:red",
    }
    ax_lad.bar(
        x_lad,
        lad_dfs,
        color=[lad_colors.get(t, "0.6") for t in lad_types],
        alpha=0.82,
    )
    ax_lad.axhline(
        ablation_oe["C_stocks+pool_delta14C+resp_delta14C"]["dfs_total"],
        lw=0.9,
        linestyle=":",
        color="0.35",
        label="Full-data DFS",
    )
    ax_lad.set_xticks(x_lad)
    ax_lad.set_xticklabels(lad_labels, rotation=45, ha="right", fontsize=7)
    ax_lad.set_ylabel("Degrees of freedom for signal (DFS)", fontsize=9)
    ax_lad.set_xlabel("Single observation block", fontsize=9)
    ax_lad.set_title("(f) One-constraint-at-a-time ladder", fontsize=9, loc="left")
    ax_lad.tick_params(axis="y", labelsize=8)
    ax_lad.grid(axis="y", lw=0.4, alpha=0.4)
    ax_lad.set_ylim(bottom=0)
    ax_lad.legend(
        handles=[
            Patch(facecolor=lad_colors[OBS_C_STOCKS], label=_short_obs_label(OBS_C_STOCKS), alpha=0.82),
            Patch(facecolor=lad_colors[OBS_POOL_D14C], label=_short_obs_label(OBS_POOL_D14C), alpha=0.82),
            Patch(facecolor=lad_colors[OBS_RESP_D14C], label=_short_obs_label(OBS_RESP_D14C), alpha=0.82),
            Line2D([0], [0], color="0.35", linestyle=":", lw=0.9, label="Full-data DFS"),
        ],
        fontsize=7.5,
        framealpha=0.9,
    )

    fig.suptitle(site_title, fontsize=11)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")


__all__ = [
    "run_hf_canonical",
    "run_barrow_canonical",
    "oe_style_ablation",
    "build_hf_c_stocks_israd",
    "site_information_analysis",
    "site_figure",
]
