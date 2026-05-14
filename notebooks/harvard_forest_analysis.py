"""
harvard_forest_analysis.py — Information-content and model-complexity analysis
for radiocarbon-constrained soil carbon turnover inference at Harvard Forest.

Run from the repository root:
    python notebooks/harvard_forest_analysis.py

What this script does
---------------------
1. Loads the Harvard Forest soil-only model and 1996-initialised state.
2. Optimises τ and external-input partition against pool Δ¹⁴C + respired Δ¹⁴C.
3. Runs the information-content analysis at the MAP estimate:
     a. Observation ablation  — 5 scenarios spanning the obs-type space.
     b. Per-parameter DFS     — which pools are actually constrained?
     c. Uncertainty reduction — prior σ vs. posterior σ for log_tau.
     d. FIM eigenspectrum     — effective rank of the information matrix.
     e. Age diagnostics       — stored (pool) vs. respired (flux-weighted) Δ¹⁴C.
     f. Parameter complexity ladder — DFS vs. parameter-vector dimension.
4. Saves a 6-panel figure and prints a comprehensive text summary.

The central scientific question (§7 of the analysis spec):
  Does pool Δ¹⁴C constrain stored-carbon age structure independently from
  the active decomposition source mixture constrained by respired CO₂ Δ¹⁴C?

Data files required (relative to repo root)
-------------------------------------------
  data/harvard_forest/AMF_US-Ha1_FLUXNET_FULLSET_1991-2020_3-5/
      AMF_US-Ha1_FLUXNET_FULLSET_HR_1991-2020_3-5.csv
  data/harvard_forest/hf271-07-soils.csv
  data/harvard_forest/hf212-03-14c-org.csv
  data/harvard_forest/hf212-01-14c-no-treat.csv
  data/shared/atm_14C/Hua_2021.csv
  data/shared/atm_14C/Graven_2017.csv
  data/shared/atm_14C/intcal20.14c

Output
------
  notebooks/harvard_forest_analysis.png  — 6-panel information-content figure
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

# ── Make src importable when run as a script ─────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from ecosystem_complexity.api import build_model, run_model, optimize
from ecosystem_complexity.config import load_config, PoolIndex
from ecosystem_complexity.data.parsers import attach_atm14C, load_harvard_forest
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.schemas import ForcingData, ObservationData
from ecosystem_complexity.state import make_default_params, make_initial_state
from ecosystem_complexity.information import (
    OBS_C_STOCKS, OBS_POOL_D14C, OBS_RESP_D14C,
    PARAM_GROUP_TAU, PARAM_GROUP_PARTITION, PARAM_GROUP_TRANSFER, PARAM_GROUP_ENV,
    compute_fisher, compute_dof, compute_posterior,
    make_prior_covariance, get_param_names, get_param_groups,
    _default_fields,
)
from ecosystem_complexity.analysis import (
    run_ablation_study, summarize_ablation, param_group_dfs,
    run_complexity_ladder, compute_age_diagnostics, age_diagnostics_summary,
)

# ── File paths (relative to repo root) ──────────────────────────────────────
HF_HR_PATH = (
    "data/harvard_forest/AMF_US-Ha1_FLUXNET_FULLSET_1991-2020_3-5/"
    "AMF_US-Ha1_FLUXNET_FULLSET_HR_1991-2020_3-5.csv"
)
HF_SOIL_CONFIG   = "configs/harvard_forest_soil_only.yaml"
HF_SOIL_C_PATH   = "data/harvard_forest/hf271-07-soils.csv"
HF_SOIL_14C_PATH = "data/harvard_forest/hf212-03-14c-org.csv"
HF_RESP_14C_PATH = "data/harvard_forest/hf212-01-14c-no-treat.csv"
HUA_PATH     = "data/shared/atm_14C/Hua_2021.csv"
GRAVEN_PATH  = "data/shared/atm_14C/Graven_2017.csv"
INTCAL_PATH  = "data/shared/atm_14C/intcal20.14c"

_R_STD = 1.176e-12   # IAEA modern ¹⁴C/¹²C standard (matches tracer_14C.py)

# Horizon → model pool name (hf212-03 → harvard_forest_soil_only.yaml)
_HORIZON_TO_POOL = {
    "Oi":    "organic_litter",
    "Oe":    "organic_fast",
    "A-lf":  "mineral_A_fast",
    "A-min": "mineral_A_slow",
}

# Pool visual styles: (display label, colour, marker)
_POOL_STYLES = {
    "organic_litter":  ("Org litter (Oi)",    "tab:green",  "o"),
    "organic_fast":    ("Org fast (Oe)",       "tab:olive",  "s"),
    "mineral_A_fast":  ("Min-A fast (A-lf)",   "tab:orange", "^"),
    "mineral_A_slow":  ("Min-A slow (A-min)",  "tab:brown",  "D"),
    "mineral_B_slow":  ("Min-B slow",           "tab:gray",   "v"),
    "mineral_B_passive": ("Min-B passive",      "tab:purple", "P"),
}

# Parameter group colours for plots
_GROUP_COLORS = {
    PARAM_GROUP_TAU:       "steelblue",
    PARAM_GROUP_PARTITION: "darkorange",
    PARAM_GROUP_TRANSFER:  "seagreen",
    PARAM_GROUP_ENV:       "mediumpurple",
}

# Complexity ladder: (label, parameter fields to include)
_LADDER_RUNGS = [
    ("τ only\n(6 params)",           ["log_tau"]),
    ("τ + partition\n(10 params)",   ["log_tau", "log_external_input_partition"]),
    ("τ + partition\n+ transfer",    ["log_tau", "log_external_input_partition",
                                      "log_f_transfer"]),
]


# ════════════════════════════════════════════════════════════════════════════
# Data helpers  (adapted from harvard_forest_inversion.py)
# ════════════════════════════════════════════════════════════════════════════

def _slice_forcing(forcing: ForcingData, start: int, end: int) -> ForcingData:
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


def _build_soil_14C_obs(soil_14c_path: str, forcing_time) -> dict[str, jnp.ndarray]:
    """Map hf212-03 pool Δ¹⁴C to time-aligned arrays."""
    df = pd.read_csv(soil_14c_path)
    df["horizon"] = df["horizon"].str.strip()
    time_np = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T = len(time_np)
    obs_dict: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        hz = row["horizon"]
        if hz not in _HORIZON_TO_POOL:
            continue
        pool = _HORIZON_TO_POOL[hz]
        yr = float(row["year"]) + 0.5
        val = float(row["of.14c.12c"])
        t_idx = int(np.argmin(np.abs(years_np - yr)))
        if pool not in obs_dict:
            obs_dict[pool] = np.full(T, np.nan, dtype=np.float32)
        obs_dict[pool][t_idx] = val
    return {k: jnp.array(v) for k, v in obs_dict.items()}


def _build_resp_14C_obs(resp_14c_path: str, forcing_time) -> jnp.ndarray:
    """Map hf212-01 NWN respired Δ¹⁴C to a time-aligned array."""
    df = pd.read_csv(resp_14c_path)
    nwn = df[df["site"] == "NWN"].groupby("year.date")["delta.14c"].mean()
    time_np = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    T = len(time_np)
    arr = np.full(T, np.nan, dtype=np.float32)
    for yr, val in nwn.items():
        t_idx = int(np.argmin(np.abs(years_np - float(yr))))
        arr[t_idx] = float(val)
    return jnp.array(arr)


def _build_state0(config, pool_index, site_cfg):
    """Initialise state from 1996 hf271 C stocks and hf212-03 Δ¹⁴C."""
    soils = pd.read_csv(HF_SOIL_C_PATH)
    org = soils.loc[soils["horizon"] == "O", "carbon.gm2"].dropna().mean()
    min_ = soils.loc[soils["horizon"] == "M", "carbon.gm2"].dropna().mean()
    minB = 1.5 * min_

    C12_init = {
        "organic_litter": 0.40 * org,  "organic_fast":    0.60 * org,
        "mineral_A_fast": 0.30 * min_, "mineral_A_slow":  0.70 * min_,
        "mineral_B_slow": 0.20 * minB, "mineral_B_passive": 0.80 * minB,
    }

    d14c_df = pd.read_csv(HF_SOIL_14C_PATH)
    d14c_df["horizon"] = d14c_df["horizon"].str.strip()
    d96 = d14c_df[d14c_df["year"] == 1996]

    d14C_init = {
        "organic_litter":    float(d96.loc[d96["horizon"] == "Oi",    "of.14c.12c"].iloc[0]),
        "organic_fast":      float(d96.loc[d96["horizon"] == "Oe",    "of.14c.12c"].iloc[0]),
        "mineral_A_fast":    float(d96.loc[d96["horizon"] == "A-lf",  "of.14c.12c"].iloc[0]),
        "mineral_A_slow":    float(d96.loc[d96["horizon"] == "A-min", "of.14c.12c"].iloc[0]),
        "mineral_B_slow":    -50.0,
        "mineral_B_passive": -300.0,
    }

    base = make_initial_state(config, site_cfg)
    n_pools = len(pool_index)
    pool_set = set(pool_index.pool_names)
    C12_arr = np.zeros(n_pools, dtype=np.float32)
    C14_arr = np.zeros(n_pools, dtype=np.float32)
    for name, c12 in C12_init.items():
        if name not in pool_set:
            continue
        i = pool_index[name]
        d14c = d14C_init.get(name, 0.0)
        C12_arr[i] = max(c12, 0.0)
        C14_arr[i] = C12_arr[i] * _R_STD * (1.0 + d14c / 1000.0)
    return base._replace(C12=jnp.array(C12_arr), C14=jnp.array(C14_arr))


# ════════════════════════════════════════════════════════════════════════════
# Core workflow
# ════════════════════════════════════════════════════════════════════════════

def load_and_optimize():
    """Build model, load data, optimise, return everything needed for analysis."""
    os.chdir(_REPO_ROOT)

    # ── Atmospheric ¹⁴C record ───────────────────────────────────────────────
    print("Loading atmospheric ¹⁴C record…")
    atm14C = load_full_14C_record(
        hua_path=HUA_PATH, graven_path=GRAVEN_PATH, intcal_path=INTCAL_PATH,
        hemisphere="NH", start_year=1500.0, end_year=2025.0,
    )

    # ── Build model ──────────────────────────────────────────────────────────
    print("Building model…")
    model = build_model(HF_SOIL_CONFIG)
    config = model.config
    idx = model.pool_index
    with open(HF_SOIL_CONFIG) as fh:
        site_cfg = yaml.safe_load(fh).get("site", {})

    # ── Load forcing (1996 onward) ───────────────────────────────────────────
    print("Loading forcing…")
    forcing_full, _ = load_harvard_forest(
        hr_path=HF_HR_PATH, config=config, qc_threshold=2,
        include_gpp_forcing=True,
    )
    years_daily, d14c_daily = atm14C
    forcing_full = attach_atm14C(forcing_full, d14c_daily, years_daily)

    time_np = np.array(forcing_full.time)
    years_all = 1970.0 + time_np / 365.25
    start_idx = int(np.searchsorted(years_all, 1996.0))
    forcing = _slice_forcing(forcing_full, start_idx, len(time_np))
    T = int(forcing.time.shape[0])
    time_years = 1970.0 + np.array(forcing.time) / 365.25
    print(f"  Forcing: {T} days  ({time_years[0]:.1f}–{time_years[-1]:.1f})")

    # ── Observations ─────────────────────────────────────────────────────────
    print("Building observations…")
    delta14C_obs = _build_soil_14C_obs(HF_SOIL_14C_PATH, forcing.time)
    delta14C_resp = _build_resp_14C_obs(HF_RESP_14C_PATH, forcing.time)
    n_pool_obs = int(jnp.sum(~jnp.isnan(
        jnp.concatenate([v for v in delta14C_obs.values()]))))
    n_resp_obs = int(jnp.sum(~jnp.isnan(delta14C_resp)))
    print(f"  Pool Δ¹⁴C: {n_pool_obs} obs across {len(delta14C_obs)} pools")
    print(f"  Respired Δ¹⁴C: {n_resp_obs} dates (hf212-01 NWN)")

    obs_full = ObservationData(
        time=forcing.time,
        NEE=jnp.full(T, jnp.nan), GPP=jnp.full(T, jnp.nan),
        ER=jnp.full(T, jnp.nan),  NEE_unc=jnp.full(T, jnp.nan),
        delta14C_obs=delta14C_obs, deltaD14C_obs={}, C_pools_obs={},
        delta14C_resp=delta14C_resp,
    )

    # ── Initial state from 1996 observations ────────────────────────────────
    state0 = _build_state0(config, idx, site_cfg)
    print(f"  State0 total C12: {float(jnp.sum(state0.C12)):.0f} gC m⁻²")

    # ── Prior forward run ────────────────────────────────────────────────────
    print("\nRunning prior forward model…")
    t0 = time.perf_counter()
    params_prior = make_default_params(config)
    out_prior = run_model(model, forcing, state0=state0, params=params_prior)
    jax.block_until_ready(out_prior.delta14C)
    print(f"  Done [{time.perf_counter()-t0:.1f}s]")

    # ── Optimisation: pool Δ¹⁴C + respired Δ¹⁴C ────────────────────────────
    opt_fields = ("log_tau", "log_external_input_partition")
    print(f"\nOptimising [{', '.join(opt_fields)}] against pool + resp Δ¹⁴C…")
    t0 = time.perf_counter()
    result = optimize(model, forcing, obs_full, state0=state0, fields=opt_fields)
    print(f"  Done [{time.perf_counter()-t0:.1f}s]  "
          f"loss {result.loss_history[0]:.4f} → {result.loss_history[-1]:.4f}"
          f"  (converged={result.converged})")

    params_opt = result.params_opt
    out_opt = run_model(model, forcing, state0=state0, params=params_opt)
    jax.block_until_ready(out_opt.delta14C)

    return dict(
        model=model, config=config, idx=idx, site_cfg=site_cfg,
        forcing=forcing, time_years=time_years,
        obs_full=obs_full, delta14C_obs=delta14C_obs, delta14C_resp=delta14C_resp,
        state0=state0,
        params_prior=params_prior, out_prior=out_prior,
        params_opt=params_opt, out_opt=out_opt,
        opt_result=result,
    )


# ════════════════════════════════════════════════════════════════════════════
# Information-content analysis
# ════════════════════════════════════════════════════════════════════════════

def run_information_analysis(data: dict) -> dict:
    """Run all information-content analyses and return structured results."""
    model = data["model"]
    forcing = data["forcing"]
    state0 = data["state0"]
    params_opt = data["params_opt"]
    obs_full = data["obs_full"]

    # Use optimised params as the linearisation point
    fields_standard = ["log_tau", "log_external_input_partition"]

    print("\n" + "─" * 60)
    print("Information-content analysis")
    print("─" * 60)

    # ── 1. Observation ablation ──────────────────────────────────────────────
    print("Running observation ablation study…")
    t0 = time.perf_counter()
    ablation = run_ablation_study(
        model, forcing, state0, params_opt, obs_full,
        fields=fields_standard,
    )
    print(f"  Done [{time.perf_counter()-t0:.1f}s]")

    ablation_table = summarize_ablation(ablation)
    print(f"\n  {'Scenario':<45s}  {'DFS':>6}  {'mean UR':>8}")
    print("  " + "─" * 65)
    for key, row in ablation_table.items():
        print(f"  {key:<45s}  {row['dfs_total']:6.3f}  "
              f"{row['uncertainty_reduction_mean']:8.3f}")

    # ── 2. Full FIM, DFS, and posterior ─────────────────────────────────────
    print("\nComputing full FIM (all obs types)…")
    t0 = time.perf_counter()
    prior_sigma = make_prior_covariance(params_opt, fields_standard, model)
    param_groups = get_param_groups(params_opt, fields_standard, model)

    fisher_full = compute_fisher(
        model, forcing, state0, params_opt, obs_full,
        fields=fields_standard,
    )
    dof_full = compute_dof(fisher_full, prior_sigma, param_groups=param_groups)
    post_full = compute_posterior(fisher_full, prior_sigma)
    print(f"  Done [{time.perf_counter()-t0:.1f}s]  DFS = {dof_full.dfs_total:.3f}")

    # Print per-parameter breakdown
    param_names = fisher_full.param_names or []
    print(f"\n  {'Parameter':<40s}  {'prior σ':>8}  {'post σ':>8}  {'UR':>6}  {'DFS':>6}")
    print("  " + "─" * 75)
    for i, name in enumerate(param_names):
        ps = prior_sigma[i]
        qs = post_full.posterior_sigma[i]
        ur = post_full.uncertainty_reduction[i]
        dfs_i = dof_full.dfs_per_param[i]
        print(f"  {name:<40s}  {ps:8.3f}  {qs:8.3f}  {ur:6.3f}  {dfs_i:6.3f}")

    # DFS by parameter group
    grp_dfs = param_group_dfs(dof_full)
    print(f"\n  DFS by parameter group:")
    for grp, val in grp_dfs.items():
        print(f"    {grp:<30s}  {val:.3f}")

    # ── 3. Parameter complexity ladder ──────────────────────────────────────
    print("\nRunning parameter complexity ladder…")
    ladder_results = []
    for label, fields in _LADDER_RUNGS:
        n_params = sum(
            int(np.prod(getattr(params_opt, f).shape)) for f in fields
        )
        t0 = time.perf_counter()
        fisher_k = compute_fisher(
            model, forcing, state0, params_opt, obs_full,
            fields=fields,
        )
        prior_k = make_prior_covariance(params_opt, fields, model)
        dof_k = compute_dof(fisher_k, prior_k)
        dt = time.perf_counter() - t0
        clean_label = label.replace("\n", " ")
        print(f"  {clean_label:<35s}  n_params={n_params:3d}  "
              f"DFS={dof_k.dfs_total:.3f}  [{dt:.1f}s]")
        ladder_results.append({
            "label": label,
            "n_params": n_params,
            "dfs": dof_k.dfs_total,
            "fisher": fisher_k,
            "dof": dof_k,
        })

    # ── 4. Age diagnostics ────────────────────────────────────────────────────
    print("\nComputing age diagnostics…")
    age_diag_prior = compute_age_diagnostics(data["out_prior"], data["params_prior"], model)
    age_diag_opt   = compute_age_diagnostics(data["out_opt"],   params_opt,            model)

    summary_opt = age_diagnostics_summary(age_diag_opt)
    print(f"\n  Time-mean age diagnostics (optimised params):")
    print(f"  {'Quantity':<40s}  {'mean Δ¹⁴C (‰)':>14}")
    print("  " + "─" * 58)
    for key, entry in sorted(summary_opt.items()):
        if not np.isnan(entry["mean"]):
            print(f"  {key:<40s}  {entry['mean']:14.1f}")

    return dict(
        ablation=ablation,
        ablation_table=ablation_table,
        fisher_full=fisher_full,
        dof_full=dof_full,
        post_full=post_full,
        prior_sigma=prior_sigma,
        param_groups=param_groups,
        grp_dfs=grp_dfs,
        ladder_results=ladder_results,
        age_diag_prior=age_diag_prior,
        age_diag_opt=age_diag_opt,
        fields_standard=fields_standard,
    )


# ════════════════════════════════════════════════════════════════════════════
# Figure
# ════════════════════════════════════════════════════════════════════════════

def make_analysis_figure(data: dict, analysis: dict, out_path: str) -> None:
    """6-panel information-content figure.

    (a) DFS by obs-type scenario (ablation)
    (b) Per-parameter DFS — which params are constrained?
    (c) Uncertainty reduction: prior vs. posterior σ for log_tau
    (d) FIM eigenspectrum — effective rank
    (e) Age diagnostics — stored vs. respired Δ¹⁴C (prior vs. optimised)
    (f) Parameter complexity ladder — DFS vs. n_params
    """
    from matplotlib.patches import Patch

    model = data["model"]
    pool_names = model.pool_index.pool_names
    time_years = data["time_years"]

    ablation     = analysis["ablation"]
    ablation_table = analysis["ablation_table"]
    fisher_full  = analysis["fisher_full"]
    dof_full     = analysis["dof_full"]
    post_full    = analysis["post_full"]
    prior_sigma  = analysis["prior_sigma"]
    param_groups = analysis["param_groups"]
    grp_dfs      = analysis["grp_dfs"]
    ladder       = analysis["ladder_results"]
    age_prior    = analysis["age_diag_prior"]
    age_opt      = analysis["age_diag_opt"]
    delta14C_obs = data["delta14C_obs"]
    delta14C_resp = data["delta14C_resp"]

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.42, wspace=0.38,
        left=0.06, right=0.97, top=0.92, bottom=0.07,
    )
    ax_abl  = fig.add_subplot(gs[0, 0])
    ax_dfs  = fig.add_subplot(gs[0, 1])
    ax_ur   = fig.add_subplot(gs[0, 2])
    ax_eig  = fig.add_subplot(gs[1, 0])
    ax_age  = fig.add_subplot(gs[1, 1])
    ax_lad  = fig.add_subplot(gs[1, 2])

    # ── (a) DFS ablation — stacked bars coloured by obs-type contribution ────
    obs_type_colors = {
        OBS_C_STOCKS:   "tab:blue",
        OBS_POOL_D14C:  "tab:green",
        OBS_RESP_D14C:  "tab:red",
    }
    scenario_labels = [t["scenario"] for t in ablation_table.values()]

    # Shorten labels for readability
    def _short_label(s):
        return (s.replace("C_stocks", "C-stock")
                 .replace("pool_delta14C", "pool Δ¹⁴C")
                 .replace("resp_delta14C", "resp Δ¹⁴C")
                 .replace(" + ", "\n+ "))

    short_labels = [_short_label(lab) for lab in scenario_labels]
    x = np.arange(len(short_labels))
    bar_h = 0.55

    dfs_totals = [row["dfs_total"] for row in ablation_table.values()]
    ax_abl.bar(x, dfs_totals, bar_h, color="0.75", alpha=0.5, label="Total DFS")

    # Overlay per-obs-type DFS contribution (from the single-type scenarios)
    single_type_dfs = {}
    for key, res in ablation.items():
        if len(res.obs_types) == 1:
            single_type_dfs[res.obs_types[0]] = res.dfs_total

    for obs_type, color in obs_type_colors.items():
        if obs_type in single_type_dfs:
            ax_abl.axhline(single_type_dfs[obs_type], lw=1.2, linestyle="--",
                           color=color, alpha=0.7,
                           label=f"{_short_label(obs_type)} alone: {single_type_dfs[obs_type]:.2f}")

    ax_abl.set_xticks(x)
    ax_abl.set_xticklabels(short_labels, fontsize=7.5)
    ax_abl.set_ylabel("Degrees of freedom for signal (DFS)", fontsize=9)
    ax_abl.set_title("(a) DFS by observation subset (ablation)", fontsize=9, loc="left")
    ax_abl.legend(fontsize=7.5, framealpha=0.9)
    ax_abl.tick_params(axis="y", labelsize=8)
    ax_abl.grid(axis="y", lw=0.4, alpha=0.4)
    ax_abl.set_ylim(bottom=0)

    # ── (b) Per-parameter DFS — horizontal bar chart coloured by group ───────
    param_names = fisher_full.param_names or []
    dfs_per = dof_full.dfs_per_param

    # Build colour array by group
    n_params = len(param_names)
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
    ax_dfs.set_xlabel("DFS per parameter (diagonal of averaging kernel)", fontsize=8)
    ax_dfs.set_title("(b) Per-parameter DFS (all obs combined)", fontsize=9, loc="left")
    ax_dfs.tick_params(axis="x", labelsize=8)
    ax_dfs.grid(axis="x", lw=0.4, alpha=0.4)
    ax_dfs.axvline(0, lw=0.6, color="black")

    # Group legend
    legend_patches = [
        Patch(facecolor=_GROUP_COLORS[g], label=g, alpha=0.8)
        for g in param_groups if g in _GROUP_COLORS
    ]
    ax_dfs.legend(handles=legend_patches, fontsize=7.5, framealpha=0.9,
                  loc="lower right")

    # ── (c) Uncertainty reduction: prior σ vs. posterior σ for log_tau ───────
    tau_idxs = param_groups.get(PARAM_GROUP_TAU, [])
    if tau_idxs:
        tau_names = [param_names[i].replace("log_tau[", "").rstrip("]")
                     for i in tau_idxs]
        prior_s = prior_sigma[tau_idxs]
        post_s = post_full.posterior_sigma[tau_idxs]
        n_tau = len(tau_idxs)
        x_tau = np.arange(n_tau)
        w = 0.35

        ax_ur.bar(x_tau - w / 2, prior_s, w, label="Prior σ (log τ)", color="0.7", alpha=0.85)
        ax_ur.bar(x_tau + w / 2, post_s,  w, label="Posterior σ (log τ)", color="steelblue", alpha=0.85)

        # Annotate % reduction
        for i, (ps, qs) in enumerate(zip(prior_s, post_s)):
            ur_pct = (1 - qs / ps) * 100
            ax_ur.text(i, max(ps, qs) + 0.005, f"{ur_pct:.0f}%",
                       ha="center", va="bottom", fontsize=6.5, color="0.3")

        ax_ur.set_xticks(x_tau)
        ax_ur.set_xticklabels(tau_names, fontsize=7, rotation=30, ha="right")
        ax_ur.set_ylabel("Standard deviation in log-space", fontsize=9)
        ax_ur.set_title("(c) Uncertainty reduction for log_τ", fontsize=9, loc="left")
        ax_ur.legend(fontsize=8, framealpha=0.9)
        ax_ur.tick_params(axis="y", labelsize=8)
        ax_ur.grid(axis="y", lw=0.4, alpha=0.4)
        ax_ur.set_ylim(bottom=0)

    # ── (d) FIM eigenspectrum — log-scale, sorted descending ─────────────────
    eigvals = fisher_full.eigenvalues
    eigvals_pos = np.maximum(eigvals, 1e-12)  # clamp for log-plot
    k = np.arange(1, len(eigvals_pos) + 1)
    ax_eig.semilogy(k, eigvals_pos, "o-", color="steelblue", lw=1.5, ms=5,
                    alpha=0.9, label="FIM eigenvalues")

    # Per-obs-type FIM eigenvalues (dashed)
    eig_colors = {OBS_C_STOCKS: "tab:blue", OBS_POOL_D14C: "tab:green",
                  OBS_RESP_D14C: "tab:red"}
    for obs_type, FIM_k in fisher_full.FIM_per_type.items():
        ev_k = np.linalg.eigvalsh(FIM_k)[::-1]
        ev_k = np.maximum(ev_k, 1e-12)
        color = eig_colors.get(obs_type, "0.5")
        label = _short_label(obs_type)
        ax_eig.semilogy(k, ev_k, "--", color=color, lw=1.0, alpha=0.7,
                        label=f"{label} only")

    # Mark the effective rank (where cumulative sum reaches 90% of DFS)
    dfs_cum = np.cumsum(dof_full.dfs_per_param[np.argsort(dof_full.dfs_per_param)[::-1]])
    eff_rank = int(np.searchsorted(dfs_cum, 0.9 * dof_full.dfs_total)) + 1
    ax_eig.axvline(eff_rank, lw=1.0, linestyle=":", color="firebrick", alpha=0.7,
                   label=f"90% DFS rank = {eff_rank}")

    ax_eig.set_xlabel("Eigenvalue rank", fontsize=9)
    ax_eig.set_ylabel("FIM eigenvalue (log scale)", fontsize=9)
    ax_eig.set_title("(d) FIM eigenspectrum by obs type", fontsize=9, loc="left")
    ax_eig.legend(fontsize=7.5, framealpha=0.9)
    ax_eig.tick_params(labelsize=8)
    ax_eig.grid(axis="both", lw=0.4, alpha=0.4)
    ax_eig.set_xlim(0.5, len(eigvals) + 0.5)

    # ── (e) Age diagnostics — stored (pool) vs. respired Δ¹⁴C ───────────────
    # Bulk mass-weighted Δ¹⁴C — prior (dotted) vs. optimised (solid)
    ax_age.plot(time_years, age_prior.bulk_delta14C,
                lw=0.9, color="0.5", linestyle=":", alpha=0.7,
                label="Bulk (stored) — prior")
    ax_age.plot(time_years, age_opt.bulk_delta14C,
                lw=1.4, color="sienna", alpha=0.9,
                label="Bulk (stored) — optimised")

    # Flux-weighted respired Δ¹⁴C
    ax_age.plot(time_years, age_prior.respired_delta14C,
                lw=0.9, color="0.4", linestyle="--", alpha=0.6,
                label="Respired Rh — prior")
    ax_age.plot(time_years, age_opt.respired_delta14C,
                lw=1.4, color="tomato", linestyle="--", alpha=0.9,
                label="Respired Rh — optimised")

    # hf212-03 pool obs (scatter per pool, offset slightly for visibility)
    for pool_name, (label, color, marker) in _POOL_STYLES.items():
        if pool_name not in delta14C_obs:
            continue
        obs_arr = np.array(delta14C_obs[pool_name])
        valid = np.where(np.isfinite(obs_arr))[0]
        for t_i in valid:
            ax_age.scatter(time_years[t_i], obs_arr[t_i],
                           s=60, color=color, marker=marker,
                           edgecolors="black", linewidths=0.7, zorder=7,
                           label=f"{label} (hf212-03)")

    # hf212-01 respired CO₂ obs
    resp_arr = np.array(delta14C_resp)
    valid_resp = np.where(np.isfinite(resp_arr))[0]
    if len(valid_resp) > 0:
        ax_age.scatter(time_years[valid_resp], resp_arr[valid_resp],
                       s=20, color="black", marker="x", alpha=0.5, zorder=6,
                       label="Resp obs (hf212-01 NWN)")

    ax_age.axhline(0, lw=0.5, color="gray", linestyle=":")
    ax_age.set_ylabel("Δ¹⁴C (‰)", fontsize=9)
    ax_age.set_xlabel("Year", fontsize=9)
    ax_age.set_title("(e) Storage age vs. respiration age", fontsize=9, loc="left")
    ax_age.tick_params(labelsize=8)
    ax_age.grid(axis="y", lw=0.4, alpha=0.4)
    # Deduplicate legend entries
    handles, labels = ax_age.get_legend_handles_labels()
    seen = {}
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen[lab] = h
    ax_age.legend(list(seen.values()), list(seen.keys()),
                  fontsize=7, framealpha=0.9, ncol=2, loc="upper right")

    # ── (f) Parameter complexity ladder — DFS vs. n_params ───────────────────
    lad_n = [r["n_params"] for r in ladder]
    lad_dfs = [r["dfs"] for r in ladder]
    lad_labels = [r["label"] for r in ladder]

    ax_lad.plot(lad_n, lad_dfs, "o-", color="steelblue", lw=2.0, ms=8, zorder=5)
    for n, dfs, lab in zip(lad_n, lad_dfs, lad_labels):
        ax_lad.annotate(
            lab, (n, dfs),
            textcoords="offset points", xytext=(8, 4),
            fontsize=7.5, color="0.3",
        )

    # Diagonal reference (DFS = n_params means fully prior-dominated — no info)
    # and a "saturation" line at the observed DFS maximum
    n_range = np.linspace(min(lad_n) * 0.9, max(lad_n) * 1.1, 50)
    ax_lad.fill_between(n_range, 0, n_range, alpha=0.06, color="tomato",
                         label="Prior-dominated (DFS = n_params)")
    ax_lad.plot(n_range, n_range, lw=0.8, linestyle="--", color="tomato", alpha=0.5)
    ax_lad.axhline(max(lad_dfs), lw=0.8, linestyle=":", color="steelblue", alpha=0.6,
                   label=f"Max DFS observed = {max(lad_dfs):.2f}")

    ax_lad.set_xlabel("Parameter vector dimension", fontsize=9)
    ax_lad.set_ylabel("Degrees of freedom for signal (DFS)", fontsize=9)
    ax_lad.set_title("(f) Parameter complexity ladder", fontsize=9, loc="left")
    ax_lad.legend(fontsize=8, framealpha=0.9)
    ax_lad.tick_params(labelsize=8)
    ax_lad.grid(axis="both", lw=0.4, alpha=0.4)
    ax_lad.set_xlim(left=0)
    ax_lad.set_ylim(bottom=0)

    fig.suptitle(
        "Harvard Forest Soil-Only Model — Information-Content Analysis\n"
        "Radiocarbon observations constrain stored-carbon age (pool Δ¹⁴C)"
        " vs. active decomposition (respired CO₂ Δ¹⁴C)",
        fontsize=11,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    try:
        plt.show()
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
# Summary report
# ════════════════════════════════════════════════════════════════════════════

def print_summary_report(data: dict, analysis: dict) -> None:
    """Print a structured text report to stdout."""
    model = data["model"]
    dof_full = analysis["dof_full"]
    post_full = analysis["post_full"]
    prior_sigma = analysis["prior_sigma"]
    ladder = analysis["ladder_results"]
    age_opt = analysis["age_diag_opt"]

    print("\n" + "═" * 70)
    print("  INFORMATION-CONTENT ANALYSIS  —  Harvard Forest Soil-Only")
    print("═" * 70)

    print(f"\n  Model: {model.config.site_id}  ({len(model.pool_index)} pools)")
    print(f"  Linearisation: MAP estimate (pool Δ¹⁴C + resp Δ¹⁴C inversion)")
    print(f"  Parameter fields: log_tau + log_external_input_partition")
    print(f"  Total DFS: {dof_full.dfs_total:.3f}")

    print("\n  DFS by observation type:")
    for obs_type, dfs_k in dof_full.dfs_per_obs_type.items():
        print(f"    {obs_type:<30s}  {dfs_k:.3f}")

    print("\n  DFS by parameter group:")
    for grp, dfs_g in (analysis["grp_dfs"] or {}).items():
        print(f"    {grp:<30s}  {dfs_g:.3f}")

    print("\n  Parameter complexity ladder:")
    print(f"  {'Rung':<40s}  {'n_params':>8}  {'DFS':>6}  {'DFS/n':>7}")
    for r in ladder:
        clean = r["label"].replace("\n", " ")
        print(f"  {clean:<40s}  {r['n_params']:8d}  {r['dfs']:6.3f}"
              f"  {r['dfs']/r['n_params']:7.3f}")

    print("\n  Age-structure diagnostics (time-mean, optimised params):")
    print(f"    Bulk stored Δ¹⁴C (mass-weighted):    "
          f"{float(np.nanmean(age_opt.bulk_delta14C)):+.1f} ‰")
    print(f"    Respired Rh Δ¹⁴C (flux-weighted):   "
          f"{float(np.nanmean(age_opt.respired_delta14C)):+.1f} ‰")
    separation = (float(np.nanmean(age_opt.bulk_delta14C))
                  - float(np.nanmean(age_opt.respired_delta14C)))
    print(f"    Stored − respired (age separation):  {separation:+.1f} ‰")
    print()
    if abs(separation) > 20:
        print("  → Stored carbon is significantly older than respired carbon.")
        print("    Pool Δ¹⁴C and respired Δ¹⁴C should constrain distinct parameters.")
    else:
        print("  → Stored and respired carbon are similar in age.")
        print("    Pool Δ¹⁴C and respired Δ¹⁴C may be partially redundant.")

    print("\n" + "═" * 70)


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Step 1: load data, run optimisation
    data = load_and_optimize()

    # Step 2: information-content analysis (ablation + complexity ladder + age)
    analysis = run_information_analysis(data)

    # Step 3: text summary
    print_summary_report(data, analysis)

    # Step 4: 6-panel figure
    out_path = os.path.join("notebooks", "harvard_forest_analysis.png")
    make_analysis_figure(data, analysis, out_path)
