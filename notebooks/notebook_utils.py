"""
notebook_utils.py — Shared utilities for site-specific inversion notebooks.

Provides:
  find_data_root()        — walk up the directory tree to locate the data/ folder
  softmax_np()            — pure-numpy softmax for display use
  build_mean_ss_modifier()— compute mean decomp modifier + mean input from forcing
  build_oe5_obs_sets()    — construct the five ObservationData variants (OE1–OE5)
  run_oe5_sequence()      — run OE1–OE5, collect results and outputs
  print_prior_ss_check()  — print analytical SS sanity table
  print_dfs_table()       — print DFS per experiment
  print_tau_summary()     — print τ before/after table
  print_rh_diagnostics()  — print annual Rh sim vs obs table
  make_oe5_figure()       — 8-panel OE comparison figure (site-parameterised)
"""
from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_data_root(start: str) -> str:
    """
    Walk up the directory tree from *start* looking for a ``data/`` subdirectory.

    Returns the first ancestor that contains ``data/``, or *start* if none found
    within 4 levels (fallback prevents infinite climb on unexpected layouts).
    """
    candidate = start
    for _ in range(4):
        if os.path.isdir(os.path.join(candidate, "data")):
            return candidate
        candidate = os.path.dirname(candidate)
    return start


# ─────────────────────────────────────────────────────────────────────────────
# Numeric helpers
# ─────────────────────────────────────────────────────────────────────────────

def softmax_np(x: np.ndarray) -> np.ndarray:
    """Pure-NumPy softmax (for display/printing only — not JAX-traced)."""
    e = np.exp(x - x.max())
    return e / e.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Analytical SS helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_mean_ss_modifier(forcing, model, t_nan_fill: float = 5.0,
                           theta_nan_fill: float = 0.3) -> tuple[float, float]:
    """
    Compute the mean decomposition modifier and mean NPP input used by the
    analytical steady-state initialiser.

    Returns
    -------
    (mean_modifier, mean_input_val)
        mean_modifier : float
            Time-mean of f_temp × f_moisture × thawed_frac evaluated at the
            prior parameters over the forcing window.
        mean_input_val : float
            Mean GPP × CUE (gC m⁻² day⁻¹).
    """
    from ecosystem_complexity.oe_utils import build_mean_ss_modifier as _build_mean_ss_modifier
    from ecosystem_complexity.state import make_default_params

    params0 = make_default_params(model.config)
    mean_modifier, mean_gpp = _build_mean_ss_modifier(
        forcing,
        params0,
        t_nan_fill=t_nan_fill,
        theta_nan_fill=theta_nan_fill,
    )
    cue_val = float(model.config.external_inputs.CUE)
    mean_input_val = mean_gpp * cue_val
    return mean_modifier, mean_input_val


# ─────────────────────────────────────────────────────────────────────────────
# OE5 observation set builder
# ─────────────────────────────────────────────────────────────────────────────

def build_oe5_obs_sets(forcing, delta14C_obs: dict, delta14C_resp,
                       c_pools_obs: dict, er_sliced):
    """
    Build the five ``ObservationData`` variants used in OE1–OE5.

    OE1 — C stocks only
    OE2 — C stocks + pool Δ¹⁴C
    OE3 — C stocks + resp Δ¹⁴C  (orthogonality test)
    OE4 — C stocks + pool Δ¹⁴C + resp Δ¹⁴C
    OE5 — OE4 + FluxNet ER (free f_hetero)

    Parameters
    ----------
    forcing :
        ``ForcingData`` for the inversion window.
    delta14C_obs :
        ``{pool_name: jnp.array}`` pool Δ¹⁴C observations.
    delta14C_resp :
        ``jnp.array`` surface-resp Δ¹⁴C time series (NaN outside obs dates).
    c_pools_obs :
        ``{pool_name: (mu, sigma)}`` individual C stock constraints.
    er_sliced :
        ``jnp.array`` daily ER (NaN for masked years/gaps).

    Returns
    -------
    tuple[ObservationData, ...]
        Five obs sets in order OE1, OE2, OE3, OE4, OE5.
    """
    from ecosystem_complexity.oe_utils import build_oe_observation_sets

    return build_oe_observation_sets(
        forcing, delta14C_obs, delta14C_resp, c_pools_obs, er_sliced
    )


# ─────────────────────────────────────────────────────────────────────────────
# OE1–OE5 sequence runner
# ─────────────────────────────────────────────────────────────────────────────

def run_oe5_sequence(
    model,
    forcing,
    state0,
    obs_sets,
    carbon_sum_blocks: list,
    extra_blocks: list,
    oe_fields: tuple,
    oe_fields_er: tuple,
    make_ss_state_fn,
    print_cstock_diag_fn,
    n_pool_obs: int,
    n_resp_obs: int,
) -> dict:
    """
    Run OE1 through OE5 in sequence, collect results and forward outputs.

    Parameters
    ----------
    model, forcing, state0 :
        Standard model / forcing objects.
    obs_sets :
        Tuple of five ``ObservationData`` objects (OE1 … OE5) as returned by
        ``build_oe5_obs_sets``.
    carbon_sum_blocks :
        Extra ObsBlocks for sum constraints — used in OE1/OE3 (no ISRaD).
    extra_blocks :
        All extra ObsBlocks — used in OE2/OE4/OE5 (sum + ISRaD if present).
    oe_fields, oe_fields_er :
        Field tuples for OE1–4 and OE5 respectively.
    make_ss_state_fn :
        Callable ``(params_opt) -> EcosystemState`` that returns the initial
        state for a diagnostic forward run.  Site-specific (e.g. Barrow pins
        the passive pool to the inventory prior).
    print_cstock_diag_fn :
        Callable ``(label, out)`` that prints site-specific C-stock residuals.
    n_pool_obs, n_resp_obs :
        Counts of pool/resp Δ¹⁴C observations (for header strings).

    Returns
    -------
    dict
        Keys: result_oe1 … result_oe5, out_oe1 … out_oe5, lh1 … lh5,
              params_opt (best run = OE5).
    """
    from ecosystem_complexity.api import optimize_oe
    from ecosystem_complexity.api import run_model

    obs_oe1, obs_oe2, obs_oe3, obs_oe4, obs_oe5 = obs_sets
    n_cstock1 = len(obs_oe1.C_pools_obs) + len(carbon_sum_blocks)
    n_israd   = len(extra_blocks) - len(carbon_sum_blocks)

    # ── OE1 — C stocks only ──────────────────────────────────────────────────
    print(f"\nOE 1 — C stocks only  "
          f"({len(obs_oe1.C_pools_obs)} individual + {len(carbon_sum_blocks)} sum"
          f" = {n_cstock1} constraints)…")
    t0 = time.perf_counter()
    res1 = optimize_oe(model, forcing, obs_oe1, state0=state0,
                       fields=oe_fields, extra_obs_blocks=carbon_sum_blocks)
    ch1 = np.array(res1.cost_history)
    print(f"  Done [{time.perf_counter()-t0:.0f}s]  J {ch1[0]:.2f} → {ch1[-1]:.2f}"
          f"  ({'converged' if res1.converged else 'max-iter'})")
    out1 = run_model(model, forcing, state0=make_ss_state_fn(res1.params_opt),
                     params=res1.params_opt)
    jax.block_until_ready(out1.delta14C)
    print_cstock_diag_fn("OE1", out1)

    # ── OE2 — C stocks + pool Δ¹⁴C (+ ISRaD if present) ────────────────────
    israd_note = f" + {n_israd} ISRaD" if n_israd > 0 else ""
    print(f"\nOE 2 — C stocks + pool Δ¹⁴C{israd_note}  "
          f"({n_cstock1} C + {n_pool_obs} pool Δ¹⁴C obs)…")
    t0 = time.perf_counter()
    res2 = optimize_oe(model, forcing, obs_oe2, state0=state0,
                       fields=oe_fields, extra_obs_blocks=extra_blocks)
    ch2 = np.array(res2.cost_history)
    print(f"  Done [{time.perf_counter()-t0:.0f}s]  J {ch2[0]:.2f} → {ch2[-1]:.2f}"
          f"  ({'converged' if res2.converged else 'max-iter'})")
    out2 = run_model(model, forcing, state0=make_ss_state_fn(res2.params_opt),
                     params=res2.params_opt)
    jax.block_until_ready(out2.delta14C)
    print_cstock_diag_fn("OE2", out2)

    # ── OE3 — C stocks + resp Δ¹⁴C ──────────────────────────────────────────
    print(f"\nOE 3 — C stocks + resp Δ¹⁴C  "
          f"({n_cstock1} C + {n_resp_obs} resp Δ¹⁴C obs)…")
    t0 = time.perf_counter()
    res3 = optimize_oe(model, forcing, obs_oe3, state0=state0,
                       fields=oe_fields, extra_obs_blocks=carbon_sum_blocks)
    ch3 = np.array(res3.cost_history)
    print(f"  Done [{time.perf_counter()-t0:.0f}s]  J {ch3[0]:.2f} → {ch3[-1]:.2f}"
          f"  ({'converged' if res3.converged else 'max-iter'})")
    out3 = run_model(model, forcing, state0=make_ss_state_fn(res3.params_opt),
                     params=res3.params_opt)
    jax.block_until_ready(out3.delta14C)
    print_cstock_diag_fn("OE3", out3)

    # ── OE4 — full Δ¹⁴C + C stocks ──────────────────────────────────────────
    print(f"\nOE 4 — full OE: pool + resp Δ¹⁴C + C stocks{israd_note}  "
          f"({n_cstock1} C + {n_pool_obs} pool + {n_resp_obs} resp obs)…")
    t0 = time.perf_counter()
    res4 = optimize_oe(model, forcing, obs_oe4, state0=state0,
                       fields=oe_fields, extra_obs_blocks=extra_blocks)
    ch4 = np.array(res4.cost_history)
    print(f"  Done [{time.perf_counter()-t0:.0f}s]  J {ch4[0]:.2f} → {ch4[-1]:.2f}"
          f"  ({'converged' if res4.converged else 'max-iter'})")
    print(f"  Posterior σ (diagonal Sₓ): "
          + "  ".join(f"{n}={float(np.sqrt(v)):.3f}"
                      for n, v in zip(res4.state_names[:6],
                                      np.diag(np.array(res4.Sx))[:6])))
    out4 = run_model(model, forcing, state0=make_ss_state_fn(res4.params_opt),
                     params=res4.params_opt)
    jax.block_until_ready(out4.delta14C)
    print_cstock_diag_fn("OE4", out4)

    # ── OE5 — OE4 + FluxNet ER (free f_hetero) ──────────────────────────────
    print(f"\nOE 5 — OE4 + FluxNet ER (free f_hetero)…")
    t0 = time.perf_counter()
    res5 = optimize_oe(model, forcing, obs_oe5, state0=state0,
                       fields=oe_fields_er, extra_obs_blocks=extra_blocks)
    ch5 = np.array(res5.cost_history)
    print(f"  Done [{time.perf_counter()-t0:.0f}s]  J {ch5[0]:.2f} → {ch5[-1]:.2f}"
          f"  ({'converged' if res5.converged else 'max-iter'})")
    print(f"  Posterior σ (diagonal Sₓ): "
          + "  ".join(f"{n}={float(np.sqrt(v)):.3f}"
                      for n, v in zip(res5.state_names[:6],
                                      np.diag(np.array(res5.Sx))[:6])))
    out5 = run_model(model, forcing, state0=make_ss_state_fn(res5.params_opt),
                     params=res5.params_opt)
    jax.block_until_ready(out5.delta14C)
    print_cstock_diag_fn("OE5", out5)

    return dict(
        result_oe1=res1, result_oe2=res2, result_oe3=res3,
        result_oe4=res4, result_oe5=res5,
        out_oe1=out1, out_oe2=out2, out_oe3=out3,
        out_oe4=out4, out_oe5=out5,
        lh1=ch1, lh2=ch2, lh3=ch3, lh4=ch4, lh5=ch5,
        params_opt=res5.params_opt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic printers
# ─────────────────────────────────────────────────────────────────────────────

def print_prior_ss_check(prior_ss, params_prior, idx, ext_target_idx: list):
    """Print prior analytical SS values with input fraction annotations."""
    prior_part = jax.nn.softmax(params_prior.log_external_input_partition)
    n_pools    = len(idx)
    print("\nSanity checks (prior):")
    print(f"  Partition sum: {float(prior_part.sum()):.6f}  (must be 1.000000)")
    f_input_full = np.zeros(n_pools)
    for k, ti in enumerate(ext_target_idx):
        f_input_full[ti] = float(prior_part[k])
    for i, nm in enumerate(idx.pool_names):
        print(f"  SS C_{nm:<14} = {float(prior_ss[i]):7.1f} gC m⁻²"
              f"   (f_input={f_input_full[i]:.3f})")


def print_dfs_table(results: dict):
    """Print DFS trace(A) for each of the five OE experiments."""
    print("\nDFS by experiment:")
    for label, key in [
        ("OE1 C-stocks",    "result_oe1"),
        ("OE2 +pool Δ¹⁴C",  "result_oe2"),
        ("OE3 +resp Δ¹⁴C",  "result_oe3"),
        ("OE4 full",         "result_oe4"),
        ("OE5 full+ER flux", "result_oe5"),
    ]:
        A   = np.array(results[key].averaging_kernel)
        dfs = np.trace(A)
        n   = A.shape[0]
        print(f"  {label:<22} DFS={dfs:.3f} / {n} params  (DFS/n={dfs/n:.3f})")


def print_tau_summary(params_prior, params_opt, pool_names: list):
    """Print turnover-time before/after table plus f_hetero."""
    tau_p   = np.exp(np.array(params_prior.log_tau))
    tau_opt = np.exp(np.array(params_opt.log_tau))
    print(f"\n{'Pool':<16}  {'τ prior (yr)':>13}  {'τ opt (yr)':>12}")
    print("  " + "─" * 43)
    for i, name in enumerate(pool_names):
        print(f"  {name:<16}  {tau_p[i]/365:>13.1f}  {tau_opt[i]/365:>12.1f}")
    f_het_prior = float(jax.nn.sigmoid(params_prior.log_f_hetero))
    f_het_opt   = float(jax.nn.sigmoid(params_opt.log_f_hetero))
    print(f"\n  f_hetero: prior={f_het_prior:.3f}  optimised={f_het_opt:.3f}"
          f"  (ER → Rh fraction)")
    return tau_p, tau_opt


def print_rh_diagnostics(out_oe5, forcing, er_sliced, params_opt):
    """Print annual Rh sim vs obs table using the OE5 posterior f_hetero."""
    f_het     = float(jax.nn.sigmoid(params_opt.log_f_hetero))
    er_np     = np.array(er_sliced)
    gpp_np    = np.array(forcing.GPP_obs)
    years_np  = 1970.0 + np.array(forcing.time) / 365.25
    rh5       = np.array(out_oe5.Rh)
    mean_gpp  = float(np.nanmean(gpp_np))
    print(f"\nAnnual Rh diagnostics (OE5):")
    print(f"  f_hetero (OE5 posterior): {f_het:.3f}  mean_GPP: {mean_gpp:.3f} gC m⁻² day⁻¹")
    for yr in range(int(years_np[0]), int(years_np[-1]) + 1):
        mask   = (years_np >= yr) & (years_np < yr + 1)
        er_yr  = er_np[mask]; er_yr  = er_yr[np.isfinite(er_yr)]
        gpp_yr = gpp_np[mask]; gpp_yr = gpp_yr[np.isfinite(gpp_yr)]
        if len(er_yr) < 30:
            continue
        rh_obs = float(np.mean(er_yr)) * f_het
        rh_sim = float(np.mean(rh5[mask]))
        gpp_m  = float(np.mean(gpp_yr)) if len(gpp_yr) > 10 else float("nan")
        gpp_note = (f"  GPP={gpp_m:.3f} (Δ={gpp_m-mean_gpp:+.3f})"
                    if np.isfinite(gpp_m) else "")
        print(f"  {yr}: Rh_obs={rh_obs:.3f}  Rh_sim={rh_sim:.3f}"
              f"  diff={rh_sim-rh_obs:+.3f} gC m⁻² day⁻¹{gpp_note}")


def print_averaging_kernel(result_oe5):
    """Print OE averaging kernel diagonal and total DFS for OE5."""
    print("\nInformation content (OE averaging kernel, OE5 run)…")
    A_full    = np.array(result_oe5.averaging_kernel)
    dfs_oe    = float(np.trace(A_full))
    n_params  = A_full.shape[0]
    print(f"  n_params={n_params}  DFS=trace(A)={dfs_oe:.3f}"
          f"  DFS/n={dfs_oe/n_params:.3f}")
    Sx_diag = np.diag(np.array(result_oe5.Sx))
    for name, a_ii, sx_ii in zip(result_oe5.state_names[:n_params],
                                  np.diag(A_full), Sx_diag):
        if a_ii > 0.01:
            print(f"    {name}: A_ii={a_ii:.3f}  posterior_σ={np.sqrt(sx_ii):.4f}")
    return dfs_oe, n_params


# ─────────────────────────────────────────────────────────────────────────────
# Shared 8-panel OE5 figure
# ─────────────────────────────────────────────────────────────────────────────

def make_oe5_figure(
    r: dict,
    site_title: str,
    site_subtitle: str,
    resp_obs_label: str,
    pool_colors: list,
    pool_markers: list,
    rh_er_frac: float,
    out_path: str,
):
    """
    Save the standard 8-panel OE comparison figure for a site.

    Parameters
    ----------
    r :
        Results dict as returned by the site's ``run_optimal_inversion()``.
    site_title :
        First line of the figure ``suptitle`` (e.g. ``"Harvard Forest — 3-Pool …"``).
    site_subtitle :
        Second line of the figure ``suptitle``.
    resp_obs_label :
        Legend label for the resp Δ¹⁴C observation scatter (data-source citation).
    pool_colors, pool_markers :
        Per-pool colour/marker lists (same length as pool_names).
    rh_er_frac :
        Fraction used for the "Rh est." line in the GPP panel (e.g. 0.55 or 0.80).
    out_path :
        Output PNG path.
    """
    C_PRIOR = "0.55"
    C_OE1   = "steelblue"
    C_OE2   = "tomato"
    C_OE3   = "mediumseagreen"
    C_OE4   = "darkorange"
    C_OE5   = "mediumpurple"

    time_years   = r["time_years"]
    out_prior    = r["out_prior"]
    out_oe1      = r["out_oe1"];   out_oe2 = r["out_oe2"]
    out_oe3      = r["out_oe3"];   out_oe4 = r["out_oe4"]
    out_oe5      = r["out_oe5"]
    delta14C_obs = r["delta14C_obs"]
    c_pools_obs  = r["c_pools_obs"]
    d14C_resp    = r["delta14C_resp"]
    er_arr_fig   = np.array(r["forcing_ER"])
    pool_idx     = r["pool_idx"]
    age_diag     = r["age_diag"]
    tau_p        = r["tau_p"];     tau_opt = r["tau_opt"]
    lh1, lh2, lh3, lh4, lh5 = (r["lh1"], r["lh2"], r["lh3"],
                                  r["lh4"], r["lh5"])
    pool_names = pool_idx.pool_names

    fig = plt.figure(figsize=(16, 19))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.32)
    axes = [fig.add_subplot(gs[row, col]) for row in range(4) for col in range(2)]
    ax_loss, ax_tau, ax_pool14C, ax_resp14C, ax_cstockbar, ax_age, ax_gpp, ax_cstock = axes

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
    x, w = np.arange(len(pool_names)), 0.35
    ax_tau.bar(x - w/2, tau_p   / 365, width=w, color=C_PRIOR, label="prior",            alpha=0.85)
    ax_tau.bar(x + w/2, tau_opt / 365, width=w, color=C_OE5,   label="optimised (OE5)", alpha=0.85)
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
        ax_pool14C.plot(time_years, np.array(out_prior.delta14C)[:, pi],
                        color=C_PRIOR, lw=1.0, alpha=0.5, linestyle="--",
                        label="prior" if i == 0 else None)
        ax_pool14C.plot(time_years, np.array(out_oe2.delta14C)[:, pi],
                        color=color, lw=1.0, alpha=0.6, linestyle=":")
        ax_pool14C.plot(time_years, np.array(out_oe4.delta14C)[:, pi],
                        color=color, lw=1.4, alpha=0.8, linestyle="-.")
        ax_pool14C.plot(time_years, np.array(out_oe5.delta14C)[:, pi],
                        color=color, lw=1.8,
                        label=f"{pool_name.replace('soil_','')} (OE5)")
        if pool_name in delta14C_obs:
            obs_arr  = np.array(delta14C_obs[pool_name])
            obs_mask = ~np.isnan(obs_arr)
            ax_pool14C.scatter(time_years[obs_mask], obs_arr[obs_mask],
                               color=color, marker=marker, s=60, zorder=5,
                               label=f"{pool_name.replace('soil_','')} obs")
    ax_pool14C.set_xlabel("Year"); ax_pool14C.set_ylabel("Δ¹⁴C (‰)")
    ax_pool14C.set_title("(c) Pool Δ¹⁴C  (prior– OE2··· OE4-·- OE5—)")
    ax_pool14C.legend(fontsize=8, ncol=2)

    # ── (d) Respired CO₂ Δ¹⁴C ───────────────────────────────────────────────
    resp_arr  = np.array(d14C_resp)
    resp_mask = ~np.isnan(resp_arr)
    ax_resp14C.plot(time_years, r["d14C_resp_prior"], color=C_PRIOR, lw=1.0,
                    linestyle="--", label="prior", alpha=0.7)
    for d14c_r, col, lw, ls, lab in [
        (r["d14C_resp_oe1"], C_OE1, 1.0, ":",           "OE1: C stocks only"),
        (r["d14C_resp_oe2"], C_OE2, 1.2, "-.",           "OE2: +pool Δ¹⁴C"),
        (r["d14C_resp_oe3"], C_OE3, 1.4, (0, (3, 1)),   "OE3: +resp Δ¹⁴C"),
        (r["d14C_resp_oe4"], C_OE4, 1.6, "-.",           "OE4: full"),
        (r["d14C_resp_oe5"], C_OE5, 2.0, "-",            "OE5: full + ER flux"),
    ]:
        ax_resp14C.plot(time_years, d14c_r, color=col, lw=lw, linestyle=ls,
                        label=lab, alpha=0.85 if lw < 2 else 1.0)
    ax_resp14C.scatter(time_years[resp_mask], resp_arr[resp_mask],
                       color="k", marker="x", s=30, zorder=5, label=resp_obs_label)
    ax_resp14C.set_xlabel("Year"); ax_resp14C.set_ylabel("Δ¹⁴C (‰)")
    ax_resp14C.set_title("(d) Respired CO₂ Δ¹⁴C")
    ax_resp14C.legend(fontsize=8)

    # ── (e) Carbon stocks — bar chart per OE run ─────────────────────────────
    bar_labels = ["OE1\nC stocks", "OE2\n+pool Δ¹⁴C", "OE3\n+resp Δ¹⁴C",
                  "OE4\nfull", "OE5\n+ER flux"]
    x_base = np.arange(len(bar_labels))
    bar_w  = 0.18
    n_pools_fig = len(pool_names)
    for j, (pool_name, color, marker) in enumerate(zip(pool_names, pool_colors, pool_markers)):
        if pool_name not in pool_idx.pool_names:
            continue
        pi = pool_idx[pool_name]
        means_sim = [float(np.mean(np.array(out.C12)[:, pi]))
                     for out in [out_oe1, out_oe2, out_oe3, out_oe4, out_oe5]]
        offset = (j - (n_pools_fig - 1) / 2.0) * bar_w
        ax_cstockbar.bar(x_base + offset, means_sim, width=bar_w,
                         color=color, alpha=0.75, label=pool_name.replace("soil_", ""))
        if pool_name in c_pools_obs:
            c_mu, c_sig = c_pools_obs[pool_name]
            ax_cstockbar.axhline(c_mu, color=color, lw=1.5, linestyle="--", alpha=0.9)
            ax_cstockbar.axhspan(c_mu - c_sig, c_mu + c_sig, color=color, alpha=0.12,
                                  label=f"{pool_name.replace('soil_','')} obs ±1σ")
    ax_cstockbar.set_xticks(x_base)
    ax_cstockbar.set_xticklabels(bar_labels, fontsize=8)
    ax_cstockbar.set_ylabel("Mean C12 (gC m⁻²)")
    ax_cstockbar.set_title("(e) Carbon stocks: model vs. observed")
    ax_cstockbar.legend(fontsize=8)

    # ── (f) Age diagnostics ───────────────────────────────────────────────────
    ax_age.plot(time_years, age_diag.bulk_delta14C,     color="tab:blue", lw=1.8,
                label="Stored bulk Δ¹⁴C (mass-weighted)")
    ax_age.plot(time_years, age_diag.respired_delta14C, color="tab:red",  lw=1.8,
                label="Respired Rh Δ¹⁴C (flux-weighted)")
    ax_age.fill_between(time_years, age_diag.respired_delta14C, age_diag.bulk_delta14C,
                         alpha=0.12, color="tab:purple", label="Stored−respired gap")
    if resp_mask.any():
        ax_age.scatter(time_years[resp_mask], resp_arr[resp_mask],
                       color="k", marker="x", s=25, zorder=5, label="resp obs")
    ax_age.set_xlabel("Year"); ax_age.set_ylabel("Δ¹⁴C (‰)")
    ax_age.set_title("(f) Stored vs. respired carbon age")
    ax_age.legend(fontsize=8)
    ax_age.axhline(0, color="0.7", lw=0.8)

    # ── (g) GPP and ER forcing ────────────────────────────────────────────────
    gpp_arr    = np.array(r["forcing_GPP"])
    yrs_unique = np.arange(int(time_years[0]), int(time_years[-1]) + 1)
    gpp_annual, er_annual = [], []
    for yr in yrs_unique:
        mask = (time_years >= yr) & (time_years < yr + 1)
        gpp_annual.append(float(np.nanmean(gpp_arr[mask])) if mask.sum() > 10 else np.nan)
        er_yr = er_arr_fig[mask]; er_yr = er_yr[np.isfinite(er_yr)]
        er_annual.append(float(np.mean(er_yr)) if len(er_yr) >= 10 else np.nan)
    gpp_annual = np.array(gpp_annual)
    er_annual  = np.array(er_annual)
    ax_gpp.plot(time_years, gpp_arr, color="0.75", lw=0.5, alpha=0.5, label="daily GPP")
    ax_gpp.plot(yrs_unique + 0.5, gpp_annual, color="tab:green", lw=2.0, label="GPP annual")
    ax_gpp.plot(yrs_unique + 0.5, er_annual,  color="tab:red",   lw=2.0,
                linestyle="--", label="ER annual")
    ax_gpp.plot(yrs_unique + 0.5, er_annual * rh_er_frac, color="saddlebrown", lw=1.5,
                linestyle=":", label=f"Rh est. (ER×{rh_er_frac})")
    ax_gpp.set_xlabel("Year"); ax_gpp.set_ylabel("gC m⁻² day⁻¹")
    ax_gpp.set_title("(g) GPP and ER forcing")
    ax_gpp.legend(fontsize=7)
    ax_gpp.set_xlim(time_years[0], time_years[-1])

    # ── (h) C stock time series — prior vs OE5 ───────────────────────────────
    for i, (pool_name, color, marker) in enumerate(zip(pool_names, pool_colors, pool_markers)):
        if pool_name not in pool_idx.pool_names:
            continue
        pi = pool_idx[pool_name]
        ax_cstock.plot(time_years, np.array(out_prior.C12)[:, pi],
                       color=color, lw=1.0, linestyle="--", alpha=0.5)
        ax_cstock.plot(time_years, np.array(out_oe5.C12)[:, pi],
                       color=color, lw=1.8, label=pool_name.replace("soil_", ""))
        if pool_name in c_pools_obs:
            c_mu, c_sig = c_pools_obs[pool_name]
            ax_cstock.axhline(c_mu, color=color, lw=1.5, linestyle=":", alpha=0.9)
            ax_cstock.axhspan(c_mu - c_sig, c_mu + c_sig, color=color, alpha=0.12,
                              label=f"{pool_name.replace('soil_','')} obs ±σ")
    ax_cstock.set_xlabel("Year"); ax_cstock.set_ylabel("C stock (gC m⁻²)")
    ax_cstock.set_title("(h) C stocks: prior (--) vs OE5 (—), obs bands (·)")
    ax_cstock.legend(fontsize=8, ncol=2)
    ax_cstock.set_xlim(time_years[0], time_years[-1])

    fig.suptitle(f"{site_title}\n{site_subtitle}", fontsize=12, y=0.995)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.close(fig)
