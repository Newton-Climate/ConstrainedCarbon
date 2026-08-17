"""The shared canonical OE inversion driver.

``run_oe_canonical`` is THE single ``optimize_oe`` call behind every canonical
site (Harvard Forest, Barrow, Eight Mile Lake, Howland). It previously lived in
``notebooks/sites/canonical.py`` as ``_run_oe_canonical``, where it was
importable only through a ``sys.path`` hack.

Both the prior and the MAP are run from their *own* analytical steady state —
the exact operating points ``optimize_oe`` costs against — so the returned
diagnostics are self-consistent regardless of site.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ecosystem_complexity.api import build_model, optimize_oe, run_model
from ecosystem_complexity.data.custom_14c import (
    build_custom_14c_observations,
    load_custom_14c_manifest,
)
from ecosystem_complexity.data.israd_14c import (
    _bulk_pool_ic_seeds,
    build_bulk_14C_blocks,
    build_fraction_14C_blocks,
    build_incubation_14C_blocks,
    build_resp_14C_obs,
)
from ecosystem_complexity.data.israd_12c import build_fraction_12C_blocks
from ecosystem_complexity.data.israd_incubation import build_incubation_rate_blocks
from ecosystem_complexity.data.parsers import attach_atm14C
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.paths import (
    GRAVEN_PATH,
    HUA_PATH,
    INTCAL_PATH,
)
from ecosystem_complexity.data.paths import (
    REPO_ROOT as _REPO_ROOT,
)
from ecosystem_complexity.data.schemas import ObservationData
from ecosystem_complexity.data.soc_stocks import (
    build_measured_soc_total,
    build_soilgrids_soc_total,
)
from ecosystem_complexity.oe_utils import ss_state_for_params
from ecosystem_complexity.sites.forcing import (
    load_site_observations,
    load_site_forcing,
    resolve_forcing_file,
)
from ecosystem_complexity.sites.soc import build_soc_prior
from ecosystem_complexity.sites.spec import SiteSpec
from ecosystem_complexity.state import make_default_params

logger = logging.getLogger(__name__)


def run_oe_canonical(
    model, forcing, state0, obs_full,
    extra_blocks: list,
    opt_fields: tuple,
    site_label: str,
) -> dict:
    """Single canonical ``optimize_oe`` call with a structured return.

    Returns the prior and MAP parameter sets, the forward runs at each, the
    steady states they were run from, and the raw ``OEResult``.
    """
    params_prior = make_default_params(model.config)
    logger.info("[%s] prior forward simulation…", site_label)
    # Run the prior from its own analytical steady state — the same operating
    # point optimize_oe costs the prior against (y_prior = _forward(xa)). Using
    # the observed-stock state0 here would produce a "prior" that is not at
    # steady state and does not match the prior the inversion actually sees.
    state_at_prior = ss_state_for_params(model, forcing, state0, params_prior)
    out_prior = run_model(model, forcing, state0=state_at_prior, params=params_prior)
    jax.block_until_ready(out_prior.delta14C)

    logger.info(
        "[%s] optimize_oe fields=%s extras=%s",
        site_label, opt_fields, [b.name for b in extra_blocks],
    )
    t0 = time.perf_counter()
    result = optimize_oe(
        model, forcing, obs_full, state0=state0,
        fields=opt_fields, extra_obs_blocks=extra_blocks,
    )
    ch = np.array(result.cost_history)
    logger.info(
        "  Done [%.1fs]  J %.2f → %.2f  (%d iter, converged=%s)",
        time.perf_counter() - t0, ch[0], ch[-1], result.n_iter, result.converged,
    )

    params_opt = result.params_opt
    state_at_map = ss_state_for_params(model, forcing, state0, params_opt)
    out_opt = run_model(model, forcing, state0=state_at_map, params=params_opt)
    jax.block_until_ready(out_opt.delta14C)

    return {
        "params_prior": params_prior, "params_opt": params_opt,
        "out_prior": out_prior, "out_opt": out_opt,
        "state_at_prior": state_at_prior, "state_at_map": state_at_map,
        "oe_result": result,
    }


# ════════════════════════════════════════════════════════════════════════════
# Config-driven multi-site driver
# ════════════════════════════════════════════════════════════════════════════

_R_STD = 1.176e-12
OPT_FIELDS = ("log_tau", "log_f_transfer")
# Fallback initial Δ¹⁴C per pool role (‰) when a pool has no layer obs.
_FALLBACK_D14C_BY_ROLE = {"active": 120.0, "slow": 60.0, "passive": 0.0}

def build_state0(model, state_ss, pool_blocks, ic_seeds=None):
    """Use the site SOC prior state and seed ¹⁴C from the first available obs.

    Per-pool ¹⁴C seeding priority: (1) a fraction block that names the pool
    (``israd_fraction_soil_slow`` → ``soil_slow``); (2) ``ic_seeds`` — profile-
    derived seeds for whole-sample bulk sites, whose block names carry no pool
    (see ``_bulk_pool_ic_seeds``); (3) a modern fallback keyed on the pool's
    kinetic *role* rather than its full name, so a config whose layer is not
    called "soil" still gets the intended seed instead of silently falling
    through to 0‰. Bulk-only sites previously skipped straight to (3),
    initialising even the passive pool at a modern value — wrong for
    aged/permafrost carbon.
    """
    ic_seeds = ic_seeds or {}
    # first observed Δ¹⁴C per pool (blocks are ordered by year)
    first_d14c: dict[str, float] = {}
    for b in pool_blocks:
        for pool in model.pool_index.pool_names:
            if pool in b.name and pool not in first_d14c:
                first_d14c[pool] = float(np.array(b.y)[0])
    c12 = np.array(state_ss.C12, dtype=float)
    c14 = np.zeros_like(c12)
    for name in model.pool_index.pool_names:
        i = model.pool_index[name]
        role = name.rsplit("_", 1)[-1]
        d14 = first_d14c.get(
            name, ic_seeds.get(name, _FALLBACK_D14C_BY_ROLE.get(role, 0.0))
        )
        c14[i] = c12[i] * _R_STD * (1.0 + d14 / 1000.0)
    return state_ss._replace(C14=jnp.array(c14, dtype=jnp.float32))


def run_site_canonical(
    spec: SiteSpec,
    observation_path: str = "bulk_resp",
    include_er_constraint: bool = False,
    include_incubation_constraint: bool = False,
    include_incubation_14c_constraint: bool = False,
    incubation_duration_types: frozenset[str] | None = None,
    include_fraction_12c_constraint: bool | None = None,
) -> dict:
    if observation_path not in {"bulk_resp", "fraction", "combined"}:
        raise ValueError(
            "observation_path must be 'bulk_resp', 'fraction', or 'combined'"
        )
    logger.info(
        "\n══ %s — %s OE inversion ══════════════════════",
        spec.label, observation_path,
    )
    config_path = spec.config_path
    model = build_model(config_path)
    idx = model.pool_index
    logger.info("%s", f"  Config: {os.path.relpath(config_path, _REPO_ROOT)}")

    forcing_path = resolve_forcing_file(spec)
    logger.info("%s", f"  Forcing: {os.path.relpath(forcing_path, _REPO_ROOT)}")
    forcing = load_site_forcing(spec, forcing_path, model)
    tower_obs = (
        load_site_observations(spec, forcing_path, model, forcing=forcing)
        if include_er_constraint
        else None
    )
    mean_gpp = float(np.nanmean(np.array(forcing.GPP_obs)))

    hemisphere = "NH" if spec.lat >= 0 else "SH"
    years_daily, d14c_daily = load_full_14C_record(
        hua_path=HUA_PATH, graven_path=GRAVEN_PATH, intcal_path=INTCAL_PATH,
        hemisphere=hemisphere, start_year=1500.0, end_year=2025.0,
    )
    forcing = attach_atm14C(forcing, d14c_daily, years_daily)
    time_years = 1970.0 + np.array(forcing.time) / 365.25
    logger.info(
        "  Record: %d days (%.1f–%.1f)  mean GPP %.0f gC m⁻² yr⁻¹  [%s]",
        len(time_years), time_years[0], time_years[-1], mean_gpp * 365, hemisphere,
    )

    soc_prior_state, _c_pools_prior, ss_years, c_total_obs = build_soc_prior(
        model, forcing
    )
    # Co-located kinetic pools ⇒ per-pool stock is unobservable; the stock
    # constraint is the column total Σ_i C12_i only (ObservationData.C_total_obs).
    soc_source = "model steady state (self-referential)"
    manifest_path = (
        Path(config_path).parent / spec.radiocarbon_manifest
        if spec.radiocarbon_manifest
        else None
    )
    if manifest_path is not None:
        custom_data = load_custom_14c_manifest(manifest_path)
        pool_blocks, resp = build_custom_14c_observations(
            custom_data, forcing.time, idx
        )
        block_label = "custom"
        logger.info("  Custom ¹⁴C manifest: %s", custom_data.manifest_path)
    elif observation_path == "fraction":
        pool_blocks = build_fraction_14C_blocks(
            spec.israd_name, forcing.time, model, spec.fraction_rules
        )
        resp = jnp.full(forcing.time.shape[0], jnp.nan, dtype=jnp.float32)
        block_label = "fraction"
    elif observation_path == "combined":
        pool_blocks = (
            build_fraction_14C_blocks(
                spec.israd_name, forcing.time, model, spec.fraction_rules
            )
            + build_bulk_14C_blocks(spec.israd_name, forcing.time, model)
        )
        resp = build_resp_14C_obs(spec.israd_name, forcing.time)
        # Stock-source priority: a direct ISRaD measurement at the site beats a
        # SoilGrids prediction, which in turn beats the model's own steady state
        # (which is self-referential and carries no independent information — at
        # σ=0.50 it is a deliberate no-op).
        measured = build_measured_soc_total(spec.israd_name, model)
        soilgrids = build_soilgrids_soc_total(spec.israd_name, model)
        if measured is not None:
            mean_soc, sigma_soc, depth_cov = measured
            c_total_obs = (mean_soc, sigma_soc)
            soc_source = f"ISRaD MEASURED total ({depth_cov:.0%} of column)"
        elif soilgrids is not None:
            mean_soc, sigma_soc, depth_cov = soilgrids
            c_total_obs = (mean_soc, sigma_soc)
            soc_source = f"SoilGrids total ({depth_cov:.0%} of column)"
        block_label = "fraction+bulk"
    else:
        pool_blocks = build_bulk_14C_blocks(spec.israd_name, forcing.time, model)
        resp = build_resp_14C_obs(spec.israd_name, forcing.time)
        block_label = "bulk"
    n_resp = int(jnp.sum(~jnp.isnan(resp)))
    soc_total = c_total_obs[0] if c_total_obs else 0.0
    logger.info(
        "  Obs: %d %s Δ¹⁴C blocks, %d respiration Δ¹⁴C, 1 total-SOC constraint "
        "(Σ=%.0f±%.0f gC m⁻² — %s; %s annual-mean years)",
        len(pool_blocks), block_label, n_resp,
        soc_total, c_total_obs[1], soc_source, ss_years,
    )
    incubation_rows = (
        build_incubation_rate_blocks(
            spec.israd_name,
            model,
            duration_types=incubation_duration_types,
        )
        if include_incubation_constraint
        else []
    )
    incubation_blocks = [row["block"] for row in incubation_rows]
    n_incubation = len(incubation_blocks)
    if include_incubation_constraint:
        duration_label = (
            ", ".join(sorted(incubation_duration_types))
            if incubation_duration_types
            else "all duration classes"
        )
        logger.info(
            "  ISRaD incubation constraint: %d block(s) [%s]",
            n_incubation,
            duration_label,
        )
        mixed = [
            row for row in incubation_rows
            if len(row.get("duration_mix", {})) > 1
        ]
        if mixed:
            logger.warning(
                "  Incubation blocks for %s pool multiple duration classes; "
                "pass incubation_duration_types to avoid mixing protocol biases.",
                spec.label,
            )
    incubation_14c_blocks = (
        build_incubation_14C_blocks(spec.israd_name, forcing.time)
        if include_incubation_14c_constraint
        else []
    )
    if include_incubation_14c_constraint:
        logger.info("  ISRaD incubation Δ¹⁴C constraint: %d dated block(s) [σ≥20‰]", len(incubation_14c_blocks))
    # Density-fraction ¹²C partition (and optional per-pool stock) blocks.
    # Default on for `fraction`/`combined` paths, where fraction data is already
    # loaded for ¹⁴C; off for `bulk_resp`, which deliberately avoids the
    # fraction table. Explicit True/False from the caller overrides.
    if include_fraction_12c_constraint is None:
        include_fraction_12c_constraint = (
            manifest_path is None and observation_path in {"fraction", "combined"}
        )
    fraction_12c_blocks = (
        build_fraction_12C_blocks(spec.israd_name, model, spec.fraction_rules)
        if include_fraction_12c_constraint
        else []
    )
    if (
        not pool_blocks
        or (
            observation_path == "bulk_resp"
            and n_resp == 0
            and n_incubation == 0
            and len(incubation_14c_blocks) == 0
        )
    ):
        logger.info("%s", "  SKIP — insufficient radiocarbon/incubation obs.")
        return {"spec": spec, "skipped": True}

    T = int(forcing.time.shape[0])
    er_obs = (
        tower_obs.ER
        if (tower_obs is not None and tower_obs.ER is not None)
        else jnp.full(T, jnp.nan)
    )
    n_er_finite = int(jnp.sum(jnp.isfinite(er_obs)))
    if include_er_constraint:
        logger.info("  Tower ER constraint: %d finite daily ER values", n_er_finite)
    obs_full = ObservationData(
        time=forcing.time,
        NEE=jnp.full(T, jnp.nan), GPP=jnp.full(T, jnp.nan),
        ER=er_obs, NEE_unc=jnp.full(T, jnp.nan),
        delta14C_obs={}, deltaD14C_obs={}, C_pools_obs={}, delta14C_resp=resp,
        C_total_obs=c_total_obs,
    )

    # Whole-sample bulk sites carry no per-pool ¹⁴C split in their block names,
    # so seed the pool ICs from the observed layer profile (aged carbon → passive)
    # instead of the modern _FALLBACK_D14C. Fraction blocks still win by name.
    ic_seeds = {} if manifest_path is not None else _bulk_pool_ic_seeds(spec.israd_name)
    state0 = build_state0(model, soc_prior_state, pool_blocks, ic_seeds=ic_seeds)

    t0 = time.perf_counter()
    result = optimize_oe(
        model, forcing, obs_full, state0=state0,
        fields=OPT_FIELDS,
        extra_obs_blocks=pool_blocks + incubation_blocks + incubation_14c_blocks + fraction_12c_blocks,
    )
    ch = np.array(result.cost_history)
    logger.info(
        "  optimize_oe done [%.1fs]  J %.1f→%.1f  (%d iter, converged=%s)",
        time.perf_counter() - t0, ch[0], ch[-1], result.n_iter, result.converged,
    )

    tau_days = np.exp(np.array(result.params_opt.log_tau))
    logger.info("%s", "  optimised τ (yr): " + ", ".join(
        f"{n}={t/365.25:.1f}" for n, t in zip(idx.pool_names, tau_days)))

    return {
        "spec": spec, "skipped": False, "model": model,
        "observation_path": observation_path,
        "include_er_constraint": include_er_constraint,
        "config_path": config_path,
        "mean_gpp_gCm2yr": mean_gpp * 365.0,
        "soc_total_gCm2": soc_total,
        "n_cstock": 1 if c_total_obs else 0,
        "soc_source": soc_source,
        "n_pool_blocks": len(pool_blocks), "n_resp": n_resp,
        "n_er_finite": n_er_finite,
        "include_incubation_constraint": include_incubation_constraint,
        "n_incubation": n_incubation,
        "n_incubation_14c": len(incubation_14c_blocks),
        "n_fraction_12c": len(fraction_12c_blocks),
        "tau_years": {n: float(t / 365.25) for n, t in zip(idx.pool_names, tau_days)},
        "cost0": float(ch[0]), "cost_final": float(ch[-1]),
        "converged": bool(result.converged), "n_iter": int(result.n_iter),
        "oe_result": result,
        # pieces needed for downstream information diagnostics (constraint ladder)
        "forcing": forcing, "state0": state0,
        "obs_full": obs_full,
        "pool_blocks": pool_blocks + incubation_blocks + incubation_14c_blocks + fraction_12c_blocks,
        "params_opt": result.params_opt,
    }


def summary_row(result: dict) -> dict:
    """Flatten one ``run_site_canonical`` result into a summary-table row.

    Per-pool τ columns are emitted for whichever pools the site's config
    defines, so a non-3-pool config still summarises instead of raising a
    KeyError on the canonical active/slow/passive names.
    """
    spec = result["spec"]
    row = {
        "site": spec.israd_name, "label": spec.label,
        "tower_id": spec.tower_id, "biome": spec.biome,
        "mean_GPP_gCm2yr": round(result["mean_gpp_gCm2yr"]),
        "SOC_gCm2": round(result["soc_total_gCm2"]),
        "n_cstock": result["n_cstock"],
        "n_pool_blocks": result["n_pool_blocks"],
        "n_resp": result["n_resp"],
        "n_incubation": result["n_incubation"],
        "n_incubation_14c": result["n_incubation_14c"],
    }
    for pool, tau in result["tau_years"].items():
        # soil_active → tau_active_yr; passive is slow-moving so keep 1 decimal.
        short = pool.replace("soil_", "")
        row[f"tau_{short}_yr"] = round(tau, 1 if tau >= 100 else 2)
    row.update({
        "J0": round(result["cost0"], 1),
        "J_final": round(result["cost_final"], 1),
        "converged": result["converged"], "n_iter": result["n_iter"],
    })
    return row


def _run_one(
    spec: SiteSpec,
    observation_path: str | None,
    include_er_constraint: bool = False,
    include_incubation_constraint: bool = False,
    include_incubation_14c_constraint: bool = False,
    incubation_duration_types: frozenset[str] | None = None,
    include_fraction_12c_constraint: bool | None = None,
    reduce: Callable[[dict], Any] | None = None,
) -> tuple[SiteSpec, Any, Exception | None]:
    """Run one site, capturing rather than raising, for use by both schedulers.

    ``reduce`` is applied *inside* the worker. That is not an optimisation: the
    raw result dict holds the built model, whose ``_compiled_step`` is a closure
    local to ``EcosystemModel.__post_init__`` and therefore unpicklable, so a
    parallel worker cannot return it. Reducing before the process boundary is
    what makes the result transferable at all.
    """
    path = observation_path or spec.observation_path
    try:
        result = run_site_canonical(
            spec,
            observation_path=path,
            include_er_constraint=include_er_constraint,
            include_incubation_constraint=include_incubation_constraint,
            include_incubation_14c_constraint=include_incubation_14c_constraint,
            incubation_duration_types=incubation_duration_types,
            include_fraction_12c_constraint=include_fraction_12c_constraint,
        )
    except Exception as exc:  # noqa: BLE001 — one bad site must not stop the rest
        return spec, None, exc
    if result.get("skipped"):
        return spec, None, None
    return spec, (reduce(result) if reduce is not None else result), None


def run_sites(
    specs: list[SiteSpec],
    observation_path: str | None = None,
    include_er_constraint: bool = False,
    include_incubation_constraint: bool = False,
    include_incubation_14c_constraint: bool = False,
    incubation_duration_types: frozenset[str] | None = None,
    include_fraction_12c_constraint: bool | None = None,
    workers: int = 1,
    reduce: Callable[[dict], Any] | None = None,
) -> tuple[list[Any], list[tuple[SiteSpec, Exception]]]:
    """Run the canonical inversion over several sites, isolating failures.

    Returns ``(results, failures)``. One site blowing up (missing forcing file,
    empty ISRaD selection) must not abandon the remaining sites, so exceptions
    are captured per site and returned for the caller to report and to set an
    exit status from — the previous ``main`` printed them and still exited 0.

    ``observation_path`` overrides each spec's configured path when given.

    ``workers`` > 1 runs sites concurrently in separate *processes*. Processes
    rather than threads because the work is CPU-bound inside JAX/XLA, and
    because each site builds its own model and JAX state — sharing one
    interpreter would contend on the GIL for the Python-level driver work and
    have the per-site XLA compilations fight over the same default device.
    Results are collected as they finish and then re-ordered to match ``specs``,
    so the summary table does not reshuffle with scheduling.

    ``reduce`` maps each result dict to the value collected, and is **required**
    when ``workers`` > 1: the raw result holds the compiled model, which cannot
    be pickled back from a worker (see ``_run_one``). Pass ``summary_row`` for
    the summary table, or a callable extracting whatever the caller needs.
    """
    if workers > 1 and reduce is None:
        raise ValueError(
            "run_sites(workers>1) needs a `reduce` callable: the full result "
            "holds the compiled model and cannot cross a process boundary. "
            "Pass reduce=summary_row, or run with workers=1."
        )

    results_by_stem: dict[str, Any] = {}
    failures: list[tuple[SiteSpec, Exception]] = []

    if workers > 1 and len(specs) > 1:
        # `spawn` keeps each worker's JAX/XLA initialisation independent; forking
        # a process that has already initialised a JAX backend is unsafe.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(
                    _run_one,
                    spec,
                    observation_path,
                    include_er_constraint,
                    include_incubation_constraint,
                    include_incubation_14c_constraint,
                    incubation_duration_types,
                    include_fraction_12c_constraint,
                    reduce,
                ): spec
                for spec in specs
            }
            for fut in as_completed(futures):
                spec = futures[fut]
                try:
                    spec, result, exc = fut.result()
                except Exception as exc:  # noqa: BLE001 — worker died outright
                    logger.error("ERROR at %s: %s", spec.label, exc)
                    failures.append((spec, exc))
                    continue
                if exc is not None:
                    logger.error("ERROR at %s: %s", spec.label, exc)
                    failures.append((spec, exc))
                elif result is not None:
                    results_by_stem[spec.config_stem] = result
    else:
        for spec in specs:
            spec, result, exc = _run_one(
                spec,
                observation_path,
                include_er_constraint,
                include_incubation_constraint,
                include_incubation_14c_constraint,
                incubation_duration_types,
                include_fraction_12c_constraint,
                reduce,
            )
            if exc is not None:
                logger.error("ERROR at %s: %s", spec.label, exc)
                failures.append((spec, exc))
            elif result is not None:
                results_by_stem[spec.config_stem] = result

    results = [
        results_by_stem[s.config_stem]
        for s in specs
        if s.config_stem in results_by_stem
    ]
    return results, failures
