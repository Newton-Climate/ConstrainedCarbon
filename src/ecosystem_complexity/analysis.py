"""
Scientific analysis workflows for radiocarbon-constrained soil carbon inversion.

Builds on information.py to provide:
  1. Observation ablation — quantifies the marginal information value of each
     observation type by comparing DFS with and without each type.
  2. Parameter-group DFS — reports which model parameter groups are constrained.
  3. Model-complexity ladder — compares information content across model structures
     (e.g. 3-pool vs. 6-pool vs. full-transfer).
  4. Storage-age vs. respiration-age diagnostics — decomposes model output into
     stored-carbon age structure and actively-respired carbon age.

Scientific context
------------------
The key hypothesis is that pool Δ¹⁴C constrains *stored* carbon age structure
(turnover times and transfer fractions) while respired CO₂ Δ¹⁴C constrains the
*active* decomposition source mixture. This module provides tools to test that
hypothesis quantitatively using the Fisher/DFS framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from .config import ModelConfig, PoolIndex
from .model import EcosystemModel
from .state import ModelParams, make_default_params
from .api import run_model, ModelOutput
from .data.schemas import ForcingData, ObservationData
from .information import (
    FisherResult,
    DofResult,
    PosteriorResult,
    OBS_C_STOCKS,
    OBS_POOL_D14C,
    OBS_RESP_D14C,
    ALL_OBS_TYPES,
    PARAM_GROUP_TAU,
    PARAM_GROUP_PARTITION,
    PARAM_GROUP_TRANSFER,
    PARAM_GROUP_ENV,
    _DEFAULT_PRIOR_SIGMA,
    _DEFAULT_C_SIGMA_REL,
    _DEFAULT_D14C_SIGMA,
    _DEFAULT_RESP_SIGMA,
    analyze_information_content,
    compute_fisher,
    compute_dof,
    compute_posterior,
    get_param_groups,
    make_prior_covariance,
    _default_fields,
)


# ── Ablation analysis ──────────────────────────────────────────────────────────


@dataclass
class AblationResult:
    """Result for a single observation-subset scenario.

    Attributes
    ----------
    scenario : str
        Human-readable label (e.g. "C_stocks + pool_delta14C").
    obs_types : list[str]
        Active observation types in this scenario.
    fisher : FisherResult
    dof : DofResult
    posterior : PosteriorResult
    dfs_total : float
        Total DFS for this scenario (convenience access).
    """

    scenario: str
    obs_types: list[str]
    fisher: FisherResult
    dof: DofResult
    posterior: PosteriorResult
    dfs_total: float


def run_ablation_study(
    model: EcosystemModel,
    forcing: ForcingData,
    state0,
    params: ModelParams,
    observations: ObservationData,
    fields: Sequence[str] | None = None,
    prior_sigma: np.ndarray | None = None,
    default_prior_sigma: float = _DEFAULT_PRIOR_SIGMA,
    obs_sigma_C: float = _DEFAULT_C_SIGMA_REL,
    obs_sigma_d14C: float = _DEFAULT_D14C_SIGMA,
    obs_sigma_resp: float = _DEFAULT_RESP_SIGMA,
) -> dict[str, AblationResult]:
    """Run observation ablation study across five canonical scenarios.

    Scenarios
    ---------
    1. C_stocks                                — soil C stocks only
    2. pool_delta14C                           — pool Δ¹⁴C only
    3. resp_delta14C                           — respired CO₂ Δ¹⁴C only
    4. C_stocks + pool_delta14C
    5. C_stocks + pool_delta14C + resp_delta14C — full dataset

    Returns
    -------
    dict[scenario_key, AblationResult]
        Ordered from fewest to most observations.
    """
    if fields is None:
        fields = _default_fields(model)

    if prior_sigma is None:
        prior_sigma = make_prior_covariance(
            params, fields, model, default_sigma=default_prior_sigma
        )

    param_groups = get_param_groups(params, fields, model)

    # Define scenarios in logical order
    scenarios: list[tuple[str, list[str]]] = [
        (OBS_C_STOCKS, [OBS_C_STOCKS]),
        (OBS_POOL_D14C, [OBS_POOL_D14C]),
        (OBS_RESP_D14C, [OBS_RESP_D14C]),
        (f"{OBS_C_STOCKS}+{OBS_POOL_D14C}", [OBS_C_STOCKS, OBS_POOL_D14C]),
        (
            f"{OBS_C_STOCKS}+{OBS_POOL_D14C}+{OBS_RESP_D14C}",
            [OBS_C_STOCKS, OBS_POOL_D14C, OBS_RESP_D14C],
        ),
    ]

    results: dict[str, AblationResult] = {}
    for key, obs_types in scenarios:
        fisher = compute_fisher(
            model, forcing, state0, params, observations,
            fields=fields,
            obs_sigma_C=obs_sigma_C,
            obs_sigma_d14C=obs_sigma_d14C,
            obs_sigma_resp=obs_sigma_resp,
            active_obs_types=obs_types,
        )
        dof = compute_dof(fisher, prior_sigma, param_groups=param_groups)
        posterior = compute_posterior(fisher, prior_sigma)
        label = " + ".join(obs_types)
        results[key] = AblationResult(
            scenario=label,
            obs_types=obs_types,
            fisher=fisher,
            dof=dof,
            posterior=posterior,
            dfs_total=dof.dfs_total,
        )

    return results


def summarize_ablation(
    ablation: dict[str, AblationResult],
) -> dict[str, dict]:
    """Return a compact comparison table from an ablation study.

    Returns
    -------
    dict
        Keys are scenario names.  Values are dicts with::

            {
                "dfs_total": float,
                "dfs_per_obs_type": dict[str, float],
                "dfs_by_group": dict[str, float] or None,
                "uncertainty_reduction_mean": float,
            }
    """
    table: dict[str, dict] = {}
    for key, res in ablation.items():
        ur = res.posterior.uncertainty_reduction
        table[key] = {
            "dfs_total": res.dfs_total,
            "dfs_per_obs_type": res.dof.dfs_per_obs_type,
            "dfs_by_group": res.dof.dfs_by_group,
            "uncertainty_reduction_mean": float(np.nanmean(ur)),
        }
    return table


# ── Parameter-group DFS ────────────────────────────────────────────────────────


def param_group_dfs(
    dof: DofResult,
) -> dict[str, float]:
    """Extract per-parameter-group DFS from a DofResult.

    If ``dof.dfs_by_group`` is already populated, returns it directly.
    Otherwise computes it from ``dfs_per_param`` and ``param_groups``.

    Returns
    -------
    dict[group_name, dfs]
    """
    if dof.dfs_by_group is not None:
        return dof.dfs_by_group
    if dof.param_groups is None:
        return {}
    return {
        grp: float(np.sum(dof.dfs_per_param[idxs]))
        for grp, idxs in dof.param_groups.items()
    }


# ── Complexity ladder ──────────────────────────────────────────────────────────


@dataclass
class ComplexityRung:
    """Result for a single model configuration on the complexity ladder.

    Attributes
    ----------
    label : str
        Human-readable model description.
    config_path : str or None
    n_params : int
        Dimension of the parameter vector.
    n_obs : int
        Number of scalar observation summaries.
    fisher : FisherResult
    dof : DofResult
    posterior : PosteriorResult
    dfs_total : float
    effective_params : int
        Parameters that are at least half-constrained: sum(dfs_per_param > 0.5).
    """

    label: str
    config_path: str | None
    n_params: int
    n_obs: int
    fisher: FisherResult
    dof: DofResult
    posterior: PosteriorResult
    dfs_total: float
    effective_params: int


def run_complexity_ladder(
    ladder: list[tuple[EcosystemModel, str]],
    forcing: ForcingData,
    state0_by_model: dict[str, object],
    params_by_model: dict[str, ModelParams],
    observations: ObservationData,
    fields_by_model: dict[str, Sequence[str]] | None = None,
    default_prior_sigma: float = _DEFAULT_PRIOR_SIGMA,
    obs_sigma_C: float = _DEFAULT_C_SIGMA_REL,
    obs_sigma_d14C: float = _DEFAULT_D14C_SIGMA,
    obs_sigma_resp: float = _DEFAULT_RESP_SIGMA,
) -> list[ComplexityRung]:
    """Compare information content across model configurations.

    Parameters
    ----------
    ladder : list of (EcosystemModel, label)
        Models to compare in order from simplest to most complex.
    state0_by_model : dict[label, EcosystemState]
        Initial state for each model.
    params_by_model : dict[label, ModelParams]
        Parameters (e.g. MAP estimate) for each model.
    fields_by_model : dict[label, fields], optional
        Parameter fields to include for each model. Defaults to
        ``_default_fields(model)`` per model.

    Returns
    -------
    list[ComplexityRung]
        One entry per model, in ladder order.
    """
    rungs: list[ComplexityRung] = []

    for model, label in ladder:
        state0 = state0_by_model[label]
        params = params_by_model[label]
        fields = (fields_by_model or {}).get(label, _default_fields(model))

        prior_sigma = make_prior_covariance(
            params, fields, model, default_sigma=default_prior_sigma
        )
        param_groups = get_param_groups(params, fields, model)

        fisher = compute_fisher(
            model, forcing, state0, params, observations,
            fields=fields,
            obs_sigma_C=obs_sigma_C,
            obs_sigma_d14C=obs_sigma_d14C,
            obs_sigma_resp=obs_sigma_resp,
        )
        dof = compute_dof(fisher, prior_sigma, param_groups=param_groups)
        posterior = compute_posterior(fisher, prior_sigma)

        n_params = fisher.FIM_total.shape[0]
        n_obs = sum(fisher.n_obs_per_type.values())
        effective = int(np.sum(dof.dfs_per_param > 0.5))

        rungs.append(
            ComplexityRung(
                label=label,
                config_path=getattr(model, "_config_path", None),
                n_params=n_params,
                n_obs=n_obs,
                fisher=fisher,
                dof=dof,
                posterior=posterior,
                dfs_total=dof.dfs_total,
                effective_params=effective,
            )
        )

    return rungs


# ── Age diagnostics ────────────────────────────────────────────────────────────


@dataclass
class AgeDiagnostics:
    """Age-structure diagnostics derived from model output.

    All arrays are time series of length T (one value per timestep).

    Attributes
    ----------
    stored_delta14C : (T, n_pools) ndarray
        Δ¹⁴C of each pool (stored carbon age structure).
    bulk_delta14C : (T,) ndarray
        Mass-weighted mean Δ¹⁴C across all soil pools.
    respired_delta14C : (T,) ndarray
        Flux-weighted Δ¹⁴C of heterotrophic respiration (Rh-weighted mean).
    reco_delta14C : (T,) ndarray or None
        Δ¹⁴C of total ecosystem respiration (Ra + Rh weighted).  None if Ra is
        unavailable or identically zero.
    pool_names : list[str]
    """

    stored_delta14C: np.ndarray
    bulk_delta14C: np.ndarray
    respired_delta14C: np.ndarray
    reco_delta14C: np.ndarray | None
    pool_names: list[str]


def compute_age_diagnostics(
    output: ModelOutput,
    params: ModelParams,
    model: EcosystemModel,
) -> AgeDiagnostics:
    """Compute age-related diagnostics from a model forward run.

    Separates *stored* carbon age structure (pool-level Δ¹⁴C) from
    *respired* carbon age (flux-weighted mean Δ¹⁴C of Rh and Reco).

    Parameters
    ----------
    output : ModelOutput
        From ``run_model``.
    params : ModelParams
        Parameters used for the run (needed for τ values).
    model : EcosystemModel

    Returns
    -------
    AgeDiagnostics
    """
    pool_names = model.pool_index.pool_names
    n_pools = len(pool_names)

    # Pool-level Δ¹⁴C — direct from output: (T, n_pools)
    stored_d14C = np.array(output.delta14C)

    # Bulk (mass-weighted) mean Δ¹⁴C across all soil pools
    C12 = np.array(output.C12)  # (T, n_pools)
    C12_sum = C12.sum(axis=-1, keepdims=True) + 1e-30
    bulk_d14C = (stored_d14C * C12).sum(axis=-1) / C12_sum[:, 0]

    # Flux-weighted respired Δ¹⁴C (Rh-weighted)
    # Rh contribution from pool i ∝ C12_i / τ_i
    tau = np.exp(np.array(params.log_tau))  # (n_pools,)
    rh_weights = C12 / (tau[None, :] + 1e-30)  # (T, n_pools)
    rh_sum = rh_weights.sum(axis=-1) + 1e-30
    resp_d14C = (stored_d14C * rh_weights).sum(axis=-1) / rh_sum

    # Reco Δ¹⁴C (autotrophic + heterotrophic) if Ra is non-trivial
    Ra = np.array(output.Ra)  # (T,)
    Rh = np.array(output.Rh)  # (T,)

    reco_d14C: np.ndarray | None = None
    if np.any(Ra > 1e-10):
        # Autotrophic respiration draws from current-year photosynthate at
        # atmospheric Δ¹⁴C; approximate as zero offset from atmosphere.
        # We don't have per-timestep atm Δ¹⁴C in ModelOutput, so we use
        # the Rh-weighted soil signal scaled by Rh/(Ra+Rh) + 0 * Ra/(Ra+Rh).
        # Users can override by replacing Ra with a proper Ra Δ¹⁴C if available.
        Rtotal = Ra + Rh + 1e-30
        rh_frac = Rh / Rtotal
        # Reco Δ¹⁴C ≈ Rh_frac × Rh_Δ¹⁴C  (Ra Δ¹⁴C ≈ 0 at modern atmosphere)
        reco_d14C = rh_frac * resp_d14C

    return AgeDiagnostics(
        stored_delta14C=stored_d14C,
        bulk_delta14C=bulk_d14C,
        respired_delta14C=resp_d14C,
        reco_delta14C=reco_d14C,
        pool_names=pool_names,
    )


def age_diagnostics_summary(
    diag: AgeDiagnostics,
    percentiles: tuple[float, ...] = (5.0, 50.0, 95.0),
) -> dict[str, dict]:
    """Return time-mean and percentile statistics for each age diagnostic.

    Returns
    -------
    dict with keys "bulk_delta14C", "respired_delta14C", "reco_delta14C",
    and per-pool entries "stored_delta14C[pool_name]".
    """
    summary: dict[str, dict] = {}

    def _stats(arr: np.ndarray) -> dict:
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return {"mean": np.nan, "percentiles": {p: np.nan for p in percentiles}}
        pct = np.percentile(valid, percentiles)
        return {
            "mean": float(np.mean(valid)),
            "percentiles": {float(p): float(v) for p, v in zip(percentiles, pct)},
        }

    summary["bulk_delta14C"] = _stats(diag.bulk_delta14C)
    summary["respired_delta14C"] = _stats(diag.respired_delta14C)

    if diag.reco_delta14C is not None:
        summary["reco_delta14C"] = _stats(diag.reco_delta14C)

    for i, pool_name in enumerate(diag.pool_names):
        summary[f"stored_delta14C[{pool_name}]"] = _stats(diag.stored_delta14C[:, i])

    return summary
