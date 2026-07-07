"""
Information-theoretic analysis for radiocarbon-constrained soil carbon inversion.

Implements:
  - Fisher Information Matrix (FIM) decomposed by observation type
  - Averaging kernel and degrees of freedom for signal (DFS)
  - Gaussian posterior covariance and uncertainty reduction
  - Helper utilities for parameter flattening and prior specification

Scientific context
------------------
The central question is whether pool Δ¹⁴C, respired CO₂ Δ¹⁴C, and soil C stocks
constrain *distinct* elements of the state vector — particularly whether stored
carbon age structure (constrained by pool Δ¹⁴C) can be separated from the active
decomposition source mixture (constrained by respired Δ¹⁴C).

Mathematical framework
----------------------
Given a (possibly nonlinear) forward model h(θ), linearised at the current
parameter estimate θ₀:

    y = h(θ₀) + H·(θ - θ₀) + ε,    ε ~ N(0, S_obs)

where H = ∂h/∂θ|_{θ₀} is the Jacobian (sensitivity matrix).

With a Gaussian prior θ ~ N(θ_prior, C_prior):

    Posterior covariance: C_post = (FIM + C_prior⁻¹)⁻¹
    Fisher Information:   FIM = Hᵀ S_obs⁻¹ H
    Averaging kernel:     A   = C_post · FIM
    Degrees of freedom:   DFS = trace(A)

Per observation-type contribution (type k):

    FIM_k = Hₖᵀ S_obs_k⁻¹ Hₖ
    A_k   = C_post · FIM_k
    DFS_k = trace(A_k)

Observation types
-----------------
OBS_C_STOCKS   — soil carbon pool masses (gC m⁻²)
OBS_POOL_D14C  — pool-level radiocarbon (Δ¹⁴C ‰)
OBS_RESP_D14C  — flux-weighted respired CO₂ Δ¹⁴C (‰)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .config import ModelConfig, PoolIndex
from .model import EcosystemModel
from .state import ModelParams, make_default_params
from .data.schemas import ForcingData, ObservationData

# ── Result dataclasses ─────────────────────────────────────────────────────────


@dataclass
class FisherResult:
    """Fisher Information Matrix decomposed by observation type.

    Attributes
    ----------
    FIM_total : (n_params, n_params) ndarray
        Total FIM summed across all active observation types.
    FIM_per_type : dict[str, (n_params, n_params) ndarray]
        Per observation-type FIM contributions.
    eigenvalues : (n_params,) ndarray
        Eigenvalues of FIM_total in descending order.
    eigenvectors : (n_params, n_params) ndarray
        Column eigenvectors (columns correspond to eigenvalues).
    param_names : list[str] or None
        Human-readable labels for each parameter dimension.
    obs_types : list[str] or None
        Observation types included in this result.
    n_obs_per_type : dict[str, int]
        Number of scalar observations contributed by each type.
    """

    FIM_total: np.ndarray
    FIM_per_type: dict[str, np.ndarray]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    param_names: list[str] | None = None
    obs_types: list[str] | None = None
    n_obs_per_type: dict[str, int] = field(default_factory=dict)


@dataclass
class DofResult:
    """Averaging kernel and degrees of freedom for signal.

    Attributes
    ----------
    dfs_total : float
        Total degrees of freedom for signal = trace(A).
    dfs_per_obs_type : dict[str, float]
        DFS contribution from each observation type: trace(C_post @ FIM_k).
    dfs_per_param : (n_params,) ndarray
        Diagonal of the averaging kernel A; per-parameter information.
    averaging_kernel : (n_params, n_params) ndarray
        Full averaging kernel A = C_post @ FIM_total.
    averaging_kernel_per_type : dict[str, (n_params, n_params) ndarray]
        Per observation-type averaging kernels.
    dfs_by_group : dict[str, float] or None
        DFS summed over named parameter groups.
    param_names : list[str] or None
    param_groups : dict[str, list[int]] or None
        Maps group name → list of parameter indices.
    """

    dfs_total: float
    dfs_per_obs_type: dict[str, float]
    dfs_per_param: np.ndarray
    averaging_kernel: np.ndarray
    averaging_kernel_per_type: dict[str, np.ndarray]
    dfs_by_group: dict[str, float] | None = None
    param_names: list[str] | None = None
    param_groups: dict[str, list[int]] | None = None


@dataclass
class PosteriorResult:
    """Gaussian posterior covariance and uncertainty reduction.

    Attributes
    ----------
    C_post : (n_params, n_params) ndarray
        Full posterior covariance matrix.
    posterior_sigma : (n_params,) ndarray
        Posterior standard deviation for each parameter.
    prior_sigma : (n_params,) ndarray
        Prior standard deviation for each parameter.
    uncertainty_reduction : (n_params,) ndarray
        Fractional uncertainty reduction: 1 - σ_post / σ_prior ∈ [0, 1].
    uncertainty_reduction_per_type : dict[str, (n_params,) ndarray]
        Uncertainty reduction when using each obs type in addition to prior.
    correlation_matrix : (n_params, n_params) ndarray
        Posterior correlation matrix.
    C_post_per_type : dict[str, (n_params, n_params) ndarray]
        Posterior covariance when using only each obs type (+ prior).
    param_names : list[str] or None
    """

    C_post: np.ndarray
    posterior_sigma: np.ndarray
    prior_sigma: np.ndarray
    uncertainty_reduction: np.ndarray
    uncertainty_reduction_per_type: dict[str, np.ndarray]
    correlation_matrix: np.ndarray
    C_post_per_type: dict[str, np.ndarray]
    param_names: list[str] | None = None



from .sensitivity import (
    OBS_C_STOCKS,
    OBS_POOL_D14C,
    OBS_RESP_D14C,
    ALL_OBS_TYPES,
    PARAM_GROUP_TAU,
    PARAM_GROUP_PARTITION,
    PARAM_GROUP_TRANSFER,
    PARAM_GROUP_ENV,
    _DEFAULT_C_SIGMA_REL,
    _DEFAULT_D14C_SIGMA,
    _DEFAULT_RESP_SIGMA,
    _DEFAULT_PRIOR_SIGMA,
    flatten_params,
    unflatten_params,
    get_param_names,
    get_param_groups,
    make_prior_covariance,
    _build_obs_config,
    _build_obs_fn,
    compute_jacobian,
)

# ── Core information functions ─────────────────────────────────────────────────


def _fim_from_jacobian(
    H: np.ndarray,
    obs_sigma: np.ndarray,
) -> np.ndarray:
    """Compute FIM = Hᵀ diag(1/σ²) H from Jacobian H (n_obs, n_params)."""
    S_inv = 1.0 / (obs_sigma ** 2 + 1e-300)
    return (H * S_inv[:, None]).T @ H


def _obs_sigma_vector(obs_config: dict) -> np.ndarray:
    """Build the observation error σ vector in observation-type order."""
    sigmas: list[float] = []
    for obs_type in ALL_OBS_TYPES:
        if obs_type not in obs_config:
            continue
        for _, _, sigma in obs_config[obs_type]:
            sigmas.append(sigma)
    return np.array(sigmas, dtype=np.float64)


def compute_fisher(
    model: EcosystemModel,
    forcing: ForcingData,
    state0,
    params: ModelParams,
    observations: ObservationData,
    fields: Sequence[str] | None = None,
    obs_sigma_C: float = _DEFAULT_C_SIGMA_REL,
    obs_sigma_d14C: float = _DEFAULT_D14C_SIGMA,
    obs_sigma_resp: float = _DEFAULT_RESP_SIGMA,
    active_obs_types: Sequence[str] | None = None,
) -> FisherResult:
    """Compute the Fisher Information Matrix decomposed by observation type.

    Parameters
    ----------
    model : EcosystemModel
    forcing : ForcingData
    state0 : EcosystemState
        Linearisation point for the state (usually the spun-up state).
    params : ModelParams
        Linearisation point for the parameters (usually the MAP estimate).
    observations : ObservationData
        Observed C stocks, pool Δ¹⁴C, and/or respired Δ¹⁴C.
    fields : sequence of str, optional
        ModelParams field names included in the parameter vector.
        Defaults to ``["log_tau", "log_f_transfer", "log_external_input_partition"]``
        (the radiocarbon-relevant subset).
    obs_sigma_C : float
        Default relative uncertainty for C stocks (fraction of obs value).
    obs_sigma_d14C : float
        Default absolute uncertainty for pool Δ¹⁴C (‰).
    obs_sigma_resp : float
        Default absolute uncertainty for respired Δ¹⁴C (‰).
    active_obs_types : sequence of str, optional
        Restrict analysis to a subset of obs types. Default: all available.

    Returns
    -------
    FisherResult
    """
    if fields is None:
        fields = _default_fields(model)

    obs_config = _build_obs_config(
        observations, model,
        obs_sigma_C=obs_sigma_C,
        obs_sigma_d14C=obs_sigma_d14C,
        obs_sigma_resp=obs_sigma_resp,
    )

    if active_obs_types is not None:
        obs_config = {k: v for k, v in obs_config.items() if k in active_obs_types}

    if not obs_config:
        n_params = sum(
            int(math.prod(getattr(params, f).shape)) for f in fields
        )
        empty_fim = np.zeros((n_params, n_params))
        return FisherResult(
            FIM_total=empty_fim,
            FIM_per_type={},
            eigenvalues=np.zeros(n_params),
            eigenvectors=np.eye(n_params),
            param_names=get_param_names(params, fields, model),
            obs_types=[],
            n_obs_per_type={},
        )

    H_total, H_per_type = compute_jacobian(
        model, forcing, state0, params, fields, obs_config
    )

    obs_sigma = _obs_sigma_vector(obs_config)
    FIM_total = _fim_from_jacobian(H_total, obs_sigma)

    # Per-type FIMs with their own σ vectors
    FIM_per_type: dict[str, np.ndarray] = {}
    n_obs_per_type: dict[str, int] = {}
    for obs_type, H_k in H_per_type.items():
        sigmas_k = np.array(
            [sigma for _, _, sigma in obs_config[obs_type]], dtype=np.float64
        )
        FIM_per_type[obs_type] = _fim_from_jacobian(H_k, sigmas_k)
        n_obs_per_type[obs_type] = len(obs_config[obs_type])

    # Eigendecomposition of total FIM
    eigenvalues, eigenvectors = np.linalg.eigh(FIM_total)
    # Sort descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return FisherResult(
        FIM_total=FIM_total,
        FIM_per_type=FIM_per_type,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        param_names=get_param_names(params, fields, model),
        obs_types=list(obs_config.keys()),
        n_obs_per_type=n_obs_per_type,
    )


def compute_dof(
    fisher: FisherResult,
    prior_sigma: np.ndarray,
    param_groups: dict[str, list[int]] | None = None,
) -> DofResult:
    """Compute averaging kernel and degrees of freedom for signal.

    Parameters
    ----------
    fisher : FisherResult
    prior_sigma : (n_params,) ndarray
        Prior standard deviation for each parameter in the flat vector.
    param_groups : dict[str, list[int]], optional
        Maps group name → flat-vector indices for group-level DFS.

    Returns
    -------
    DofResult
    """
    C_prior = np.diag(prior_sigma ** 2)
    C_prior_inv = np.diag(1.0 / (prior_sigma ** 2 + 1e-300))

    C_post = _invert_posterior(fisher.FIM_total, C_prior_inv)
    A = C_post @ fisher.FIM_total  # averaging kernel

    dfs_total = float(np.trace(A))
    dfs_per_param = np.diag(A)

    # Per-type averaging kernels and DFS
    A_per_type: dict[str, np.ndarray] = {}
    dfs_per_type: dict[str, float] = {}
    for obs_type, FIM_k in fisher.FIM_per_type.items():
        A_k = C_post @ FIM_k
        A_per_type[obs_type] = A_k
        dfs_per_type[obs_type] = float(np.trace(A_k))

    # Group DFS
    dfs_by_group: dict[str, float] | None = None
    if param_groups is not None:
        dfs_by_group = {
            grp: float(np.sum(dfs_per_param[idxs]))
            for grp, idxs in param_groups.items()
        }

    return DofResult(
        dfs_total=dfs_total,
        dfs_per_obs_type=dfs_per_type,
        dfs_per_param=dfs_per_param,
        averaging_kernel=A,
        averaging_kernel_per_type=A_per_type,
        dfs_by_group=dfs_by_group,
        param_names=fisher.param_names,
        param_groups=param_groups,
    )


def compute_posterior(
    fisher: FisherResult,
    prior_sigma: np.ndarray,
) -> PosteriorResult:
    """Compute Gaussian posterior covariance and uncertainty reduction.

    Parameters
    ----------
    fisher : FisherResult
    prior_sigma : (n_params,) ndarray

    Returns
    -------
    PosteriorResult
    """
    C_prior_inv = np.diag(1.0 / (prior_sigma ** 2 + 1e-300))

    C_post = _invert_posterior(fisher.FIM_total, C_prior_inv)
    post_sigma = np.sqrt(np.diag(C_post))
    ur = 1.0 - post_sigma / prior_sigma

    # Correlation matrix
    D = np.diag(1.0 / (post_sigma + 1e-300))
    corr = D @ C_post @ D

    # Per-type posterior and uncertainty reduction (using each type in isolation)
    C_post_per_type: dict[str, np.ndarray] = {}
    ur_per_type: dict[str, np.ndarray] = {}
    for obs_type, FIM_k in fisher.FIM_per_type.items():
        C_post_k = _invert_posterior(FIM_k, C_prior_inv)
        C_post_per_type[obs_type] = C_post_k
        sigma_k = np.sqrt(np.diag(C_post_k))
        ur_per_type[obs_type] = 1.0 - sigma_k / prior_sigma

    return PosteriorResult(
        C_post=C_post,
        posterior_sigma=post_sigma,
        prior_sigma=prior_sigma,
        uncertainty_reduction=ur,
        uncertainty_reduction_per_type=ur_per_type,
        correlation_matrix=corr,
        C_post_per_type=C_post_per_type,
        param_names=fisher.param_names,
    )


# ── Convenience entry point ────────────────────────────────────────────────────


def analyze_information_content(
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
    active_obs_types: Sequence[str] | None = None,
) -> tuple[FisherResult, DofResult, PosteriorResult]:
    """One-shot information-content analysis.

    Computes FIM, DFS, and posterior covariance in a single call.

    Parameters
    ----------
    fields : sequence of str, optional
        ModelParams fields to include. Defaults to the radiocarbon-relevant set.
    prior_sigma : (n_params,) ndarray, optional
        Per-parameter prior σ. If None, built from YAML ``tau_prior_std``
        with ``default_prior_sigma`` for other parameters.

    Returns
    -------
    (FisherResult, DofResult, PosteriorResult)
    """
    if fields is None:
        fields = _default_fields(model)

    fisher = compute_fisher(
        model, forcing, state0, params, observations,
        fields=fields,
        obs_sigma_C=obs_sigma_C,
        obs_sigma_d14C=obs_sigma_d14C,
        obs_sigma_resp=obs_sigma_resp,
        active_obs_types=active_obs_types,
    )

    if prior_sigma is None:
        prior_sigma = make_prior_covariance(
            params, fields, model, default_sigma=default_prior_sigma
        )

    param_groups = get_param_groups(params, fields, model)

    dof = compute_dof(fisher, prior_sigma, param_groups=param_groups)
    posterior = compute_posterior(fisher, prior_sigma)

    return fisher, dof, posterior


# ── Internal helpers ───────────────────────────────────────────────────────────


def _default_fields(model: EcosystemModel) -> list[str]:
    """Fields that are scientifically relevant for the radiocarbon analysis."""
    fields = ["log_tau", "log_f_transfer"]
    ext = model.config.external_inputs
    if ext is not None and ext.enabled and ext.optimize_partition:
        fields.append("log_external_input_partition")
    return fields


def _invert_posterior(
    FIM: np.ndarray,
    C_prior_inv: np.ndarray,
) -> np.ndarray:
    """Compute C_post = (FIM + C_prior⁻¹)⁻¹ stably via Cholesky when possible."""
    M = FIM + C_prior_inv
    try:
        L = np.linalg.cholesky(M + 1e-10 * np.eye(M.shape[0]))
        L_inv = np.linalg.inv(L)
        return L_inv.T @ L_inv
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)
