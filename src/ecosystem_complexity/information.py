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

# ── Observation type constants ─────────────────────────────────────────────────

OBS_C_STOCKS = "C_stocks"
OBS_POOL_D14C = "pool_delta14C"
OBS_RESP_D14C = "resp_delta14C"

ALL_OBS_TYPES: tuple[str, ...] = (OBS_C_STOCKS, OBS_POOL_D14C, OBS_RESP_D14C)

# ── Parameter group names ──────────────────────────────────────────────────────

PARAM_GROUP_TAU = "turnover_times"
PARAM_GROUP_PARTITION = "external_input_partition"
PARAM_GROUP_TRANSFER = "transfer_fractions"
PARAM_GROUP_ENV = "environmental"

# Default observation uncertainties used when none are supplied
_DEFAULT_C_SIGMA_REL: float = 0.2    # 20 % relative uncertainty for C stocks
_DEFAULT_D14C_SIGMA: float = 20.0    # 20 ‰ for pool Δ¹⁴C
_DEFAULT_RESP_SIGMA: float = 20.0    # 20 ‰ for respired Δ¹⁴C

# Default prior standard deviation in log-space for unspecified parameters
_DEFAULT_PRIOR_SIGMA: float = 1.0


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


# ── Parameter utilities ────────────────────────────────────────────────────────


def _get_field_shape(params: ModelParams, field_name: str) -> tuple[int, ...]:
    return getattr(params, field_name).shape


def flatten_params(params: ModelParams, fields: Sequence[str]) -> np.ndarray:
    """Flatten selected ModelParams fields into a 1-D vector."""
    parts = [np.ravel(getattr(params, f)) for f in fields]
    return np.concatenate(parts) if parts else np.zeros(0)


def unflatten_params(
    flat: np.ndarray | jnp.ndarray,
    template: ModelParams,
    fields: Sequence[str],
) -> ModelParams:
    """Reconstruct ModelParams from a flat vector, filling all other fields from template."""
    updates: dict[str, jnp.ndarray] = {}
    offset = 0
    for f in fields:
        shape = _get_field_shape(template, f)
        size = int(math.prod(shape)) if shape else 1
        updates[f] = jnp.asarray(flat[offset : offset + size]).reshape(shape)
        offset += size
    return template._replace(**updates)


def get_param_names(
    params: ModelParams,
    fields: Sequence[str],
    model: EcosystemModel,
) -> list[str]:
    """Generate human-readable parameter names for each scalar in the flat vector.

    Produces names like:
        log_tau[organic_litter], log_f_transfer[organic_litter→organic_slow], …
    """
    pool_names = model.pool_index.pool_names
    n_pools = len(pool_names)
    names: list[str] = []

    for f in fields:
        val = getattr(params, f)
        shape = val.shape

        if f == "log_tau":
            names.extend(f"log_tau[{p}]" for p in pool_names)

        elif f == "log_f_transfer":
            # shape: (n_pools, n_pools + 1); last column = respiration
            for i, src in enumerate(pool_names):
                for j in range(n_pools):
                    names.append(f"log_f_transfer[{src}→{pool_names[j]}]")
                names.append(f"log_f_transfer[{src}→resp]")

        elif f == "log_alloc":
            ag_names = [p.name for p in model.config.aboveground_pools]
            names.extend(f"log_alloc[{p}]" for p in ag_names)

        elif f == "log_Q10":
            layer_names = [lay.name for lay in model.config.soil_layers]
            names.extend(f"log_Q10[{l}]" for l in layer_names)

        elif f == "log_theta_opt":
            layer_names = [lay.name for lay in model.config.soil_layers]
            names.extend(f"log_theta_opt[{l}]" for l in layer_names)

        elif f == "log_gamma_moist":
            layer_names = [lay.name for lay in model.config.soil_layers]
            names.extend(f"log_gamma_moist[{l}]" for l in layer_names)

        elif f == "log_external_input_partition":
            ext = model.config.external_inputs
            if ext is not None and ext.enabled:
                names.extend(
                    f"log_ext_partition[{p}]" for p in ext.target_pool_names
                )
            else:
                pass  # empty array — no names to add

        elif f == "alpha_priming":
            n_mic = int(math.prod(shape))
            layer_names = [lay.name for lay in model.config.soil_layers]
            mic_layers = layer_names[: n_mic]
            names.extend(f"alpha_priming[{l}]" for l in mic_layers)

        else:
            # Generic fallback: scalar or unknown shape
            size = int(math.prod(shape)) if shape else 1
            if size == 1:
                names.append(f"{f}")
            else:
                names.extend(f"{f}[{i}]" for i in range(size))

    return names


def get_param_groups(
    params: ModelParams,
    fields: Sequence[str],
    model: EcosystemModel,
) -> dict[str, list[int]]:
    """Return a mapping from parameter-group name to flat-vector indices.

    Groups:
        turnover_times          — log_tau
        external_input_partition — log_external_input_partition
        transfer_fractions      — log_f_transfer
        environmental           — log_Q10, log_theta_opt, log_gamma_moist, alpha_priming
    """
    groups: dict[str, list[int]] = {
        PARAM_GROUP_TAU: [],
        PARAM_GROUP_PARTITION: [],
        PARAM_GROUP_TRANSFER: [],
        PARAM_GROUP_ENV: [],
    }

    _group_map = {
        "log_tau": PARAM_GROUP_TAU,
        "log_external_input_partition": PARAM_GROUP_PARTITION,
        "log_CUE": PARAM_GROUP_PARTITION,
        "log_soil_input_fraction": PARAM_GROUP_PARTITION,
        "log_f_transfer": PARAM_GROUP_TRANSFER,
        "log_alloc": PARAM_GROUP_ENV,
        "log_Q10": PARAM_GROUP_ENV,
        "log_theta_opt": PARAM_GROUP_ENV,
        "log_gamma_moist": PARAM_GROUP_ENV,
        "alpha_priming": PARAM_GROUP_ENV,
    }

    offset = 0
    for f in fields:
        val = getattr(params, f)
        shape = val.shape
        size = int(math.prod(shape)) if shape else 1
        grp = _group_map.get(f, PARAM_GROUP_ENV)
        groups[grp].extend(range(offset, offset + size))
        offset += size

    # Remove empty groups
    return {k: v for k, v in groups.items() if v}


def make_prior_covariance(
    params: ModelParams,
    fields: Sequence[str],
    model: EcosystemModel,
    default_sigma: float = _DEFAULT_PRIOR_SIGMA,
) -> np.ndarray:
    """Build a diagonal prior standard-deviation vector for the flat parameter space.

    For ``log_tau`` entries the prior σ is taken from the YAML
    ``tau_prior_std`` field of each SOM pool (default: ``default_sigma``).
    All other parameters receive ``default_sigma``.

    Returns
    -------
    prior_sigma : (n_params,) ndarray
        Per-parameter prior standard deviation in log-space.
    """
    pool_names = model.pool_index.pool_names

    # Build per-pool tau prior σ from YAML
    tau_sigma: dict[str, float] = {}
    for layer in model.config.soil_layers:
        for sp in layer.som_pools:
            compound = f"{layer.name}_{sp.name}"
            tau_sigma[compound] = float(sp.tau_prior_std) if sp.tau_prior_std > 0 else default_sigma
    # AG pools: use default
    for ag in model.config.aboveground_pools:
        tau_sigma[ag.name] = default_sigma

    sigmas: list[float] = []
    for f in fields:
        val = getattr(params, f)
        shape = val.shape
        size = int(math.prod(shape)) if shape else 1

        if f == "log_tau":
            sigmas.extend(tau_sigma.get(p, default_sigma) for p in pool_names)
        else:
            sigmas.extend([default_sigma] * size)

    return np.array(sigmas, dtype=np.float64)


# ── Observation extraction ─────────────────────────────────────────────────────


def _extract_scalar_obs(
    raw_value: object,
) -> tuple[float, float]:
    """Extract (value, uncertainty) from the various formats used in ObservationData.

    Handles:
    - tuple/list (value, uncertainty, ...) — scalar point observation
    - scalar float/int — value; uncertainty = NaN
    - 1-D array — nanmean as value; uncertainty = NaN (caller applies default)
    """
    if isinstance(raw_value, (tuple, list)):
        v = float(raw_value[0])
        u = float(raw_value[1]) if len(raw_value) >= 2 else float("nan")
        return v, u

    arr = np.asarray(raw_value, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr), float("nan")

    valid = ~np.isnan(arr)
    if not np.any(valid):
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float("nan")


def _build_obs_config(
    observations: ObservationData,
    model: EcosystemModel,
    obs_sigma_C: float = _DEFAULT_C_SIGMA_REL,
    obs_sigma_d14C: float = _DEFAULT_D14C_SIGMA,
    obs_sigma_resp: float = _DEFAULT_RESP_SIGMA,
) -> dict[str, list[tuple[str, float, float]]]:
    """Extract observations into a structured config.

    Returns
    -------
    obs_config : dict
        Keys are OBS_* constants.  Values are lists of (pool_name, obs_value, obs_sigma)
        for pool-level obs, or [(None, value, sigma)] for respired Δ¹⁴C.
    """
    pool_name_set = set(model.pool_index.pool_names)
    obs_config: dict[str, list[tuple[str | None, float, float]]] = {}

    # C stocks
    c_entries = []
    for pool_name, raw in (observations.C_pools_obs or {}).items():
        if pool_name not in pool_name_set:
            continue
        val, unc = _extract_scalar_obs(raw)
        if np.isnan(val):
            continue
        sigma = unc if not np.isnan(unc) else obs_sigma_C * abs(val)
        if sigma <= 0:
            sigma = obs_sigma_C * max(abs(val), 1.0)
        c_entries.append((pool_name, val, sigma))
    if c_entries:
        obs_config[OBS_C_STOCKS] = c_entries

    # Pool Δ¹⁴C
    d14C_entries = []
    for pool_name, raw in (observations.delta14C_obs or {}).items():
        if pool_name not in pool_name_set:
            continue
        val, unc = _extract_scalar_obs(raw)
        if np.isnan(val):
            continue
        sigma = unc if not np.isnan(unc) else obs_sigma_d14C
        if sigma <= 0:
            sigma = obs_sigma_d14C
        d14C_entries.append((pool_name, val, sigma))
    if d14C_entries:
        obs_config[OBS_POOL_D14C] = d14C_entries

    # Respired Δ¹⁴C
    if observations.delta14C_resp is not None:
        resp_arr = np.asarray(observations.delta14C_resp, dtype=np.float64)
        valid = ~np.isnan(resp_arr)
        if np.any(valid):
            mean_val = float(np.nanmean(resp_arr[valid]))
            obs_config[OBS_RESP_D14C] = [(None, mean_val, obs_sigma_resp)]

    return obs_config


# ── Jacobian computation ───────────────────────────────────────────────────────


def _build_obs_fn(
    model: EcosystemModel,
    forcing: ForcingData,
    state0,
    fields: Sequence[str],
    obs_config: dict,
):
    """Build a JAX-differentiable function: flat_params → flat_obs_summaries.

    The output vector is the concatenation of time-mean scalar summaries for
    each active observation type in the order [C_stocks, pool_delta14C, resp_delta14C].

    Parameters
    ----------
    obs_config :
        Structured observation config from ``_build_obs_config``.
        Only types present as keys are included in the output.
    """
    from .api import run_model  # avoid circular import at module level

    template = make_default_params(model.config)
    fields_tuple = tuple(fields)

    # Pre-compute pool indices for efficiency inside JIT
    c_indices = []
    if OBS_C_STOCKS in obs_config:
        c_indices = [model.pool_index[p] for p, _, _ in obs_config[OBS_C_STOCKS]]

    d14C_indices = []
    if OBS_POOL_D14C in obs_config:
        d14C_indices = [model.pool_index[p] for p, _, _ in obs_config[OBS_POOL_D14C]]

    has_resp = OBS_RESP_D14C in obs_config

    def obs_fn(flat_params: jnp.ndarray) -> jnp.ndarray:
        params = unflatten_params(flat_params, template, fields_tuple)
        output = run_model(model, forcing, state0=state0, params=params)

        parts: list[jnp.ndarray] = []

        # C stocks — time-mean C12 for each observed pool
        for idx in c_indices:
            parts.append(jnp.mean(output.C12[:, idx]))

        # Pool Δ¹⁴C — time-mean delta14C for each observed pool
        for idx in d14C_indices:
            parts.append(jnp.mean(output.delta14C[:, idx]))

        # Respired Δ¹⁴C — flux-weighted time-mean
        if has_resp:
            tau = jnp.exp(params.log_tau)
            weights = output.C12 / (tau[None, :] + 1e-30)
            w_sum = weights.sum(axis=-1, keepdims=False) + 1e-30
            d14C_resp = (output.delta14C * weights).sum(axis=-1) / w_sum
            parts.append(jnp.mean(d14C_resp))

        if not parts:
            return jnp.zeros(0)
        return jnp.stack(parts)

    return obs_fn


def compute_jacobian(
    model: EcosystemModel,
    forcing: ForcingData,
    state0,
    params: ModelParams,
    fields: Sequence[str],
    obs_config: dict,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute the Jacobian H = ∂h/∂θ at the given parameters.

    Uses reverse-mode autodiff (``jax.jacrev``) for efficiency when
    n_obs < n_params.

    Parameters
    ----------
    obs_config :
        Output of ``_build_obs_config``.

    Returns
    -------
    H_total : (n_obs_total, n_params) ndarray
    H_per_type : dict[str, (n_obs_k, n_params) ndarray]
    """
    flat0 = jnp.array(flatten_params(params, fields))
    obs_fn = _build_obs_fn(model, forcing, state0, fields, obs_config)

    # Compute full Jacobian (reverse-mode: one forward + n_obs backward passes)
    H_total_jax = jax.jacrev(obs_fn)(flat0)
    H_total = np.array(H_total_jax)

    # Split by observation type
    H_per_type: dict[str, np.ndarray] = {}
    offset = 0
    for obs_type in ALL_OBS_TYPES:
        if obs_type not in obs_config:
            continue
        n_k = len(obs_config[obs_type])
        H_per_type[obs_type] = H_total[offset : offset + n_k, :]
        offset += n_k

    return H_total, H_per_type


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
