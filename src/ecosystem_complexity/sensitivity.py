"""
Parameter sensitivity utilities for information-theoretic analysis.

Provides:
  - Observation type constants (OBS_C_STOCKS, OBS_POOL_D14C, OBS_RESP_D14C)
  - Parameter group constants (PARAM_GROUP_TAU, etc.)
  - Default observation uncertainty constants
  - Parameter vector operations: flatten_params, unflatten_params
  - Parameter label utilities: get_param_names, get_param_groups
  - Prior covariance builder: make_prior_covariance
  - Observation extraction: _extract_scalar_obs, _build_obs_config
  - Forward model and Jacobian for Fisher analysis: _build_obs_fn, compute_jacobian
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .data.schemas import ForcingData, ObservationData
from .state import ModelParams, make_default_params

if TYPE_CHECKING:
    from .model import EcosystemModel

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
_DEFAULT_C_SIGMA_REL: float = 0.2  # 20 % relative uncertainty for C stocks
_DEFAULT_D14C_SIGMA: float = 20.0  # 20 ‰ for pool Δ¹⁴C
_DEFAULT_RESP_SIGMA: float = 20.0  # 20 ‰ for respired Δ¹⁴C

# Default prior standard deviation in log-space for unspecified parameters
_DEFAULT_PRIOR_SIGMA: float = 1.0


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
    """Reconstruct ModelParams from a flat vector.

    All other fields are filled from the template.
    """
    updates: dict[str, jnp.ndarray] = {}
    offset = 0
    for f in fields:
        shape = _get_field_shape(template, f)
        size = int(math.prod(shape)) if shape else 1
        updates[f] = jnp.asarray(flat[offset : offset + size]).reshape(shape)
        offset += size
    return template._replace(**updates)


def get_param_names(  # noqa: C901
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
            names.extend(f"log_Q10[{lyr}]" for lyr in layer_names)

        elif f == "log_theta_opt":
            layer_names = [lay.name for lay in model.config.soil_layers]
            names.extend(f"log_theta_opt[{lyr}]" for lyr in layer_names)

        elif f == "log_gamma_moist":
            layer_names = [lay.name for lay in model.config.soil_layers]
            names.extend(f"log_gamma_moist[{lyr}]" for lyr in layer_names)

        elif f == "log_external_input_partition":
            ext = model.config.external_inputs
            if ext is not None and ext.enabled:
                names.extend(f"log_ext_partition[{p}]" for p in ext.target_pool_names)
            else:
                pass  # empty array — no names to add

        elif f == "alpha_priming":
            n_mic = int(math.prod(shape))
            layer_names = [lay.name for lay in model.config.soil_layers]
            mic_layers = layer_names[:n_mic]
            names.extend(f"alpha_priming[{lyr}]" for lyr in mic_layers)

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

    # Build per-pool tau prior σ from YAML, converted to log-space.
    # tau_prior_std is in days; convert via delta method: σ_log ≈ σ_days / τ_days.
    # This gives the fractional (coefficient-of-variation) uncertainty in log-space,
    # e.g. organic_litter: 60 days / 180 days ≈ 0.33 log-units.
    tau_sigma: dict[str, float] = {}
    for layer in model.config.soil_layers:
        for sp in layer.som_pools:
            compound = f"{layer.name}_{sp.name}"
            if sp.tau_prior_std > 0 and sp.tau_prior_days > 0:
                tau_sigma[compound] = float(sp.tau_prior_std) / float(sp.tau_prior_days)
            else:
                tau_sigma[compound] = default_sigma
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


def _build_obs_config(  # noqa: C901
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
