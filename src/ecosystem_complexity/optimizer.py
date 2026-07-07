"""Shared optimizer helpers for parameter selection and vector packing."""

from __future__ import annotations

import math
from typing import Optional

import jax.numpy as jnp

from .config import ModelConfig
from .state import ModelParams

_CORE_OPTIMIZED_FIELDS = (
    "log_tau",
    "log_f_transfer",
    "log_alloc",
    "log_Q10",
    "log_theta_opt",
    "log_gamma_moist",
    "alpha_priming",
)

_OE_CORE_FIELDS = (
    "log_tau",
    "log_f_transfer",
)


def get_opt_fields(config: ModelConfig) -> tuple[str, ...]:
    """Fields for the gradient-based optimiser."""
    fields = list(_CORE_OPTIMIZED_FIELDS)
    ext = config.external_inputs
    if ext is not None and ext.enabled:
        if ext.optimize_CUE:
            fields.append("log_CUE")
        if ext.optimize_soil_input_fraction:
            fields.append("log_soil_input_fraction")
        if ext.optimize_partition:
            fields.append("log_external_input_partition")
    return tuple(fields)


def get_oe_fields(
    config: ModelConfig, inv_cfg: Optional[dict] = None
) -> tuple[str, ...]:
    """Fields for the OE Levenberg-Marquardt inversion."""
    fields = list(_OE_CORE_FIELDS)
    inv = inv_cfg or {}
    ext = config.external_inputs

    if ext is not None and ext.enabled and ext.optimize_partition:
        fields.append("log_external_input_partition")

    if inv.get("optimize_f_hetero", False):
        fields.append("log_f_hetero")

    return tuple(fields)


def params_to_vector(params: ModelParams, opt_fields: tuple[str, ...]) -> jnp.ndarray:
    """Flatten optimised parameter fields into a 1-D vector."""
    parts = [jnp.ravel(getattr(params, f)) for f in opt_fields]
    return jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype=jnp.float32)


def vector_to_params(
    vec: jnp.ndarray, template: ModelParams, opt_fields: tuple[str, ...]
) -> ModelParams:
    """Unpack a 1-D optimisation vector back into ModelParams."""
    updates = {}
    offset = 0
    for f in opt_fields:
        val = getattr(template, f)
        size = int(math.prod(val.shape))
        updates[f] = vec[offset : offset + size].reshape(val.shape)
        offset += size
    return template._replace(**updates)


# Backward-compatible private aliases for existing callers.
_get_opt_fields = get_opt_fields
_get_oe_fields = get_oe_fields
_params_to_vector = params_to_vector
_vector_to_params = vector_to_params
