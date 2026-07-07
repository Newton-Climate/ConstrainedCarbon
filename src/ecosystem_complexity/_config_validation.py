"""
Internal validation helpers for ModelConfig.

These functions are called by ``config.py``'s ``_validate`` before returning
a loaded config.  Kept in a separate module to stay within the 500-line limit.

``ConfigValidationError`` is defined here so that ``_config_validation.py``
does not need to import from ``config.py``, avoiding a circular dependency.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecosystem_complexity.config import ModelConfig


class ConfigValidationError(ValueError):
    """Raised when a config file fails semantic validation."""


def _all_valid_pool_names(config: ModelConfig) -> set[str]:
    """Return the complete set of valid pool names for this config."""
    names: set[str] = set()
    for pool in config.aboveground_pools:
        names.add(pool.name)
    for layer in config.soil_layers:
        for som_pool in layer.som_pools:
            names.add(f"{layer.name}_{som_pool.name}")
        if config.microbial_pool_per_layer:
            names.add(f"{layer.name}_mic")
    return names


def _check_transfer_pool_names(
    config: ModelConfig, valid_names: set[str]
) -> None:
    """Raise if any transfer rule references an unknown pool name."""
    for source, dest, _ in config.transfer_rules:
        if source not in valid_names:
            raise ConfigValidationError(
                f"transfer_rules references unknown source pool {source!r}. "
                f"Known pools: {sorted(valid_names)}"
            )
        if dest not in valid_names:
            raise ConfigValidationError(
                f"transfer_rules references unknown destination pool {dest!r}. "
                f"Known pools: {sorted(valid_names)}"
            )


def _check_transfer_sums(config: ModelConfig) -> None:
    """Raise if any source pool's outflow fractions sum to > 1.0."""
    outflow: dict[str, float] = {}
    for source, _, fraction in config.transfer_rules:
        outflow[source] = outflow.get(source, 0.0) + fraction
    for pool_name, total in outflow.items():
        if total > 1.0 + 1e-9:
            raise ConfigValidationError(
                f"Transfer fractions for source pool {pool_name!r} sum to "
                f"{total:.6f}, which exceeds 1.0.  The remainder (1 − sum) is "
                f"treated as heterotrophic respiration and must be ≥ 0."
            )


def _check_layer_depths(config: ModelConfig) -> None:
    """Raise if layer depths are non-positive, overlapping, or non-contiguous."""
    layers = config.soil_layers
    for i, layer in enumerate(layers):
        if layer.depth_bot_m <= layer.depth_top_m:
            raise ConfigValidationError(
                f"Layer {layer.name!r} has depth_bot_m ({layer.depth_bot_m}) "
                f"≤ depth_top_m ({layer.depth_top_m}); layers must have "
                f"positive thickness."
            )
        if i > 0:
            prev = layers[i - 1]
            if not math.isclose(
                layer.depth_top_m, prev.depth_bot_m, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ConfigValidationError(
                    f"Depth gap or overlap between layer {prev.name!r} "
                    f"(depth_bot_m={prev.depth_bot_m}) and layer "
                    f"{layer.name!r} (depth_top_m={layer.depth_top_m}). "
                    f"Layers must be contiguous: depth_top of next layer must "
                    f"equal depth_bot of previous layer."
                )


def _check_alloc_coverage(config: ModelConfig) -> None:
    """Raise if any aboveground pool is missing an alloc fraction."""
    ag_names = {p.name for p in config.aboveground_pools}
    missing = ag_names - set(config.alloc.keys())
    if missing:
        raise ConfigValidationError(
            f"NPP alloc fractions are missing for aboveground pool(s): "
            f"{sorted(missing)}.  Every aboveground pool must have an entry "
            f"under parameters.alloc in the YAML."
        )


def _check_external_inputs(config: ModelConfig, valid_names: set[str]) -> None:
    """Validate the external_inputs block when present and enabled."""
    ext = config.external_inputs
    if ext is None or not ext.enabled:
        return

    valid_sources = {"GPP_obs", "NPP_obs"}
    if ext.source not in valid_sources:
        raise ConfigValidationError(
            f"external_inputs.source {ext.source!r} is not valid.  "
            f"Allowed values: {sorted(valid_sources)}"
        )

    if not (0.0 <= ext.soil_input_fraction <= 1.0):
        raise ConfigValidationError(
            f"external_inputs.soil_input_fraction={ext.soil_input_fraction} "
            f"is outside [0, 1]."
        )

    if not (0.0 < ext.CUE <= 1.0):
        raise ConfigValidationError(
            f"external_inputs.CUE={ext.CUE} is outside (0, 1]."
        )

    ag_names = {p.name for p in config.aboveground_pools}
    for pool_name in ext.partition:
        if pool_name not in valid_names:
            raise ConfigValidationError(
                f"external_inputs.partition references unknown pool "
                f"{pool_name!r}.  Known pools: {sorted(valid_names)}"
            )
        if pool_name in ag_names:
            raise ConfigValidationError(
                f"external_inputs.partition references aboveground pool "
                f"{pool_name!r}.  Only soil pools are valid targets."
            )

    fracs = list(ext.partition.values())
    if any(f < 0 for f in fracs):
        raise ConfigValidationError(
            "external_inputs.partition contains negative fractions."
        )
    total = sum(fracs)
    if total <= 0 or total > 1.0 + 1e-9:
        raise ConfigValidationError(
            f"external_inputs.partition fractions sum to {total:.6f}; "
            f"they must be positive and sum to a value in (0, 1]."
        )


def _validate(config: ModelConfig) -> None:
    """
    Raise ``ConfigValidationError`` if any semantic constraint is violated.

    Checks (in order):
    1. transfer_rules reference only known pool names.
    2. Outflow fractions from any single source pool sum to ≤ 1.0.
    3. Soil layer depths are strictly monotonic and contiguous.
    4. NPP alloc fractions exist for every aboveground pool.
    5. external_inputs block (when present and enabled) is self-consistent.
    """
    valid_names = _all_valid_pool_names(config)
    _check_transfer_pool_names(config, valid_names)
    _check_transfer_sums(config)
    _check_layer_depths(config)
    _check_alloc_coverage(config)
    _check_external_inputs(config, valid_names)
