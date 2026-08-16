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


def _check_transfer_pool_names(config: ModelConfig, valid_names: set[str]) -> None:
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
        raise ConfigValidationError(f"external_inputs.CUE={ext.CUE} is outside (0, 1].")

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


def _check_warming(config: ModelConfig) -> None:
    """Shape-check the optional ``warming`` block."""
    block = config.warming_raw
    if not block:
        return
    if "horizon_years" in block:
        v = block["horizon_years"]
        if not isinstance(v, (int, float)) or v <= 0:
            raise ConfigValidationError(
                f"warming.horizon_years must be a positive number, got {v!r}."
            )
    if "warming_delta_c" in block:
        v = block["warming_delta_c"]
        if not isinstance(v, (int, float)):
            raise ConfigValidationError(
                f"warming.warming_delta_c must be numeric, got {v!r}."
            )
    if "metric" in block and block["metric"] not in {"vulnerability", "transit"}:
        raise ConfigValidationError(
            f"warming.metric must be 'vulnerability' or 'transit', "
            f"got {block['metric']!r}."
        )
    if "include_constraints" in block and not isinstance(
        block["include_constraints"], dict
    ):
        raise ConfigValidationError(
            "warming.include_constraints must be a mapping of "
            "{constraint_name: bool}."
        )


def _check_mcmc(config: ModelConfig) -> None:
    """Shape-check the optional ``mcmc`` block."""
    block = config.mcmc_raw
    if not block:
        return
    positive_ints = (
        "rng_seed",
        "mc_iterations",
        "null_iterations",
        "posterior_draw_count",
        "prior_draw_count",
    )
    for key in positive_ints:
        if key in block:
            v = block[key]
            if not isinstance(v, int) or v < 0:
                raise ConfigValidationError(
                    f"mcmc.{key} must be a non-negative integer, got {v!r}."
                )
    for key in ("warming_horizon_years", "warming_delta_c"):
        if key in block and not isinstance(block[key], (int, float)):
            raise ConfigValidationError(
                f"mcmc.{key} must be numeric, got {block[key]!r}."
            )
    if "old_pools" in block:
        v = block["old_pools"]
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ConfigValidationError(
                "mcmc.old_pools must be a list of pool-name strings."
            )
        valid = _all_valid_pool_names(config)
        unknown = [name for name in v if name not in valid]
        if unknown:
            raise ConfigValidationError(
                f"mcmc.old_pools references unknown pool(s): {unknown}. "
                f"Known pools: {sorted(valid)}"
            )


def _check_information(config: ModelConfig) -> None:
    """Shape-check the optional ``information`` block."""
    block = config.information_raw
    if not block:
        return
    if "metrics" in block and not isinstance(block["metrics"], dict):
        raise ConfigValidationError(
            "information.metrics must be a mapping of {metric_name: bool}."
        )
    shapley = block.get("shapley")
    if shapley is not None:
        if not isinstance(shapley, dict):
            raise ConfigValidationError("information.shapley must be a mapping.")
        rule = shapley.get("sigma_rule")
        if rule is not None:
            if not isinstance(rule, str) or rule.count(":") != 1:
                raise ConfigValidationError(
                    f"information.shapley.sigma_rule must be 'REL:ABS' "
                    f"(e.g. '0.20:500'), got {rule!r}."
                )
            rel_s, abs_s = rule.split(":")
            try:
                float(rel_s)
                float(abs_s)
            except ValueError as exc:
                raise ConfigValidationError(
                    f"information.shapley.sigma_rule={rule!r} has non-numeric parts."
                ) from exc
    ose = block.get("ose")
    if ose is not None:
        if not isinstance(ose, dict) or not isinstance(ose.get("scenarios", []), list):
            raise ConfigValidationError(
                "information.ose must be a mapping with a 'scenarios' list."
            )
        for i, sc in enumerate(ose.get("scenarios", [])):
            if not isinstance(sc, dict) or "name" not in sc or "include" not in sc:
                raise ConfigValidationError(
                    f"information.ose.scenarios[{i}] must have 'name' and 'include'."
                )
            if not isinstance(sc["include"], list):
                raise ConfigValidationError(
                    f"information.ose.scenarios[{i}].include must be a list."
                )


def _check_sweep(config: ModelConfig) -> None:
    """Shape-check the optional ``sweep`` block."""
    block = config.sweep_raw
    if not block:
        return
    valid_kinds = {"pool_count", "sigma", "forcing"}
    if "kind" in block and block["kind"] not in valid_kinds:
        raise ConfigValidationError(
            f"sweep.kind must be one of {sorted(valid_kinds)}, got {block['kind']!r}."
        )
    for key in ("member_dir", "member_glob", "combine_output"):
        if key in block and not isinstance(block[key], str):
            raise ConfigValidationError(
                f"sweep.{key} must be a string, got {block[key]!r}."
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
    6. Optional experiment blocks (warming, mcmc, information, sweep) are
       shape-valid when present.
    """
    valid_names = _all_valid_pool_names(config)
    _check_transfer_pool_names(config, valid_names)
    _check_transfer_sums(config)
    _check_layer_depths(config)
    _check_alloc_coverage(config)
    _check_external_inputs(config, valid_names)
    _check_warming(config)
    _check_mcmc(config)
    _check_information(config)
    _check_sweep(config)
