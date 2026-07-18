"""
Configuration system for the ecosystem-complexity carbon model.

Parses YAML files into a validated, frozen dataclass tree and exposes a
``PoolIndex`` helper that maps pool names to integer positions in the flat
state vector used by the rest of the model.

Pool naming convention
----------------------
Soil SOM pools are referenced everywhere as ``{layer_name}_{som_pool_name}``.
For example, layer ``organic`` with SOM pool ``litter`` → ``organic_litter``.
Microbial pools (when ``microbial_pool_per_layer: true``) are named
``{layer_name}_mic``.

Canonical pool order (matches EcosystemState)
----------------------------------------------
    [ag_pool_0, ag_pool_1, ... |
     layer_0_som_0, layer_0_som_1, [layer_0_mic,] |
     layer_1_som_0, ... ]

Public API
----------
    load_config(path)  -> ModelConfig
    PoolIndex(config)  -> PoolIndex
    ConfigValidationError
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import yaml

from ecosystem_complexity._config_validation import (  # noqa: F401
    ConfigValidationError,  # re-exported as public API
    _validate,
)
from ecosystem_complexity.site_config import (
    normalize_analysis_config,
    normalize_output_config,
)

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AboveGroundPoolDef:
    """A single above-ground carbon pool (leaf, wood, branch, root, …)."""

    name: str
    is_woody: bool
    litterfall_to: str  # target soil pool in '{layer}_{som}' format


@dataclass(frozen=True)
class SOMPoolDef:
    """A single soil organic matter pool within a layer."""

    name: str
    tau_prior_days: float
    tau_prior_std: float
    # Optional per-pool depth bounds (m).  When provided, depth-attenuated
    # soil temperature is used for this pool's thawed_frac rather than the
    # layer-level temperature.  Enables freeze-thaw gating at permafrost depth.
    depth_top_m: float | None = None
    depth_bot_m: float | None = None


@dataclass(frozen=True)
class SoilLayerDef:
    """A soil layer and the SOM pools it contains."""

    name: str
    depth_top_m: float
    depth_bot_m: float
    permafrost_eligible: bool
    som_pools: tuple[SOMPoolDef, ...]


@dataclass(frozen=True)
class ExternalInputsConfig:
    """
    Parsed and validated ``external_inputs`` YAML block.

    When this block is present and ``enabled=True``, the internal LUE GPP
    submodel is bypassed and carbon enters soil pools from a prescribed
    ``ForcingData`` field (``GPP_obs`` or ``NPP_obs``).

    Pool names in ``partition`` follow the ``{layer_name}_{som_pool_name}``
    convention used everywhere else in the model (e.g. ``"organic_litter"``).
    """

    enabled: bool
    source: str  # ForcingData field name: "GPP_obs" or "NPP_obs"
    CUE: float  # carbon use efficiency (0 < CUE ≤ 1)
    optimize_CUE: bool
    soil_input_fraction: float  # fraction of NPP bypassing AG pools [0, 1]
    optimize_soil_input_fraction: bool
    partition: dict[str, float]  # {pool_name: prior_fraction}; priors sum ≤ 1
    optimize_partition: bool
    target_pool_names: tuple[str, ...]  # ordered keys from partition (for ModelParams)


@dataclass(frozen=True)
class ModelConfig:
    """
    Complete, validated model configuration parsed from a YAML file.

    All structural fields are immutable after construction.  Raw YAML
    sections (data paths, inversion settings, analysis options, output)
    are preserved as plain dicts for downstream modules.
    """

    # Site metadata
    site_id: str
    site_name: str
    lat: float
    lon: float

    # Numerical / solver settings
    dt_days: float
    solver: str
    spinup_years: int
    enable_14C: bool
    microbial_pool_per_layer: bool

    # Pool / layer structure  (these fix the canonical pool order)
    aboveground_pools: tuple[AboveGroundPoolDef, ...]
    soil_layers: tuple[SoilLayerDef, ...]

    # Transfer rules: (source_pool, dest_pool, fraction)
    # fraction is the share of source outflux routed to dest;
    # 1 - sum(fractions for source) is respired.
    transfer_rules: tuple[tuple[str, str, float], ...]

    # NPP allocation fractions keyed by AG pool name
    alloc: dict[str, float]

    # Raw YAML sections – preserved for other modules
    parameters_raw: dict[str, Any]
    data_raw: dict[str, Any]
    inversion_raw: dict[str, Any]
    analysis_raw: dict[str, Any]
    output_raw: dict[str, Any]

    # Optional typed external-inputs config (None when block is absent from YAML)
    external_inputs: Optional[ExternalInputsConfig] = None
    # Mean annual soil/air temperature (°C).  Used by depth-based freeze-thaw
    # gating to anchor the temperature at depth (Fourier heat conduction
    # placeholder).  None → disables depth-gating (uses layer-level T).
    T_annual_mean_C: float | None = None


# ---------------------------------------------------------------------------
# PoolIndex
# ---------------------------------------------------------------------------


class PoolIndex:
    """
    Bidirectional mapping between pool names and flat-state-vector indices.

    Parameters
    ----------
    config:
        A validated ``ModelConfig``.

    Examples
    --------
    >>> idx = PoolIndex(config)
    >>> idx["leaf"]                       # → 0  (first AG pool)
    >>> idx.ag_slice                      # → slice(0, 4)
    >>> idx.layer_slices["organic"]       # → slice(4, 6) or similar
    >>> idx.pool_names                    # full ordered list of names
    """

    def __init__(self, config: ModelConfig) -> None:
        names: list[str] = []

        # 1. Aboveground pools — in definition order
        for pool in config.aboveground_pools:
            names.append(pool.name)
        self._ag_count: int = len(config.aboveground_pools)

        # 2. Soil pools — layer by layer, SOM first, optional microbial last
        self._layer_slices: dict[str, slice] = {}
        for layer in config.soil_layers:
            start = len(names)
            for som_pool in layer.som_pools:
                names.append(f"{layer.name}_{som_pool.name}")
            if config.microbial_pool_per_layer:
                names.append(f"{layer.name}_mic")
            self._layer_slices[layer.name] = slice(start, len(names))

        self._names: list[str] = names
        self._index: dict[str, int] = {n: i for i, n in enumerate(names)}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> int:
        """Return the integer index for *name*, or raise ``KeyError``."""
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(f"Unknown pool name: {name!r}") from None

    @property
    def ag_slice(self) -> slice:
        """Slice that selects exactly the aboveground pools."""
        return slice(0, self._ag_count)

    @property
    def layer_slices(self) -> dict[str, slice]:
        """
        Mapping from layer name → slice covering all pools in that layer
        (SOM pools + microbial pool if enabled).
        """
        return dict(self._layer_slices)

    @property
    def pool_names(self) -> list[str]:
        """All pool names in canonical order."""
        return list(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        return f"PoolIndex(n_pools={len(self)}, names={self._names!r})"


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------


def _parse_aboveground_pools(
    raw: list[dict[str, Any]],
) -> tuple[AboveGroundPoolDef, ...]:
    return tuple(
        AboveGroundPoolDef(
            name=str(entry["name"]),
            is_woody=bool(entry.get("is_woody", False)),
            litterfall_to=str(entry["litterfall_to"]),
        )
        for entry in raw
    )


def _parse_soil_layers(
    raw: list[dict[str, Any]],
) -> tuple[SoilLayerDef, ...]:
    layers: list[SoilLayerDef] = []
    for entry in raw:
        som_pools = tuple(
            SOMPoolDef(
                name=str(p["name"]),
                tau_prior_days=float(p["tau_prior_days"]),
                tau_prior_std=float(p["tau_prior_std"]),
                depth_top_m=float(p["depth_top_m"]) if "depth_top_m" in p else None,
                depth_bot_m=float(p["depth_bot_m"]) if "depth_bot_m" in p else None,
            )
            for p in entry["som_pools"]
        )
        layers.append(
            SoilLayerDef(
                name=str(entry["name"]),
                depth_top_m=float(entry["depth_top_m"]),
                depth_bot_m=float(entry["depth_bot_m"]),
                permafrost_eligible=bool(entry.get("permafrost_eligible", False)),
                som_pools=som_pools,
            )
        )
    return tuple(layers)


def _parse_transfer_rules(
    raw: list[list[Any]],
) -> tuple[tuple[str, str, float], ...]:
    return tuple((str(row[0]), str(row[1]), float(row[2])) for row in raw)


def _parse_alloc(
    parameters_raw: dict[str, Any],
    ag_pools: tuple[AboveGroundPoolDef, ...],
) -> dict[str, float]:
    """Extract numeric allocation fractions for AG pool names only.

    The YAML alloc block may contain non-numeric keys such as ``optimize``;
    these are silently ignored.
    """
    alloc_raw: dict[str, Any] = parameters_raw.get("alloc", {})
    ag_names = {p.name for p in ag_pools}
    return {k: float(v) for k, v in alloc_raw.items() if k in ag_names}


def _parse_external_inputs(
    raw: Optional[dict[str, Any]],
) -> Optional[ExternalInputsConfig]:
    """Parse the optional external_inputs YAML block into a typed dataclass."""
    if raw is None:
        return None
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return None
    source = str(raw.get("source", "GPP_obs"))
    CUE = float(raw.get("CUE", 0.5))
    optimize_CUE = bool(raw.get("optimize_CUE", False))
    soil_input_fraction = float(raw.get("soil_input_fraction", 0.0))
    optimize_soil_input_fraction = bool(raw.get("optimize_soil_input_fraction", False))
    partition_raw: dict[str, Any] = raw.get("partition", {})
    partition = {k: float(v) for k, v in partition_raw.items()}
    optimize_partition = bool(raw.get("optimize_partition", False))
    target_pool_names = tuple(partition.keys())
    return ExternalInputsConfig(
        enabled=enabled,
        source=source,
        CUE=CUE,
        optimize_CUE=optimize_CUE,
        soil_input_fraction=soil_input_fraction,
        optimize_soil_input_fraction=optimize_soil_input_fraction,
        partition=partition,
        optimize_partition=optimize_partition,
        target_pool_names=target_pool_names,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str) -> ModelConfig:
    """
    Parse a YAML configuration file and return a validated ``ModelConfig``.

    Parameters
    ----------
    path:
        Filesystem path to the ``.yaml`` config file.

    Returns
    -------
    ModelConfig
        Fully validated, frozen configuration object.

    Raises
    ------
    ConfigValidationError
        If any of the following semantic constraints are violated:

        * A ``transfer_rules`` entry references an unknown pool name.
        * The outflow fractions for any source pool sum to > 1.0.
        * Soil layer depths are non-contiguous, overlapping, or
          depth_bot ≤ depth_top for any layer.
        * The ``parameters.alloc`` section omits one or more aboveground
          pool names.

    FileNotFoundError
        If *path* does not exist.
    yaml.YAMLError
        If the file contains invalid YAML.
    """
    with open(path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    site: dict[str, Any] = raw.get("site", {})
    model: dict[str, Any] = raw.get("model", {})
    params: dict[str, Any] = raw.get("parameters", {})

    ag_pools = _parse_aboveground_pools(model.get("aboveground_pools", []))
    soil_layers = _parse_soil_layers(model.get("soil_layers", []))
    transfer_rules = _parse_transfer_rules(model.get("transfer_rules", []))
    alloc = _parse_alloc(params, ag_pools)

    external_inputs = _parse_external_inputs(raw.get("external_inputs"))

    config = ModelConfig(
        # Site
        site_id=str(site.get("id", "")),
        site_name=str(site.get("name", "")),
        lat=float(site.get("lat", 0.0)),
        lon=float(site.get("lon", 0.0)),
        T_annual_mean_C=(
            float(site["T_annual_mean_C"]) if "T_annual_mean_C" in site else None
        ),
        # Model structure
        dt_days=float(model.get("dt_days", 1.0)),
        solver=str(model.get("solver", "euler")),
        spinup_years=int(model.get("spinup_years", 200)),
        enable_14C=bool(model.get("enable_14C", True)),
        microbial_pool_per_layer=bool(model.get("microbial_pool_per_layer", False)),
        # Pools and layers
        aboveground_pools=ag_pools,
        soil_layers=soil_layers,
        transfer_rules=transfer_rules,
        alloc=alloc,
        # Raw sections for downstream use
        parameters_raw=dict(params),
        data_raw=dict(raw.get("data", {})),
        inversion_raw=dict(raw.get("inversion", {})),
        analysis_raw=normalize_analysis_config(raw.get("analysis")),
        output_raw=normalize_output_config(raw.get("output")),
        # Typed external-inputs config
        external_inputs=external_inputs,
    )

    _validate(config)
    return config
