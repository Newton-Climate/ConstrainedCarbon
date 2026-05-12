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

import math
from dataclasses import dataclass
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConfigValidationError(ValueError):
    """Raised when a config file fails semantic validation."""


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


@dataclass(frozen=True)
class SoilLayerDef:
    """A soil layer and the SOM pools it contains."""

    name: str
    depth_top_m: float
    depth_bot_m: float
    permafrost_eligible: bool
    som_pools: tuple[SOMPoolDef, ...]


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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


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


def _validate(config: ModelConfig) -> None:
    """
    Raise ``ConfigValidationError`` if any semantic constraint is violated.

    Checks (in order):
    1. transfer_rules reference only known pool names.
    2. Outflow fractions from any single source pool sum to ≤ 1.0.
    3. Soil layer depths are strictly monotonic and contiguous.
    4. NPP alloc fractions exist for every aboveground pool.
    """
    valid_names = _all_valid_pool_names(config)

    # ── 1. Unknown pool names in transfer_rules ───────────────────────────
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

    # ── 2. Transfer row sums > 1.0 ────────────────────────────────────────
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

    # ── 3. Overlapping or non-monotonic layer depths ──────────────────────
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

    # ── 4. Alloc fractions must cover every AG pool ───────────────────────
    ag_names = {p.name for p in config.aboveground_pools}
    missing = ag_names - set(config.alloc.keys())
    if missing:
        raise ConfigValidationError(
            f"NPP alloc fractions are missing for aboveground pool(s): "
            f"{sorted(missing)}.  Every aboveground pool must have an entry "
            f"under parameters.alloc in the YAML."
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

    config = ModelConfig(
        # Site
        site_id=str(site.get("id", "")),
        site_name=str(site.get("name", "")),
        lat=float(site.get("lat", 0.0)),
        lon=float(site.get("lon", 0.0)),
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
        analysis_raw=dict(raw.get("analysis", {})),
        output_raw=dict(raw.get("output", {})),
    )

    _validate(config)
    return config
