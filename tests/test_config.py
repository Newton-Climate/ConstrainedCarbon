"""
Tests for src/ecosystem_complexity/config.py.

Coverage
--------
* harvard_forest.yaml and barrow_alaska.yaml load without exception.
* ConfigValidationError is raised for:
    - a transfer rule whose fractions for one source sum to > 1.0
    - a transfer rule referencing an unknown pool name
    - soil layers whose depth_top / depth_bot values overlap
* PoolIndex contract:
    - every pool name maps to a unique integer in [0, n_total_pools)
    - ag_slice covers exactly the aboveground pools
"""
from __future__ import annotations

import pathlib
import textwrap

import pytest

from ecosystem_complexity.config import (
    ConfigValidationError,
    ModelConfig,
    PoolIndex,
    load_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def _write_yaml(tmp_path: pathlib.Path, content: str) -> str:
    """Write a YAML string to a temp file and return its path."""
    p = tmp_path / "test_config.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


# Minimal valid YAML snippet shared by several invalid-config tests so that
# only the section under test is broken.
_VALID_BASE = """\
site:
  id: "TEST"
  name: "Test Site"
  lat: 0.0
  lon: 0.0

model:
  dt_days: 1.0
  solver: euler
  spinup_years: 10
  enable_14C: false
  microbial_pool_per_layer: false

  aboveground_pools:
    - name: leaf
      is_woody: false
      litterfall_to: org_litter

  soil_layers:
    - name: org
      depth_top_m: 0.00
      depth_bot_m: 0.10
      permafrost_eligible: false
      som_pools:
        - name: litter
          tau_prior_days: 365
          tau_prior_std: 100

  transfer_rules:
    - [leaf, org_litter, 0.90]

parameters:
  alloc:
    leaf: 1.0
"""


# ---------------------------------------------------------------------------
# Happy-path: real config files
# ---------------------------------------------------------------------------


def test_harvard_forest_loads() -> None:
    """configs/harvard_forest.yaml parses and validates without exception."""
    cfg = load_config(str(CONFIGS_DIR / "harvard_forest.yaml"))
    assert isinstance(cfg, ModelConfig)
    assert cfg.site_id == "US-Ha1"
    assert len(cfg.aboveground_pools) > 0
    assert len(cfg.soil_layers) > 0


def test_barrow_alaska_loads() -> None:
    """configs/barrow_alaska.yaml parses and validates without exception."""
    cfg = load_config(str(CONFIGS_DIR / "barrow_alaska.yaml"))
    assert isinstance(cfg, ModelConfig)
    assert cfg.site_id == "US-Brw"
    assert len(cfg.aboveground_pools) > 0
    assert len(cfg.soil_layers) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_bad_transfer_sum_raises(tmp_path: pathlib.Path) -> None:
    """Fractions from one source summing to > 1.0 raises ConfigValidationError."""
    yaml_text = _VALID_BASE.replace(
        "transfer_rules:\n    - [leaf, org_litter, 0.90]",
        # leaf → org_litter: 0.70 + 0.50 = 1.20 → exceeds 1.0
        (
            "transfer_rules:\n"
            "    - [leaf, org_litter, 0.70]\n"
            "    - [leaf, org_litter, 0.50]"
        ),
    )
    with pytest.raises(ConfigValidationError, match="sum"):
        load_config(_write_yaml(tmp_path, yaml_text))


def test_unknown_pool_in_transfer_raises(tmp_path: pathlib.Path) -> None:
    """A transfer rule referencing a non-existent pool raises ConfigValidationError."""
    yaml_text = _VALID_BASE.replace(
        "transfer_rules:\n    - [leaf, org_litter, 0.90]",
        "transfer_rules:\n    - [leaf, DOES_NOT_EXIST, 0.90]",
    )
    with pytest.raises(ConfigValidationError, match="unknown"):
        load_config(_write_yaml(tmp_path, yaml_text))


def test_overlapping_layer_depths_raises(tmp_path: pathlib.Path) -> None:
    """Overlapping soil layer depths raise ConfigValidationError."""
    # Two layers whose depth ranges overlap: first ends at 0.10, second starts at 0.05
    yaml_text = _VALID_BASE.replace(
        """\
  soil_layers:
    - name: org
      depth_top_m: 0.00
      depth_bot_m: 0.10
      permafrost_eligible: false
      som_pools:
        - name: litter
          tau_prior_days: 365
          tau_prior_std: 100""",
        """\
  soil_layers:
    - name: org
      depth_top_m: 0.00
      depth_bot_m: 0.10
      permafrost_eligible: false
      som_pools:
        - name: litter
          tau_prior_days: 365
          tau_prior_std: 100
    - name: mineral
      depth_top_m: 0.05
      depth_bot_m: 0.30
      permafrost_eligible: false
      som_pools:
        - name: fast
          tau_prior_days: 1000
          tau_prior_std: 300""",
    )
    with pytest.raises(ConfigValidationError, match="[Oo]verlap|[Gg]ap|contiguous"):
        load_config(_write_yaml(tmp_path, yaml_text))


# ---------------------------------------------------------------------------
# PoolIndex contract — Harvard Forest config
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def harvard_config() -> ModelConfig:
    return load_config(str(CONFIGS_DIR / "harvard_forest.yaml"))


@pytest.fixture(scope="module")
def harvard_index(harvard_config: ModelConfig) -> PoolIndex:
    return PoolIndex(harvard_config)


def test_pool_index_unique_integers(
    harvard_index: PoolIndex,
) -> None:
    """Every pool name maps to a unique integer in [0, n_total_pools)."""
    names = harvard_index.pool_names
    n = len(names)
    indices = [harvard_index[name] for name in names]

    # All unique
    assert len(set(indices)) == n, "Pool indices are not unique"
    # All in [0, n)
    assert all(0 <= idx < n for idx in indices), (
        "Some pool index is out of range [0, n_total_pools)"
    )


def test_pool_index_ag_slice(
    harvard_config: ModelConfig,
    harvard_index: PoolIndex,
) -> None:
    """ag_slice covers exactly the aboveground pool names, in definition order."""
    ag_names_from_config = [p.name for p in harvard_config.aboveground_pools]
    all_names = harvard_index.pool_names
    ag_names_from_slice = all_names[harvard_index.ag_slice]

    assert ag_names_from_slice == ag_names_from_config, (
        f"ag_slice gives {ag_names_from_slice}, "
        f"expected {ag_names_from_config}"
    )


def test_pool_index_layer_slices_cover_all_soil_pools(
    harvard_config: ModelConfig,
    harvard_index: PoolIndex,
) -> None:
    """layer_slices together cover every non-AG pool exactly once."""
    n_ag = len(harvard_config.aboveground_pools)
    all_names = harvard_index.pool_names
    soil_names_via_slices: list[str] = []
    for layer in harvard_config.soil_layers:
        sl = harvard_index.layer_slices[layer.name]
        soil_names_via_slices.extend(all_names[sl])

    expected_soil_names = all_names[n_ag:]
    assert soil_names_via_slices == expected_soil_names


# ---------------------------------------------------------------------------
# Integration: n_total_pools matches known pool structure (both sites)
# ---------------------------------------------------------------------------
#
# Counts derived from the YAML structure (microbial_pool_per_layer: false):
#
#   Harvard Forest  — 4 AG (leaf, wood, branch, root)
#                   + organic(litter, fast)     = 2
#                   + mineral_A(fast, slow)      = 2
#                   + mineral_B(slow, passive)   = 2
#                   ──────────────────────────────
#                   = 10 total pools
#
#   Barrow Alaska   — 2 AG (leaf, root)
#                   + organic(litter, fast)      = 2
#                   + active(fast, slow)         = 2
#                   + permafrost(slow, passive)  = 2
#                   ──────────────────────────────
#                   = 8 total pools
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def barrow_config() -> ModelConfig:
    return load_config(str(CONFIGS_DIR / "barrow_alaska.yaml"))


@pytest.fixture(scope="module")
def barrow_index(barrow_config: ModelConfig) -> PoolIndex:
    return PoolIndex(barrow_config)


def test_harvard_n_total_pools(harvard_config: ModelConfig) -> None:
    """
    Harvard Forest pool count matches hand-counted YAML structure.

    4 AG + 3 layers × 2 SOM pools (microbial_pool_per_layer=False) = 10.
    """
    idx = PoolIndex(harvard_config)
    assert len(idx) == 10, (
        f"Expected 10 pools for Harvard Forest, got {len(idx)}. "
        f"Pool names: {idx.pool_names}"
    )


def test_barrow_n_total_pools(barrow_config: ModelConfig) -> None:
    """
    Barrow Alaska pool count matches hand-counted YAML structure.

    2 AG + 3 layers × 2 SOM pools (microbial_pool_per_layer=False) = 8.
    """
    idx = PoolIndex(barrow_config)
    assert len(idx) == 8, (
        f"Expected 8 pools for Barrow Alaska, got {len(idx)}. "
        f"Pool names: {idx.pool_names}"
    )


def test_pool_names_no_duplicates_harvard(harvard_index: PoolIndex) -> None:
    """pool_names must not contain duplicate strings (Harvard Forest)."""
    names = harvard_index.pool_names
    assert len(names) == len(set(names)), (
        f"Duplicate pool names detected: "
        f"{[n for n in names if names.count(n) > 1]}"
    )


def test_pool_names_no_duplicates_barrow(barrow_index: PoolIndex) -> None:
    """pool_names must not contain duplicate strings (Barrow Alaska)."""
    names = barrow_index.pool_names
    assert len(names) == len(set(names)), (
        f"Duplicate pool names detected: "
        f"{[n for n in names if names.count(n) > 1]}"
    )


def test_layer_slices_keys_match_yaml_harvard(
    harvard_config: ModelConfig, harvard_index: PoolIndex
) -> None:
    """layer_slices keys match the YAML soil layer names in definition order."""
    yaml_layer_names = [layer.name for layer in harvard_config.soil_layers]
    slice_keys = list(harvard_index.layer_slices.keys())
    assert slice_keys == yaml_layer_names, (
        f"layer_slices keys {slice_keys} != YAML layer names {yaml_layer_names}"
    )


def test_layer_slices_keys_match_yaml_barrow(
    barrow_config: ModelConfig, barrow_index: PoolIndex
) -> None:
    """layer_slices keys match the YAML soil layer names in definition order."""
    yaml_layer_names = [layer.name for layer in barrow_config.soil_layers]
    slice_keys = list(barrow_index.layer_slices.keys())
    assert slice_keys == yaml_layer_names, (
        f"layer_slices keys {slice_keys} != YAML layer names {yaml_layer_names}"
    )


# ---------------------------------------------------------------------------
# external_inputs — validation and soil-only mode
# ---------------------------------------------------------------------------

_SOIL_ONLY_PATH = str(CONFIGS_DIR / "harvard_forest_soil_only.yaml")


def test_soil_only_model_builds_without_aboveground_pools() -> None:
    """
    configs/harvard_forest_soil_only.yaml (aboveground_pools=[]) loads and
    validates without error; external_inputs config is set.
    """
    cfg = load_config(_SOIL_ONLY_PATH)
    assert isinstance(cfg, ModelConfig)
    assert len(cfg.aboveground_pools) == 0, "Soil-only config should have no AG pools"
    assert cfg.external_inputs is not None, "external_inputs block should be parsed"
    assert cfg.external_inputs.enabled is True


def test_external_inputs_config_fields_correct() -> None:
    """ExternalInputsConfig values match what the YAML specifies."""
    cfg = load_config(_SOIL_ONLY_PATH)
    ext = cfg.external_inputs
    assert ext.source == "GPP_obs"
    assert ext.CUE == pytest.approx(0.47, rel=1e-5)
    assert ext.soil_input_fraction == pytest.approx(1.0, rel=1e-5)
    assert ext.optimize_partition is True
    assert "organic_litter" in ext.target_pool_names
    assert "mineral_B_passive" not in ext.target_pool_names  # not in partition


def test_external_inputs_config_validation_bad_pool_name(tmp_path: pathlib.Path) -> None:
    """A partition entry referencing a non-existent pool raises ConfigValidationError."""
    yaml_text = """\
site:
  id: "TEST"
  name: "Test"
  lat: 0.0
  lon: 0.0

model:
  dt_days: 1.0
  solver: euler
  spinup_years: 10
  enable_14C: false
  microbial_pool_per_layer: false
  aboveground_pools: []

  soil_layers:
    - name: org
      depth_top_m: 0.00
      depth_bot_m: 0.10
      permafrost_eligible: false
      som_pools:
        - name: litter
          tau_prior_days: 365
          tau_prior_std: 100

  transfer_rules: []

parameters:
  alloc: {}

external_inputs:
  enabled: true
  source: GPP_obs
  CUE: 0.5
  optimize_CUE: false
  soil_input_fraction: 0.8
  optimize_soil_input_fraction: false
  partition:
    DOES_NOT_EXIST: 1.0
  optimize_partition: false
"""
    with pytest.raises(ConfigValidationError, match="[Uu]nknown|[Ii]nvalid|not found|not a soil"):
        load_config(_write_yaml(tmp_path, yaml_text))


def test_external_inputs_partition_sums_check(tmp_path: pathlib.Path) -> None:
    """A partition with all-zero fractions raises ConfigValidationError."""
    yaml_text = """\
site:
  id: "TEST"
  name: "Test"
  lat: 0.0
  lon: 0.0

model:
  dt_days: 1.0
  solver: euler
  spinup_years: 10
  enable_14C: false
  microbial_pool_per_layer: false
  aboveground_pools: []

  soil_layers:
    - name: org
      depth_top_m: 0.00
      depth_bot_m: 0.10
      permafrost_eligible: false
      som_pools:
        - name: litter
          tau_prior_days: 365
          tau_prior_std: 100

  transfer_rules: []

parameters:
  alloc: {}

external_inputs:
  enabled: true
  source: GPP_obs
  CUE: 0.5
  optimize_CUE: false
  soil_input_fraction: 0.8
  optimize_soil_input_fraction: false
  partition:
    org_litter: 0.0
  optimize_partition: false
"""
    with pytest.raises(ConfigValidationError, match="[Pp]artition|[Ss]um|zero"):
        load_config(_write_yaml(tmp_path, yaml_text))


def test_external_inputs_absent_has_no_effect() -> None:
    """When external_inputs block is absent, ModelConfig.external_inputs is None."""
    cfg = load_config(str(CONFIGS_DIR / "harvard_forest.yaml"))
    assert cfg.external_inputs is None, (
        "harvard_forest.yaml has no external_inputs block — field should be None"
    )
