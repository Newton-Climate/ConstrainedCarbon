"""
Tests for src/ecosystem_complexity/transfer.py.

Coverage
--------
build_transfer_matrix
  - Output shape matches n_total_pools for both sites.
  - Row sums are all <= 1.0.
  - Known explicit transfer fractions match the YAML rules.
  - Self-transfer raises ValueError.
  - Row-sum > 1.0 raises ValueError.

get_transfer_matrix
  - Output shape is (n_pools, n_pools).
  - Row sums are all <= 1.0 (respiration column absorbed).
  - jax.grad through get_transfer_matrix produces finite gradients.
  - Consistent with build_transfer_matrix for default ModelParams.

validate_transfer_rules
  - Returns a list (may be empty or contain strings).
  - Sink-pool warning is emitted for a pool with no outgoing rules.
"""

from __future__ import annotations

import pathlib
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ecosystem_complexity.model.configuration import PoolIndex, load_config
from ecosystem_complexity.model.state import make_default_params
from ecosystem_complexity.model.transfers import (
    build_transfer_matrix,
    get_transfer_matrix,
    validate_transfer_rules,
)

CONFIGS_DIR = pathlib.Path(__file__).resolve().parents[2] / "configs"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def harvard_config():
    return load_config(str(CONFIGS_DIR / "harvard_forest.yaml"))


@pytest.fixture(scope="module")
def barrow_config():
    return load_config(str(CONFIGS_DIR / "barrow_alaska.yaml"))


@pytest.fixture(scope="module")
def harvard_matrix(harvard_config):
    return build_transfer_matrix(harvard_config)


@pytest.fixture(scope="module")
def harvard_params(harvard_config):
    return make_default_params(harvard_config)


# ---------------------------------------------------------------------------
# build_transfer_matrix — shape and values
# ---------------------------------------------------------------------------


def test_build_matrix_shape_harvard(harvard_config, harvard_matrix):
    n_pools = len(PoolIndex(harvard_config))
    assert harvard_matrix.shape == (n_pools, n_pools)


def test_build_matrix_shape_barrow(barrow_config):
    matrix = build_transfer_matrix(barrow_config)
    n_pools = len(PoolIndex(barrow_config))
    assert matrix.shape == (n_pools, n_pools)


def test_build_matrix_row_sums_leq_one(harvard_matrix):
    row_sums = np.array(harvard_matrix).sum(axis=1)
    assert np.all(
        row_sums <= 1.0 + 1e-6
    ), f"Row sums exceed 1.0: {row_sums[row_sums > 1.0 + 1e-6]}"


def test_build_matrix_dtype_float32(harvard_matrix):
    assert harvard_matrix.dtype == jnp.float32


def test_build_matrix_no_diagonal(harvard_matrix):
    """No pool should transfer carbon to itself."""
    diag = np.diag(np.array(harvard_matrix))
    assert np.all(diag == 0.0), f"Non-zero diagonal entries: {diag}"


def test_build_matrix_known_fraction(harvard_config, harvard_matrix):
    """
    Spot-check a specific rule from harvard_forest.yaml:
    leaf → organic_litter: 0.95
    """
    idx = PoolIndex(harvard_config)
    i = idx["leaf"]
    j = idx["organic_litter"]
    assert float(harvard_matrix[i, j]) == pytest.approx(0.95, abs=1e-6)


# ---------------------------------------------------------------------------
# build_transfer_matrix — validation errors
# ---------------------------------------------------------------------------


def _config_from_yaml(tmp_path: pathlib.Path, content: str):
    """Write a minimal YAML and return a loaded ModelConfig."""
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(content))
    return load_config(str(p))


_MINIMAL_YAML = """\
site:
  id: "T"
  name: "Test"
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


def test_self_transfer_raises(tmp_path):
    """A rule transferring a pool to itself should raise ValueError."""
    yaml_text = _MINIMAL_YAML.replace(
        "transfer_rules:\n    - [leaf, org_litter, 0.90]",
        "transfer_rules:\n    - [org_litter, org_litter, 0.50]",
    )
    cfg = _config_from_yaml(tmp_path, yaml_text)
    with pytest.raises(ValueError, match="[Ss]elf"):
        build_transfer_matrix(cfg)


def test_row_sum_exceeds_one_raises():
    """
    Fractions from one source summing to > 1.0 should raise ValueError.

    We construct ModelConfig directly (bypassing load_config) because
    load_config's own _check_transfer_sums would raise ConfigValidationError
    first.  build_transfer_matrix must be an independent line of defence.
    """
    from ecosystem_complexity.model.configuration import (
        AboveGroundPoolDef,
        ModelConfig,
        SoilLayerDef,
        SOMPoolDef,
    )

    som = (SOMPoolDef(name="litter", tau_prior_days=365.0, tau_prior_std=100.0),)
    layer = SoilLayerDef(
        name="org",
        depth_top_m=0.0,
        depth_bot_m=0.1,
        permafrost_eligible=False,
        som_pools=som,
    )
    ag = (AboveGroundPoolDef(name="leaf", is_woody=False, litterfall_to="org_litter"),)
    # Two rules for the same source → total fraction = 1.20
    bad_rules: tuple[tuple[str, str, float], ...] = (
        ("leaf", "org_litter", 0.70),
        ("leaf", "org_litter", 0.50),
    )
    cfg = ModelConfig(
        site_id="T",
        site_name="Test",
        lat=0.0,
        lon=0.0,
        dt_days=1.0,
        solver="euler",
        spinup_years=10,
        enable_14C=False,
        microbial_pool_per_layer=False,
        aboveground_pools=ag,
        soil_layers=(layer,),
        transfer_rules=bad_rules,
        alloc={"leaf": 1.0},
        parameters_raw={},
        data_raw={},
        inversion_raw={},
        analysis_raw={},
        output_raw={},
    )
    with pytest.raises(ValueError, match="[Rr]ow|sum|exceed"):
        build_transfer_matrix(cfg)


# ---------------------------------------------------------------------------
# get_transfer_matrix — shape, row sums, differentiability
# ---------------------------------------------------------------------------


def test_get_matrix_shape(harvard_config, harvard_params):
    n_pools = len(PoolIndex(harvard_config))
    f_mat = get_transfer_matrix(harvard_params.log_f_transfer, n_pools)
    assert f_mat.shape == (n_pools, n_pools)


def test_get_matrix_row_sums_leq_one(harvard_config, harvard_params):
    n_pools = len(PoolIndex(harvard_config))
    f_mat = get_transfer_matrix(harvard_params.log_f_transfer, n_pools)
    row_sums = f_mat.sum(axis=-1)
    assert jnp.all(
        row_sums <= 1.0 + 1e-5
    ), f"get_transfer_matrix row sums exceed 1.0: {row_sums}"


def test_get_matrix_grad_finite(harvard_config, harvard_params):
    """jax.grad must flow through get_transfer_matrix without NaN or inf."""
    n_pools = len(PoolIndex(harvard_config))

    def loss(log_f: jnp.ndarray) -> jnp.ndarray:
        return get_transfer_matrix(log_f, n_pools).sum()

    grad = jax.grad(loss)(harvard_params.log_f_transfer)
    assert jnp.all(jnp.isfinite(grad)), "Gradient contains NaN or inf"


def test_get_matrix_consistent_with_build(harvard_config, harvard_params):
    """
    Entries that are zero in build_transfer_matrix should be near-zero in
    get_transfer_matrix (they receive only the _EPS floor, which is tiny
    after normalisation).
    """
    idx = PoolIndex(harvard_config)
    n_pools = len(idx)
    f_build = np.array(build_transfer_matrix(harvard_config))
    f_get = np.array(get_transfer_matrix(harvard_params.log_f_transfer, n_pools))

    # For every (i, j) where build gives a meaningful fraction (> 0.01),
    # get_transfer_matrix should be in the same ballpark.
    for i in range(n_pools):
        for j in range(n_pools):
            if f_build[i, j] > 0.01:
                assert f_get[i, j] == pytest.approx(f_build[i, j], rel=0.05), (
                    f"Mismatch at ({idx.pool_names[i]}, {idx.pool_names[j]}): "
                    f"build={f_build[i, j]:.4f}, get={f_get[i, j]:.4f}"
                )


# ---------------------------------------------------------------------------
# validate_transfer_rules
# ---------------------------------------------------------------------------


def test_validate_returns_list(harvard_config):
    warnings = validate_transfer_rules(harvard_config)
    assert isinstance(warnings, list)
    assert all(isinstance(w, str) for w in warnings)


def test_validate_sink_pool_warning(tmp_path):
    """A pool with no outgoing transfers should produce a sink-pool warning."""
    # org_litter has no outgoing rule in this minimal config
    yaml_text = _MINIMAL_YAML  # only leaf → org_litter; org_litter has no outgoing
    cfg = _config_from_yaml(tmp_path, yaml_text)
    warnings = validate_transfer_rules(cfg)
    warning_text = " ".join(warnings)
    assert "org_litter" in warning_text or any("org_litter" in w for w in warnings)
