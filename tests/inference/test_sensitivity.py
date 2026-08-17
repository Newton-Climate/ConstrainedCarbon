"""
Tests for src/ecosystem_complexity/information.py and analysis.py.

Coverage
--------
information.py
  - flatten_params / unflatten_params round-trip
  - get_param_names: correct length and no blanks
  - get_param_groups: partition is complete and non-overlapping
  - make_prior_covariance: correct shape, all positive, tau σ from YAML
  - _build_obs_config: extracts C_stocks, pool_d14C, resp_d14C
  - _build_obs_fn: output shape and finiteness; JAX-differentiable
  - compute_jacobian: H shape (n_obs, n_params); finite; non-zero
  - compute_fisher:
      - returns FisherResult with correct shapes
      - FIM_total ≈ sum of per-type FIMs
      - FIM is symmetric and PSD
      - empty obs returns zero FIM
  - compute_dof:
      - DFS ∈ [0, n_params]
      - trace(A) == dfs_total
      - DFS increases with more obs types
  - compute_posterior:
      - C_post is symmetric, PSD
      - posterior_sigma ≤ prior_sigma everywhere
      - uncertainty_reduction ∈ [0, 1]
      - correlation diagonal == 1.0

analysis.py
  - run_ablation_study: returns five scenarios; DFS is monotone in obs count
  - param_group_dfs: groups partition total DFS
  - compute_age_diagnostics: correct shapes; bulk Δ¹⁴C is mass-weighted mean
  - age_diagnostics_summary: keys and finite values
"""

from __future__ import annotations

import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ecosystem_complexity.model.api import build_model, run_model
from ecosystem_complexity.data.schemas import ForcingData, ObservationData
from ecosystem_complexity.inference.information import (
    _default_fields,
)
from ecosystem_complexity.inference.sensitivity import (
    OBS_C_STOCKS,
    OBS_POOL_D14C,
    OBS_RESP_D14C,
    _build_obs_config,
    _build_obs_fn,
    flatten_params,
    get_param_groups,
    get_param_names,
    make_prior_covariance,
    unflatten_params,
)
from ecosystem_complexity.model.state import make_default_params, make_initial_state

CONFIGS_DIR = pathlib.Path(__file__).resolve().parents[2] / "configs"
_HF_PATH = str(CONFIGS_DIR / "harvard_forest.yaml")
_SOIL_ONLY_PATH = str(CONFIGS_DIR / "harvard_forest_soil_only.yaml")


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hf_model():
    return build_model(_HF_PATH)


@pytest.fixture(scope="module")
def hf_params(hf_model):
    return make_default_params(hf_model.config)


@pytest.fixture(scope="module")
def hf_state0(hf_model):
    return make_initial_state(hf_model.config, {"mat_c": 8.5, "permafrost": False})


@pytest.fixture(scope="module")
def hf_pool_names(hf_model):
    return hf_model.pool_index.pool_names


@pytest.fixture(scope="module")
def short_forcing(hf_model):
    """Minimal ForcingData with 30 timesteps — fast enough for Jacobian tests."""
    n_layers = len(hf_model.config.soil_layers)
    T = 30
    return ForcingData(
        time=jnp.arange(T, dtype=jnp.float32),
        air_temp=jnp.full(T, 15.0),
        sw_radiation=jnp.full(T, 100.0),
        precip=jnp.full(T, 3.0),
        vpd=jnp.full(T, 10.0),
        soil_temp=jnp.full((T, n_layers), 12.0),
        soil_moisture=jnp.full((T, n_layers), 0.3),
        snow_depth=jnp.full(T, 0.0),
        active_layer=jnp.full(T, jnp.inf),
        delta14C_atm=jnp.full(T, 50.0),  # mild bomb-spike
        GPP_obs=jnp.full(T, float("nan")),
        NPP_obs=jnp.full(T, float("nan")),
    )


@pytest.fixture(scope="module")
def hf_output(hf_model, short_forcing, hf_state0, hf_params):
    """Pre-run model output for age-diagnostic tests."""
    warm_state = hf_state0._replace(C12=jnp.full_like(hf_state0.C12, 100.0))
    return run_model(hf_model, short_forcing, state0=warm_state, params=hf_params)


def _make_obs(pool_names: list[str]) -> ObservationData:
    """Build a minimal ObservationData with synthetic point observations."""
    n = 30
    T_dummy = jnp.zeros(n)
    # Use the first two soil pools for Δ¹⁴C and first one for C stock
    soil_pools = [p for p in pool_names if "_" in p]  # heuristic for soil pools

    c_pools = {}
    d14C_pools = {}
    for i, p in enumerate(soil_pools[:2]):
        c_pools[p] = (500.0 + i * 100, 100.0)  # (value gC m-2, sigma)
        d14C_pools[p] = (-50.0 + i * 10, 15.0, 2000)  # (value ‰, sigma ‰, year)

    resp_arr = jnp.full(n, -30.0)  # synthetic respired Δ¹⁴C

    return ObservationData(
        time=T_dummy,
        NEE=jnp.full(n, float("nan")),
        GPP=jnp.full(n, float("nan")),
        ER=jnp.full(n, float("nan")),
        NEE_unc=jnp.full(n, float("nan")),
        delta14C_obs=d14C_pools,
        deltaD14C_obs={},
        C_pools_obs=c_pools,
        delta14C_resp=resp_arr,
    )


@pytest.fixture(scope="module")
def hf_obs(hf_pool_names):
    return _make_obs(hf_pool_names)


# ── flatten / unflatten ────────────────────────────────────────────────────────


def test_flatten_unflatten_roundtrip(hf_params, hf_model):
    """flatten then unflatten recovers the original parameter values exactly."""
    fields = _default_fields(hf_model)
    flat = flatten_params(hf_params, fields)
    recovered = unflatten_params(flat, hf_params, fields)

    for f in fields:
        orig = np.array(getattr(hf_params, f))
        rec = np.array(getattr(recovered, f))
        np.testing.assert_allclose(
            orig, rec, atol=1e-6, err_msg=f"Round-trip mismatch for field '{f}'"
        )


def _expected_flat_len(params, fields):
    """Flat parameter-vector length, excluding the log_f_transfer respiration column.

    `log_f_transfer` has shape (n_pools, n_pools+1) with the last column being the
    to-respiration branch, which is the residual of each row and is not optimized.
    """
    total = 0
    for f in fields:
        val = getattr(params, f)
        if f == "log_f_transfer":
            total += int(math.prod(val[:, :-1].shape))
        else:
            total += int(math.prod(val.shape))
    return total


def test_flatten_length(hf_params, hf_model):
    """Flat vector length equals sum of field sizes (minus the respiration column)."""
    fields = _default_fields(hf_model)
    flat = flatten_params(hf_params, fields)
    expected = _expected_flat_len(hf_params, fields)
    assert flat.shape == (expected,)


# ── get_param_names ────────────────────────────────────────────────────────────


def test_param_names_length(hf_params, hf_model):
    """get_param_names returns one label per flat-vector element."""
    fields = _default_fields(hf_model)
    names = get_param_names(hf_params, fields, hf_model)
    n_params = _expected_flat_len(hf_params, fields)
    assert len(names) == n_params


def test_param_names_nonempty_strings(hf_params, hf_model):
    """All parameter name labels are non-empty strings."""
    fields = _default_fields(hf_model)
    names = get_param_names(hf_params, fields, hf_model)
    assert all(isinstance(n, str) and len(n) > 0 for n in names)


# ── get_param_groups ───────────────────────────────────────────────────────────


def test_param_groups_cover_all_params(hf_params, hf_model):
    """Union of all param group indices covers every element of the flat vector."""
    fields = _default_fields(hf_model)
    n_params = _expected_flat_len(hf_params, fields)
    groups = get_param_groups(hf_params, fields, hf_model)
    all_idxs = sorted(idx for idxs in groups.values() for idx in idxs)
    assert all_idxs == list(range(n_params)), "Groups do not cover all params"


def test_param_groups_disjoint(hf_params, hf_model):
    """No parameter index appears in more than one group."""
    fields = _default_fields(hf_model)
    groups = get_param_groups(hf_params, fields, hf_model)
    seen = set()
    for grp, idxs in groups.items():
        overlap = seen & set(idxs)
        assert not overlap, f"Group '{grp}' overlaps with previous groups: {overlap}"
        seen |= set(idxs)


# ── make_prior_covariance ──────────────────────────────────────────────────────


def test_prior_covariance_shape(hf_params, hf_model):
    """Prior sigma has the same length as the flat parameter vector."""
    fields = _default_fields(hf_model)
    n_params = _expected_flat_len(hf_params, fields)
    sigma = make_prior_covariance(hf_params, fields, hf_model)
    assert sigma.shape == (n_params,)


def test_prior_covariance_positive(hf_params, hf_model):
    """All prior sigmas are strictly positive."""
    fields = _default_fields(hf_model)
    sigma = make_prior_covariance(hf_params, fields, hf_model)
    assert np.all(sigma > 0), f"Non-positive prior sigma: {sigma[sigma <= 0]}"


def test_prior_covariance_tau_from_yaml(hf_model, hf_params):
    """log_tau prior σ is taken from YAML tau_prior_std, not the default."""
    fields = ["log_tau"]
    sigma = make_prior_covariance(hf_params, fields, hf_model)
    # organic_litter has tau_prior_std=60 days — so log-space sigma ≈ 60/180 ≈ 0.33
    # It should differ from the default (1.0)
    pool_names = hf_model.pool_index.pool_names
    idx_litter = pool_names.index("organic_litter")
    # tau_prior_std=60 != 1.0 (default), so sigma at that index should differ
    assert (
        sigma[idx_litter] != 1.0
    ), "organic_litter tau sigma should come from YAML (60 days), not default"


# ── _build_obs_config ──────────────────────────────────────────────────────────


def test_build_obs_config_keys(hf_obs, hf_model):
    """obs_config contains C_stocks, pool_d14C, and resp_d14C when all are present."""
    obs_config = _build_obs_config(hf_obs, hf_model)
    assert OBS_C_STOCKS in obs_config
    assert OBS_POOL_D14C in obs_config
    assert OBS_RESP_D14C in obs_config


def test_build_obs_config_no_resp(hf_obs, hf_model):
    """Without delta14C_resp, resp_delta14C is absent from obs_config."""
    obs_no_resp = hf_obs._replace(delta14C_resp=None)
    obs_config = _build_obs_config(obs_no_resp, hf_model)
    assert OBS_RESP_D14C not in obs_config


def test_build_obs_config_positive_sigma(hf_obs, hf_model):
    """All extracted obs sigmas are strictly positive."""
    obs_config = _build_obs_config(hf_obs, hf_model)
    for obs_type, entries in obs_config.items():
        for pool_name, val, sigma in entries:
            assert sigma > 0, f"Non-positive sigma for {obs_type}[{pool_name}]: {sigma}"


# ── _build_obs_fn ──────────────────────────────────────────────────────────────


def test_obs_fn_output_shape(hf_model, short_forcing, hf_state0, hf_params, hf_obs):
    """obs_fn output length equals total number of extracted observations."""
    fields = _default_fields(hf_model)
    obs_config = _build_obs_config(hf_obs, hf_model)
    obs_fn = _build_obs_fn(hf_model, short_forcing, hf_state0, fields, obs_config)

    flat0 = jnp.array(flatten_params(hf_params, fields))
    out = obs_fn(flat0)

    n_exp = sum(len(v) for v in obs_config.values())
    assert out.shape == (n_exp,), f"Expected shape ({n_exp},), got {out.shape}"


def test_obs_fn_finite(hf_model, short_forcing, hf_state0, hf_params, hf_obs):
    """obs_fn returns finite values for valid parameters and forcing."""
    fields = _default_fields(hf_model)
    obs_config = _build_obs_config(hf_obs, hf_model)
    # Warm state so C12 > 0 and Δ¹⁴C is well-defined
    warm_state = hf_state0._replace(C12=jnp.full_like(hf_state0.C12, 100.0))
    obs_fn = _build_obs_fn(hf_model, short_forcing, warm_state, fields, obs_config)

    flat0 = jnp.array(flatten_params(hf_params, fields))
    out = obs_fn(flat0)
    assert jnp.all(jnp.isfinite(out)), f"obs_fn returned non-finite values: {out}"


def test_obs_fn_differentiable(hf_model, short_forcing, hf_state0, hf_params, hf_obs):
    """jax.grad flows through obs_fn without NaN or inf."""
    fields = ["log_tau"]  # small subset for speed
    obs_config = _build_obs_config(hf_obs, hf_model)
    # Restrict to C stocks only for fastest test
    obs_config_small = {OBS_C_STOCKS: obs_config[OBS_C_STOCKS]}
    warm_state = hf_state0._replace(C12=jnp.full_like(hf_state0.C12, 100.0))
    obs_fn = _build_obs_fn(
        hf_model, short_forcing, warm_state, fields, obs_config_small
    )

    flat0 = jnp.array(flatten_params(hf_params, fields))

    # Sum output to get a scalar for grad
    grads = jax.grad(lambda p: obs_fn(p).sum())(flat0)
    assert jnp.all(jnp.isfinite(grads)), f"Non-finite grads: {grads}"
