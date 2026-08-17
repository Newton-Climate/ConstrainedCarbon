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
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ecosystem_complexity.inference._helpers import _build_sa_diag, build_oe_prior_sigma
from ecosystem_complexity.synthesis.analysis import (
    age_diagnostics_summary,
    compute_age_diagnostics,
    param_group_dfs,
    run_ablation_study,
)
from ecosystem_complexity.model.api import build_model, run_model
from ecosystem_complexity.data.schemas import ForcingData, ObservationData
from ecosystem_complexity.inference.information import (
    _default_fields,
    compute_dof,
    compute_fisher,
    compute_posterior,
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
_HF_3POOL_PATH = str(CONFIGS_DIR / "harvard_3pool_config.yaml")


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hf_model():
    return build_model(_HF_PATH)


@pytest.fixture(scope="module")
def hf_3pool_model():
    return build_model(_HF_3POOL_PATH)


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


def test_flatten_length(hf_params, hf_model):
    """Flat vector length equals sum of field sizes."""
    fields = _default_fields(hf_model)
    flat = flatten_params(hf_params, fields)
    expected = 0
    for f in fields:
        val = getattr(hf_params, f)
        expected += (
            int(math.prod(val[:, :-1].shape))
            if f == "log_f_transfer"
            else int(math.prod(val.shape))
        )
    assert flat.shape == (expected,)


# ── get_param_names ────────────────────────────────────────────────────────────


def test_param_names_length(hf_params, hf_model):
    """get_param_names returns one label per flat-vector element."""
    fields = _default_fields(hf_model)
    names = get_param_names(hf_params, fields, hf_model)
    n_params = flat_len = len(flatten_params(hf_params, fields))
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
    n_params = len(flatten_params(hf_params, fields))
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
    n_params = len(flatten_params(hf_params, fields))
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
    pool_names = hf_model.pool_index.pool_names
    idx_litter = pool_names.index("organic_litter")
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
    warm_state = hf_state0._replace(C12=jnp.full_like(hf_state0.C12, 100.0))
    obs_fn = _build_obs_fn(hf_model, short_forcing, warm_state, fields, obs_config)

    flat0 = jnp.array(flatten_params(hf_params, fields))
    out = obs_fn(flat0)
    assert jnp.all(jnp.isfinite(out)), f"obs_fn returned non-finite values: {out}"


def test_obs_fn_differentiable(hf_model, short_forcing, hf_state0, hf_params, hf_obs):
    """jax.grad flows through obs_fn without NaN or inf."""
    fields = ["log_tau"]
    obs_config = _build_obs_config(hf_obs, hf_model)
    obs_config_small = {OBS_C_STOCKS: obs_config[OBS_C_STOCKS]}
    warm_state = hf_state0._replace(C12=jnp.full_like(hf_state0.C12, 100.0))
    obs_fn = _build_obs_fn(
        hf_model, short_forcing, warm_state, fields, obs_config_small
    )

    flat0 = jnp.array(flatten_params(hf_params, fields))
    grads = jax.grad(lambda p: obs_fn(p).sum())(flat0)
    assert jnp.all(jnp.isfinite(grads)), f"Non-finite grads: {grads}"


# ── compute_fisher ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hf_fisher(hf_model, short_forcing, hf_state0, hf_params, hf_obs):
    warm_state = hf_state0._replace(C12=jnp.full_like(hf_state0.C12, 100.0))
    return compute_fisher(
        hf_model,
        short_forcing,
        warm_state,
        hf_params,
        hf_obs,
        fields=["log_tau"],  # small subset so tests stay fast
    )


def test_fisher_fim_shape(hf_fisher, hf_model, hf_params):
    n_params = len(hf_model.pool_index.pool_names)  # == len(log_tau)
    assert hf_fisher.FIM_total.shape == (n_params, n_params)


def test_fisher_fim_symmetric(hf_fisher):
    diff = np.abs(hf_fisher.FIM_total - hf_fisher.FIM_total.T)
    assert np.max(diff) < 1e-6, f"FIM not symmetric, max|A-Aᵀ| = {np.max(diff):.2e}"


def test_fisher_fim_psd(hf_fisher):
    eigvals = np.linalg.eigvalsh(hf_fisher.FIM_total)
    assert np.all(
        eigvals >= -1e-6
    ), f"FIM has negative eigenvalues: {eigvals.min():.2e}"


def test_fisher_eigenvalues_descending(hf_fisher):
    ev = hf_fisher.eigenvalues
    assert np.all(np.diff(ev) <= 1e-8), "Eigenvalues are not in descending order"


def test_fisher_fim_total_equals_sum_of_types(hf_fisher):
    """FIM_total ≈ sum of per-type FIMs."""
    fim_sum = sum(hf_fisher.FIM_per_type.values())
    np.testing.assert_allclose(
        hf_fisher.FIM_total,
        fim_sum,
        rtol=1e-5,
        err_msg="FIM_total != sum(FIM_per_type)",
    )


def test_fisher_empty_obs(hf_model, short_forcing, hf_state0, hf_params):
    """Empty observations return a zero FIM and empty obs_types."""
    empty_obs = ObservationData(
        time=jnp.zeros(1),
        NEE=jnp.array([float("nan")]),
        GPP=jnp.array([float("nan")]),
        ER=jnp.array([float("nan")]),
        NEE_unc=jnp.array([float("nan")]),
        delta14C_obs={},
        deltaD14C_obs={},
        C_pools_obs={},
        delta14C_resp=None,
    )
    fisher = compute_fisher(
        hf_model,
        short_forcing,
        hf_state0,
        hf_params,
        empty_obs,
        fields=["log_tau"],
    )
    assert np.allclose(fisher.FIM_total, 0.0), "Expected zero FIM for empty obs"
    assert fisher.obs_types == []


# ── compute_dof ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hf_prior_sigma(hf_fisher, hf_model, hf_params):
    return make_prior_covariance(hf_params, ["log_tau"], hf_model)


@pytest.fixture(scope="module")
def hf_dof(hf_fisher, hf_prior_sigma):
    return compute_dof(hf_fisher, hf_prior_sigma)


def test_dof_total_in_range(hf_dof, hf_fisher):
    n_params = hf_fisher.FIM_total.shape[0]
    assert (
        0.0 <= hf_dof.dfs_total <= n_params + 1e-6
    ), f"DFS {hf_dof.dfs_total:.3f} outside [0, {n_params}]"


def test_dof_trace_equals_total(hf_dof):
    """trace(A) == dfs_total."""
    trace_A = float(np.trace(hf_dof.averaging_kernel))
    assert (
        abs(trace_A - hf_dof.dfs_total) < 1e-5
    ), f"trace(A)={trace_A:.4f} != dfs_total={hf_dof.dfs_total:.4f}"


def test_build_oe_prior_sigma_matches_sa_diag(hf_3pool_model):
    params0 = make_default_params(hf_3pool_model.config)
    fields = ("log_tau", "log_f_transfer")
    sigma = np.array(build_oe_prior_sigma(hf_3pool_model.config, params0, fields))
    sa_diag = np.array(_build_sa_diag(hf_3pool_model.config, params0, fields))
    np.testing.assert_allclose(sa_diag, sigma**2, rtol=1e-6)


def test_build_oe_prior_sigma_transfer_and_tau_values(hf_3pool_model):
    params0 = make_default_params(hf_3pool_model.config)
    fields = ("log_tau", "log_f_transfer")
    sigma = np.array(build_oe_prior_sigma(hf_3pool_model.config, params0, fields))
    pool_names = hf_3pool_model.pool_index.pool_names
    n_pools = len(pool_names)

    tau_sigma = sigma[:n_pools]
    transfer_sigma = sigma[n_pools:]

    idx_slow_tau = pool_names.index("soil_slow")
    assert tau_sigma[idx_slow_tau] == pytest.approx(3000.0 / 7300.0)

    idx_active = pool_names.index("soil_active")
    idx_slow = pool_names.index("soil_slow")
    idx_passive = pool_names.index("soil_passive")

    explicit_active_to_slow = idx_active * n_pools + idx_slow
    explicit_slow_to_passive = idx_slow * n_pools + idx_passive
    structural_active_to_passive = idx_active * n_pools + idx_passive
    structural_passive_to_active = idx_passive * n_pools + idx_active

    assert transfer_sigma[explicit_active_to_slow] == pytest.approx(0.5)
    assert transfer_sigma[explicit_slow_to_passive] == pytest.approx(0.5)
    assert transfer_sigma[structural_active_to_passive] == pytest.approx(0.02)
    assert transfer_sigma[structural_passive_to_active] == pytest.approx(0.02)


def test_dof_per_param_sums_to_total(hf_dof):
    """Sum of per-param DFS == dfs_total."""
    assert abs(float(np.sum(hf_dof.dfs_per_param)) - hf_dof.dfs_total) < 1e-5


def test_dof_per_obs_type_nonneg(hf_dof):
    for obs_type, dfs_k in hf_dof.dfs_per_obs_type.items():
        assert dfs_k >= -1e-6, f"Negative DFS for obs type {obs_type}: {dfs_k:.4f}"


def test_dof_monotone_with_obs(hf_model, short_forcing, hf_state0, hf_params, hf_obs):
    """DFS with C_stocks + pool_delta14C ≥ DFS with C_stocks alone."""
    warm_state = hf_state0._replace(C12=jnp.full_like(hf_state0.C12, 100.0))
    prior_sigma = make_prior_covariance(hf_params, ["log_tau"], hf_model)

    fisher_C = compute_fisher(
        hf_model,
        short_forcing,
        warm_state,
        hf_params,
        hf_obs,
        fields=["log_tau"],
        active_obs_types=[OBS_C_STOCKS],
    )
    fisher_both = compute_fisher(
        hf_model,
        short_forcing,
        warm_state,
        hf_params,
        hf_obs,
        fields=["log_tau"],
        active_obs_types=[OBS_C_STOCKS, OBS_POOL_D14C],
    )

    dof_C = compute_dof(fisher_C, prior_sigma)
    dof_both = compute_dof(fisher_both, prior_sigma)

    assert (
        dof_both.dfs_total >= dof_C.dfs_total - 1e-6
    ), f"DFS(C+Δ14C)={dof_both.dfs_total:.3f} < DFS(C)={dof_C.dfs_total:.3f}"


# ── compute_posterior ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hf_posterior(hf_fisher, hf_prior_sigma):
    return compute_posterior(hf_fisher, hf_prior_sigma)


def test_posterior_cpost_shape(hf_posterior, hf_fisher):
    n = hf_fisher.FIM_total.shape[0]
    assert hf_posterior.C_post.shape == (n, n)


def test_posterior_cpost_symmetric(hf_posterior):
    diff = np.abs(hf_posterior.C_post - hf_posterior.C_post.T)
    assert np.max(diff) < 1e-6, f"C_post not symmetric, max|A-Aᵀ| = {np.max(diff):.2e}"


def test_posterior_cpost_psd(hf_posterior):
    eigvals = np.linalg.eigvalsh(hf_posterior.C_post)
    assert np.all(eigvals >= -1e-6), f"C_post has negative eigvals: {eigvals.min():.2e}"


def test_posterior_sigma_le_prior(hf_posterior):
    """Posterior σ ≤ prior σ everywhere (observations can only reduce uncertainty)."""
    np.testing.assert_array_less(
        hf_posterior.posterior_sigma - 1e-6,
        hf_posterior.prior_sigma,
        err_msg="Some posterior σ > prior σ — observations increased uncertainty",
    )


def test_posterior_uncertainty_reduction_in_01(hf_posterior):
    ur = hf_posterior.uncertainty_reduction
    assert np.all(ur >= -1e-6), f"Negative UR: {ur.min():.4f}"
    assert np.all(ur <= 1.0 + 1e-6), f"UR > 1: {ur.max():.4f}"


def test_posterior_correlation_diagonal_ones(hf_posterior):
    diag = np.diag(hf_posterior.correlation_matrix)
    np.testing.assert_allclose(
        diag, 1.0, atol=1e-5, err_msg="Correlation matrix diagonal != 1"
    )


def test_posterior_per_type_psd(hf_posterior):
    """Per-type posterior covariance matrices are also PSD."""
    for obs_type, C_k in hf_posterior.C_post_per_type.items():
        eigvals = np.linalg.eigvalsh(C_k)
        assert np.all(
            eigvals >= -1e-6
        ), f"C_post[{obs_type}] has negative eigvals: {eigvals.min():.2e}"


# ── run_ablation_study ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hf_ablation(hf_model, short_forcing, hf_state0, hf_params, hf_obs):
    warm_state = hf_state0._replace(C12=jnp.full_like(hf_state0.C12, 100.0))
    return run_ablation_study(
        hf_model,
        short_forcing,
        warm_state,
        hf_params,
        hf_obs,
        fields=["log_tau"],
    )


def test_ablation_five_scenarios(hf_ablation):
    """Ablation study returns exactly five scenarios."""
    assert len(hf_ablation) == 5


def test_ablation_scenario_types(hf_ablation):
    """Each scenario has a non-empty obs_types list and a finite DFS."""
    for key, res in hf_ablation.items():
        assert isinstance(res.obs_types, list)
        assert math.isfinite(res.dfs_total), f"Scenario {key} has non-finite DFS"


def test_ablation_dfs_monotone(hf_ablation):
    """DFS(C_stocks + Δ14C) ≥ DFS(C_stocks alone)."""
    from ecosystem_complexity.inference.information import OBS_C_STOCKS, OBS_POOL_D14C

    key_C = OBS_C_STOCKS
    key_both = f"{OBS_C_STOCKS}+{OBS_POOL_D14C}"
    dfs_C = hf_ablation[key_C].dfs_total
    dfs_both = hf_ablation[key_both].dfs_total
    assert dfs_both >= dfs_C - 1e-6, f"DFS(C+Δ14C)={dfs_both:.3f} < DFS(C)={dfs_C:.3f}"


# ── param_group_dfs ────────────────────────────────────────────────────────────


def test_param_group_dfs_sums_to_total(hf_dof):
    """Sum of group DFS == dfs_total (when groups cover all params)."""
    group_dfs = param_group_dfs(hf_dof)
    if group_dfs:
        total = sum(group_dfs.values())
        assert (
            abs(total - hf_dof.dfs_total) < 1e-5
        ), f"sum(group DFS)={total:.4f} != dfs_total={hf_dof.dfs_total:.4f}"


# ── compute_age_diagnostics ────────────────────────────────────────────────────


def test_age_diagnostics_shapes(hf_output, hf_params, hf_model):
    diag = compute_age_diagnostics(hf_output, hf_params, hf_model)
    T = hf_output.C12.shape[0]
    n_pools = hf_output.C12.shape[1]
    assert diag.stored_delta14C.shape == (T, n_pools)
    assert diag.bulk_delta14C.shape == (T,)
    assert diag.respired_delta14C.shape == (T,)


def test_age_diagnostics_bulk_is_mass_weighted(hf_output, hf_params, hf_model):
    """bulk_delta14C is the mass-weighted mean of pool Δ¹⁴C at each timestep."""
    diag = compute_age_diagnostics(hf_output, hf_params, hf_model)
    C12 = np.array(hf_output.C12)
    d14C = np.array(hf_output.delta14C)
    expected = (d14C * C12).sum(axis=-1) / (C12.sum(axis=-1) + 1e-30)
    np.testing.assert_allclose(
        diag.bulk_delta14C,
        expected,
        rtol=1e-4,
        err_msg="bulk_delta14C is not the mass-weighted mean",
    )


def test_age_diagnostics_pool_names(hf_output, hf_params, hf_model):
    diag = compute_age_diagnostics(hf_output, hf_params, hf_model)
    assert diag.pool_names == hf_model.pool_index.pool_names


# ── age_diagnostics_summary ────────────────────────────────────────────────────


def test_age_diagnostics_summary_keys(hf_output, hf_params, hf_model):
    diag = compute_age_diagnostics(hf_output, hf_params, hf_model)
    summary = age_diagnostics_summary(diag)
    assert "bulk_delta14C" in summary
    assert "respired_delta14C" in summary
    # Per-pool entries
    for pool_name in hf_model.pool_index.pool_names:
        assert f"stored_delta14C[{pool_name}]" in summary


def test_age_diagnostics_summary_finite(hf_output, hf_params, hf_model):
    diag = compute_age_diagnostics(hf_output, hf_params, hf_model)
    summary = age_diagnostics_summary(diag)
    for key, entry in summary.items():
        mean_val = entry["mean"]
        if not np.isnan(mean_val):
            assert np.isfinite(mean_val), f"summary[{key}]['mean'] is not finite"
