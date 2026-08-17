"""Respiration and carbon-conservation invariants under the log_f_transfer reparameterization.

`log_f_transfer` has shape (n_pools, n_pools+1); the last column is the
to-respiration branch. The optimizer excludes that column from the free
parameter vector and reconstructs the full matrix with it held at its template
value (see optimizer.params_to_vector / vector_to_params). Because the transfer
fractions come from a softmax over all n_pools+1 columns, each pool's decay is
still partitioned into transfers + respiration summing to 1 — so mass is
conserved and respiration is the exact transfer complement for *any* reduced
vector. These tests guard that the reparameterization preserves those laws.

Complements the existing default-parameter mass-balance tests
(test_model_integration.test_mass_balance_*, test_model.test_step_12C_mass_balance)
by exercising the reparameterized / perturbed parameter path.
"""
from __future__ import annotations

import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ecosystem_complexity.model.api import build_model, run_model
from ecosystem_complexity.data.schemas import ForcingData
from ecosystem_complexity.inference.parameters import params_to_vector, vector_to_params
from ecosystem_complexity.processes.soil import het_respiration
from ecosystem_complexity.model.state import make_default_params, make_initial_state
from ecosystem_complexity.model.transfers import get_transfer_matrix

CONFIGS_DIR = pathlib.Path(__file__).resolve().parents[2] / "configs"
_HF_3POOL_PATH = str(CONFIGS_DIR / "harvard_3pool_config.yaml")
_FIELDS = ("log_tau", "log_f_transfer")


@pytest.fixture(scope="module")
def model():
    return build_model(_HF_3POOL_PATH)


@pytest.fixture(scope="module")
def params(model):
    return make_default_params(model.config)


def _n_pools(model) -> int:
    return len(model.pool_index)


def _perturb(params, model, scale: float, seed: int):
    """Reconstruct params from a randomly perturbed *reduced* parameter vector."""
    vec = params_to_vector(params, _FIELDS)
    noise = jax.random.normal(jax.random.PRNGKey(seed), vec.shape) * scale
    return vector_to_params(vec + noise, params, _FIELDS)


# ── The reparameterization drops exactly the respiration column ─────────────

def test_reduced_vector_excludes_respiration_column(model, params):
    n_pools = _n_pools(model)
    vec = np.asarray(params_to_vector(params, _FIELDS))
    n_tau = int(math.prod(params.log_tau.shape))

    reduced = n_tau + n_pools * n_pools           # last column dropped
    full = n_tau + n_pools * (n_pools + 1)         # if it were kept
    assert vec.shape == (reduced,)
    assert full - reduced == n_pools               # exactly one column removed


def test_roundtrip_preserves_full_matrix_including_respiration_logit(model, params):
    p2 = vector_to_params(params_to_vector(params, _FIELDS), params, _FIELDS)
    # Whole matrix round-trips, and the (un-optimized) respiration column is
    # carried over from the template byte-for-byte.
    np.testing.assert_allclose(
        np.asarray(p2.log_f_transfer), np.asarray(params.log_f_transfer), atol=1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(p2.log_f_transfer[:, -1]), np.asarray(params.log_f_transfer[:, -1])
    )


# ── Mass conservation of the decay partition, for ANY reduced vector ────────

@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_arbitrary_reduced_vector_conserves_decay_partition(model, params, seed):
    """transfers + respiration = 1 per pool, respiration ≥ 0 — even far from the prior."""
    n_pools = _n_pools(model)
    p2 = _perturb(params, model, scale=3.0, seed=seed)  # large perturbation

    F = np.asarray(get_transfer_matrix(p2.log_f_transfer, n_pools))
    row_sums = F.sum(axis=1)
    f_resp = 1.0 - row_sums

    assert np.all(F >= -1e-6), "negative transfer fraction (carbon created)"
    assert np.all(f_resp >= -1e-6), "negative respiration fraction (carbon created)"
    assert np.all(row_sums <= 1.0 + 1e-6), "transfers exceed 1 (no respiration sink)"
    np.testing.assert_allclose(row_sums + f_resp, 1.0, atol=1e-6)


# ── Respiration equals the decay-weighted transfer complement ───────────────

@pytest.mark.parametrize("seed", [0, 3, 11])
def test_respiration_equals_decay_weighted_transfer_complement(model, params, seed):
    """het_respiration == Σ_i (1 - Σ_j F_ij)·C_i/τ_i, including after reparam perturbation."""
    n_pools = _n_pools(model)
    p2 = _perturb(params, model, scale=1.0, seed=seed)

    C12 = jnp.full((n_pools,), 500.0)
    ones = jnp.ones((n_pools,))
    Rh = float(het_respiration(C12, p2.log_tau, p2.log_f_transfer, ones, ones, ones, n_pools))

    F = np.asarray(get_transfer_matrix(p2.log_f_transfer, n_pools))
    decomp = np.asarray(C12) / np.exp(np.asarray(p2.log_tau))
    Rh_manual = float(np.sum((1.0 - F.sum(axis=1)) * decomp))

    assert Rh >= 0.0
    np.testing.assert_allclose(Rh, Rh_manual, rtol=1e-5, atol=1e-6)


# ── A full forward run under reparam-perturbed params stays conservative ────

def _constant_forcing(model, T: int = 60) -> ForcingData:
    n_layers = len(model.config.soil_layers)
    return ForcingData(
        time=jnp.arange(T, dtype=jnp.float32),
        air_temp=jnp.full(T, 12.0),
        sw_radiation=jnp.full(T, 150.0),
        precip=jnp.full(T, 3.0),
        vpd=jnp.full(T, 8.0),
        soil_temp=jnp.full((T, n_layers), 10.0),
        soil_moisture=jnp.full((T, n_layers), 0.3),
        snow_depth=jnp.full(T, 0.0),
        active_layer=jnp.full(T, jnp.inf),
        delta14C_atm=jnp.full(T, 0.0),
        GPP_obs=jnp.full(T, 5.0),
        NPP_obs=jnp.full(T, float("nan")),
    )


def test_reparam_run_is_conservative_and_respiration_bookkeeping(model, params):
    """A run with reparam-perturbed params conserves carbon and keeps Rh accounting exact."""
    p2 = _perturb(params, model, scale=0.5, seed=5)
    forcing = _constant_forcing(model)
    state0 = make_initial_state(model.config, {"mat_c": 8.5, "permafrost": False})
    state0 = state0._replace(C12=jnp.full_like(state0.C12, 200.0))
    dt = float(model.config.dt_days)

    out = run_model(model, forcing, state0=state0, params=p2)
    C12 = np.asarray(out.C12)
    Rh, Ra, GPP, NEE, ER = (np.asarray(getattr(out, k)) for k in ("Rh", "Ra", "GPP", "NEE", "ER"))

    # Physicality: no NaNs, non-negative stocks and respiration.
    assert np.all(np.isfinite(C12)) and np.all(C12 >= -1e-3)
    assert np.all(np.isfinite(Rh)) and np.all(Rh >= -1e-6)

    # Exact diagnostic bookkeeping (respiration partition of the fluxes).
    np.testing.assert_allclose(ER, Ra + Rh, atol=1e-4)
    np.testing.assert_allclose(NEE, Rh + Ra - GPP, atol=1e-4)

    # Carbon conservation: per-step change in total stock == (NPP - Rh)·dt = -NEE·dt.
    NPP = GPP - Ra
    dC_total = np.diff(C12.sum(axis=1))
    expected = (NPP[:-1] - Rh[:-1]) * dt
    max_res = float(np.max(np.abs(dC_total - expected)))
    assert max_res < 1e-3, f"carbon not conserved under reparam params: max resid {max_res:.2e}"
