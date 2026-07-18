from __future__ import annotations

import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

from ecosystem_complexity.api import build_model
from ecosystem_complexity.data.schemas import ForcingData, ObservationData
from ecosystem_complexity.oe_diagnostics import (
    fit_param_subset_labels,
    oe_constraint_ladder,
    oe_gain_matrix_diagnostics,
    oe_style_ablation,
)
from ecosystem_complexity.oe_utils import (
    build_mean_ss_modifier,
    build_oe_observation_sets,
    ss_state_for_params,
)
from ecosystem_complexity.optimizer import (
    get_oe_fields,
    get_opt_fields,
    params_to_vector,
    vector_to_params,
)
from ecosystem_complexity.state import make_default_params, make_initial_state

CONFIGS_DIR = pathlib.Path(__file__).parent.parent / "configs"
_HF_3POOL_PATH = str(CONFIGS_DIR / "harvard_3pool_config.yaml")
_SOIL_ONLY_PATH = str(CONFIGS_DIR / "harvard_forest_soil_only.yaml")


@pytest.fixture(scope="module")
def hf_3pool_model():
    return build_model(_HF_3POOL_PATH)


@pytest.fixture(scope="module")
def soil_only_model():
    return build_model(_SOIL_ONLY_PATH)


@pytest.fixture(scope="module")
def short_forcing():
    t_len = 30
    n_layers = 1
    return ForcingData(
        time=jnp.arange(t_len, dtype=jnp.float32),
        air_temp=jnp.full(t_len, 12.0),
        sw_radiation=jnp.full(t_len, 100.0),
        precip=jnp.full(t_len, 3.0),
        vpd=jnp.full(t_len, 1.0),
        soil_temp=jnp.full((t_len, n_layers), 10.0),
        soil_moisture=jnp.full((t_len, n_layers), 0.3),
        snow_depth=jnp.zeros(t_len),
        active_layer=jnp.full(t_len, jnp.inf),
        delta14C_atm=jnp.full(t_len, 50.0),
        GPP_obs=jnp.full(t_len, 2.5),
        NPP_obs=jnp.full(t_len, jnp.nan),
    )


def _make_obs(t_len: int, pool_names: list[str]) -> ObservationData:
    arr = jnp.full(t_len, jnp.nan)
    pool_obs = {}
    for pool_name, value in zip(pool_names[:2], (-40.0, -10.0)):
        obs_arr = np.full(t_len, np.nan, dtype=np.float32)
        obs_arr[5] = value
        pool_obs[pool_name] = jnp.array(obs_arr)
    c_obs = {pool_names[0]: (500.0, 100.0)}
    resp = arr.at[10].set(-25.0)
    return ObservationData(
        time=jnp.arange(t_len, dtype=jnp.float32),
        NEE=arr,
        GPP=arr,
        ER=arr,
        NEE_unc=arr,
        delta14C_obs=pool_obs,
        deltaD14C_obs={},
        C_pools_obs=c_obs,
        delta14C_resp=resp,
    )


def test_optimizer_field_selection(soil_only_model, hf_3pool_model):
    opt_fields = get_opt_fields(soil_only_model.config)
    assert "log_tau" in opt_fields
    assert "log_f_transfer" in opt_fields
    assert "log_Q10" in opt_fields

    oe_fields = get_oe_fields(hf_3pool_model.config)
    assert oe_fields == ("log_tau", "log_f_transfer")


def test_optimizer_vector_roundtrip(hf_3pool_model):
    params = make_default_params(hf_3pool_model.config)
    fields = ("log_tau", "log_f_transfer")
    flat = params_to_vector(params, fields)
    recovered = vector_to_params(flat, params, fields)
    for field in fields:
        np.testing.assert_allclose(
            np.array(getattr(params, field)),
            np.array(getattr(recovered, field)),
            atol=1e-6,
        )


def test_build_oe_observation_sets(short_forcing, hf_3pool_model):
    pool_names = hf_3pool_model.pool_index.pool_names
    resp = jnp.full(short_forcing.time.shape[0], -20.0)
    c_obs = {pool_names[0]: (500.0, 100.0)}
    er = jnp.full(short_forcing.time.shape[0], 1.5)
    obs_sets = build_oe_observation_sets(short_forcing, {}, resp, c_obs, er)
    assert len(obs_sets) == 5
    assert obs_sets[0].delta14C_resp is None
    assert obs_sets[1].delta14C_resp is None
    assert obs_sets[2].delta14C_resp is not None
    assert obs_sets[4].ER is not None


def test_build_mean_ss_modifier_and_ss_state_for_params(hf_3pool_model, short_forcing):
    params = make_default_params(hf_3pool_model.config)
    mean_mod, mean_gpp = build_mean_ss_modifier(short_forcing, params)
    assert mean_mod > 0.0
    assert mean_gpp == pytest.approx(2.5)

    state0 = make_initial_state(hf_3pool_model.config, {})
    state_ss = ss_state_for_params(hf_3pool_model, short_forcing, state0, params)
    assert state_ss.C12.shape == state0.C12.shape
    assert np.all(np.isfinite(np.array(state_ss.C12)))


def test_oe_diagnostics_outputs(hf_3pool_model, short_forcing):
    params = make_default_params(hf_3pool_model.config)
    state0 = make_initial_state(hf_3pool_model.config, {})
    obs = _make_obs(
        int(short_forcing.time.shape[0]), hf_3pool_model.pool_index.pool_names
    )

    ablation = oe_style_ablation(
        hf_3pool_model, short_forcing, state0, params, obs, opt_fields=("log_tau",)
    )
    assert set(ablation) == {
        "C_stocks",
        "pool_delta14C",
        "resp_delta14C",
        "C_stocks+pool_delta14C",
        "C_stocks+pool_delta14C+resp_delta14C",
    }
    assert ablation["C_stocks"]["n_obs"] > 0

    ladder = oe_constraint_ladder(
        hf_3pool_model, short_forcing, state0, params, obs, opt_fields=("log_tau",)
    )
    labels = [row["label"] for row in ladder]
    assert labels == ["pool_14C", "resp_14C", "c_stock"]
    assert all(row["dfs"] >= 0.0 for row in ladder)


def test_oe_gain_matrix_subset_outputs(hf_3pool_model, short_forcing):
    params = make_default_params(hf_3pool_model.config)
    state0 = make_initial_state(hf_3pool_model.config, {})
    obs = _make_obs(
        int(short_forcing.time.shape[0]), hf_3pool_model.pool_index.pool_names
    )

    diag = oe_gain_matrix_diagnostics(
        hf_3pool_model,
        short_forcing,
        state0,
        params,
        obs,
        opt_fields=("log_tau", "log_f_transfer"),
    )

    assert diag["subset_state_names"] == [
        "log_tau[soil_active]",
        "log_tau[soil_slow]",
        "log_tau[soil_passive]",
        "log_f_transfer[soil_active→soil_slow]",
        "log_f_transfer[soil_slow→soil_passive]",
    ]
    assert fit_param_subset_labels() == [
        r"$\tau_{\mathrm{active}}$",
        r"$\tau_{\mathrm{slow}}$",
        r"$\tau_{\mathrm{passive}}$",
        r"$f_{\mathrm{a\to s}}$",
        r"$f_{\mathrm{s\to p}}$",
    ]
    assert diag["subset_averaging_kernel"].shape == (5, 5)
    assert diag["subset_gain_matrix"].shape[0] == 5
    assert diag["gain_matrix"].shape[0] == diag["averaging_kernel"].shape[0]
    assert diag["constraint_labels"] and (
        len(diag["constraint_labels"]) == diag["K"].shape[0]
    )
    assert len(diag["obs_annotations"]) == diag["y_obs"].shape[0]
    assert diag["obs_annotations"][0]["obs_index"] == 0
    assert {
        "obs_block_name",
        "obs_family",
        "obs_label_full",
        "y_obs",
        "y_prior",
        "y_opt",
        "obs_sigma",
        "obs_variance",
    } <= set(diag["obs_annotations"][0])
    n_rows = len(diag["obs_annotations"]) * len(diag["subset_state_names"])
    assert n_rows == diag["subset_gain_matrix"].shape[0] * diag["subset_gain_matrix"].shape[1]
