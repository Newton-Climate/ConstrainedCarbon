"""Tests for the site-agnostic per-layer mixture bulk-¹⁴C operator."""
from __future__ import annotations

import pathlib

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from ecosystem_complexity.model.api import build_model
from ecosystem_complexity.data.israd_observations import (
    build_perlayer_mixture_obs_blocks,
)

CONFIGS_DIR = pathlib.Path(__file__).parent.parent / "configs"
_HF_3POOL_PATH = str(CONFIGS_DIR / "harvard_3pool_config.yaml")


@pytest.fixture(scope="module")
def hf_3pool_model():
    return build_model(_HF_3POOL_PATH)


class _FakeOut:
    """Minimal ModelOutput stand-in exposing delta14C and C12 (T, n_pools)."""

    def __init__(self, delta14C: np.ndarray, C12: np.ndarray) -> None:
        self.delta14C = jnp.asarray(delta14C, dtype=jnp.float32)
        self.C12 = jnp.asarray(C12, dtype=jnp.float32)


def _const_out(delta_per_pool, c12_per_pool, T=20):
    d = np.tile(np.asarray(delta_per_pool, dtype=float), (T, 1))
    c = np.tile(np.asarray(c12_per_pool, dtype=float), (T, 1))
    return _FakeOut(d, c)


def test_measured_weights_linear_mixture(hf_3pool_model):
    """Surface bin with density fractions → fixed measured-weight mixture."""
    forcing_time = np.arange(20, dtype=float)
    layer_df = pd.DataFrame({
        "lyr_top": [0.0, 0.0],
        "lyr_bot": [10.0, 10.0],
        "lyr_14c": [40.0, 60.0],
        "lyr_soc": [1.0, 1.0],
    })
    fraction_df = pd.DataFrame({
        "lyr_top": [0.0, 0.0, 0.0],
        "lyr_bot": [10.0, 10.0, 10.0],
        "frc_property": ["free light", "occluded light", "heavy"],
        "frc_mass_perc": [10.0, 20.0, 70.0],
    })
    rows = build_perlayer_mixture_obs_blocks(
        layer_df, hf_3pool_model.pool_index, forcing_time,
        depth_bins=[("b1", (0.0, 10.0))],
        obs_year=1970.0,
        fraction_df=fraction_df,
        fraction_to_pool={
            "free light": "soil_active",
            "occluded light": "soil_slow",
            "heavy": "soil_passive",
        },
        pool_order=("soil_active", "soil_slow", "soil_passive"),
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["weight_source"] == "measured"
    assert r["weights"] == pytest.approx([0.1, 0.2, 0.7])
    assert r["mean"] == pytest.approx(50.0)  # SOC-weighted (equal masses) mean of 40,60

    block = r["block"]
    assert block.name == "israd_perlayer_b1"
    assert block.y.shape == (1,) and block.Se.shape == (1,)

    # Predictor is Σ_p w_p · Δ_p with FIXED weights (independent of C12).
    out = _const_out([100.0, 0.0, -100.0], [999.0, 1.0, 1.0])
    pred = float(np.asarray(block.predict(out, None))[0])
    assert pred == pytest.approx(0.1 * 100 + 0.2 * 0 + 0.7 * -100)  # = -60


def test_model_c12_fallback_when_no_fractions(hf_3pool_model):
    """Deep bin without fractions → model-C12 partition mixture predictor."""
    forcing_time = np.arange(20, dtype=float)
    layer_df = pd.DataFrame({
        "lyr_top": [20.0, 20.0],
        "lyr_bot": [40.0, 40.0],
        "lyr_14c": [-100.0, -140.0],
        "lyr_soc": [1.0, 1.0],
    })
    rows = build_perlayer_mixture_obs_blocks(
        layer_df, hf_3pool_model.pool_index, forcing_time,
        depth_bins=[("deep", (20.0, 40.0))],
        obs_year=1970.0,
        fraction_df=None,
        pool_order=("soil_active", "soil_slow", "soil_passive"),
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["weight_source"] == "model_c12"
    assert r["weights"] is None

    # Predictor is the C12-mass-weighted mixture Σ_p (C12_p/ΣC12)·Δ_p.
    out = _const_out([100.0, 0.0, -100.0], [1000.0, 3000.0, 6000.0])
    pred = float(np.asarray(r["block"].predict(out, None))[0])
    assert pred == pytest.approx(0.1 * 100 + 0.3 * 0 + 0.6 * -100)  # = -50


def test_soc_weighted_observed_value(hf_3pool_model):
    """Observed bulk value is carbon-mass (lyr_soc) weighted across layers."""
    forcing_time = np.arange(20, dtype=float)
    layer_df = pd.DataFrame({
        "lyr_top": [0.0, 0.0],
        "lyr_bot": [10.0, 10.0],
        "lyr_14c": [50.0, -10.0],
        "lyr_soc": [1.0, 9.0],
    })
    rows = build_perlayer_mixture_obs_blocks(
        layer_df, hf_3pool_model.pool_index, forcing_time,
        depth_bins=[("b1", (0.0, 10.0))],
        obs_year=1970.0,
    )
    # (1*50 + 9*-10) / 10 = -4.0
    assert rows[0]["mean"] == pytest.approx(-4.0)


def test_min_layers_skips_thin_bins(hf_3pool_model):
    forcing_time = np.arange(20, dtype=float)
    layer_df = pd.DataFrame({
        "lyr_top": [0.0],
        "lyr_bot": [10.0],
        "lyr_14c": [50.0],
        "lyr_soc": [1.0],
    })
    rows = build_perlayer_mixture_obs_blocks(
        layer_df, hf_3pool_model.pool_index, forcing_time,
        depth_bins=[("b1", (0.0, 10.0))],
        obs_year=1970.0,
        min_layers=2,
    )
    assert rows == []
