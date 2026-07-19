"""Tests for ISRaD incubation rates as a bulk turnover constraint.

Coverage: unit conversion, row filtering, temperature grouping, and the
forward operator's structure (lab conditions applied, gradient reaching τ).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from ecosystem_complexity.api import build_model
from ecosystem_complexity.climate import f_temp
from ecosystem_complexity.data.israd_incubation import (
    build_incubation_rate_blocks,
    load_incubation_rates,
)
from ecosystem_complexity.state import make_default_params

_SITE = "test_site"


def _write_inc_csv(tmp_path, rows: list[dict]) -> str:
    cols = {
        "site_name": _SITE,
        "inc_type": "root-picked soil",
        "inc_anaerobic": np.nan,
        "inc_flux": np.nan,
        "inc_flux_units": "mgC/gC soil/day",
        "inc_temp": 15.0,
        "lyr_c_org": np.nan,
        "inc_duration_type": "<1 year",
    }
    df = pd.DataFrame([{**cols, **r} for r in rows])
    path = tmp_path / "inc.csv"
    df.to_csv(path, index=False)
    return str(path)


# ── unit conversion ────────────────────────────────────────────────────────


def test_per_gc_units_convert_by_1e_minus_3(tmp_path):
    """mgC/gC/day → day⁻¹ is a factor of 1e-3."""
    p = _write_inc_csv(tmp_path, [{"inc_flux": 2.0}])
    out = load_incubation_rates(_SITE, path=p)
    assert out["k_obs"].iloc[0] == pytest.approx(2.0e-3)


def test_per_dry_soil_units_divide_by_carbon_fraction(tmp_path):
    """mgC/g dry soil/day is normalised by lyr_c_org (a percent)."""
    p = _write_inc_csv(
        tmp_path,
        [{"inc_flux": 2.0, "inc_flux_units": "mgC/g dry soil/day", "lyr_c_org": 5.0}],
    )
    out = load_incubation_rates(_SITE, path=p)
    # 2 mgC/g soil/day ÷ 0.05 gC/g soil = 40 mgC/gC/day = 4e-2 day⁻¹
    assert out["k_obs"].iloc[0] == pytest.approx(4.0e-2)


def test_per_dry_soil_without_carbon_content_is_dropped(tmp_path):
    """A per-dry-soil row with no lyr_c_org cannot be converted, so it is dropped."""
    p = _write_inc_csv(
        tmp_path, [{"inc_flux": 2.0, "inc_flux_units": "mgC/g dry soil/day"}]
    )
    assert load_incubation_rates(_SITE, path=p).empty


# ── row filtering ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_type", ["live roots", "soil w/ live roots", "litter"])
def test_non_soil_incubation_types_excluded(tmp_path, bad_type):
    """Autotrophic-contaminated and non-soil reservoirs are not bulk soil rates."""
    p = _write_inc_csv(tmp_path, [{"inc_flux": 2.0, "inc_type": bad_type}])
    assert load_incubation_rates(_SITE, path=p).empty


def test_anaerobic_incubations_excluded(tmp_path):
    p = _write_inc_csv(tmp_path, [{"inc_flux": 2.0, "inc_anaerobic": "yes"}])
    assert load_incubation_rates(_SITE, path=p).empty


def test_nonpositive_and_missing_flux_excluded(tmp_path):
    p = _write_inc_csv(
        tmp_path,
        [{"inc_flux": 0.0}, {"inc_flux": -1.0}, {"inc_flux": np.nan}, {"inc_flux": 1.0}],
    )
    assert len(load_incubation_rates(_SITE, path=p)) == 1


def test_duration_types_filter(tmp_path):
    """Protocol classes differ ~35× in implied rate, so they must be separable."""
    p = _write_inc_csv(
        tmp_path,
        [
            {"inc_flux": 1.0, "inc_duration_type": "<2 weeks"},
            {"inc_flux": 2.0, "inc_duration_type": "<1 year"},
        ],
    )
    assert len(load_incubation_rates(_SITE, path=p)) == 2
    only_long = load_incubation_rates(
        _SITE, path=p, duration_types=frozenset({"<1 year"})
    )
    assert len(only_long) == 1
    assert only_long["k_obs"].iloc[0] == pytest.approx(2.0e-3)


def test_duration_mix_is_reported(tmp_path, model):
    """A block pooling two protocol classes must say so."""
    p = _write_inc_csv(
        tmp_path,
        [
            {"inc_flux": 1.0, "inc_duration_type": "<2 weeks"},
            {"inc_flux": 2.0, "inc_duration_type": "<1 year"},
        ],
    )
    (block,) = build_incubation_rate_blocks(_SITE, model, path=p)
    assert block["duration_mix"] == {"<2 weeks": 1, "<1 year": 1}


def test_missing_temperature_excluded(tmp_path):
    """Without inc_temp the lab condition is unknown, so the row is unusable."""
    p = _write_inc_csv(tmp_path, [{"inc_flux": 2.0, "inc_temp": np.nan}])
    assert load_incubation_rates(_SITE, path=p).empty


# ── block construction ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model():
    return build_model("configs/harvard_3pool_config.yaml")


def test_blocks_group_by_temperature(tmp_path, model):
    """Rows at different inc_temp cannot share a block — the predictor differs."""
    p = _write_inc_csv(
        tmp_path,
        [
            {"inc_flux": 1.0, "inc_temp": 4.0},
            {"inc_flux": 3.0, "inc_temp": 4.0},
            {"inc_flux": 2.0, "inc_temp": 20.0},
        ],
    )
    blocks = build_incubation_rate_blocks(_SITE, model, path=p)
    assert len(blocks) == 2
    by_temp = {b["inc_temp"]: b for b in blocks}
    assert by_temp[4.0]["n"] == 2
    assert by_temp[4.0]["k_obs"] == pytest.approx(2.0e-3)  # mean of 1 and 3
    assert by_temp[20.0]["n"] == 1


def test_sigma_floor_applied_to_single_row(tmp_path, model):
    """One replicate gives zero scatter, so the relative floor must take over."""
    p = _write_inc_csv(tmp_path, [{"inc_flux": 2.0}])
    (block,) = build_incubation_rate_blocks(_SITE, model, path=p)
    assert block["sigma"] == pytest.approx(0.5 * block["k_obs"])


def test_missing_site_yields_no_blocks(tmp_path, model):
    p = _write_inc_csv(tmp_path, [{"inc_flux": 2.0}])
    assert build_incubation_rate_blocks("no_such_site", model, path=p) == []


# ── forward operator ───────────────────────────────────────────────────────


def _fake_output(model, c_by_pool: dict[str, float]):
    """Minimal stand-in for ModelOutput exposing the C12 trajectory only."""
    names = model.pool_index.pool_names
    c = np.array([c_by_pool.get(n, 0.0) for n in names], dtype=np.float32)

    class _Out:
        C12 = jnp.asarray(np.tile(c, (3, 1)))

    return _Out()


def _predict(model, path, out, params, temp=15.0):
    (block,) = build_incubation_rate_blocks(_SITE, model, path=path)
    assert block["inc_temp"] == temp
    return float(block["block"].predict(out, params)[0])


def test_predictor_matches_hand_computed_rate(tmp_path, model):
    """k = Σ resp_frac·(C/τ)·f_temp / ΣC over soil pools, at lab conditions."""
    p = _write_inc_csv(tmp_path, [{"inc_flux": 2.0}])
    params = make_default_params(model.config)
    stocks = {"soil_active": 500.0, "soil_slow": 3000.0, "soil_passive": 8000.0}
    out = _fake_output(model, stocks)

    n = len(model.pool_index)
    f_full = jax.nn.softmax(params.log_f_transfer, axis=-1)
    resp_frac = np.array(1.0 - f_full[:, :n].sum(axis=-1))
    tau = np.exp(np.array(params.log_tau))
    ft = float(f_temp(jnp.asarray(15.0), params.log_Q10)[0])

    cols = [model.pool_index[k] for k in stocks]
    C = np.array([stocks[k] for k in stocks])
    expected = (resp_frac[cols] * (C / tau[cols]) * ft).sum() / C.sum()

    assert _predict(model, p, out, params) == pytest.approx(expected, rel=1e-5)


def test_temperature_scales_prediction_by_q10(tmp_path, model):
    """Doubling from T_ref=15 to 25 °C must scale k by the model's own Q10."""
    params = make_default_params(model.config)
    out = _fake_output(model, {"soil_active": 500.0, "soil_slow": 3000.0})

    dir15 = tmp_path / "t15"
    dir25 = tmp_path / "t25"
    dir15.mkdir()
    dir25.mkdir()
    p15 = _write_inc_csv(dir15, [{"inc_flux": 2.0, "inc_temp": 15.0}])
    p25 = _write_inc_csv(dir25, [{"inc_flux": 2.0, "inc_temp": 25.0}])

    k15 = _predict(model, p15, out, params, temp=15.0)
    k25 = _predict(model, p25, out, params, temp=25.0)
    q10 = float(np.exp(np.array(params.log_Q10)[0]))
    assert k25 / k15 == pytest.approx(q10, rel=1e-5)


def test_gradient_tracks_each_pool_respiration_share(tmp_path, model):
    """∂k/∂log τ_i = −resp_frac_i·(C_i/τ_i)·f_temp / ΣC.

    Each pool's leverage on the constraint is exactly its share of the
    incubation respiration flux, so how much this observation says about a given
    τ depends on the stock partition — it is not fixed by the operator. For a
    realistic equilibrium partition the fast pool dominates by orders of
    magnitude; the identity below is what actually holds in general.
    """
    p = _write_inc_csv(tmp_path, [{"inc_flux": 2.0}])
    params = make_default_params(model.config)
    stocks = {"soil_active": 500.0, "soil_slow": 3000.0, "soil_passive": 8000.0}
    out = _fake_output(model, stocks)
    (block,) = build_incubation_rate_blocks(_SITE, model, path=p)

    g = jax.grad(lambda pp: jnp.sum(block["block"].predict(out, pp)))(params)
    d_tau = np.array(g.log_tau)
    assert np.all(np.isfinite(d_tau))

    n = len(model.pool_index)
    f_full = jax.nn.softmax(params.log_f_transfer, axis=-1)
    resp_frac = np.array(1.0 - f_full[:, :n].sum(axis=-1))
    tau = np.exp(np.array(params.log_tau))
    ft = float(f_temp(jnp.asarray(15.0), params.log_Q10)[0])
    total_c = sum(stocks.values())

    for name, c in stocks.items():
        i = model.pool_index[name]
        expected = -resp_frac[i] * (c / tau[i]) * ft / total_c
        assert d_tau[i] == pytest.approx(expected, rel=1e-4)
