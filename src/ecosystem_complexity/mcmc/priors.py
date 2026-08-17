"""Site-level priors and context construction for the MCMC pipeline."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from ecosystem_complexity import mcmc as _mcmc
from ecosystem_complexity.inference._helpers import build_oe_prior_sigma
from ecosystem_complexity.model.api import build_model
from ecosystem_complexity.data.israd_14c import build_bulk_14C_blocks, build_resp_14C_obs
from ecosystem_complexity.data.parsers import attach_atm14C
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.paths import GRAVEN_PATH, HUA_PATH, INTCAL_PATH
from ecosystem_complexity.data.schemas import ObservationData
from ecosystem_complexity.inference.parameters import params_to_vector
from ecosystem_complexity.sites.driver import OPT_FIELDS, build_state0
from ecosystem_complexity.sites.forcing import (
    load_site_forcing,
    load_site_observations,
    resolve_forcing_file,
)
from ecosystem_complexity.sites.soc import build_soc_prior
from ecosystem_complexity.sites.spec import load_site_spec
from ecosystem_complexity.model.state import make_default_params
from ecosystem_complexity.synthesis.warming import repeat_forcing, warm_forcing


def _build_site_context(
    config_path: str,
    include_er_constraint: bool,
    include_incubation_constraint: bool,
) -> dict:
    spec = load_site_spec(config_path)
    model = build_model(config_path)
    forcing_path = resolve_forcing_file(spec)
    forcing = load_site_forcing(spec, forcing_path, model)
    tower_obs = (
        load_site_observations(spec, forcing_path, model, forcing=forcing)
        if include_er_constraint
        else None
    )
    hemisphere = "NH" if spec.lat >= 0 else "SH"
    years_daily, d14c_daily = load_full_14C_record(
        hua_path=HUA_PATH,
        graven_path=GRAVEN_PATH,
        intcal_path=INTCAL_PATH,
        hemisphere=hemisphere,
        start_year=1500.0,
        end_year=2025.0,
    )
    forcing = attach_atm14C(forcing, d14c_daily, years_daily)
    soc_prior_state, _c_pools_prior, _ss_years, c_total_obs = build_soc_prior(model, forcing)
    pool_blocks = []
    if spec.observation_path in {"fraction", "combined"}:
        from ecosystem_complexity.data.israd_14c import build_fraction_14C_blocks

        pool_blocks.extend(build_fraction_14C_blocks(spec.israd_name, forcing.time, model, spec.fraction_rules))
    if spec.observation_path in {"bulk_resp", "combined"}:
        pool_blocks.extend(build_bulk_14C_blocks(spec.israd_name, forcing.time, model))
    resp = (
        build_resp_14C_obs(spec.israd_name, forcing.time)
        if spec.observation_path in {"bulk_resp", "combined"}
        else jnp.full(forcing.time.shape[0], jnp.nan, dtype=jnp.float32)
    )
    if include_incubation_constraint:
        from ecosystem_complexity.data.israd_incubation import build_incubation_rate_blocks
        from ecosystem_complexity.data.israd_14c import build_incubation_14C_blocks

        incubation_rows = build_incubation_rate_blocks(spec.israd_name, model)
        pool_blocks = pool_blocks + [row["block"] for row in incubation_rows]
        pool_blocks.extend(build_incubation_14C_blocks(spec.israd_name, forcing.time))
    T = int(forcing.time.shape[0])
    er_obs = (
        tower_obs.ER
        if (tower_obs is not None and tower_obs.ER is not None)
        else jnp.full(T, jnp.nan)
    )
    obs_full = ObservationData(
        time=forcing.time,
        NEE=jnp.full(T, jnp.nan),
        GPP=jnp.full(T, jnp.nan),
        ER=er_obs,
        NEE_unc=jnp.full(T, jnp.nan),
        delta14C_obs={},
        deltaD14C_obs={},
        C_pools_obs={},
        delta14C_resp=resp,
        C_total_obs=c_total_obs,
    )
    from ecosystem_complexity.data.israd_14c import _bulk_pool_ic_seeds

    state0 = build_state0(model, soc_prior_state, pool_blocks, ic_seeds=_bulk_pool_ic_seeds(spec.israd_name))
    forcing_proj = repeat_forcing(forcing, _mcmc.WARMING_HORIZON_YEARS)
    forcing_warm = warm_forcing(forcing_proj, _mcmc.WARMING_DELTA_C)
    params_prior = make_default_params(model.config)
    return {
        "spec": spec,
        "model": model,
        "forcing": forcing,
        "forcing_proj": forcing_proj,
        "forcing_warm": forcing_warm,
        "state0": state0,
        "params_prior": params_prior,
        "opt_fields": OPT_FIELDS,
        "obs_full": obs_full,
    }


def _draw_gaussian_samples(
    rng: np.random.Generator,
    x_mean: np.ndarray,
    cov: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    cov_sym = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov_sym)
    eigvals = np.clip(eigvals, 0.0, None)
    transform = eigvecs @ np.diag(np.sqrt(eigvals))
    z = rng.standard_normal((n_samples, x_mean.size))
    return x_mean[None, :] + z @ transform.T


def _prior_mean_and_cov(context: dict) -> tuple[np.ndarray, np.ndarray]:
    x_mean = np.array(params_to_vector(context["params_prior"], context["opt_fields"]), dtype=np.float64)
    sigma = np.array(
        build_oe_prior_sigma(context["model"].config, context["params_prior"], context["opt_fields"]),
        dtype=np.float64,
    )
    return x_mean, np.diag(np.square(sigma))


def _observed_gap_for_spec(config_path: str) -> dict[str, float]:
    spec = load_site_spec(config_path)
    if spec.observation_path != "bulk_resp":
        return {
            "obs_bulk_mean": np.nan,
            "obs_resp_mean": np.nan,
            "obs_offset_resp_minus_bulk": np.nan,
            "n_bulk_vals": 0,
            "n_resp_vals": 0,
        }
    model = build_model(config_path)
    forcing_path = resolve_forcing_file(spec)
    forcing = load_site_forcing(spec, forcing_path, model)
    hemisphere = "NH" if spec.lat >= 0 else "SH"
    years_daily, d14c_daily = load_full_14C_record(
        hua_path=HUA_PATH,
        graven_path=GRAVEN_PATH,
        intcal_path=INTCAL_PATH,
        hemisphere=hemisphere,
        start_year=1500.0,
        end_year=2025.0,
    )
    forcing = attach_atm14C(forcing, d14c_daily, years_daily)
    bulk_blocks = build_bulk_14C_blocks(spec.israd_name, forcing.time, model)
    resp = np.array(build_resp_14C_obs(spec.israd_name, forcing.time), dtype=float)
    bulk_vals: list[float] = []
    for block in bulk_blocks:
        vals = np.array(block.y, dtype=float).ravel()
        bulk_vals.extend(vals[np.isfinite(vals)].tolist())
    resp_vals = resp[np.isfinite(resp)]
    if not bulk_vals or resp_vals.size == 0:
        return {
            "obs_bulk_mean": np.nan,
            "obs_resp_mean": np.nan,
            "obs_offset_resp_minus_bulk": np.nan,
            "n_bulk_vals": len(bulk_vals),
            "n_resp_vals": int(resp_vals.size),
        }
    return {
        "obs_bulk_mean": float(np.mean(bulk_vals)),
        "obs_resp_mean": float(np.mean(resp_vals)),
        "obs_offset_resp_minus_bulk": float(np.mean(resp_vals) - np.mean(bulk_vals)),
        "n_bulk_vals": int(len(bulk_vals)),
        "n_resp_vals": int(resp_vals.size),
    }
