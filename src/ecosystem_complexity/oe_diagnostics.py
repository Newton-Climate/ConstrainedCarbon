"""Reusable OE diagnostics shared by notebooks and tests."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from .api import run_model
from .optimizer import params_to_vector, vector_to_params
from .state import make_default_params
from ._oe_helpers import _build_obs_blocks, _build_sa_diag, _analytical_c12_ss
from .oe_utils import build_mean_ss_modifier
from .sensitivity import OBS_C_STOCKS, OBS_POOL_D14C, OBS_RESP_D14C

_BLOCK_TO_OBSTYPE = {
    "pool_14C": OBS_POOL_D14C,
    "resp_14C": OBS_RESP_D14C,
    "c_stock": OBS_C_STOCKS,
    "c_sum": OBS_C_STOCKS,
    "israd": OBS_POOL_D14C,
}


def classify_block(block_name: str) -> str:
    """Map an OE observation-block name to its canonical observation type."""
    for prefix, label in _BLOCK_TO_OBSTYPE.items():
        if block_name.startswith(prefix):
            return label
    return block_name


def oe_style_ablation(
    model, forcing, state0, params_opt, observations,
    opt_fields: tuple,
    extra_obs_blocks: list | None = None,
) -> dict:
    """OE-style DFS-by-observation-type ablation at the MAP estimate."""
    inv_cfg = getattr(model.config, "inversion_raw", {}) or {}
    sigma_pool = float(inv_cfg.get("sigma_pool_14C", 5.0))
    sigma_resp = float(inv_cfg.get("sigma_resp_14C", 10.0))
    sigma_carbon = float(inv_cfg.get("sigma_carbon_gCm2", 1000.0))

    obs_blocks = _build_obs_blocks(
        observations, model, sigma_pool, sigma_resp, sigma_carbon,
        f_hetero=0.0, sigma_er_frac=0.15,
    )
    if extra_obs_blocks:
        obs_blocks = obs_blocks + list(extra_obs_blocks)
    if not obs_blocks:
        raise ValueError("oe_style_ablation: no obs blocks built")

    obs_type_per_block = [classify_block(b.name) for b in obs_blocks]
    params0 = make_default_params(model.config)
    sa_diag = np.array(_build_sa_diag(model.config, params0, tuple(opt_fields)))
    sa_inv = 1.0 / (sa_diag + 1e-30)

    n_pools = len(model.pool_index)
    cue = float(getattr(model.config.external_inputs, "CUE", 0.47))
    mean_mod, mean_gpp = build_mean_ss_modifier(forcing, params0)
    mean_input = mean_gpp * cue
    target_names = list(model.config.external_inputs.partition.keys())
    target_idx = [model.pool_index[n] for n in target_names] or None

    def _forward(x_vec):
        p = vector_to_params(x_vec, params0, tuple(opt_fields))
        c12_ss = _analytical_c12_ss(
            p, n_pools, mean_input, mean_mod, target_indices=target_idx
        )
        state_ss = state0._replace(C12=c12_ss)
        out = run_model(model, forcing, state0=state_ss, params=p)
        return jnp.concatenate([b.predict(out, p) for b in obs_blocks])

    x_opt = params_to_vector(params_opt, tuple(opt_fields))
    k = np.array(jax.jacobian(_forward)(x_opt))
    se = np.array(jnp.concatenate([b.Se for b in obs_blocks]))

    block_lens = [int(b.y.shape[0]) for b in obs_blocks]
    block_starts = np.cumsum([0] + block_lens)

    def _dfs_for_subset(active_types: list[str]) -> tuple[float, int, np.ndarray]:
        row_mask = np.zeros(k.shape[0], dtype=bool)
        for i, obs_type in enumerate(obs_type_per_block):
            if obs_type in active_types:
                row_mask[block_starts[i]:block_starts[i + 1]] = True
        if not row_mask.any():
            return 0.0, 0, np.zeros(k.shape[1])
        k_sub = k[row_mask, :]
        se_sub = se[row_mask]
        ktsek = (k_sub.T / se_sub) @ k_sub
        h = ktsek + np.diag(sa_inv)
        sx = np.linalg.inv(h)
        a = sx @ ktsek
        return float(np.trace(a)), int(row_mask.sum()), np.diag(a)

    scenarios = [
        ("C_stocks", [OBS_C_STOCKS]),
        ("pool_delta14C", [OBS_POOL_D14C]),
        ("resp_delta14C", [OBS_RESP_D14C]),
        ("C_stocks+pool_delta14C", [OBS_C_STOCKS, OBS_POOL_D14C]),
        ("C_stocks+pool_delta14C+resp_delta14C", [OBS_C_STOCKS, OBS_POOL_D14C, OBS_RESP_D14C]),
    ]
    return {
        label: {"dfs_total": dfs, "n_obs": n_obs, "dfs_per_param": dfs_diag}
        for label, obs_types in scenarios
        for dfs, n_obs, dfs_diag in [_dfs_for_subset(obs_types)]
    }


def oe_constraint_ladder(
    model, forcing, state0, params_opt, observations,
    opt_fields: tuple,
    extra_obs_blocks: list | None = None,
) -> list[dict]:
    """One-constraint-at-a-time OE ladder at the MAP estimate."""
    inv_cfg = getattr(model.config, "inversion_raw", {}) or {}
    sigma_pool = float(inv_cfg.get("sigma_pool_14C", 5.0))
    sigma_resp = float(inv_cfg.get("sigma_resp_14C", 10.0))
    sigma_carbon = float(inv_cfg.get("sigma_carbon_gCm2", 1000.0))

    obs_blocks = _build_obs_blocks(
        observations, model, sigma_pool, sigma_resp, sigma_carbon,
        f_hetero=0.0, sigma_er_frac=0.15,
    )
    if extra_obs_blocks:
        obs_blocks = obs_blocks + list(extra_obs_blocks)
    if not obs_blocks:
        raise ValueError("oe_constraint_ladder: no obs blocks built")

    obs_type_per_block = [classify_block(b.name) for b in obs_blocks]
    params0 = make_default_params(model.config)
    sa_diag = np.array(_build_sa_diag(model.config, params0, tuple(opt_fields)))
    sa_inv = 1.0 / (sa_diag + 1e-30)

    n_pools = len(model.pool_index)
    cue = float(getattr(model.config.external_inputs, "CUE", 0.47))
    mean_mod, mean_gpp = build_mean_ss_modifier(forcing, params0)
    mean_input = mean_gpp * cue
    target_names = list(model.config.external_inputs.partition.keys())
    target_idx = [model.pool_index[n] for n in target_names] or None

    def _forward(x_vec):
        p = vector_to_params(x_vec, params0, tuple(opt_fields))
        c12_ss = _analytical_c12_ss(
            p, n_pools, mean_input, mean_mod, target_indices=target_idx
        )
        state_ss = state0._replace(C12=c12_ss)
        out = run_model(model, forcing, state0=state_ss, params=p)
        return jnp.concatenate([b.predict(out, p) for b in obs_blocks])

    x_opt = params_to_vector(params_opt, tuple(opt_fields))
    t0 = time.perf_counter()
    k = np.array(jax.jacobian(_forward)(x_opt))
    se = np.array(jnp.concatenate([b.Se for b in obs_blocks]))
    _ = t0  # preserve timing hook without printing in library code

    block_lens = [int(b.y.shape[0]) for b in obs_blocks]
    block_starts = np.cumsum([0] + block_lens)
    results = []
    for i, (block, obs_type) in enumerate(zip(obs_blocks, obs_type_per_block)):
        row_slice = slice(block_starts[i], block_starts[i + 1])
        k_sub = k[row_slice, :]
        se_sub = se[row_slice]
        ktsek = (k_sub.T / se_sub) @ k_sub
        h = ktsek + np.diag(sa_inv)
        sx = np.linalg.inv(h)
        a = sx @ ktsek
        results.append({
            "label": block.name,
            "obs_type": obs_type,
            "n_obs": int(block.y.shape[0]),
            "dfs": float(np.trace(a)),
            "dfs_per_param": np.diag(a),
        })
    return results
