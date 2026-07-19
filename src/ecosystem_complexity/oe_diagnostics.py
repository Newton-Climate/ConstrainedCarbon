"""Reusable OE diagnostics shared by notebooks and tests."""

from __future__ import annotations

import time
from itertools import combinations
from math import factorial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ._oe_helpers import (
    ObsBlock,
    _analytical_c12_ss,
    _build_obs_blocks,
    _build_sa_diag,
    analytical_c14_ss,
    apply_ss_c12_c14,
    prepare_c14_spinup,
)
from .api import run_model
from .data.schemas import ForcingData, ObservationData
from .model import EcosystemModel
from .oe_utils import build_mean_ss_modifier
from .optimizer import params_to_vector, vector_to_params
from .sensitivity import OBS_C_STOCKS, OBS_POOL_D14C, OBS_RESP_D14C, get_param_names
from .state import EcosystemState, ModelParams, make_default_params

_BLOCK_TO_OBSTYPE = {
    "pool_14C": OBS_POOL_D14C,
    "resp_14C": OBS_RESP_D14C,
    "er_annual": "ER_annual",
    "c_stock": OBS_C_STOCKS,
    "c_sum": OBS_C_STOCKS,
    "israd": OBS_POOL_D14C,
}

_FIT_PARAM_SUBSET_NAMES = (
    "log_tau[soil_active]",
    "log_tau[soil_slow]",
    "log_tau[soil_passive]",
    "log_f_transfer[soil_active→soil_slow]",
    "log_f_transfer[soil_slow→soil_passive]",
)


def classify_block(block_name: str) -> str:
    """Map an OE observation-block name to its canonical observation type."""
    for prefix, label in _BLOCK_TO_OBSTYPE.items():
        if block_name.startswith(prefix):
            return label
    return block_name


def _build_forward_oe_context(
    model: EcosystemModel,
    forcing: ForcingData,
    state0: EcosystemState,
    observations: ObservationData,
    opt_fields: tuple[str, ...],
    extra_obs_blocks: Optional[list[ObsBlock]] = None,
) -> dict[str, Any]:
    """Rebuild the OE forward-model context used by the canonical diagnostics."""
    inv_cfg = getattr(model.config, "inversion_raw", {}) or {}
    sigma_pool = float(inv_cfg.get("sigma_pool_14C", 5.0))
    sigma_resp = float(inv_cfg.get("sigma_resp_14C", 10.0))
    sigma_carbon = float(inv_cfg.get("sigma_carbon_gCm2", 1000.0))
    f_hetero = float(inv_cfg.get("f_hetero", 0.0))
    sigma_er_frac = float(inv_cfg.get("sigma_er_frac", 0.15))

    obs_blocks = _build_obs_blocks(
        observations,
        model,
        sigma_pool,
        sigma_resp,
        sigma_carbon,
        f_hetero=f_hetero,
        sigma_er_frac=sigma_er_frac,
    )
    if extra_obs_blocks:
        obs_blocks = obs_blocks + list(extra_obs_blocks)
    if not obs_blocks:
        raise ValueError("oe_diagnostics: no obs blocks built")

    params0 = make_default_params(model.config)
    sa_diag = np.array(_build_sa_diag(model.config, params0, tuple(opt_fields)))
    n_pools = len(model.pool_index)
    cue = float(getattr(model.config.external_inputs, "CUE", 0.47))
    mean_mod, mean_gpp = build_mean_ss_modifier(forcing, params0)
    mean_input = mean_gpp * cue
    _atm_years, _atm_d14c, _t0_year = prepare_c14_spinup(forcing)
    ext_cfg = model.config.external_inputs
    assert ext_cfg is not None
    target_names = list(ext_cfg.partition.keys())
    target_idx = [model.pool_index[n] for n in target_names] or None

    def _forward(x_vec: jnp.ndarray) -> jnp.ndarray:
        p = vector_to_params(x_vec, params0, tuple(opt_fields))
        c12_ss = _analytical_c12_ss(
            p, n_pools, mean_input, mean_mod, target_indices=target_idx
        )
        c14_ss = analytical_c14_ss(
            p, c12_ss, n_pools, mean_input, mean_mod,
            _atm_years, _atm_d14c, _t0_year, target_indices=target_idx,
        )
        state_ss = apply_ss_c12_c14(state0, c12_ss, c14_ss)
        out = run_model(model, forcing, state0=state_ss, params=p)
        return jnp.concatenate([b.predict(out, p) for b in obs_blocks])

    return {
        "obs_blocks": obs_blocks,
        "params0": params0,
        "sa_diag": sa_diag,
        "forward": _forward,
    }


def _expand_constraint_labels(obs_blocks: list[ObsBlock]) -> list[str]:
    """Return one human-readable label per scalar observation in OE order."""
    labels: list[str] = []
    for block in obs_blocks:
        n_obs = int(block.y.shape[0])
        if n_obs == 1:
            labels.append(block.name)
            continue
        labels.extend(f"{block.name}[{i + 1}]" for i in range(n_obs))
    return labels


def _expand_obs_annotations(
    obs_blocks: list[ObsBlock],
    y_obs: np.ndarray,
    y_prior: np.ndarray,
    y_opt: np.ndarray,
    se_diag: np.ndarray,
) -> list[dict[str, Any]]:
    """Return one annotation record per scalar observation in OE order."""
    rows: list[dict[str, Any]] = []
    offset = 0
    for block in obs_blocks:
        obs_family = classify_block(block.name)
        n_obs = int(block.y.shape[0])
        for i in range(n_obs):
            obs_index = offset + i
            obs_label_full = block.name if n_obs == 1 else f"{block.name}[{i + 1}]"
            rows.append(
                {
                    "obs_index": obs_index,
                    "obs_block_name": block.name,
                    "obs_family": obs_family,
                    "obs_label_full": obs_label_full,
                    "y_obs": float(y_obs[obs_index]),
                    "y_prior": float(y_prior[obs_index]),
                    "y_opt": float(y_opt[obs_index]),
                    "obs_variance": float(se_diag[obs_index]),
                    "obs_sigma": float(np.sqrt(max(se_diag[obs_index], 0.0))),
                }
            )
        offset += n_obs
    return rows


def fit_param_subset_indices(state_names: list[str]) -> list[int]:
    """Indices of the fixed fitted-parameter subset used in AK/gain plots."""
    missing = [name for name in _FIT_PARAM_SUBSET_NAMES if name not in state_names]
    if missing:
        raise ValueError(f"Missing fit-parameter subset names: {missing}")
    return [state_names.index(name) for name in _FIT_PARAM_SUBSET_NAMES]


def fit_param_subset_labels() -> list[str]:
    """Short labels for the canonical fitted-parameter AK/gain subset."""
    return [
        r"$\tau_{\mathrm{active}}$",
        r"$\tau_{\mathrm{slow}}$",
        r"$\tau_{\mathrm{passive}}$",
        r"$f_{\mathrm{a\to s}}$",
        r"$f_{\mathrm{s\to p}}$",
    ]


def oe_gain_matrix_diagnostics(
    model: EcosystemModel,
    forcing: ForcingData,
    state0: EcosystemState,
    params_opt: ModelParams,
    observations: ObservationData,
    opt_fields: tuple[str, ...],
    extra_obs_blocks: Optional[list[ObsBlock]] = None,
) -> dict[str, Any]:
    """Return full/subset AK and gain diagnostics at the OE MAP estimate."""
    ctx = _build_forward_oe_context(
        model, forcing, state0, observations, opt_fields, extra_obs_blocks
    )
    obs_blocks = ctx["obs_blocks"]
    params0 = ctx["params0"]
    forward = ctx["forward"]

    x_opt = params_to_vector(params_opt, tuple(opt_fields))
    K = np.array(jax.jacobian(forward)(x_opt), dtype=float)
    y_prior = np.array(forward(params_to_vector(params0, tuple(opt_fields))), dtype=float)
    y_opt = np.array(forward(x_opt), dtype=float)
    se_diag = np.array(jnp.concatenate([b.Se for b in obs_blocks]), dtype=float)
    se_inv = 1.0 / (se_diag + 1e-30)
    sa_diag = np.array(ctx["sa_diag"], dtype=float)
    sa_inv = 1.0 / (sa_diag + 1e-30)
    ktsek = (K.T * se_inv) @ K
    sx = np.linalg.inv(ktsek + np.diag(sa_inv))
    averaging_kernel = sx @ ktsek
    gain_matrix = sx @ (K.T * se_inv)

    state_names = get_param_names(params0, tuple(opt_fields), model)

    y_obs = np.array(jnp.concatenate([b.y for b in obs_blocks]), dtype=float)
    subset_idx = fit_param_subset_indices(state_names)
    constraint_labels = _expand_constraint_labels(obs_blocks)
    obs_annotations = _expand_obs_annotations(
        obs_blocks, y_obs=y_obs, y_prior=y_prior, y_opt=y_opt, se_diag=se_diag
    )
    return {
        "K": K,
        "Sx": sx,
        "averaging_kernel": averaging_kernel,
        "gain_matrix": gain_matrix,
        "y_obs": y_obs,
        "y_prior": y_prior,
        "y_opt": y_opt,
        "Se_diag": se_diag,
        "Se_inv_diag": se_inv,
        "obs_blocks": obs_blocks,
        "obs_annotations": obs_annotations,
        "constraint_labels": constraint_labels,
        "subset_indices": subset_idx,
        "subset_state_names": [state_names[i] for i in subset_idx],
        "subset_averaging_kernel": averaging_kernel[np.ix_(subset_idx, subset_idx)],
        "subset_gain_matrix": gain_matrix[subset_idx, :],
    }


def oe_style_ablation(
    model: EcosystemModel,
    forcing: ForcingData,
    state0: EcosystemState,
    params_opt: ModelParams,
    observations: ObservationData,
    opt_fields: tuple[str, ...],
    extra_obs_blocks: Optional[list[ObsBlock]] = None,
) -> dict[str, Any]:
    """OE-style DFS-by-observation-type ablation at the MAP estimate."""
    inv_cfg = getattr(model.config, "inversion_raw", {}) or {}
    sigma_pool = float(inv_cfg.get("sigma_pool_14C", 5.0))
    sigma_resp = float(inv_cfg.get("sigma_resp_14C", 10.0))
    sigma_carbon = float(inv_cfg.get("sigma_carbon_gCm2", 1000.0))

    obs_blocks = _build_obs_blocks(
        observations,
        model,
        sigma_pool,
        sigma_resp,
        sigma_carbon,
        f_hetero=0.0,
        sigma_er_frac=0.15,
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
    _atm_years, _atm_d14c, _t0_year = prepare_c14_spinup(forcing)
    ext_cfg = model.config.external_inputs
    assert ext_cfg is not None
    target_names = list(ext_cfg.partition.keys())
    target_idx = [model.pool_index[n] for n in target_names] or None

    def _forward(x_vec: jnp.ndarray) -> jnp.ndarray:
        p = vector_to_params(x_vec, params0, tuple(opt_fields))
        c12_ss = _analytical_c12_ss(
            p, n_pools, mean_input, mean_mod, target_indices=target_idx
        )
        c14_ss = analytical_c14_ss(
            p, c12_ss, n_pools, mean_input, mean_mod,
            _atm_years, _atm_d14c, _t0_year, target_indices=target_idx,
        )
        state_ss = apply_ss_c12_c14(state0, c12_ss, c14_ss)
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
                row_mask[block_starts[i] : block_starts[i + 1]] = True
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
        (
            "C_stocks+pool_delta14C+resp_delta14C",
            [OBS_C_STOCKS, OBS_POOL_D14C, OBS_RESP_D14C],
        ),
    ]
    return {
        label: {"dfs_total": dfs, "n_obs": n_obs, "dfs_per_param": dfs_diag}
        for label, obs_types in scenarios
        for dfs, n_obs, dfs_diag in [_dfs_for_subset(obs_types)]
    }


# ── cumulative constraint ladder ─────────────────────────────────────────────
# Maps an OE observation-block name to a ladder constraint *family*. Unlike
# ``classify_block`` (which lumps bulk-layer and density-fraction Δ¹⁴C together as
# OBS_POOL_D14C), the ladder must tell bulk 14C apart from fraction 14C, so it
# keys on the block-name prefixes emitted by the multisite builders
# (``israd_bulk_<year>...`` vs ``israd_fraction...``).
_LADDER_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("c_stock", "C_stocks"),
    ("c_sum", "C_stocks"),
    ("israd_bulk", "bulk_14C"),
    ("israd_fraction", "fraction_14C"),
    # Must precede the generic ``israd`` catch-all below, which would otherwise
    # swallow ``israd_inc_rate_*`` into bulk_14C — silently attributing a
    # *rate* constraint to a radiocarbon family.
    ("israd_inc_rate", "inc_rate"),
    ("resp_14C", "resp_14C"),
    ("er_annual", "ER_annual"),
    ("pool_14C", "bulk_14C"),
    ("israd", "bulk_14C"),
)

# Cumulative rungs, in the order constraints are piled on:
#   1. ¹²C stocks
#   2. + bulk-layer Δ¹⁴C
#   3. + density-fraction Δ¹⁴C
#   4. + respired-CO₂ Δ¹⁴C
LADDER_STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("C_stocks", ("C_stocks",)),
    ("C_stocks+bulk_14C", ("C_stocks", "bulk_14C")),
    ("C_stocks+bulk_14C+fraction_14C", ("C_stocks", "bulk_14C", "fraction_14C")),
    (
        "C_stocks+bulk_14C+fraction_14C+resp_14C",
        ("C_stocks", "bulk_14C", "fraction_14C", "resp_14C"),
    ),
    (
        "C_stocks+bulk_14C+fraction_14C+resp_14C+ER_annual",
        ("C_stocks", "bulk_14C", "fraction_14C", "resp_14C", "ER_annual"),
    ),
    # Incubation rates go last deliberately. Every rung above is an isotopic or
    # stock constraint on the same τ/transfer directions; a lab rate constraint
    # is the one family that observes a decay constant directly, so putting it
    # at the end makes the cumulative increment read as "what a direct rate
    # measurement adds once radiocarbon has said all it can". (Use the Shapley
    # attribution for the order-independent number.)
    (
        "C_stocks+bulk_14C+fraction_14C+resp_14C+ER_annual+inc_rate",
        (
            "C_stocks", "bulk_14C", "fraction_14C", "resp_14C",
            "ER_annual", "inc_rate",
        ),
    ),
)

# The same ladder with the ¹²C-stock rung withheld — the radiocarbon-only ladder.
# Comparing it against LADDER_STEPS isolates how much of the stock constraint is
# genuinely new information versus information the Δ¹⁴C data already carries.
LADDER_STEPS_NO_STOCKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bulk_14C", ("bulk_14C",)),
    ("bulk_14C+fraction_14C", ("bulk_14C", "fraction_14C")),
    (
        "bulk_14C+fraction_14C+resp_14C",
        ("bulk_14C", "fraction_14C", "resp_14C"),
    ),
    (
        "bulk_14C+fraction_14C+resp_14C+inc_rate",
        ("bulk_14C", "fraction_14C", "resp_14C", "inc_rate"),
    ),
)

C14_FAMILIES: tuple[str, ...] = ("bulk_14C", "fraction_14C", "resp_14C")
STOCK_FAMILIES: tuple[str, ...] = ("C_stocks",)
# Laboratory incubation specific-respiration rates. Not radiocarbon and not a
# stock — the only family that observes a rate constant directly.
RATE_FAMILIES: tuple[str, ...] = ("inc_rate",)


def ladder_family(block_name: str) -> str:
    """Map an OE observation-block name to its ladder constraint family."""
    for prefix, family in _LADDER_FAMILY_PREFIXES:
        if block_name.startswith(prefix):
            return family
    return block_name


def oe_ladder_context(
    model: EcosystemModel,
    forcing: ForcingData,
    state0: EcosystemState,
    params_opt: ModelParams,
    observations: ObservationData,
    opt_fields: tuple[str, ...],
    extra_obs_blocks: Optional[list[ObsBlock]] = None,
) -> dict[str, Any]:
    """Linearise the OE forward model once for reuse by every ladder diagnostic.

    Computing the Jacobian is by far the expensive step, so every ladder /
    orthogonality view of a site should share one context rather than re-running
    ``jax.jacobian`` per view.

    Returns the *prewhitened* Jacobian K̃ = Se^(-1/2) · K · Sa^(1/2), in which the
    prior covariance becomes the identity. In these coordinates the information
    matrix of an observation subset is M = K̃ᵀK̃ and its averaging kernel is
    A = (M + I)⁻¹M, so DFS = tr(A). Both tr(A) and diag(A) are unchanged by the
    prewhitening (it is a similarity transform by a diagonal matrix), so the
    numbers match the un-whitened formulation exactly.
    """
    fwd_ctx = _build_forward_oe_context(
        model, forcing, state0, observations, opt_fields, extra_obs_blocks
    )
    obs_blocks = fwd_ctx["obs_blocks"]
    forward = fwd_ctx["forward"]
    sa_diag = np.array(fwd_ctx["sa_diag"], dtype=float)

    x_opt = params_to_vector(params_opt, tuple(opt_fields))
    k = np.array(jax.jacobian(forward)(x_opt), dtype=float)
    se = np.array(jnp.concatenate([b.Se for b in obs_blocks]), dtype=float)
    k_tilde = (k / np.sqrt(se)[:, None]) * np.sqrt(sa_diag)[None, :]

    block_lens = [int(b.y.shape[0]) for b in obs_blocks]
    block_starts = np.cumsum([0] + block_lens)
    rows_by_family: dict[str, list[int]] = {}
    for i, block in enumerate(obs_blocks):
        family = ladder_family(block.name)
        rows_by_family.setdefault(family, []).extend(
            range(int(block_starts[i]), int(block_starts[i + 1]))
        )
    return {
        "k_tilde": k_tilde,
        "rows_by_family": rows_by_family,
        "n_obs_per_family": {f: len(r) for f, r in rows_by_family.items()},
        "n_state": int(k.shape[1]),
        "state_names": get_param_names(fwd_ctx["params0"], tuple(opt_fields), model),
    }


def _rows_for_families(ctx: dict[str, Any], families: tuple[str, ...]) -> list[int]:
    rows: list[int] = []
    for family in families:
        rows.extend(ctx["rows_by_family"].get(family, []))
    return sorted(rows)


def _dfs_from_rows(
    k_tilde: np.ndarray, rows: list[int]
) -> tuple[float, np.ndarray]:
    """DFS and per-parameter DFS for an observation subset (prewhitened space)."""
    n_state = k_tilde.shape[1]
    if not rows:
        return 0.0, np.zeros(n_state)
    kt = k_tilde[rows, :]
    m = kt.T @ kt
    a = np.linalg.solve(m + np.eye(n_state), m)
    return float(np.trace(a)), np.diag(a)


def cumulative_ladder_from_context(
    ctx: dict[str, Any],
    steps: tuple[tuple[str, tuple[str, ...]], ...] = LADDER_STEPS,
) -> list[dict[str, Any]]:
    """Cumulative DFS ladder over ``steps``, using a prebuilt ladder context."""
    results: list[dict[str, Any]] = []
    prev_dfs = 0.0
    for rung, (label, active_families) in enumerate(steps, start=1):
        rows = _rows_for_families(ctx, active_families)
        dfs, dfs_diag = _dfs_from_rows(ctx["k_tilde"], rows)
        added_family = active_families[-1]
        results.append(
            {
                "rung": rung,
                "label": label,
                "added_family": added_family,
                "active_families": list(active_families),
                "n_obs_cumulative": len(rows),
                "n_obs_added_family": ctx["n_obs_per_family"].get(added_family, 0),
                "dfs_cumulative": dfs,
                "dfs_increment": dfs - prev_dfs,
                "dfs_per_param": dfs_diag,
            }
        )
        prev_dfs = dfs
    return results


def _orthonormal_row_basis(mat: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Orthonormal basis (columns) of the row space of ``mat``."""
    if mat.size == 0 or mat.shape[0] == 0:
        return np.zeros((mat.shape[1] if mat.ndim == 2 else 0, 0))
    _, s, vt = np.linalg.svd(mat, full_matrices=False)
    if s.size == 0:
        return np.zeros((mat.shape[1], 0))
    rank = int(np.sum(s > tol * max(float(s[0]), 1e-30)))
    return np.asarray(vt[:rank].T)


def _principal_angles_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between the row spaces of ``a`` and ``b``."""
    qa = _orthonormal_row_basis(a)
    qb = _orthonormal_row_basis(b)
    if qa.shape[1] == 0 or qb.shape[1] == 0:
        return np.array([])
    sv = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.asarray(np.degrees(np.arccos(np.clip(sv, -1.0, 1.0))), dtype=float)


def constraint_orthogonality_from_context(
    ctx: dict[str, Any],
    group_a: tuple[str, ...] = STOCK_FAMILIES,
    group_b: tuple[str, ...] = C14_FAMILIES,
) -> dict[str, Any]:
    """How independent is constraint ``group_a`` from ``group_b``?

    Reports two complementary views, because either alone can mislead:

    * **DFS redundancy** — how much the two groups' information *overlaps*:
      ``redundancy = DFS(A) + DFS(B) − DFS(A∪B)``. Zero means the DFS simply add
      (independent information); large positive means both groups constrain the
      same parameter directions. ``unique_a`` is what A still adds once B is
      already in hand (A added *last*), and ``uniqueness_a = unique_a / DFS(A)``
      normalises that against what A is worth on its own — 1 = fully independent,
      0 = entirely redundant with B. This is magnitude-aware and is the number to
      lead with.
    * **Principal angles** — the geometry of which parameter *directions* each
      group sees, ignoring how strongly it sees them. 90° = orthogonal
      directions, 0° = the groups pin exactly the same direction. A subspace can
      sit at a wide angle yet still be redundant in DFS terms if its sensitivity
      is too weak to beat the prior, hence reporting both.
    """
    k_tilde = ctx["k_tilde"]
    rows_a = _rows_for_families(ctx, group_a)
    rows_b = _rows_for_families(ctx, group_b)
    rows_ab = sorted(set(rows_a) | set(rows_b))

    dfs_a, _ = _dfs_from_rows(k_tilde, rows_a)
    dfs_b, _ = _dfs_from_rows(k_tilde, rows_b)
    dfs_ab, _ = _dfs_from_rows(k_tilde, rows_ab)

    unique_a = dfs_ab - dfs_b
    unique_b = dfs_ab - dfs_a
    angles = _principal_angles_deg(k_tilde[rows_a, :], k_tilde[rows_b, :])

    return {
        "group_a": list(group_a),
        "group_b": list(group_b),
        "n_obs_a": len(rows_a),
        "n_obs_b": len(rows_b),
        "dfs_a_alone": dfs_a,
        "dfs_b_alone": dfs_b,
        "dfs_joint": dfs_ab,
        "redundancy": dfs_a + dfs_b - dfs_ab,
        "unique_a": unique_a,
        "unique_b": unique_b,
        "uniqueness_a": unique_a / dfs_a if dfs_a > 1e-12 else float("nan"),
        "uniqueness_b": unique_b / dfs_b if dfs_b > 1e-12 else float("nan"),
        "principal_angles_deg": angles,
        "min_principal_angle_deg": (
            float(angles.min()) if angles.size else float("nan")
        ),
        "mean_principal_angle_deg": (
            float(angles.mean()) if angles.size else float("nan")
        ),
    }


ALL_FAMILIES: tuple[str, ...] = STOCK_FAMILIES + C14_FAMILIES
ALL_FAMILIES = ALL_FAMILIES + ("ER_annual",) + RATE_FAMILIES


def shapley_dfs_attribution_from_context(
    ctx: dict[str, Any],
    families: tuple[str, ...] = ALL_FAMILIES,
) -> list[dict[str, Any]]:
    """Order-independent attribution of the joint DFS to each constraint family.

    A cumulative ladder charges each family only for what it adds *at its own
    position*, so whichever family goes first collects credit for information the
    later ones would have supplied anyway — the ordering artifact that makes ¹²C
    stocks look dominant when they are in fact near-redundant with Δ¹⁴C. The
    Shapley value removes that dependence by averaging a family's marginal DFS
    over every possible ordering:

        φ_i = Σ_{S ⊆ N\\{i}}  |S|!·(n−|S|−1)!/n! · [ v(S ∪ {i}) − v(S) ]

    with ``v(S)`` the DFS of observation-family subset S. By the efficiency
    property the φ_i sum exactly to the joint DFS ``v(N)``, so they read as a
    clean decomposition of the total information. With the six families in
    ``ALL_FAMILIES`` this is 2⁶ = 64 evaluations of ``v`` against the cached
    Jacobian — still cheap, since the Jacobian is computed once by
    ``oe_ladder_context`` and only row subsets are re-traced.

    Also reports each family's standalone DFS (added first) and unique DFS (added
    last); the spread between those two is precisely the redundancy the Shapley
    value averages over. Families with no observations get exactly zero — check
    ``n_obs`` before reading a zero as "uninformative" rather than "absent".
    """
    fams = tuple(families)
    n = len(fams)
    cache: dict[frozenset[str], float] = {}

    def v(subset: frozenset[str]) -> float:
        if subset not in cache:
            rows = _rows_for_families(ctx, tuple(sorted(subset)))
            dfs, _ = _dfs_from_rows(ctx["k_tilde"], rows)
            cache[subset] = dfs
        return cache[subset]

    total = v(frozenset(fams))
    results: list[dict[str, Any]] = []
    for family in fams:
        others = [g for g in fams if g != family]
        phi = 0.0
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for subset in combinations(others, size):
                base = frozenset(subset)
                phi += weight * (v(base | {family}) - v(base))
        standalone = v(frozenset({family}))
        unique_last = total - v(frozenset(others))
        results.append(
            {
                "family": family,
                "n_obs": ctx["n_obs_per_family"].get(family, 0),
                "shapley_dfs": phi,
                "shapley_share": phi / total if total > 1e-12 else float("nan"),
                "dfs_standalone": standalone,
                "dfs_unique_last": unique_last,
                "dfs_joint_total": total,
            }
        )
    return results


def oe_cumulative_constraint_ladder(
    model: EcosystemModel,
    forcing: ForcingData,
    state0: EcosystemState,
    params_opt: ModelParams,
    observations: ObservationData,
    opt_fields: tuple[str, ...],
    extra_obs_blocks: Optional[list[ObsBlock]] = None,
    steps: tuple[tuple[str, tuple[str, ...]], ...] = LADDER_STEPS,
) -> list[dict[str, Any]]:
    """Cumulative OE constraint ladder (DFS as constraints are piled on).

    Builds every observation block once (¹²C stocks, bulk-layer Δ¹⁴C,
    density-fraction Δ¹⁴C, respired-CO₂ Δ¹⁴C), linearises the forward model at
    the MAP estimate, then reports the total degrees-of-freedom-for-signal (DFS,
    the trace of the averaging kernel over the optimised parameters) obtained
    from each successively larger subset of constraint families:

        1. ¹²C stocks
        2. + bulk-layer Δ¹⁴C
        3. + density-fraction Δ¹⁴C
        4. + respired-CO₂ Δ¹⁴C

    Each rung reports its cumulative DFS and the marginal increment over the
    previous rung — the piece-meal information each family adds. Families with no
    observations at a site contribute a zero-increment rung. Pass
    ``steps=LADDER_STEPS_NO_STOCKS`` for the radiocarbon-only ladder.

    Convenience wrapper over ``oe_ladder_context`` + ``cumulative_ladder_from_context``;
    callers needing several views of one site should build the context once instead.
    """
    ctx = oe_ladder_context(
        model, forcing, state0, params_opt, observations, opt_fields, extra_obs_blocks
    )
    return cumulative_ladder_from_context(ctx, steps=steps)


def oe_constraint_ladder(
    model: EcosystemModel,
    forcing: ForcingData,
    state0: EcosystemState,
    params_opt: ModelParams,
    observations: ObservationData,
    opt_fields: tuple[str, ...],
    extra_obs_blocks: Optional[list[ObsBlock]] = None,
) -> list[dict[str, Any]]:
    """One-constraint-at-a-time OE ladder at the MAP estimate."""
    inv_cfg = getattr(model.config, "inversion_raw", {}) or {}
    sigma_pool = float(inv_cfg.get("sigma_pool_14C", 5.0))
    sigma_resp = float(inv_cfg.get("sigma_resp_14C", 10.0))
    sigma_carbon = float(inv_cfg.get("sigma_carbon_gCm2", 1000.0))

    obs_blocks = _build_obs_blocks(
        observations,
        model,
        sigma_pool,
        sigma_resp,
        sigma_carbon,
        f_hetero=0.0,
        sigma_er_frac=0.15,
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
    _atm_years, _atm_d14c, _t0_year = prepare_c14_spinup(forcing)
    ext_cfg = model.config.external_inputs
    assert ext_cfg is not None
    target_names = list(ext_cfg.partition.keys())
    target_idx = [model.pool_index[n] for n in target_names] or None

    def _forward(x_vec: jnp.ndarray) -> jnp.ndarray:
        p = vector_to_params(x_vec, params0, tuple(opt_fields))
        c12_ss = _analytical_c12_ss(
            p, n_pools, mean_input, mean_mod, target_indices=target_idx
        )
        c14_ss = analytical_c14_ss(
            p, c12_ss, n_pools, mean_input, mean_mod,
            _atm_years, _atm_d14c, _t0_year, target_indices=target_idx,
        )
        state_ss = apply_ss_c12_c14(state0, c12_ss, c14_ss)
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
        results.append(
            {
                "label": block.name,
                "obs_type": obs_type,
                "n_obs": int(block.y.shape[0]),
                "dfs": float(np.trace(a)),
                "dfs_per_param": np.diag(a),
            }
        )
    return results
