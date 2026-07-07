"""
Transfer matrix utilities for the ecosystem-complexity carbon model.

Two representations of the carbon transfer matrix are maintained:

build_transfer_matrix  (numpy, build-time)
    Called once when constructing the model from a YAML config.  Returns a
    dense (n_pools × n_pools) float32 matrix where F[i, j] is the fraction
    of pool-i outflux routed to pool j.  Suitable for inspection, spin-up
    logic, and building the initial ``log_f_transfer`` logits.

get_transfer_matrix  (JAX, runtime / JIT)
    Converts the raw ``ModelParams.log_f_transfer`` logits
    (shape n_pools × n_pools+1) to per-step transfer fractions via softmax,
    then strips the respiration column.  Fully differentiable — intended for
    use inside ``jax.lax.scan`` and JIT-compiled forward steps.

validate_transfer_rules  (diagnostic, build-time)
    Returns a list of human-readable warning strings without raising, so the
    caller can decide whether to surface or suppress them.
"""

from __future__ import annotations

import jax.nn
import jax.numpy as jnp
import numpy as np

from ecosystem_complexity.config import ModelConfig, PoolIndex

# ---------------------------------------------------------------------------
# Build-time matrix (numpy)
# ---------------------------------------------------------------------------


def build_transfer_matrix(config: ModelConfig) -> jnp.ndarray:
    """
    Construct the dense transfer matrix from YAML ``transfer_rules``.

    Entry ``F[i, j]`` is the fraction of pool-i outflux routed to pool j.
    The complement ``1 - row_sum(F[i])`` is heterotrophic respiration.

    This function uses NumPy and is **not** JIT-compatible.  Call it once
    at model-build time.

    Parameters
    ----------
    config :
        Validated ``ModelConfig`` from ``load_config``.

    Returns
    -------
    jnp.ndarray
        Shape ``(n_total_pools, n_total_pools)``, dtype ``float32``.

    Raises
    ------
    ValueError
        If any transfer rule creates a self-loop (``F[i, i] > 0``).
    ValueError
        If any row's fractions sum to > 1.0 (no respiration sink possible).
    """
    idx = PoolIndex(config)
    n_pools = len(idx)
    f_mat = np.zeros((n_pools, n_pools), dtype=np.float32)

    for source, dest, fraction in config.transfer_rules:
        i = idx[source]
        j = idx[dest]
        f_mat[i, j] += float(fraction)

    # ── Validate: no self-transfers ───────────────────────────────────────
    diag = np.diag(f_mat)
    if np.any(diag > 0.0):
        bad = [idx.pool_names[k] for k in range(n_pools) if diag[k] > 0.0]
        raise ValueError(
            f"Self-transfer detected for pool(s) {bad}. "
            f"A pool cannot transfer carbon to itself."
        )

    # ── Validate: row sums ≤ 1.0 ─────────────────────────────────────────
    row_sums = f_mat.sum(axis=1)
    bad_rows = [
        (idx.pool_names[i], float(row_sums[i]))
        for i in range(n_pools)
        if row_sums[i] > 1.0 + 1e-6
    ]
    if bad_rows:
        details = ", ".join(f"{name!r}: {s:.6f}" for name, s in bad_rows)
        raise ValueError(
            f"Transfer row sums exceed 1.0 (remainder must be ≥ 0 for "
            f"heterotrophic respiration): {details}"
        )

    return jnp.array(f_mat, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Runtime matrix (JAX / JIT-compatible)
# ---------------------------------------------------------------------------


def get_transfer_matrix(log_f_transfer: jnp.ndarray, n_pools: int) -> jnp.ndarray:
    """
    Convert ``ModelParams.log_f_transfer`` logits to transfer fractions.

    Applies softmax over the last axis of the ``(n_pools, n_pools+1)``
    logit matrix — the final column absorbs the respiration fraction —
    then returns the first ``n_pools`` columns as the ``(n_pools, n_pools)``
    transfer matrix.

    This function is **fully differentiable** and contains no Python control
    flow on traced values, making it safe for ``jax.jit`` and
    ``jax.lax.scan``.

    Parameters
    ----------
    log_f_transfer :
        Shape ``(n_pools, n_pools + 1)``.  Raw logits from ``ModelParams``.
    n_pools :
        Number of carbon pools (used to slice off the respiration column).

    Returns
    -------
    jnp.ndarray
        Shape ``(n_pools, n_pools)``, dtype inherited from input.
        Row sums are ≤ 1.0 by construction (remainder = respiration).
    """
    # softmax normalises each row across all n_pools+1 destinations
    # (last column = fraction respired).
    f_full = jax.nn.softmax(log_f_transfer, axis=-1)  # (n_pools, n_pools+1)
    return f_full[:, :n_pools]  # (n_pools, n_pools)


# ---------------------------------------------------------------------------
# Diagnostic validator (build-time, returns warnings)
# ---------------------------------------------------------------------------


def validate_transfer_rules(config: ModelConfig) -> list[str]:
    """
    Return a list of diagnostic warning strings for the transfer rules.

    Does **not** raise — the caller decides whether to log or surface warnings.

    Checks:
    * Any pool with no outgoing transfer rules (all outflux is respired —
      may be intentional for leaf/root litter pools, worth flagging).
    * Any non-aboveground pool with no incoming transfers (it can never
      accumulate carbon and will remain at zero forever).

    Parameters
    ----------
    config :
        Validated ``ModelConfig`` from ``load_config``.

    Returns
    -------
    list[str]
        Empty list if no issues found; otherwise one string per warning.
    """
    idx = PoolIndex(config)
    pool_names = idx.pool_names
    ag_names = {p.name for p in config.aboveground_pools}

    sources: set[str] = set()
    destinations: set[str] = set()
    for source, dest, _ in config.transfer_rules:
        sources.add(source)
        destinations.add(dest)

    warnings: list[str] = []

    for name in pool_names:
        if name not in sources:
            warnings.append(
                f"Pool {name!r} has no outgoing transfer rules — "
                f"its entire outflux will be respired."
            )

    for name in pool_names:
        if name not in ag_names and name not in destinations:
            warnings.append(
                f"Non-aboveground pool {name!r} has no incoming transfers — "
                f"it will never receive carbon from other pools."
            )

    return warnings
