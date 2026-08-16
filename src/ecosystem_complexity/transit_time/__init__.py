"""Carbon transit-time diagnostics for intrinsic and forced soil systems.

Both functions report the expected age of a carbon atom when it exits the
modelled soil system through respiration.  ``intrinsic_mean_transit_time``
uses the reference-environment turnover parameters; ``realized_mean_transit_time``
uses a repeating sequence of daily decomposition modifiers.  External input
weights specify where and when carbon enters the system—they are not decay
modifiers.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _transfer_fractions(log_f_transfer: np.ndarray, n_pools: int) -> np.ndarray:
    logits = np.asarray(log_f_transfer, dtype=np.float64)
    if logits.shape != (n_pools, n_pools + 1):
        raise ValueError("log_f_transfer must have shape (n_pools, n_pools + 1)")
    logits = logits - logits.max(axis=1, keepdims=True)
    fractions = np.exp(logits)
    return fractions / fractions.sum(axis=1, keepdims=True)


def _input_weights(input_fraction: np.ndarray, n_pools: int) -> np.ndarray:
    weights = np.asarray(input_fraction, dtype=np.float64)
    if (
        weights.shape != (n_pools,)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("input_fraction must be finite, non-negative, and sum to > 0")
    return weights / weights.sum()


def intrinsic_mean_transit_time(
    log_tau: np.ndarray,
    log_f_transfer: np.ndarray,
    input_fraction: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return reference-environment mean transit time in days.

    ``log_tau`` stores intrinsic pool turnover time in log-days.  For transfer
    matrix ``F`` (source rows, destination columns), source-pool expected exit
    ages satisfy ``t = tau + F t``.  The scalar is the external-input-weighted
    mean; the vector gives the expected exit age for carbon entering each pool.
    """
    tau = np.exp(np.asarray(log_tau, dtype=np.float64))
    if tau.ndim != 1 or not np.isfinite(tau).all() or (tau <= 0).any():
        raise ValueError("log_tau must describe finite, positive turnover times")
    fractions = _transfer_fractions(log_f_transfer, tau.size)
    transfer = fractions[:, :tau.size]
    expected = np.linalg.solve(np.eye(tau.size) - transfer, tau)
    weights = _input_weights(input_fraction, tau.size)
    return float(weights @ expected), expected


def realized_mean_transit_time(
    log_tau: np.ndarray,
    log_f_transfer: np.ndarray,
    decomposition_modifier: np.ndarray,
    input_phase_weights: np.ndarray,
    input_fraction: np.ndarray,
    *,
    dt_days: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Return periodic-forcing mean transit time in days.

    Parameters
    ----------
    decomposition_modifier
        ``(n_phase, n_pools)`` multiplier for intrinsic decomposition at every
        phase of a repeating forcing cycle, e.g. ``f_temp * f_moisture *
        f_thaw``.  GPP is represented separately by ``input_phase_weights``.
    input_phase_weights
        Non-negative weights for the timing of soil-C inputs over the forcing
        cycle; daily GPP-derived soil input is the usual choice.

    Notes
    -----
    This solves the periodic recursion ``E_s = dt + B_s.T E_(s+1)`` exactly,
    where ``B_s`` is the one-step retained-carbon transition matrix.  It
    therefore preserves seasonal input timing, environmental limitation, and
    transfer routing.
    """
    tau = np.exp(np.asarray(log_tau, dtype=np.float64))
    if tau.ndim != 1 or not np.isfinite(tau).all() or (tau <= 0).any():
        raise ValueError("log_tau must describe finite, positive turnover times")
    if not np.isfinite(dt_days) or dt_days <= 0:
        raise ValueError("dt_days must be finite and > 0")
    n_pools = tau.size
    modifier = np.asarray(decomposition_modifier, dtype=np.float64)
    if modifier.ndim != 2 or modifier.shape[1] != n_pools or modifier.shape[0] == 0:
        raise ValueError("decomposition_modifier must have shape (n_phase, n_pools)")
    if not np.isfinite(modifier).all() or (modifier < 0).any():
        raise ValueError("decomposition_modifier must be finite and non-negative")
    phase_weights = np.asarray(input_phase_weights, dtype=np.float64)
    if phase_weights.shape != (modifier.shape[0],) or not np.isfinite(phase_weights).all() or (phase_weights < 0).any() or phase_weights.sum() <= 0:
        raise ValueError("input_phase_weights must be finite, non-negative, and sum to > 0")
    weights = _input_weights(input_fraction, n_pools)
    fractions = _transfer_fractions(log_f_transfer, n_pools)
    transfer = fractions[:, :n_pools]
    n_phase = modifier.shape[0]
    decomp_fraction = modifier * (dt_days / tau)[None, :]
    if (decomp_fraction > 1.0 + 1e-12).any():
        raise ValueError("forcing violates Euler stability: modifier * dt_days / tau > 1")

    system = sparse.lil_matrix((n_phase * n_pools, n_phase * n_pools))
    rhs = np.full(n_phase * n_pools, dt_days)
    identity = np.eye(n_pools)
    for phase in range(n_phase):
        retained = np.diag(1.0 - decomp_fraction[phase]) + transfer.T * decomp_fraction[phase][None, :]
        row = slice(phase * n_pools, (phase + 1) * n_pools)
        next_phase = (phase + 1) % n_phase
        col = slice(next_phase * n_pools, (next_phase + 1) * n_pools)
        system[row, row] = identity
        system[row, col] = -retained.T
    expected = spsolve(system.tocsr(), rhs).reshape(n_phase, n_pools)
    phase_weights = phase_weights / phase_weights.sum()
    return float(phase_weights @ (expected @ weights)), expected
