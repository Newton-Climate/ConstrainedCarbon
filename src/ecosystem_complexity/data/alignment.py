"""
Map measured depth-profile data onto the model layer grid.

All depths must be in metres, positive downward from surface, before calling
align_to_layers(). Unit/sign conversion is the caller's responsibility.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from ecosystem_complexity.model.configuration import ModelConfig


def align_to_layers(
    meas_depths_top_m: list[float] | np.ndarray,
    meas_depths_bot_m: list[float] | np.ndarray,
    values: list[float] | np.ndarray,
    uncertainties: list[float] | np.ndarray,
    config: ModelConfig,
    method: str = "depth_weighted",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Map measurements at arbitrary depths onto the model layer grid.

    Parameters
    ----------
    meas_depths_top_m, meas_depths_bot_m :
        Top and bottom depth (m, positive downward) of each measurement.
    values, uncertainties :
        Measured values and their 1-sigma uncertainties.
    config :
        Model configuration (used to read soil layer boundaries).
    method :
        'depth_weighted' (default) or 'nearest'.

    Returns
    -------
    layer_values, layer_uncertainties :
        Arrays of shape (n_layers,) with NaN where no measurement maps.
    """
    tops = np.asarray(meas_depths_top_m, dtype=np.float64)
    bots = np.asarray(meas_depths_bot_m, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    uncs = np.asarray(uncertainties, dtype=np.float64)

    n_layers = len(config.soil_layers)
    out_vals = np.full(n_layers, np.nan)
    out_uncs = np.full(n_layers, np.nan)

    if method == "depth_weighted":
        for li, layer in enumerate(config.soil_layers):
            lt = layer.depth_top_m
            lb = layer.depth_bot_m
            layer_thickness = lb - lt

            weights = np.maximum(
                0.0,
                np.minimum(bots, lb) - np.maximum(tops, lt),
            ) / layer_thickness

            total_weight = weights.sum()
            if total_weight == 0.0:
                continue

            out_vals[li] = np.sum(weights * vals) / total_weight
            # Combined uncertainty: propagate as weighted 1/sigma²
            valid = uncs > 0
            if valid.any():
                inv_var = np.where(valid, 1.0 / uncs**2, 0.0)
                weighted_inv_var = np.sum(weights * inv_var)
                if weighted_inv_var > 0:
                    out_uncs[li] = 1.0 / np.sqrt(weighted_inv_var)
                else:
                    out_uncs[li] = np.nan
            else:
                out_uncs[li] = np.nan

    elif method == "nearest":
        meas_mids = 0.5 * (tops + bots)
        layer_mids = np.array(
            [0.5 * (l.depth_top_m + l.depth_bot_m) for l in config.soil_layers]
        )

        for mi, mid in enumerate(meas_mids):
            closest = int(np.argmin(np.abs(layer_mids - mid)))
            li = closest
            if np.isnan(out_vals[li]):
                out_vals[li] = vals[mi]
                out_uncs[li] = uncs[mi]
            else:
                # Average with existing via 1/sigma² weighting
                s1, s2 = out_uncs[li], uncs[mi]
                if s1 > 0 and s2 > 0:
                    w1, w2 = 1.0 / s1**2, 1.0 / s2**2
                    out_vals[li] = (w1 * out_vals[li] + w2 * vals[mi]) / (w1 + w2)
                    out_uncs[li] = 1.0 / np.sqrt(w1 + w2)
                else:
                    out_vals[li] = np.nanmean([out_vals[li], vals[mi]])
    else:
        raise ValueError(f"Unknown method {method!r}; use 'depth_weighted' or 'nearest'")

    return jnp.array(out_vals, dtype=jnp.float32), jnp.array(out_uncs, dtype=jnp.float32)
