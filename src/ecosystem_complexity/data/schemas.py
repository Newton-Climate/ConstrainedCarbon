"""
Data container types for the ecosystem-complexity carbon model.

ForcingData  — meteorological and soil-physics drivers
ObservationData — eddy-covariance fluxes, radiocarbon, and pool-C observations
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import jax.numpy as jnp


class ForcingData(NamedTuple):
    time: jnp.ndarray           # (T,) float32  days since 1970-01-01
    air_temp: jnp.ndarray       # (T,) °C
    sw_radiation: jnp.ndarray   # (T,) W m⁻²
    precip: jnp.ndarray         # (T,) mm day⁻¹
    vpd: jnp.ndarray            # (T,) hPa
    soil_temp: jnp.ndarray      # (T, n_layers) °C — NaN if unavailable
    soil_moisture: jnp.ndarray  # (T, n_layers) m³ m⁻³ — NaN if unavailable
    snow_depth: jnp.ndarray     # (T,) m — NaN if unavailable
    active_layer: jnp.ndarray   # (T,) m — jnp.inf if non-permafrost
    delta14C_atm: jnp.ndarray   # (T,) Δ¹⁴C ‰ — set by attach_atm14C()
    GPP_obs: jnp.ndarray        # (T,) gC m⁻² day⁻¹ — NaN if not available
    NPP_obs: jnp.ndarray        # (T,) gC m⁻² day⁻¹ — NaN if not available


class ObservationData(NamedTuple):
    time: jnp.ndarray       # (T,) float32  days since 1970-01-01
    NEE: jnp.ndarray        # (T,) gC m⁻² day⁻¹ — NaN if missing/rejected
    GPP: jnp.ndarray        # (T,) gC m⁻² day⁻¹
    ER: jnp.ndarray         # (T,) gC m⁻² day⁻¹
    NEE_unc: jnp.ndarray    # (T,) 1-sigma uncertainty gC m⁻² day⁻¹
    delta14C_obs: dict      # {pool_name: (Δ14C ‰, uncertainty ‰, year)}
    deltaD14C_obs: dict     # {pool_name: (Δ-Δ14C ‰, uncertainty ‰, year)}
    C_pools_obs: dict       # {pool_name: (gC m⁻², uncertainty)}
    # Optional — flux-weighted respired CO₂ Δ¹⁴C (T,); NaN at unmeasured days.
    # Defaults to None so existing ObservationData constructors need no change.
    delta14C_resp: Optional[jnp.ndarray] = None
