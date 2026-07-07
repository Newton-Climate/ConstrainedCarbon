"""
Generic data utilities for the ecosystem-complexity carbon model.

slice_forcing()          — slice all fields of a ForcingData to a time window
attach_atm14C()          — attach interpolated atmospheric Δ¹⁴C to ForcingData
validate_forcing()       — sanity-check a ForcingData, return warning strings
validate_obs_nee_gaps()  — check for long NEE gap runs in ObservationData
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import jax.numpy as jnp

from ecosystem_complexity.config import ModelConfig
from ecosystem_complexity.data.schemas import ForcingData, ObservationData

# Back-compat re-exports — loaders were extracted to data/loaders.py but
# notebooks and site modules still import them from here.
from ecosystem_complexity.data.loaders import (  # noqa: E402,F401
    load_harvard_forest,
    load_barrow_alaska,
)

# Reference epoch
_EPOCH = pd.Timestamp("1970-01-01")

# Half-hourly μmol CO₂ m⁻² s⁻¹ → gC m⁻² per half-hour
_HH_TO_GC = 1e-6 * 12.0 * 1800.0  # = 0.02160


def slice_forcing(forcing: ForcingData, start: int, end: int) -> ForcingData:
    """
    Slice all time-axis fields of a ``ForcingData`` to ``[start:end]``.

    Parameters
    ----------
    forcing :
        Source ``ForcingData`` (all fields shape ``(T, ...)``).
    start, end :
        Integer indices into the time axis.  Follows standard Python slice
        semantics: ``end`` is exclusive, negative indices are supported.

    Returns
    -------
    ForcingData
        New object with every field sliced; same dtype as input.
    """
    return ForcingData(
        time=forcing.time[start:end],
        air_temp=forcing.air_temp[start:end],
        sw_radiation=forcing.sw_radiation[start:end],
        precip=forcing.precip[start:end],
        vpd=forcing.vpd[start:end],
        soil_temp=forcing.soil_temp[start:end],
        soil_moisture=forcing.soil_moisture[start:end],
        snow_depth=forcing.snow_depth[start:end],
        active_layer=forcing.active_layer[start:end],
        delta14C_atm=forcing.delta14C_atm[start:end],
        GPP_obs=forcing.GPP_obs[start:end],
        NPP_obs=forcing.NPP_obs[start:end],
    )


def attach_atm14C(
    forcing: ForcingData,
    atm14C_record: np.ndarray,
    years_daily: np.ndarray,
) -> ForcingData:
    """
    Attach the atmospheric ¹⁴C record to a ForcingData object.

    Interpolates atm14C_record (on years_daily grid) to the exact dates
    stored in forcing.time (days since 1970-01-01).
    """
    # Convert forcing.time (days since epoch) to decimal years
    forcing_time_np = np.array(forcing.time, dtype=np.float64)
    forcing_years = 1970.0 + forcing_time_np / 365.25

    delta14C_interp = np.interp(forcing_years, years_daily, atm14C_record)

    return forcing._replace(
        delta14C_atm=jnp.array(delta14C_interp, dtype=jnp.float32)
    )


# ---------------------------------------------------------------------------
# validate_forcing
# ---------------------------------------------------------------------------


def validate_forcing(
    forcing: ForcingData,
    config: ModelConfig,
) -> list[str]:
    """
    Sanity-check a ForcingData. Returns warning strings (does not raise).
    """
    warnings_out: list[str] = []

    def _nan_frac(arr: jnp.ndarray) -> float:
        a = np.array(arr, dtype=np.float64).ravel()
        return float(np.isnan(a).mean())

    scalar_fields = {
        "air_temp": forcing.air_temp,
        "sw_radiation": forcing.sw_radiation,
        "precip": forcing.precip,
        "vpd": forcing.vpd,
        "snow_depth": forcing.snow_depth,
    }
    for name, arr in scalar_fields.items():
        frac = _nan_frac(arr)
        if frac > 0.05:
            warnings_out.append(f"{name}: {frac*100:.1f}% NaN (threshold 5%)")

    # Physical range checks
    air_temp_np = np.array(forcing.air_temp, dtype=np.float64)
    valid_at = air_temp_np[~np.isnan(air_temp_np)]
    if len(valid_at) > 0:
        if valid_at.min() < -70 or valid_at.max() > 50:
            warnings_out.append(
                f"air_temp out of range [-70, 50]°C: "
                f"min={valid_at.min():.1f}, max={valid_at.max():.1f}"
            )

    sw_np = np.array(forcing.sw_radiation, dtype=np.float64)
    valid_sw = sw_np[~np.isnan(sw_np)]
    if len(valid_sw) > 0:
        if valid_sw.min() < 0 or valid_sw.max() > 1400:
            warnings_out.append(
                f"sw_radiation out of range [0, 1400] W m⁻²: "
                f"min={valid_sw.min():.1f}, max={valid_sw.max():.1f}"
            )

    sm_np = np.array(forcing.soil_moisture, dtype=np.float64).ravel()
    valid_sm = sm_np[~np.isnan(sm_np)]
    if len(valid_sm) > 0:
        if valid_sm.min() < 0 or valid_sm.max() > 0.7:
            warnings_out.append(
                f"soil_moisture out of range [0, 0.7] m³ m⁻³: "
                f"min={valid_sm.min():.3f}, max={valid_sm.max():.3f}"
            )

    precip_np = np.array(forcing.precip, dtype=np.float64)
    valid_pr = precip_np[~np.isnan(precip_np)]
    if len(valid_pr) > 0 and valid_pr.min() < 0:
        warnings_out.append(f"precip has negative values: min={valid_pr.min():.3f}")

    d14c_np = np.array(forcing.delta14C_atm, dtype=np.float64)
    if np.all(np.isnan(d14c_np)):
        warnings_out.append("delta14C_atm is all NaN — call attach_atm14C() to populate")

    # Check for gaps > 14 consecutive NaN days in NEE — requires ObservationData,
    # but we only have ForcingData here; skip if not passed.

    return warnings_out


def validate_obs_nee_gaps(obs: ObservationData, max_gap: int = 14) -> list[str]:
    """Check for runs of > max_gap consecutive NaN days in NEE."""
    warnings_out: list[str] = []
    nee = np.array(obs.NEE, dtype=np.float64)
    is_nan = np.isnan(nee)
    run = 0
    max_run = 0
    for v in is_nan:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    if max_run > max_gap:
        warnings_out.append(
            f"NEE has a run of {max_run} consecutive NaN days (threshold {max_gap})"
        )
    return warnings_out

