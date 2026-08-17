"""ISRaD laboratory incubations as a bulk turnover-rate constraint.

Every other ISRaD constraint in the ladder — bulk ¹⁴C, fraction ¹⁴C, respired
¹⁴C — is an *isotopic* observation. Turnover time is recovered only indirectly,
through how fast a pool's Δ¹⁴C tracks the bomb curve. Incubation respiration
rates are different in kind: ``inc_flux`` reported as ``mgC/gC soil/day`` is a
specific respiration rate, i.e. a bulk decay constant, observed directly.

The observation
---------------
``inc_flux`` in ``mgC/gC soil/day`` converts to day⁻¹ by a factor of 1e-3.
The ``mgC/g dry soil/day`` variant is divided by the layer's carbon fraction
(``lyr_c_org``, %) first; rows lacking ``lyr_c_org`` cannot be converted and are
dropped rather than guessed at.

The forward model
-----------------
A jar of soil at temperature ``T_inc``, rewetted, fully thawed, and receiving no
fresh litter respires

.. math::

    k_{inc} = \\frac{\\sum_i \\mathrm{resp\\_frac}_i \\cdot (C_i/\\tau_i)
                     \\cdot f_{temp}(T_{inc})}{\\sum_i C_i}

summed over the *soil* pools only. This deliberately does **not** reuse
``out.Rh_by_pool``: that carries the field temperature, moisture and thaw
scalars baked in, and the whole point of an incubation is that it replaces them
with controlled ones. The intrinsic rate ``resp_frac_i · C_i/τ_i`` is
reconstructed from the parameters, then lab conditions are applied:

* ``f_temp`` at the reported ``inc_temp``, using the model's own retrieved Q10 —
  so the temperature correction stays internally consistent with the parameter
  being retrieved rather than assuming a literature Q10;
* ``f_moisture = 1``: incubations are rewetted to near-optimal moisture, which
  is where the model's Gaussian moisture response peaks by construction;
* ``thawed_frac = 1``: a jar on a bench is thawed.

``C_i`` is the time-mean modelled pool size over the run — the incubated soil is
the field soil, so its pool partition is the model's.

What this constrains
--------------------
The predictor is ``Σ resp_frac_i·C_i/τ_i / Σ C_i``, a C-weighted mean of the
pool rate constants. Each pool's leverage is exactly its share of the
incubation respiration flux — ``∂k/∂log τ_i = −resp_frac_i·(C_i/τ_i)·f_temp/ΣC``
— so which τ this actually informs depends on the stock partition rather than
being fixed by the operator. At a realistic equilibrium partition the fast pool
dominates by orders of magnitude (~10³ at EML); at a partition holding most
carbon in a slow pool the leverage spreads out. Either way this is orthogonal
to, not a substitute for, the ¹⁴C ladder: bulk ¹⁴C pins the *age* structure,
this pins the *rate* structure.

Caveats carried by the data, not fixable here
---------------------------------------------
* **Incubation length biases the rate, strongly.** Across the 62 usable ISRaD
  sites the implied bulk turnover time is ~5.7 yr from ``<2 weeks``
  incubations, ~23 yr from ``<1 year``, and ~203 yr from ``>1 year``: a ~35×
  spread from protocol alone. Short incubations are dominated by the rewetting
  flush of labile carbon (k too high); long ones deplete that carbon and drift
  toward the slow pool (k too low). Neither is the field rate. This is by far
  the largest error term here — larger than the σ floor — so ``duration_types``
  exists to restrict a run to one protocol class. Mixing classes within a site
  pools incompatible measurements; the returned ``duration_mix`` reports what
  went into each block so this stays visible.
* ``live roots`` and ``soil w/ live roots`` are excluded: their flux is partly
  autotrophic, which the model's Rh does not represent. This matches the
  reservoir-membership argument that excludes ``roots`` in
  :mod:`ecosystem_complexity.data.fraction_mapping`.
* Anaerobic incubations are excluded — different terminal electron acceptor,
  different rate law.
"""
from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ecosystem_complexity.inference._helpers import ObsBlock
from ecosystem_complexity.processes.climate import f_temp
from ecosystem_complexity.data.paths import ISRAD_INCUBATION

logger = logging.getLogger(__name__)

# Incubation types whose respiration is heterotrophic decomposition of the soil
# matrix. `live roots` / `soil w/ live roots` carry autotrophic flux; `litter`
# is a different reservoir from the bulk soil this rate is normalised against.
SOIL_INC_TYPES: frozenset[str] = frozenset({"root-picked soil", "soil w/ dead roots"})

# inc_flux unit → multiplier taking the reported value to day⁻¹, given carbon
# basis. `None` means the value is per gram *dry soil* and needs lyr_c_org.
_PER_GC_UNITS = "mgC/gC soil/day"
_PER_GSOIL_UNITS = "mgC/g dry soil/day"

# Relative σ floor on k. Incubation rates carry protocol variance (sieving,
# preincubation length, rewetting intensity) far exceeding analytical error, so
# a fractional floor rather than the observed scatter alone.
_SIGMA_REL_FLOOR = 0.5


def load_incubation_rates(
    israd_name: str,
    *,
    path: str | None = None,
    inc_types: frozenset[str] = SOIL_INC_TYPES,
    duration_types: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Qualifying incubation rows for one ISRaD site, converted to k in day⁻¹.

    ``duration_types`` restricts to ISRaD ``inc_duration_type`` classes
    (``"<2 weeks"``, ``"<1 month"``, ``"<1 year"``, ``">1 year"``); ``None``
    keeps all, which mixes protocols whose implied rates differ by ~35× — see
    the module docstring before doing that.

    Returns a frame with ``k_obs`` (day⁻¹) and ``inc_temp`` (°C) added, one row
    per usable incubation measurement. Empty if the site has none.
    """
    df = pd.read_csv(path or ISRAD_INCUBATION, low_memory=False)
    df = df[df["site_name"] == israd_name].copy()
    if df.empty:
        return df

    n0 = len(df)
    df = df[df["inc_type"].isin(inc_types)]
    df = df[df["inc_anaerobic"].ne("yes")]
    if duration_types is not None:
        df = df[df["inc_duration_type"].isin(duration_types)]
    for col in ("inc_flux", "inc_temp", "lyr_c_org"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df.dropna(subset=["inc_flux", "inc_temp"])
    df = df[df["inc_flux"] > 0.0]

    per_gc = df["inc_flux_units"] == _PER_GC_UNITS
    per_gsoil = df["inc_flux_units"] == _PER_GSOIL_UNITS

    k = pd.Series(np.nan, index=df.index, dtype=float)
    # mgC per gC per day → day⁻¹
    k[per_gc] = df.loc[per_gc, "inc_flux"] * 1e-3
    # mgC per g dry soil per day → divide by gC/g soil (lyr_c_org is a percent)
    c_frac = df.loc[per_gsoil, "lyr_c_org"] / 100.0
    k[per_gsoil] = (df.loc[per_gsoil, "inc_flux"] * 1e-3) / c_frac.where(c_frac > 0)

    df["k_obs"] = k
    df = df.dropna(subset=["k_obs"])
    df = df[np.isfinite(df["k_obs"]) & (df["k_obs"] > 0.0)]

    if len(df) < n0:
        logger.debug(
            "incubation rates for %s: kept %d of %d rows", israd_name, len(df), n0
        )
    return df


def _soil_pool_columns(model) -> tuple[jnp.ndarray, jnp.ndarray]:
    """(pool column indices, per-pool layer index) for the soil pools only."""
    cols: list[int] = []
    layer_of: list[int] = []
    for li, layer in enumerate(model.config.soil_layers):
        sl = model.pool_index.layer_slices[layer.name]
        for c in range(sl.start, sl.stop):
            cols.append(c)
            layer_of.append(li)
    return (
        jnp.asarray(cols, dtype=jnp.int32),
        jnp.asarray(layer_of, dtype=jnp.int32),
    )


def build_incubation_rate_blocks(
    israd_name: str,
    model,
    *,
    path: str | None = None,
    inc_types: frozenset[str] = SOIL_INC_TYPES,
    duration_types: frozenset[str] | None = None,
    temp_decimals: int = 0,
    min_rows: int = 1,
    name_prefix: str = "israd_inc_rate",
) -> list[dict[str, Any]]:
    """Bulk specific-respiration-rate observation blocks for one site.

    Rows are grouped by incubation temperature (rounded to ``temp_decimals``),
    since the predictor is temperature-dependent and pooling across temperatures
    would compare the model against a mixture it cannot reproduce. Each group
    yields one block whose observation is the group-mean ``k_obs``.

    Returns one dict per block with ``name``, ``inc_temp``, ``k_obs``,
    ``sigma``, ``n``, ``duration_mix`` and ``block`` keys — mirroring
    :func:`ecosystem_complexity.data.israd_observations.build_perlayer_mixture_obs_blocks`.
    ``duration_mix`` counts the ``inc_duration_type`` classes pooled into the
    block; more than one entry means protocols with very different biases were
    averaged together.
    """
    df = load_incubation_rates(
        israd_name, path=path, inc_types=inc_types, duration_types=duration_types
    )
    if df.empty:
        return []

    cols, layer_of = _soil_pool_columns(model)
    n_pools = len(model.pool_index)

    rows: list[dict[str, Any]] = []
    for temp, grp in df.groupby(df["inc_temp"].round(temp_decimals)):
        if len(grp) < min_rows:
            continue
        k_vals = grp["k_obs"].to_numpy(dtype=float)
        k_mean = float(k_vals.mean())
        # Scatter across replicates, floored at a fraction of the mean.
        k_sigma = float(np.std(k_vals, ddof=1)) if len(k_vals) > 1 else 0.0
        k_sigma = max(k_sigma, _SIGMA_REL_FLOOR * k_mean)

        T_inc = float(temp)

        def _predict(
            out,
            p,
            T_inc: float = T_inc,
            cols: jnp.ndarray = cols,
            layer_of: jnp.ndarray = layer_of,
            n_pools: int = n_pools,
        ) -> jnp.ndarray:
            # Intrinsic per-pool rate, free of field environmental scalars.
            f_full = jax.nn.softmax(p.log_f_transfer, axis=-1)
            resp_frac = 1.0 - f_full[:, :n_pools].sum(axis=-1)  # (n_pools,)
            tau = jnp.exp(p.log_tau)  # (n_pools,)

            C = jnp.mean(out.C12, axis=0)  # (n_pools,) time-mean stock
            C_s = C[cols]
            resp_s = resp_frac[cols]
            tau_s = tau[cols]

            # Lab conditions: model Q10 at T_inc, moisture optimal, fully thawed.
            ft = f_temp(jnp.asarray(T_inc), p.log_Q10)  # (n_layers,)
            ft_s = ft[layer_of]

            numer = jnp.sum(resp_s * (C_s / tau_s) * ft_s)
            return jnp.atleast_1d(numer / (jnp.sum(C_s) + 1e-30))

        block = ObsBlock(
            name=f"{name_prefix}_{T_inc:g}C",
            y=jnp.asarray([k_mean], dtype=jnp.float32),
            Se=jnp.asarray([k_sigma**2], dtype=jnp.float32),
            predict=_predict,
        )
        rows.append({
            "name": block.name,
            "inc_temp": T_inc,
            "k_obs": k_mean,
            "sigma": k_sigma,
            "n": int(len(k_vals)),
            "duration_mix": grp["inc_duration_type"].value_counts().to_dict(),
            "block": block,
        })
    return rows
