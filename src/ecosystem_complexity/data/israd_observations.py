"""Reusable builders for ISRaD-derived pool and fraction observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ecosystem_complexity._oe_helpers import ObsBlock
from ecosystem_complexity.config import PoolIndex


@dataclass(frozen=True)
class FractionMappingRule:
    """Rule for mapping ISRaD fraction rows onto a model pool."""

    pool_name: str
    fraction_property: str
    depth_range_cm: tuple[float, float] | None = None


def add_layer_midpoint(
    df: pd.DataFrame,
    top_col: str = "lyr_top",
    bot_col: str = "lyr_bot",
    out_col: str = "lyr_mid",
) -> pd.DataFrame:
    """Return a copy with numeric depth midpoint column added."""
    out = df.copy()
    out[out_col] = 0.5 * (
        pd.to_numeric(out[top_col], errors="coerce")
        + pd.to_numeric(out[bot_col], errors="coerce")
    )
    return out


def summarize_by_depth(
    df: pd.DataFrame,
    value_col: str,
    depth_pairs: list[tuple[str, tuple[float, float]]],
    *,
    min_n: int = 2,
    sigma_floor: float = 25.0,
    depth_col: str = "lyr_mid",
) -> dict[str, tuple[float, float, int]]:
    """Summarize a numeric ISRaD column into pool bins by depth midpoint."""
    sub = df.dropna(subset=[value_col, depth_col]).copy()
    out: dict[str, tuple[float, float, int]] = {}
    for pool_name, (z_top, z_bot) in depth_pairs:
        vals = pd.to_numeric(
            sub.loc[(sub[depth_col] >= z_top) & (sub[depth_col] < z_bot), value_col],
            errors="coerce",
        ).dropna()
        if len(vals) < min_n:
            continue
        mean_val = float(vals.mean())
        if len(vals) > 1:
            sigma_raw = float(vals.std(ddof=1))
        else:
            sigma_raw = float("nan")
        if not np.isfinite(sigma_raw) or sigma_raw <= 0.0:
            sigma_val = float(sigma_floor)
        else:
            sigma_val = float(max(sigma_raw, sigma_floor))
        out[pool_name] = (mean_val, sigma_val, int(len(vals)))
    return out


def obs_dict_from_single_year_summary(
    summary: dict[str, tuple[float, float, int]],
    forcing_time: jnp.ndarray,
    obs_year: float,
    *,
    year_offset: float = 0.5,
) -> dict[str, jnp.ndarray]:
    """Convert single-year pool summaries into sparse `(T,)` observation arrays."""
    time_np = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    t_len = len(time_np)
    obs_dict: dict[str, jnp.ndarray] = {}
    for pool_name, (mean_val, _, _) in summary.items():
        arr = np.full(t_len, np.nan, dtype=np.float32)
        t_idx = int(np.argmin(np.abs(years_np - (float(obs_year) + year_offset))))
        arr[t_idx] = mean_val
        obs_dict[pool_name] = jnp.array(arr)
    return obs_dict


def obs_blocks_from_single_year_summary(
    summary: dict[str, tuple[float, float, int]],
    pool_index: PoolIndex,
    forcing_time: jnp.ndarray,
    obs_year: float,
    *,
    name_prefix: str = "israd_layer",
) -> list[ObsBlock]:
    """Convert single-year pool summaries into one ObsBlock per pool."""
    time_np = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    t_yr = jnp.array(
        np.where((years_np >= obs_year) & (years_np < obs_year + 1))[0],
        dtype=jnp.int32,
    )
    if t_yr.shape[0] == 0:
        t_yr = jnp.array([int(np.argmin(np.abs(years_np - (float(obs_year) + 0.5))))], dtype=jnp.int32)

    blocks: list[ObsBlock] = []
    for pool_name, (mean_val, sigma_val, _) in summary.items():
        if pool_name not in pool_index.pool_names:
            continue
        pi = int(pool_index[pool_name])
        blocks.append(ObsBlock(
            name=f"{name_prefix}_{pool_name}",
            y=jnp.array([mean_val], dtype=jnp.float32),
            Se=jnp.array([sigma_val ** 2], dtype=jnp.float32),
            predict=lambda out, p, t=t_yr, pi=pi: jnp.mean(out.delta14C[t, pi], keepdims=True),
        ))
    return blocks


def bulk_mixture_obs_block(
    name: str,
    mean_val: float,
    sigma_val: float,
    t_idx: jnp.ndarray,
) -> ObsBlock:
    """Whole-sample bulk Δ¹⁴C: the C-mass-weighted mixture over ALL pools.

    A bulk soil measurement samples every kinetic pool present, so its Δ¹⁴C is
    the carbon-mass-weighted mean of the pool signatures:

        Δ¹⁴C_bulk = Σ_i C12_i · Δ¹⁴C_i / Σ_i C12_i

    This is the counterpart to ``obs_blocks_from_single_year_summary``, which
    instead asserts that a bulk layer *is* whichever single pool owns its depth
    midpoint. That assertion only holds if the pools are depth horizons; when the
    pools are co-located kinetic fractions (free light / occluded / heavy all
    coexisting at every depth) the mixture above is the correct operator and the
    depth→pool assignment is not meaningful.

    Because the mixture is a single number per time index, a co-located-pool model
    predicts ONE bulk Δ¹⁴C per date. Depth-resolved bulk profiles therefore have
    to be aggregated (C-mass-weighted) into one whole-profile value before being
    compared against it — see ``build_bulk_14C_blocks`` in the multisite driver.
    """
    def _predict(out: Any, p: Any, t: jnp.ndarray = t_idx) -> jnp.ndarray:
        c12 = out.C12[t]  # (n_t, n_pools)
        d14 = out.delta14C[t]  # (n_t, n_pools)
        num = (c12 * d14).sum(axis=-1)
        den = c12.sum(axis=-1) + 1e-30
        return jnp.mean(num / den, keepdims=True)

    return ObsBlock(
        name=name,
        y=jnp.array([mean_val], dtype=jnp.float32),
        Se=jnp.array([sigma_val**2], dtype=jnp.float32),
        predict=_predict,
    )


def build_fraction_obs_blocks(
    df: pd.DataFrame,
    forcing_time: jnp.ndarray,
    pool_index: PoolIndex,
    rules: list[FractionMappingRule],
    *,
    entry_to_year: dict[str, int] | None = None,
    exclude_pool_years: set[tuple[str, int]] | None = None,
    value_col: str = "frc_14c",
    weight_col: str | None = None,
    fraction_col: str = "frc_property",
    entry_col: str = "entry_name",
    year_col: str | None = None,
    min_sigma: float = 15.0,
    singleton_sigma: float = 50.0,
    name_prefix: str = "israd",
) -> list[dict[str, Any]]:
    """Aggregate ISRaD fraction rows into per-pool, per-year observation summaries.

    Returns a list of row dicts with `name`, `pool_name`, `obs_year`, `mean`, `sigma`, `n`,
    and `block` keys so callers can print or convert further without re-running the aggregation.
    """
    exclude_pool_years = exclude_pool_years or set()
    time_np = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    frac_series = df[fraction_col].astype(str).str.strip().str.lower()

    records: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for rule in rules:
        mask = frac_series == rule.fraction_property.strip().lower()
        if rule.depth_range_cm is not None:
            z_top, z_bot = rule.depth_range_cm
            if "lyr_mid" not in df.columns:
                raise ValueError("build_fraction_obs_blocks requires `lyr_mid` for depth-filtered rules")
            mask &= (df["lyr_mid"] >= z_top) & (df["lyr_mid"] < z_bot)
        sub = df.loc[mask].copy()
        if sub.empty:
            continue

        for _, row in sub.iterrows():
            if year_col is not None:
                raw_year = row.get(year_col, np.nan)
                if raw_year is None or (isinstance(raw_year, float) and np.isnan(raw_year)):
                    continue
                obs_year = int(float(raw_year))
            else:
                entry = str(row.get(entry_col, "")).strip()
                if entry_to_year is None or entry not in entry_to_year:
                    continue
                obs_year = int(entry_to_year[entry])

            if (rule.pool_name, obs_year) in exclude_pool_years:
                continue

            raw_val = row.get(value_col, np.nan)
            if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
                continue
            obs_val = float(raw_val)
            if not np.isfinite(obs_val):
                continue

            obs_weight = 1.0
            if weight_col is not None:
                raw_weight = row.get(weight_col, np.nan)
                try:
                    weight_val = float(raw_weight)
                    if np.isfinite(weight_val) and weight_val > 0:
                        obs_weight = weight_val
                except (TypeError, ValueError):
                    pass

            key = (rule.pool_name, obs_year)
            records.setdefault(key, []).append((obs_val, obs_weight))

    rows: list[dict[str, Any]] = []
    for (pool_name, obs_year), pts in sorted(records.items()):
        vals = np.array([v for v, _ in pts], dtype=float)
        wts = np.array([w for _, w in pts], dtype=float)
        wts = wts / wts.sum()
        mean_val = float(np.dot(wts, vals))
        n_obs = len(vals)
        if n_obs > 1:
            var_w = float(np.dot(wts, (vals - mean_val) ** 2))
            n_eff = 1.0 / float(np.dot(wts, wts))
            sigma_val = float(np.sqrt(var_w / max(n_eff, 1.0)))
        else:
            sigma_val = singleton_sigma
        sigma_val = max(sigma_val, min_sigma)

        t_idx = int(np.argmin(np.abs(years_np - (float(obs_year) + 0.5))))
        t_arr = jnp.array([t_idx], dtype=jnp.int32)
        pi = int(pool_index[pool_name])
        block = ObsBlock(
            name=f"{name_prefix}_{pool_name}_{obs_year}",
            y=jnp.array([mean_val], dtype=jnp.float32),
            Se=jnp.array([sigma_val ** 2], dtype=jnp.float32),
            predict=lambda out, p, t=t_arr, pi=pi: out.delta14C[t, pi],
        )
        rows.append({
            "name": block.name,
            "pool_name": pool_name,
            "obs_year": obs_year,
            "mean": mean_val,
            "sigma": sigma_val,
            "n": n_obs,
            "block": block,
        })
    return rows


# Fraction property → model pool: an ISRaD density-fraction property name
# ("free light", "heavy", …) mapped to the kinetic pool it best represents.
FractionToPool = dict[str, str]


def _perlayer_time_window(forcing_time: Any, obs_year: float) -> jnp.ndarray:
    """Integer time indices falling within ``obs_year`` (nearest day fallback)."""
    time_np = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    idx = np.where((years_np >= obs_year) & (years_np < obs_year + 1))[0]
    if idx.size == 0:
        idx = np.array([int(np.argmin(np.abs(years_np - (obs_year + 0.5))))])
    return jnp.asarray(idx, dtype=jnp.int32)


def _perlayer_measured_weights(
    fraction_df: pd.DataFrame,
    z_top: float,
    z_bot: float,
    pool_order: tuple[str, ...],
    *,
    value_col: str,
    pool_col: str,
    mid_col: str,
) -> np.ndarray | None:
    """Normalised carbon-mass weight vector for a depth bin, or ``None``.

    Weights are the mean measured fraction mass (e.g. ``frc_mass_perc``) per
    pool across the profiles sampled in the bin, normalised to sum to one.
    Returns ``None`` when no usable fraction rows fall in the bin.
    """
    sub = fraction_df[
        (fraction_df[mid_col] >= z_top)
        & (fraction_df[mid_col] < z_bot)
        & fraction_df[pool_col].notna()
        & fraction_df[value_col].notna()
    ]
    if len(sub) == 0:
        return None
    per_pool = sub.groupby(pool_col)[value_col].mean()
    w = np.array([float(per_pool.get(p, 0.0)) for p in pool_order], dtype=float)
    total = float(w.sum())
    if total <= 0.0:
        return None
    return w / total


def build_perlayer_mixture_obs_blocks(
    layer_df: pd.DataFrame,
    pool_index: PoolIndex,
    forcing_time: Any,
    *,
    depth_bins: list[tuple[str, tuple[float, float]]],
    obs_year: float,
    fraction_df: pd.DataFrame | None = None,
    fraction_to_pool: FractionToPool | None = None,
    pool_order: tuple[str, ...] | None = None,
    layer_value_col: str = "lyr_14c",
    layer_mass_col: str = "lyr_soc",
    layer_top_col: str = "lyr_top",
    layer_bot_col: str = "lyr_bot",
    fraction_value_col: str = "frc_mass_perc",
    fraction_property_col: str = "frc_property",
    fraction_top_col: str = "lyr_top",
    fraction_bot_col: str = "lyr_bot",
    sigma_floor: float = 15.0,
    min_layers: int = 2,
    name_prefix: str = "israd_perlayer",
) -> list[dict[str, Any]]:
    """Per-layer, carbon-mass-weighted mixture bulk Δ¹⁴C observation blocks.

    Site-agnostic implementation of the "fraction-weighted-by-layer" bulk ¹⁴C
    operator.  Each depth bin's whole-soil Δ¹⁴C is treated as a mixture of the
    model pools, ``Σ_p w_{p,z}·Δ¹⁴C_p``, with per-bin mixing weights ``w_{p,z}``:

    * **measured** — if ``fraction_df`` supplies density-fraction masses in the
      bin (mapped to pools via ``fraction_to_pool``), the normalised measured
      masses are used as *fixed* weights and the predictor is linear in the pool
      Δ¹⁴C;
    * **model C12 fallback** — otherwise the bin is weighted by the model's own
      pool masses, ``Σ_p (C12_p/ΣC12)·Δ¹⁴C_p`` (the ``bulk_delta14C`` quantity),
      which varies with the parameters during the inversion.

    The observed value per bin is the carbon-mass (``layer_mass_col``) weighted
    mean of ``layer_value_col`` across the profiles sampled in the bin, with the
    across-layer spread (floored at ``sigma_floor``) as σ.

    Because a non-depth-resolved pool model can only produce as many independent
    Δ¹⁴C values as it has pools, bulk information from these blocks is rank-limited
    to the pool count regardless of how many layers are supplied — deep, fallback
    bins are collinear and add no independent degrees of freedom.

    Parameters mirror the ISRaD flat-file column names; pass a pre-filtered
    ``layer_df`` (and optional ``fraction_df``) restricted to the site's profiles.
    Returns one row dict per emitted bin with ``name``, ``bin_label``,
    ``depth_cm``, ``mean``, ``sigma``, ``n``, ``weight_source`` (``"measured"``
    or ``"model_c12"``), ``weights`` and ``block`` keys.
    """
    if pool_order is None:
        pool_order = tuple(pool_index.pool_names)
    pool_cols = jnp.asarray([int(pool_index[p]) for p in pool_order], dtype=jnp.int32)

    lay = layer_df.copy()
    for col in (layer_value_col, layer_top_col, layer_bot_col):
        lay[col] = pd.to_numeric(lay[col], errors="coerce")
    if layer_mass_col in lay.columns:
        lay[layer_mass_col] = pd.to_numeric(lay[layer_mass_col], errors="coerce")
    lay = lay.dropna(subset=[layer_value_col, layer_top_col, layer_bot_col])
    lay = add_layer_midpoint(lay, layer_top_col, layer_bot_col, "lyr_mid")

    frac: pd.DataFrame | None = None
    if fraction_df is not None and fraction_to_pool is not None and len(fraction_df):
        frac = fraction_df.copy()
        frac[fraction_value_col] = pd.to_numeric(frac[fraction_value_col], errors="coerce")
        frac["_pool"] = frac[fraction_property_col].map(fraction_to_pool)
        frac = add_layer_midpoint(frac, fraction_top_col, fraction_bot_col, "lyr_mid")

    t_arr = _perlayer_time_window(forcing_time, obs_year)

    rows: list[dict[str, Any]] = []
    for bin_label, (z_top, z_bot) in depth_bins:
        sub = lay[(lay["lyr_mid"] >= z_top) & (lay["lyr_mid"] < z_bot)]
        vals = sub[layer_value_col].dropna()
        if len(vals) < min_layers:
            continue

        v = sub[layer_value_col].to_numpy(dtype=float)
        if layer_mass_col in sub.columns:
            mass = sub[layer_mass_col].to_numpy(dtype=float)
            good = np.isfinite(mass) & (mass > 0)
        else:
            good = np.zeros(len(v), dtype=bool)
        if int(good.sum()) >= 2:
            obs_mean = float(np.dot(mass[good], v[good]) / mass[good].sum())
        else:
            obs_mean = float(vals.mean())
        obs_sigma = max(float(vals.std(ddof=1)), sigma_floor)

        w_vec: np.ndarray | None = None
        if frac is not None:
            w_vec = _perlayer_measured_weights(
                frac, z_top, z_bot, pool_order,
                value_col=fraction_value_col, pool_col="_pool", mid_col="lyr_mid",
            )

        predict: Callable[..., jnp.ndarray]
        if w_vec is not None:
            w_j = jnp.asarray(w_vec, dtype=jnp.float32)
            predict = (
                lambda out, p, t=t_arr, cols=pool_cols, w=w_j:
                jnp.mean(out.delta14C[t][:, cols] @ w, keepdims=True)
            )
            weight_source = "measured"
            weights_out: list[float] | None = [float(x) for x in w_vec]
        else:
            predict = (
                lambda out, p, t=t_arr:
                jnp.mean(
                    jnp.sum(out.C12[t, :] * out.delta14C[t, :], axis=1)
                    / (jnp.sum(out.C12[t, :], axis=1) + 1e-30),
                    keepdims=True,
                )
            )
            weight_source = "model_c12"
            weights_out = None

        block = ObsBlock(
            name=f"{name_prefix}_{bin_label}",
            y=jnp.asarray([obs_mean], dtype=jnp.float32),
            Se=jnp.asarray([obs_sigma ** 2], dtype=jnp.float32),
            predict=predict,
        )
        rows.append({
            "name": block.name,
            "bin_label": bin_label,
            "depth_cm": (z_top, z_bot),
            "mean": obs_mean,
            "sigma": obs_sigma,
            "n": int(len(vals)),
            "weight_source": weight_source,
            "weights": weights_out,
            "block": block,
        })
    return rows
