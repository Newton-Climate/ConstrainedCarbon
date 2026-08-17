"""ISRaD density-fraction ¹²C observations.

Density fractionation partitions layer C into separates (free light, occluded
light, heavy) whose kinetic behaviour the co-located-pool ontology already
identifies with the model's active/slow/passive pools (see
:mod:`ecosystem_complexity.data.fraction_mapping`). The Δ¹⁴C on those separates
already enters the inversion via ``build_fraction_14C_blocks``, but the *mass*
partition itself — an independent observation of ``C12_i / Σ C12_j`` — was
being discarded. This module turns that partition into ``ObsBlock`` s so the
inversion can constrain per-pool C stocks (not only column-total SOC).

Two formulations, gated on what the site actually reports:

* **partition** (unitless, always emitted when ≥2 mapped pools have C-mass
  data) — constrains ``mean(C12_i) / Σ_j mean(C12_j)`` to the observed C-mass
  fraction of separate *i*. Only ``frc_c_perc`` (or ``frc_mass_perc × frc_c_org``)
  is required — no absolute-stock cross-multiplication, so a site without
  ``lyr_soc`` still contributes.
* **absolute stock** (gC m⁻², emitted only when ``lyr_soc`` is present) —
  constrains ``mean(C12_i)`` to the partition times layer SOC, aggregated over
  the profile. This is redundant with partition + total-SOC (``c_sum``) in the
  linear limit, but numerical redundancy is harmless in OE and lets sites with
  strong SOC data pull per-pool stocks more directly.

The property → pool mapping is delegated to
:func:`ecosystem_complexity.data.fraction_mapping.build_fraction_mapping`, so a
site inherits the vocabulary policy (and any ``fraction_rules`` override) that
already drives the ¹⁴C fraction blocks — the two operators cannot drift apart
in which properties they treat as kinetic.
"""
from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ecosystem_complexity.inference._helpers import ObsBlock
from ecosystem_complexity.data.fraction_mapping import build_fraction_mapping
from ecosystem_complexity.data.paths import ISRAD_FRACTION

logger = logging.getLogger(__name__)


def _load_density_fraction_df(israd_name: str) -> pd.DataFrame:
    """Density-scheme fraction rows for one site, with a resolved obs year."""
    df = pd.read_csv(ISRAD_FRACTION, low_memory=False)
    df = df[df["site_name"] == israd_name].copy()
    if df.empty:
        return df
    scheme = df["frc_scheme"].astype(str).str.lower()
    df = df[scheme == "density"].copy()
    if df.empty:
        return df
    for col in ("frc_c_perc", "frc_c_org", "frc_mass_perc", "lyr_soc",
                "lyr_top", "lyr_bot"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    frac_year = pd.to_numeric(df.get("frc_obs_date_y"), errors="coerce")
    layer_year = pd.to_numeric(df.get("lyr_obs_date_y"), errors="coerce")
    df["_obs_year"] = frac_year.fillna(layer_year)
    df = df.dropna(subset=["_obs_year"])
    return df


def _row_c_mass_weight(row: pd.Series) -> float:
    """C mass in a fraction row, in units proportional to gC per g layer.

    Preference order mirrors what ISRaD actually populates: ``frc_c_perc`` is
    already the per-fraction share of layer C (percent), so it is used as-is;
    absent that, reconstruct it from ``frc_mass_perc × frc_c_org`` (both
    percentages, product is proportional to g C per 100 g layer). Returns NaN
    when neither combination is available.
    """
    c_perc = row.get("frc_c_perc", np.nan)
    if pd.notna(c_perc) and c_perc > 0:
        return float(c_perc)
    m_perc = row.get("frc_mass_perc", np.nan)
    c_org = row.get("frc_c_org", np.nan)
    if pd.notna(m_perc) and pd.notna(c_org) and m_perc > 0 and c_org > 0:
        return float(m_perc) * float(c_org) / 100.0
    return float("nan")


def _partition_predict(pool_cols: jnp.ndarray, pool_col: jnp.int32):
    """Predictor for ``mean(C12_i) / Σ_j mean(C12_j)`` restricted to mapped pools."""
    def _predict(out, p, cols=pool_cols, i=pool_col):
        mean_c = jnp.mean(out.C12, axis=0)
        total = jnp.sum(mean_c[cols]) + 1e-30
        return jnp.asarray([mean_c[i] / total], dtype=jnp.float32)
    return _predict


def _stock_predict(pool_col: jnp.int32):
    """Predictor for ``mean(C12_i)`` — the same operator used by ``c_stock``."""
    def _predict(out, p, i=pool_col):
        return jnp.asarray([jnp.mean(out.C12, axis=0)[i]], dtype=jnp.float32)
    return _predict


def build_fraction_12C_blocks(
    israd_name: str,
    model,
    fraction_rules: dict[str, str | None] | None = None,
    *,
    sigma_partition_floor: float = 0.05,
    sigma_stock_fraction: float = 0.30,
    min_pools: int = 2,
    min_stock_depth_coverage: float = 0.8,
) -> list[ObsBlock]:
    """Density-fraction C-mass partition (and optional stock) blocks for a site.

    Parameters
    ----------
    israd_name
        Site name as it appears in ISRaD's ``site_name`` column.
    model
        Built ``EcosystemModel``; ``model.pool_index`` supplies pool ordering.
    fraction_rules
        Optional per-site overrides for the vocabulary → pool mapping (from the
        config's ``datasource.fraction_rules``). Passed straight through to
        :func:`build_fraction_mapping`.
    sigma_partition_floor
        Lower bound on σ for the unitless partition observation. The empirical
        across-profile spread of the partition is used when larger.
    sigma_stock_fraction
        Fractional σ on absolute per-pool stocks (``σ = 0.30 × mean``). Density
        fractionation mass-balance closures are typically ~90–95%, and the layer
        SOC used to cross-multiply carries its own error — 30% is a conservative
        default matching the ``c_stock`` fallback in ``_build_obs_blocks``.
    min_pools
        Emit blocks only when at least this many *mapped* pools have C-mass data
        in the observation year. A single-pool partition would be trivially 1.0.
    """
    df = _load_density_fraction_df(israd_name)
    if df.empty:
        return []

    observed = sorted(
        df[["frc_scheme", "frc_property"]].dropna().astype(str)
        .drop_duplicates().itertuples(index=False, name=None)
    )
    mapping = build_fraction_mapping(
        observed, list(model.pool_index.pool_names), overrides=fraction_rules
    )
    if not mapping.pool_by_property:
        return []

    df["_pool"] = df["frc_property"].astype(str).str.strip().str.lower().map(
        mapping.pool_by_property
    )
    df = df[df["_pool"].notna()].copy()
    if df.empty:
        return []

    df["_c_weight"] = df.apply(_row_c_mass_weight, axis=1)
    df = df[df["_c_weight"].notna() & (df["_c_weight"] > 0)].copy()
    if df.empty:
        return []

    # Depth-integration weight per row: layer C mass if we have it, else
    # thickness. Same fallback ladder as _profile_weights in israd_14c, chosen
    # per profile so a mixed profile does not silently combine bases.
    df["_thick"] = (df["lyr_bot"] - df["lyr_top"]).abs().where(
        lambda s: s > 0, 1.0
    )

    prof_key = df.get("pro_name")
    if prof_key is None or prof_key.isna().all():
        prof_key = df.get("plot_name")
    if prof_key is None or prof_key.isna().all():
        # Only one profile in the site — treat everything as one column.
        prof_key = pd.Series(["_only"] * len(df), index=df.index)
    df = df.assign(_prof=prof_key.astype(str))

    blocks: list[ObsBlock] = []
    for obs_year, year_grp in df.groupby(df["_obs_year"].astype(int)):
        per_profile: list[pd.Series] = []
        for _, prof_grp in year_grp.groupby("_prof"):
            if "lyr_soc" in prof_grp.columns and prof_grp["lyr_soc"].notna().all() \
                    and bool((prof_grp["lyr_soc"] > 0).all()):
                w_layer = prof_grp["lyr_soc"].astype(float)
            else:
                w_layer = prof_grp["_thick"].astype(float)
            # Column C in pool i = Σ_layers layer_weight × (frc_c_perc_i / 100)
            contrib = w_layer * (prof_grp["_c_weight"] / 100.0)
            pool_sums = contrib.groupby(prof_grp["_pool"]).sum()
            total = float(pool_sums.sum())
            if total <= 0:
                continue
            per_profile.append(pool_sums / total)
        if not per_profile:
            continue
        combined = pd.concat(per_profile, axis=1).fillna(0.0)
        mean_partition = combined.mean(axis=1)
        spread = combined.std(axis=1, ddof=1) if combined.shape[1] > 1 else None

        pools_here = [p for p in mean_partition.index if mean_partition[p] > 0]
        if len(pools_here) < min_pools:
            continue

        mapped_cols = jnp.asarray(
            [int(model.pool_index[p]) for p in pools_here], dtype=jnp.int32
        )

        # Partition blocks — one per mapped pool. Redundant by one under Σ=1,
        # but the diagonal Se treatment absorbs the redundancy without bias.
        for pool_name in pools_here:
            f_obs = float(mean_partition[pool_name])
            if spread is not None and np.isfinite(spread[pool_name]):
                sigma_f = max(sigma_partition_floor, float(spread[pool_name]))
            else:
                sigma_f = sigma_partition_floor
            pool_col = jnp.int32(model.pool_index[pool_name])
            blocks.append(ObsBlock(
                name=f"israd_fraction_12C_frac_{pool_name}_{int(obs_year)}",
                y=jnp.asarray([f_obs], dtype=jnp.float32),
                Se=jnp.asarray([sigma_f ** 2], dtype=jnp.float32),
                predict=_partition_predict(mapped_cols, pool_col),
            ))

        # Absolute-stock blocks — only when we can attribute the partition to a
        # measured layer SOC column. Sum (partition_i × layer_soc) over layers
        # sampled in this obs_year that have both frc rows and lyr_soc.
        # Absolute per-pool stocks require layer_soc on every layer of at least
        # one profile AND that the profile reach `min_stock_depth_coverage` of
        # the model column — a truncated per-pool total would bias mean(C12_i)
        # low (same rationale as build_measured_soc_total). lyr_soc is in
        # gC/cm² per layer, so ×1e4 → gC/m².
        column_cm = max(
            float(layer.depth_bot_m) for layer in model.config.soil_layers
        ) * 100.0
        per_pool_stock: dict[str, list[float]] = {p: [] for p in pools_here}
        for _, prof_grp in year_grp.groupby("_prof"):
            if "lyr_soc" not in prof_grp.columns:
                break
            if not prof_grp["lyr_soc"].notna().all() \
                    or not bool((prof_grp["lyr_soc"] > 0).all()):
                continue
            z_bot_max = float(prof_grp["lyr_bot"].max())
            if z_bot_max / column_cm < min_stock_depth_coverage:
                continue
            layer_soc = prof_grp["lyr_soc"].astype(float) * 1.0e4
            for pool_name in pools_here:
                mask = prof_grp["_pool"] == pool_name
                stock = float(
                    (layer_soc[mask] * (prof_grp.loc[mask, "_c_weight"] / 100.0)).sum()
                )
                per_pool_stock[pool_name].append(stock)
        if any(per_pool_stock.values()):
            for pool_name, stocks in per_pool_stock.items():
                if not stocks:
                    continue
                col_stock = float(np.mean(stocks))
                if col_stock <= 0:
                    continue
                sigma_c = max(50.0, sigma_stock_fraction * col_stock)
                pool_col = jnp.int32(model.pool_index[pool_name])
                blocks.append(ObsBlock(
                    name=f"israd_fraction_12C_stock_{pool_name}_{int(obs_year)}",
                    y=jnp.asarray([col_stock], dtype=jnp.float32),
                    Se=jnp.asarray([sigma_c ** 2], dtype=jnp.float32),
                    predict=_stock_predict(pool_col),
                ))

    if blocks:
        logger.info(
            "  [%s] density-fraction 12C: %d block(s) [%s]",
            israd_name,
            len(blocks),
            ", ".join(sorted({b.name.split("_", 3)[3].rsplit("_", 1)[0]
                              for b in blocks})),
        )
    return blocks
