"""ISRaD radiocarbon observations: bulk mixtures, density fractions, respired ¹⁴C.

The property → pool mapping lives in
:mod:`ecosystem_complexity.sites.fraction_mapping`, which keys it on the ISRaD
fractionation vocabulary rather than on the site, so a new site needs no code.
"""
from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ecosystem_complexity.api import ObsBlock
from ecosystem_complexity.data.israd_observations import (
    FractionMappingRule,
    add_layer_midpoint,
    build_fraction_obs_blocks,
    bulk_mixture_obs_block,
)
from ecosystem_complexity.sites.fraction_mapping import (
    BULK_SCHEMES,
    build_fraction_mapping,
    normalize,
)
from ecosystem_complexity.sites.paths import ISRAD_FLUX, ISRAD_FRACTION, ISRAD_LAYER

logger = logging.getLogger(__name__)

# A profile whose most-depleted layer is below this (‰) is treated as holding
# genuinely aged carbon, so its passive-pool IC is seeded from the data rather
# than the modern fallback (see _bulk_pool_ic_seeds). Above it, leave the
# fallback alone — this excludes modern and bomb-enriched profiles.
_AGED_PROFILE_D14C = -50.0


def _pool_depth_bins_cm(model) -> list[tuple[str, tuple[float, float]]]:
    """Depth→pool bins (cm) from the config pool bounds.

    The shallowest pool is extended upward (catch O-horizons at negative depth)
    and the deepest downward, so every layer maps to exactly one pool.
    """
    pools = []
    for layer in model.config.soil_layers:
        for p in layer.som_pools:
            pools.append((f"{layer.name}_{p.name}",
                          float(p.depth_top_m) * 100.0, float(p.depth_bot_m) * 100.0))
    pools.sort(key=lambda x: x[1])
    bins = []
    for i, (name, top, bot) in enumerate(pools):
        lo = -1000.0 if i == 0 else top
        hi = 1e4 if i == len(pools) - 1 else bot
        bins.append((name, (lo, hi)))
    return bins


# ── ISRaD obs ──────────────────────────────────────────────────────────────

def _layer_df(israd_name: str) -> pd.DataFrame:
    df = pd.read_csv(ISRAD_LAYER, low_memory=False)
    df = df[df["site_name"] == israd_name].copy()
    df = df.dropna(subset=["lyr_14c", "lyr_top", "lyr_bot", "lyr_obs_date_y"])
    return add_layer_midpoint(df)


def _flux_df(israd_name: str) -> pd.DataFrame:
    df = pd.read_csv(ISRAD_FLUX, low_memory=False)
    df = df[df["site_name"] == israd_name].copy()
    return df.dropna(subset=["flx_14c", "flx_obs_date_y"])


def _bulk_pool_ic_seeds(israd_name: str) -> dict[str, float]:
    """Per-pool initial Δ¹⁴C (‰) inferred from a site's observed layer profile.

    A whole-sample bulk measurement carries no per-pool split, so the pool-level
    ¹⁴C initial condition cannot be read off a bulk block by name — and the old
    behaviour then silently fell through to ``_FALLBACK_D14C`` (active +120, slow
    +60, passive **0**). Initialising a permafrost/deep-tropical passive pool at a
    *modern* value is indefensible: those sites hold strongly ¹⁴C-depleted (aged)
    carbon, and a steady-state pool relaxes only slowly (∼τ) from its initial
    value, so a modern IC forces a large, irreducible misfit against the deep
    layers (Adventdalen −716‰, Santarém −165‰, EML −196‰).

    Only the **passive** pool is reseeded, and only for genuinely aged profiles.
    Rationale: the active (τ∼2 yr) and slow (τ∼20 yr) pools equilibrate away their
    initial condition within the ∼century transient, so their IC is immaterial at
    observation time and perturbing it only destabilises otherwise-good fits.
    The passive pool (τ∼200 yr) retains ∼half its IC, so a modern seed there is
    what forces the deep-layer misfit. Under the co-located kinetic-fraction
    ontology the passive pool holds the most *aged* carbon, so it is seeded from
    the most-depleted observed layer. The gate (``_AGED_PROFILE_D14C``) keeps
    modern temperate/boreal profiles — and bomb-*enriched* sites like Howland,
    where depleting the IC is exactly wrong — on the ``_FALLBACK_D14C`` behaviour.
    Returns ``{}`` (→ fallback) when the site lacks a usably aged layer profile.
    """
    try:
        df = _layer_df(israd_name)
    except Exception:  # noqa: BLE001 — missing/ío-broken layer table → no seeds
        return {}
    vals = np.asarray(pd.to_numeric(df["lyr_14c"], errors="coerce"), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or float(np.min(vals)) >= _AGED_PROFILE_D14C:
        return {}
    return {"soil_passive": float(np.min(vals))}


def _plant_debris_df(israd_name: str) -> pd.DataFrame:
    """ISRaD plant-debris Δ¹⁴C rows, shaped like layer rows for bulk aggregation.

    Plant debris (macrofossil, roots, coarse/plant material) is recognisable
    plant fragments, not a density/kinetic fraction, so it is treated as part of
    the bulk material of its layer rather than assigned to a kinetic pool. The
    membership test is the shared ``BULK_SCHEMES`` policy, so the
    same classification drives both this fold-in and the fraction mapping's
    skip list — they cannot drift apart into double-counting a property.

    The fraction table carries no mass or C column for these rows
    (``frc_mass_perc``/``frc_c_perc``/``frc_c_tot``/``frc_c_org`` are all empty),
    so they can only ever be thickness-weighted — which is why including them
    forces the whole profile onto the thickness basis (see ``_profile_weights``).
    """
    df = pd.read_csv(ISRAD_FRACTION, low_memory=False)
    df = df[(df["site_name"] == israd_name) & df["frc_14c"].notna()].copy()
    if df.empty:
        return df
    scheme = df["frc_scheme"].astype(str).map(normalize)
    df = df[scheme.isin(BULK_SCHEMES)].copy()
    if df.empty:
        return df
    year = pd.to_numeric(df.get("frc_obs_date_y"), errors="coerce")
    df["lyr_obs_date_y"] = year.fillna(
        pd.to_numeric(df.get("lyr_obs_date_y"), errors="coerce")
    )
    df["lyr_14c"] = pd.to_numeric(df["frc_14c"], errors="coerce")
    keep = ["lyr_14c", "lyr_top", "lyr_bot", "lyr_obs_date_y"]
    df = df.dropna(subset=keep)
    return df[keep]


def _profile_weights(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Per-layer carbon-mass weights for a profile, on ONE consistent basis.

    A weighted mean is only meaningful if every row's weight is in the same units,
    so this picks a single basis for the whole profile rather than mixing per-row
    fallbacks: measured SOC stock where the profile has it throughout, else bulk
    density × organic-C × thickness, else thickness alone.

    Thickness is the weakest basis — it assumes uniform carbon density with depth,
    which over-weights deep, C-poor layers — but it is the only field present for
    every layer (``lyr_soc`` covers ~38% of ¹⁴C-bearing layers and is entirely
    absent at Harvard, EML, Solling, FLONA, ZF2, Appi and Auchencorth).
    """
    thick = (
        pd.to_numeric(df["lyr_bot"], errors="coerce")
        - pd.to_numeric(df["lyr_top"], errors="coerce")
    ).abs()

    soc = pd.to_numeric(df.get("lyr_soc", pd.Series(index=df.index)), errors="coerce")
    if soc.notna().all() and bool((soc > 0).all()):
        return np.asarray(soc, dtype=float), "lyr_soc"

    _empty = pd.Series(index=df.index, dtype=float)
    bd = pd.to_numeric(df.get("lyr_bd_samp", _empty), errors="coerce")
    c_org = pd.to_numeric(df.get("lyr_c_org", _empty), errors="coerce")
    derived = bd * (c_org / 100.0) * thick
    if derived.notna().all() and bool((derived > 0).all()):
        return np.asarray(derived, dtype=float), "bd*c_org*thickness"

    thick = thick.where(thick > 0, 1.0).fillna(1.0)
    return np.asarray(thick, dtype=float), "thickness"


def build_bulk_14C_blocks(
    israd_name: str, forcing_time, model, include_plant_debris: bool = True
) -> list[ObsBlock]:
    """Whole-sample bulk Δ¹⁴C → one C-mass-weighted ObsBlock per observation year.

    A bulk measurement samples every kinetic pool at once, so it is compared
    against the C-mass-weighted mixture Σ C12_i·Δ¹⁴C_i / Σ C12_i rather than
    against a single pool picked by depth. Because that mixture is one number per
    date, the depth-resolved layer profile is first aggregated into one
    whole-profile Δ¹⁴C per year, C-mass-weighted (``_profile_weights``).

    The aggregation is not lossy bookkeeping — it is forced by the pool ontology.
    Co-located kinetic pools have no depth dimension, so a model cannot reproduce
    a Δ¹⁴C-vs-depth gradient at all; retaining per-layer obs would leave the
    within-profile spread (up to ~760‰ at FLONA, vs σ≈15–50‰) as pure misfit.
    The spread is instead carried into the block's σ, so depth-stratified sites
    correctly report a weakly-constraining bulk observation.
    """
    df = _layer_df(israd_name)[["lyr_14c", "lyr_top", "lyr_bot", "lyr_obs_date_y",
                                "lyr_soc", "lyr_bd_samp", "lyr_c_org"]].copy()
    if include_plant_debris:
        debris = _plant_debris_df(israd_name)
        if not debris.empty:
            df = pd.concat([df, debris], ignore_index=True)

    time_np = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    blocks: list[ObsBlock] = []
    for year, grp in df.groupby(df["lyr_obs_date_y"].astype(int)):
        vals = np.asarray(pd.to_numeric(grp["lyr_14c"], errors="coerce"), dtype=float)
        w, basis = _profile_weights(grp)
        ok = np.isfinite(vals) & np.isfinite(w) & (w > 0)
        if not ok.any():
            continue
        vals, w = vals[ok], w[ok]
        w = w / w.sum()
        mean_val = float(np.dot(w, vals))
        # Spread across the profile is irreducible under co-located pools, so it
        # belongs in the observation error rather than being fit away.
        if len(vals) > 1:
            var_w = float(np.dot(w, (vals - mean_val) ** 2))
            n_eff = 1.0 / float(np.dot(w, w))
            sigma_val = float(np.sqrt(var_w / max(n_eff, 1.0)))
            sigma_spread = float(np.sqrt(var_w))
        else:
            sigma_val = 50.0
            sigma_spread = 0.0
        sigma_val = max(sigma_val, 15.0)
        t_idx = jnp.array(
            [int(np.argmin(np.abs(years_np - (float(year) + 0.5))))], dtype=jnp.int32
        )
        blocks.append(bulk_mixture_obs_block(
            f"israd_bulk_{year}_profile", mean_val, sigma_val, t_idx
        ))
        logger.info("%s", f"    bulk {year}: {mean_val:7.1f}±{sigma_val:.1f}‰ "
              f"(n={len(vals)}, weight={basis}, profile spread σ={sigma_spread:.0f}‰)")
    return blocks


def build_fraction_14C_blocks(
    israd_name: str,
    forcing_time,
    model,
    fraction_rules: dict[str, str | None] | None = None,
) -> list[ObsBlock]:
    """Map ISRaD density fractions to kinetic pools, by vocabulary not by site.

    Which ``frc_property`` values carry a kinetic interpretation is a property of
    the ISRaD fractionation vocabulary, so the mapping is resolved by
    :func:`~ecosystem_complexity.sites.fraction_mapping.build_fraction_mapping`
    against whatever properties this site actually reports. ``fraction_rules``
    (from the config's ``datasource.fraction_rules``) overrides the default for
    a site with an unusual protocol — no code change, and no per-site branch.

    Plant debris (macrofossil, roots, coarse material) is deliberately NOT
    mapped to a kinetic pool: it is recognisable plant fragments rather than a
    density fraction, so it is folded into the bulk observation by
    ``build_bulk_14C_blocks`` instead. Mapping EML's macrofossil to the active
    pool asserted +283‰ for a τ=2 yr pool while the bulk layers put the same
    pool at −176‰ in the same year — a >10σ contradiction.
    """
    df = pd.read_csv(ISRAD_FRACTION, low_memory=False)
    df = df[(df["site_name"] == israd_name) & df["frc_14c"].notna()].copy()
    if df.empty:
        return []

    # Prefer fraction sampling year; fall back to the parent layer year.
    frac_year = pd.to_numeric(df.get("frc_obs_date_y"), errors="coerce")
    layer_year = pd.to_numeric(df.get("lyr_obs_date_y"), errors="coerce")
    df["_obs_year"] = frac_year.fillna(layer_year)
    df = add_layer_midpoint(df.dropna(subset=["_obs_year", "lyr_top", "lyr_bot"]))
    if df.empty:
        return []

    observed = sorted(
        df[["frc_scheme", "frc_property"]].dropna().astype(str)
        .drop_duplicates().itertuples(index=False, name=None)
    )
    mapping = build_fraction_mapping(
        observed, list(model.pool_index.pool_names), overrides=fraction_rules
    )
    if mapping.skipped:
        logger.info(
            "  [%s] fractions skipped: %s",
            israd_name,
            "; ".join(f"{p} ({why})" for p, why in sorted(mapping.skipped.items())),
        )
    if not mapping.pool_by_property:
        logger.info(
            "  [%s] no ISRaD fraction maps to a kinetic pool (properties: %s)",
            israd_name, [p for _, p in observed],
        )
        return []

    rules = [
        FractionMappingRule(pool, prop) for pool, prop in mapping.as_rules()
    ]
    rows = build_fraction_obs_blocks(
        df,
        forcing_time,
        model.pool_index,
        rules,
        year_col="_obs_year",
        min_sigma=15.0,
        singleton_sigma=50.0,
        name_prefix="israd_fraction",
    )
    return [row["block"] for row in rows]
def build_resp_14C_obs(israd_name: str, forcing_time) -> jnp.ndarray:
    """Field-flux Δ¹⁴C → sparse (T,) respired-CO₂ observation array."""
    df = _flux_df(israd_name)
    y = pd.to_numeric(df["flx_obs_date_y"], errors="coerce")
    m = pd.to_numeric(df.get("flx_obs_date_m", np.nan), errors="coerce")
    df = df.assign(_dec=y + (m.fillna(6.5) - 0.5) / 12.0)
    by_date = df.groupby("_dec")["flx_14c"].mean()

    time_np = np.array(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    arr = np.full(len(time_np), np.nan, dtype=np.float32)
    for dec_yr, val in by_date.items():
        if np.isfinite(val):
            arr[int(np.argmin(np.abs(years_np - float(dec_yr))))] = float(val)
    return jnp.array(arr)
