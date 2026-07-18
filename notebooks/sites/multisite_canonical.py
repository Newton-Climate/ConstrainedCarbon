"""
sites/multisite_canonical.py — One shared OE inversion recipe applied to every
ISRaD field-flux + respiration site.

Each site is defined by its own transparent, version-controlled config YAML in
``configs/multisite/<site>.yaml``. Every config carries the SAME canonical model
recipe (so cross-site results stay comparable) plus a site-specific ``site`` and
``datasource`` block:

Recipe (identical at every site — the shared sections of each per-site config):
  • 3-pool config (active / slow / passive), canonical depth structure + tau priors
  • external soil input driven by observed daily GPP from the co-located flux tower
  • SOC prior built from a site-specific annual-mean forcing year, then iterated to
    steady state at the prior parameters
  • obs vector:
      - bulk Δ¹⁴C        → C-mass-weighted mixture over ALL pools, one
                           whole-profile value per year (ISRaD layer table,
                           plus macrofossil rows from the fraction table)
      - field-flux Δ¹⁴C  → respired CO₂ (ISRaD flux table)      [delta14C_resp]
      - pool C stocks    → steady-state SOC prior               [C_pools_obs]

The three pools are co-located KINETIC fractions (free light / occluded / heavy
coexisting at every depth), not depth horizons — the per-pool ``depth_top_m`` /
``depth_bot_m`` in the configs are observation-mapping metadata only and do not
enter the physics (the depth-damped soil temperature in ``_pool_env_vecs`` is
gated on ``T_annual_mean_C``, which no multisite config sets). Bulk Δ¹⁴C is
therefore a whole-sample mixture rather than a single depth-selected pool.
  • optimize_oe (Levenberg–Marquardt), fields (log_tau, log_f_transfer)

Per-site ``datasource`` block (read by this driver):
  • israd_name    — ISRaD site_name filtering the layer / flux / fraction tables
  • forcing_glob  — data/<glob> for the co-located flux tower directory or file
  • forcing_kind  — daily | harvard_hr | eml_hh (selects the forcing loader)

Forcing GPP comes from the files fetched by notebooks/download_ameriflux_sites.py
and notebooks/download_icos_sites.py. All FLUXNET DD files (AmeriFlux FULLSET and
ICOS/EUF/JPF FLUXMET) share the same column convention, so a single loader
(load_howland_forest) reads them all.

To add a site: drop a new ``configs/multisite/<site>.yaml`` (copy an existing one
and edit the ``site`` + ``datasource`` blocks); no code change required.

Run:
    python notebooks/sites/multisite_canonical.py            # all configs
    python notebooks/sites/multisite_canonical.py solling    # one config (by stem)
"""
from __future__ import annotations

import glob
import os
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

_SITES_DIR = os.path.dirname(os.path.abspath(__file__))
_NB_ROOT = os.path.dirname(_SITES_DIR)
_REPO_ROOT = os.path.dirname(_NB_ROOT)
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ecosystem_complexity.api import build_model, run_model, optimize_oe, ObsBlock
from ecosystem_complexity.data.loaders import (
    load_eight_mile_lake,
    load_harvard_forest,
    load_howland_forest,
)
from ecosystem_complexity.data.parsers import attach_atm14C
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.schemas import ForcingData, ObservationData
from ecosystem_complexity.data.israd_observations import (
    FractionMappingRule,
    add_layer_midpoint,
    build_fraction_obs_blocks,
    bulk_mixture_obs_block,
)
from ecosystem_complexity.state import make_default_params, make_initial_state
from ecosystem_complexity.oe_utils import ss_state_for_params

CONFIG_DIR = os.path.join(_REPO_ROOT, "configs", "multisite")
ISRAD_DIR = os.path.join(_REPO_ROOT, "data", "shared", "israd")
ISRAD_VERSION = "2.6.6.2024-01-25"
ISRAD_LAYER = os.path.join(ISRAD_DIR, f"ISRaD_data_flat_layer_v {ISRAD_VERSION}.csv")
ISRAD_FRACTION = os.path.join(ISRAD_DIR, f"ISRaD_data_flat_fraction_v {ISRAD_VERSION}.csv")
ISRAD_FLUX = os.path.join(ISRAD_DIR, f"ISRaD_data_flat_flux_v {ISRAD_VERSION}.csv")
HUA_PATH = os.path.join(_REPO_ROOT, "data", "shared", "atm_14C", "Hua_2021.csv")
GRAVEN_PATH = os.path.join(_REPO_ROOT, "data", "shared", "atm_14C", "Graven_2017.csv")
INTCAL_PATH = os.path.join(_REPO_ROOT, "data", "shared", "atm_14C", "intcal20.14c")

_R_STD = 1.176e-12
OPT_FIELDS = ("log_tau", "log_f_transfer")
# Fallback initial Δ¹⁴C per pool (‰) when a pool has no layer obs.
_FALLBACK_D14C = {"soil_active": 120.0, "soil_slow": 60.0, "soil_passive": 0.0}
# A profile whose most-depleted layer is below this (‰) is treated as holding
# genuinely aged carbon, so its passive-pool IC is seeded from the data rather
# than the modern fallback (see _bulk_pool_ic_seeds). Above it, leave the
# fallback alone — this excludes modern and bomb-enriched profiles.
_AGED_PROFILE_D14C = -50.0
# Matches the σ=0.5 OE prior on log_tau. build_soc_prior sets obs = C(prior params)
# and C = I·τ/modifier at steady state with I fixed by the GPP forcing, so anything
# tighter restates the τ prior at MORE confidence than the prior itself — prior
# double-counting dressed as an observation. At 0.50 the fallback is a no-op.
_SOC_PRIOR_SIGMA_FRAC = 0.50
_SS_TOL = 1e-5
_SS_MAX_YEARS = 2000


@dataclass(frozen=True)
class SiteSpec:
    config_path: str         # configs/multisite/<stem>.yaml this spec came from
    config_stem: str         # filename stem (the CLI selector for the site)
    israd_name: str          # ISRaD site_name (filters the layer/flux tables)
    forcing_glob: str        # data/<glob> for the tower directory
    lat: float
    lon: float
    label: str
    tower_id: str = ""
    biome: str = "unclassified"
    forcing_kind: str = "daily"
    observation_path: str = "bulk_resp"   # bulk_resp | fraction


# GPP columns in preference order (tropical FLUXNET only fills the DT variants).
_GPP_COLS = ("GPP_NT_VUT_REF", "GPP_DT_VUT_REF", "GPP_NT_CUT_REF", "GPP_DT_CUT_REF")


def load_site_spec(config_path: str) -> SiteSpec:
    """Build a SiteSpec from a per-site config YAML's site + datasource blocks."""
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    site = cfg.get("site") or {}
    ds = cfg.get("datasource") or {}
    missing = [k for k in ("israd_name", "forcing_glob") if not ds.get(k)]
    if missing:
        raise ValueError(
            f"{os.path.basename(config_path)}: datasource missing {missing}"
        )
    stem = os.path.splitext(os.path.basename(config_path))[0]
    return SiteSpec(
        config_path=config_path,
        config_stem=stem,
        israd_name=str(ds["israd_name"]),
        forcing_glob=str(ds["forcing_glob"]),
        lat=float(site.get("lat", 0.0)),
        lon=float(site.get("lon", 0.0)),
        label=str(site.get("name", stem)),
        tower_id=str(site.get("tower_id", "")),
        biome=str(site.get("biome", "unclassified")),
        forcing_kind=str(ds.get("forcing_kind", "daily")),
        observation_path=str(ds.get("observation_path", "bulk_resp")),
    )


def discover_site_specs(config_dir: str = CONFIG_DIR) -> dict[str, SiteSpec]:
    """Load every configs/multisite/*.yaml into a {config_stem: SiteSpec} map."""
    specs: dict[str, SiteSpec] = {}
    for path in sorted(glob.glob(os.path.join(config_dir, "*.yaml"))):
        spec = load_site_spec(path)
        specs[spec.config_stem] = spec
    return specs


def select_specs(names: list[str] | None) -> list[SiteSpec]:
    """Resolve CLI site selectors (config stem, ISRaD name, or label) to specs."""
    specs = discover_site_specs()
    if not specs:
        raise FileNotFoundError(f"No site configs found under {CONFIG_DIR}")
    if not names:
        return list(specs.values())
    by_key: dict[str, SiteSpec] = {}
    for spec in specs.values():
        by_key[spec.config_stem] = spec
        by_key[spec.israd_name] = spec
        by_key[spec.label] = spec
    out: list[SiteSpec] = []
    for name in names:
        if name not in by_key:
            raise KeyError(
                f"Unknown site {name!r}. Available: {sorted(specs)}"
            )
        out.append(by_key[name])
    return out


# ── forcing ────────────────────────────────────────────────────────────────

def _resolve_dd_file(forcing_glob: str) -> str:
    """Find the flux DD csv (with GPP) inside a tower directory."""
    root = os.path.join(_REPO_ROOT, "data", forcing_glob)
    dd = [
        f for f in glob.glob(os.path.join(root, "**", "*_DD_*.csv"), recursive=True)
        if "ERA5" not in f and "BIFVARINFO" not in f
    ]
    for f in dd:
        with open(f, encoding="latin-1") as fh:
            if "GPP_NT_VUT_REF" in fh.readline():
                return f
    if not dd:
        raise FileNotFoundError(f"No flux DD file under data/{forcing_glob}")
    return dd[0]


def _resolve_forcing_file(spec: SiteSpec) -> str:
    if spec.forcing_kind == "daily":
        return _resolve_dd_file(spec.forcing_glob)
    matches = glob.glob(os.path.join(_REPO_ROOT, "data", spec.forcing_glob))
    if not matches:
        raise FileNotFoundError(f"No forcing file matching data/{spec.forcing_glob}")
    return matches[0]


def _load_site_forcing(spec: SiteSpec, path: str, model):
    if spec.forcing_kind == "harvard_hr":
        forcing, _ = load_harvard_forest(
            path, config=model.config, qc_threshold=2, include_gpp_forcing=True,
        )
        return _sanitize_forcing(forcing, np.array(forcing.GPP_obs, dtype=float))
    if spec.forcing_kind == "eml_hh":
        forcing, _ = load_eight_mile_lake(
            path, config=model.config, include_gpp_forcing=True,
        )
        return _sanitize_forcing(forcing, np.array(forcing.GPP_obs, dtype=float))
    forcing, _ = load_howland_forest(path, config=model.config, include_gpp_forcing=True)
    return _sanitize_forcing(forcing, _load_gpp_series(path))


def _load_gpp_series(dd_path: str) -> np.ndarray:
    """Robust, fully-finite daily GPP (gC m⁻² day⁻¹), aligned to the DD rows.

    Picks the best-populated GPP partition column (tropical towers only fill the
    daytime DT variants) and gap-fills the remaining days so the forcing has no
    NaNs — a single NaN GPP day propagates through the ODE and NaNs the cost.
    """
    df = pd.read_csv(dd_path, na_values=[-9999], low_memory=False)
    series = None
    for col in _GPP_COLS:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().mean() > 0.2:
            series = pd.to_numeric(df[col], errors="coerce")
            break
    if series is None:
        raise ValueError(f"No usable GPP column in {os.path.basename(dd_path)}")
    series = series.interpolate(limit_direction="both").ffill().bfill()
    return np.asarray(series.fillna(series.mean()).values, dtype=np.float64)


def _sanitize_forcing(forcing, gpp: np.ndarray):
    """Override GPP with the robust series and make climate drivers finite.

    Fills missing soil temperature with air temperature (then 10 °C) and missing
    soil moisture with 0.30 v/v, so towers lacking TS/SWC (e.g. SJ-Adv) still run.
    """
    air = np.array(forcing.air_temp, dtype=np.float64)
    air = np.where(np.isfinite(air), air, np.nanmean(air) if np.isfinite(np.nanmean(air)) else 10.0)
    st = np.array(forcing.soil_temp, dtype=np.float64)
    st = np.where(np.isfinite(st), st, air[:, None])
    st = np.where(np.isfinite(st), st, 10.0)
    sm = np.array(forcing.soil_moisture, dtype=np.float64)
    sm = np.where(np.isfinite(sm), sm, 0.30)
    return forcing._replace(
        air_temp=jnp.array(air, dtype=jnp.float32),
        soil_temp=jnp.array(st, dtype=jnp.float32),
        soil_moisture=jnp.array(sm, dtype=jnp.float32),
        GPP_obs=jnp.array(gpp, dtype=jnp.float32),
    )


def _repeat_mean_field(arr: np.ndarray, n_days: int) -> np.ndarray:
    """Repeat the finite time mean of a forcing field into a synthetic year."""
    if arr.ndim == 1:
        if np.isposinf(arr).all():
            return np.full((n_days,), np.inf, dtype=np.float32)
        if np.isnan(arr).all():
            return np.full((n_days,), 0.0, dtype=np.float32)
        mean_val = np.nanmean(arr)
        if not np.isfinite(mean_val):
            mean_val = 0.0
        return np.full((n_days,), mean_val, dtype=np.float32)

    if np.isnan(arr).all():
        return np.zeros((n_days,) + arr.shape[1:], dtype=np.float32)
    mean_val = np.nanmean(arr, axis=0)
    mean_val = np.where(np.isfinite(mean_val), mean_val, 0.0).astype(np.float32)
    return np.repeat(mean_val[None, :], n_days, axis=0)


def build_annual_mean_forcing(forcing: ForcingData, n_days: int = 365) -> ForcingData:
    """Collapse a site forcing record to a synthetic constant annual-mean year."""
    return ForcingData(
        time=jnp.arange(n_days, dtype=jnp.float32),
        air_temp=jnp.array(_repeat_mean_field(np.array(forcing.air_temp), n_days)),
        sw_radiation=jnp.array(_repeat_mean_field(np.array(forcing.sw_radiation), n_days)),
        precip=jnp.array(_repeat_mean_field(np.array(forcing.precip), n_days)),
        vpd=jnp.array(_repeat_mean_field(np.array(forcing.vpd), n_days)),
        soil_temp=jnp.array(_repeat_mean_field(np.array(forcing.soil_temp), n_days)),
        soil_moisture=jnp.array(_repeat_mean_field(np.array(forcing.soil_moisture), n_days)),
        snow_depth=jnp.array(_repeat_mean_field(np.array(forcing.snow_depth), n_days)),
        active_layer=jnp.array(_repeat_mean_field(np.array(forcing.active_layer), n_days)),
        delta14C_atm=jnp.array(_repeat_mean_field(np.array(forcing.delta14C_atm), n_days)),
        GPP_obs=jnp.array(_repeat_mean_field(np.array(forcing.GPP_obs), n_days)),
        NPP_obs=jnp.array(_repeat_mean_field(np.array(forcing.NPP_obs), n_days)),
    )


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


def _macrofossil_df(israd_name: str) -> pd.DataFrame:
    """ISRaD macrofossil Δ¹⁴C rows, shaped like layer rows for bulk aggregation.

    Macrofossils are recognisable plant fragments, not a density/kinetic fraction,
    so they are treated as part of the bulk material of their layer rather than
    assigned to a kinetic pool. The fraction table carries no mass or C column for
    them (``frc_mass_perc``/``frc_c_perc``/``frc_c_tot``/``frc_c_org`` are all
    empty), so they can only ever be thickness-weighted — which is why including
    them forces the whole profile onto the thickness basis (see ``_profile_weights``).
    """
    df = pd.read_csv(ISRAD_FRACTION, low_memory=False)
    df = df[(df["site_name"] == israd_name) & df["frc_14c"].notna()].copy()
    if df.empty:
        return df
    prop = df["frc_property"].astype(str).str.strip().str.lower()
    df = df[prop == "macrofossil"].copy()
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
    israd_name: str, forcing_time, model, include_macrofossil: bool = True
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
    if include_macrofossil:
        macro = _macrofossil_df(israd_name)
        if not macro.empty:
            df = pd.concat([df, macro], ignore_index=True)

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
        print(f"    bulk {year}: {mean_val:7.1f}±{sigma_val:.1f}‰ "
              f"(n={len(vals)}, weight={basis}, profile spread σ={sigma_spread:.0f}‰)")
    return blocks


def build_fraction_14C_blocks(israd_name: str, forcing_time, model) -> list[ObsBlock]:
    """Map interpretable ISRaD fractions to kinetic pools by site and year."""
    df = pd.read_csv(ISRAD_FRACTION, low_memory=False)
    df = df[(df["site_name"] == israd_name) & df["frc_14c"].notna()].copy()
    if df.empty:
        return []

    # Prefer fraction sampling year; fall back to the parent layer year.
    frac_year = pd.to_numeric(df.get("frc_obs_date_y"), errors="coerce")
    layer_year = pd.to_numeric(df.get("lyr_obs_date_y"), errors="coerce")
    df["_obs_year"] = frac_year.fillna(layer_year)
    df = add_layer_midpoint(df.dropna(subset=["_obs_year", "lyr_top", "lyr_bot"]))

    rules = [
        FractionMappingRule("soil_active", "free light"),
        FractionMappingRule("soil_slow", "occluded light"),
        FractionMappingRule("soil_passive", "heavy"),
    ]
    if israd_name == "Appi forest":
        rules.append(FractionMappingRule("soil_passive", "residual"))
    # Macrofossil is deliberately NOT mapped to a kinetic pool: it is coarse plant
    # debris rather than a density fraction, so it is folded into the bulk
    # observation by build_bulk_14C_blocks instead. Mapping EML's macrofossil to
    # soil_active asserted +283‰ for a τ=2 yr pool while the bulk layers put the
    # same pool at −176‰ in the same year — a >10σ contradiction.

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


def build_soc_prior(model, forcing: ForcingData) -> tuple:
    """Build a site-specific steady-state SOC prior from annual-mean forcing."""
    params_prior = make_default_params(model.config)
    inversion = getattr(model.config, "inversion_raw", {}) or {}
    sigma_fraction = float(inversion.get("sigma_soc_fraction", _SOC_PRIOR_SIGMA_FRAC))
    forcing_mean = build_annual_mean_forcing(forcing)
    base = make_initial_state(model.config, {})
    state = ss_state_for_params(model, forcing_mean, base, params_prior)

    prev_total = None
    n_years = 0
    for n_years in range(1, _SS_MAX_YEARS + 1):
        out = run_model(model, forcing_mean, state0=state, params=params_prior)
        state = out.final_state
        total = float(np.sum(np.array(state.C12, dtype=float)))
        if prev_total is not None:
            rel = abs(total - prev_total) / (abs(prev_total) + 1e-10)
            if rel < _SS_TOL:
                break
        prev_total = total

    c12 = np.array(state.C12, dtype=float)
    c_pools_obs = {
        name: (
            float(c12[i]),
            float(sigma_fraction * c12[i]),
        )
        for i, name in enumerate(model.pool_index.pool_names)
        if float(c12[i]) > 0.0
    }
    total = float(c12.sum())
    c_total_obs = (total, float(sigma_fraction * total)) if total > 0.0 else None
    return state, c_pools_obs, n_years, c_total_obs


_MEASURED_SOC_SIGMA_FRAC = 0.15
# A measured profile is only comparable to the model's Σ_i C12_i if it samples
# most of the column the model represents. ISRaD profiles stop well short at most
# sites (MO Ozark 30 cm, Bartlett 45, EML 62, UMBS 90 vs a 130 cm model column),
# and a truncated total is a downward BIAS, not extra noise — so it is rejected
# rather than inflated.
_MIN_SOC_DEPTH_COVERAGE = 0.8


def _model_column_depth_cm(model) -> float:
    """Depth (cm) of the soil column the model's Σ_i C12_i represents."""
    return max(float(layer.depth_bot_m) for layer in model.config.soil_layers) * 100.0


def build_measured_soc_total(
    israd_name: str, model, sigma_floor_frac: float = _MEASURED_SOC_SIGMA_FRAC
) -> tuple[float, float, float] | None:
    """Measured TOTAL column SOC (gC m⁻²) → (mean, sigma, depth_coverage).

    Under co-located kinetic pools a measured stock cannot be attributed to one
    pool by depth (that was the same error the bulk Δ¹⁴C operator made), so this
    replaces ``build_measured_soc_stocks``: it sums ``lyr_soc`` over each whole
    profile and takes the across-profile mean ± σ.

    Returns None unless the profiles reach ``_MIN_SOC_DEPTH_COVERAGE`` of the
    model column. Depth coverage is the binding issue in ISRaD: profiles stop at
    different depths both across and within sites (UMBS spans 103–6049 gC m⁻²
    across profiles, cv=1.59, almost entirely because of where they stop), and
    comparing a 0–30 cm measurement against a 0–130 cm model column would bias
    the stock — and therefore ⟨τ⟩ — low.
    """
    df = pd.read_csv(ISRAD_LAYER, low_memory=False)
    df = df[df["site_name"] == israd_name].copy()
    df = df.dropna(subset=["lyr_soc", "lyr_top", "lyr_bot"])
    if df.empty:
        return None
    df["soc_gm2"] = pd.to_numeric(df["lyr_soc"], errors="coerce") * 1.0e4
    df["_profile"] = (
        df.get("entry_name", pd.Series(index=df.index)).astype(str) + "|"
        + df.get("pro_name", pd.Series(index=df.index)).astype(str) + "|"
        + df.get("lyr_obs_date_y", pd.Series(index=df.index)).astype(str)
    )
    column_cm = _model_column_depth_cm(model)
    per_profile = df.groupby("_profile").agg(
        total=("soc_gm2", "sum"), z_bot=("lyr_bot", "max")
    )
    per_profile = per_profile[per_profile["total"] > 0.0]
    if per_profile.empty:
        return None
    # Keep only profiles that actually sample most of the modelled column.
    coverage = per_profile["z_bot"] / column_cm
    keep = per_profile[coverage >= _MIN_SOC_DEPTH_COVERAGE]
    if keep.empty:
        return None
    mean_val = float(keep["total"].mean())
    std_val = float(keep["total"].std(ddof=1)) if len(keep) > 1 else 0.0
    sigma = max(std_val, sigma_floor_frac * mean_val)
    depth_cov = float(keep["z_bot"].max() / column_cm)
    return mean_val, sigma, depth_cov


SOILGRIDS_CSV = os.path.join(_NB_ROOT, "exports", "soilgrids_soc_pools.csv")
# SoilGrids is a statistical prediction, not a measurement, and it runs high on
# organic soils. Above this the value is rejected outright rather than trusted with
# a wide σ: CZ_Old_Black_Spruce comes back at 416,969 gC m⁻² and Baram Basin at
# 266,281, which exceed any real 1.3 m profile (deep tropical peat domes top out
# around 100k).
_MAX_PLAUSIBLE_SOC_GCM2 = 100_000.0
# SoilGrids per-depth uncertainties share one model and one location, so they are
# strongly correlated; adding them in quadrature would understate the total. Sum
# them linearly (≈50% of the total) instead.
_SOILGRIDS_MIN_SIGMA_FRAC = 0.50


def build_soilgrids_soc_total(
    israd_name: str, model
) -> tuple[float, float, float] | None:
    """SoilGrids v2.0 total column SOC (gC m⁻²) → (mean, sigma, depth_coverage).

    Read from ``notebooks/exports/soilgrids_soc_pools.csv`` (written by
    ``notebooks/download_soilgrids_soc.py``). Unlike ``build_soc_prior`` this is
    genuinely independent of the model, and unlike the ISRaD profiles it spans the
    full 0–130 cm column — which is why it can serve as a stock constraint where
    ISRaD cannot. The CSV stores per-pool depth bins; under co-located kinetic
    pools only their SUM is meaningful, so the bins are summed back to a column
    total here rather than used per pool.

    Returns None when the site is absent or the total is physically implausible.
    Treat the values with suspicion: at Howland — the only site with both — SoilGrids
    says 35,498±10,511 gC m⁻² against a measured 11,200±1,680, a 3.2× overestimate
    with non-overlapping intervals.
    """
    if not os.path.exists(SOILGRIDS_CSV):
        return None
    df = pd.read_csv(SOILGRIDS_CSV)
    df = df[df["site"] == israd_name]
    if df.empty:
        return None
    column_cm = _model_column_depth_cm(model)
    z_bot = float(pd.to_numeric(df["depth_bot_cm"], errors="coerce").max())
    if not np.isfinite(z_bot) or z_bot < _MIN_SOC_DEPTH_COVERAGE * column_cm:
        return None
    total = float(pd.to_numeric(df["soc_gCm2"], errors="coerce").sum())
    if not np.isfinite(total) or total <= 0.0 or total > _MAX_PLAUSIBLE_SOC_GCM2:
        return None
    sigma = float(pd.to_numeric(df["soc_sigma_gCm2"], errors="coerce").sum())
    sigma = max(sigma, _SOILGRIDS_MIN_SIGMA_FRAC * total)
    return total, sigma, z_bot / column_cm


def build_measured_soc_stocks(
    israd_name: str, model, sigma_floor_frac: float = _MEASURED_SOC_SIGMA_FRAC
) -> dict[str, tuple[float, float]]:
    """SUPERSEDED — measured ISRaD layer SOC → per-pool C-stock, by depth bin.

    Kept only for reference/back-compat; ``run_site_canonical`` now calls
    ``build_measured_soc_total`` instead. This function assigns a measured stock to
    a pool by depth (active←0–10 cm, slow←10–30, passive←30–130), which is the same
    error the bulk Δ¹⁴C operator used to make: with co-located kinetic pools, no
    observation can attribute stock to an individual pool, so only the column total
    Σ_i C12_i is observable. It also silently under-reports deep pools wherever a
    profile stops short of the modelled column (ISRaD profiles reach only 30 cm at
    MO Ozark, 45 at Bartlett, 62 at EML, 90 at UMBS).
    """
    df = pd.read_csv(ISRAD_LAYER, low_memory=False)
    df = df[df["site_name"] == israd_name].copy()
    df = df.dropna(subset=["lyr_soc", "lyr_top", "lyr_bot"])
    if df.empty:
        return {}
    df["soc_gm2"] = pd.to_numeric(df["lyr_soc"], errors="coerce") * 1.0e4
    df = add_layer_midpoint(df)
    df["_profile"] = (
        df.get("entry_name", pd.Series(index=df.index)).astype(str) + "|"
        + df.get("pro_name", pd.Series(index=df.index)).astype(str) + "|"
        + df.get("lyr_obs_date_y", pd.Series(index=df.index)).astype(str)
    )
    result: dict[str, tuple[float, float]] = {}
    for pool_name, (lo, hi) in _pool_depth_bins_cm(model):
        sub = df.loc[(df["lyr_mid"] >= lo) & (df["lyr_mid"] < hi)]
        gp = sub.groupby("_profile")["soc_gm2"].sum()
        gp = gp[gp > 0.0]
        if gp.empty:
            continue
        mean_val = float(gp.mean())
        std_val = float(gp.std(ddof=1)) if len(gp) > 1 else 0.0
        result[pool_name] = (mean_val, max(std_val, sigma_floor_frac * mean_val))
    return result


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


def build_state0(model, state_ss, pool_blocks, ic_seeds=None):
    """Use the site SOC prior state and seed ¹⁴C from the first available obs.

    Per-pool ¹⁴C seeding priority: (1) a fraction block that names the pool
    (``israd_fraction_soil_slow`` → ``soil_slow``); (2) ``ic_seeds`` — profile-
    derived seeds for whole-sample bulk sites, whose block names carry no pool
    (see ``_bulk_pool_ic_seeds``); (3) the modern ``_FALLBACK_D14C`` constant.
    Bulk-only sites previously skipped straight to (3), initialising even the
    passive pool at a modern value — wrong for aged/permafrost carbon.
    """
    ic_seeds = ic_seeds or {}
    # first observed Δ¹⁴C per pool (blocks are ordered by year)
    first_d14c: dict[str, float] = {}
    for b in pool_blocks:
        for pool in model.pool_index.pool_names:
            if pool in b.name and pool not in first_d14c:
                first_d14c[pool] = float(np.array(b.y)[0])
    c12 = np.array(state_ss.C12, dtype=float)
    c14 = np.zeros_like(c12)
    for name in model.pool_index.pool_names:
        i = model.pool_index[name]
        d14 = first_d14c.get(
            name, ic_seeds.get(name, _FALLBACK_D14C.get(name, 0.0))
        )
        c14[i] = c12[i] * _R_STD * (1.0 + d14 / 1000.0)
    return state_ss._replace(C14=jnp.array(c14, dtype=jnp.float32))


# ── driver ─────────────────────────────────────────────────────────────────

def run_site_canonical(spec: SiteSpec, observation_path: str = "bulk_resp") -> dict:
    if observation_path not in {"bulk_resp", "fraction", "combined"}:
        raise ValueError("observation_path must be 'bulk_resp', 'fraction', or 'combined'")
    print(f"\n══ {spec.label} — {observation_path} OE inversion ══════════════════════")
    config_path = spec.config_path
    model = build_model(config_path)
    idx = model.pool_index
    print(f"  Config: {os.path.relpath(config_path, _REPO_ROOT)}")

    forcing_path = _resolve_forcing_file(spec)
    print(f"  Forcing: {os.path.relpath(forcing_path, _REPO_ROOT)}")
    forcing = _load_site_forcing(spec, forcing_path, model)
    mean_gpp = float(np.nanmean(np.array(forcing.GPP_obs)))

    hemisphere = "NH" if spec.lat >= 0 else "SH"
    years_daily, d14c_daily = load_full_14C_record(
        hua_path=HUA_PATH, graven_path=GRAVEN_PATH, intcal_path=INTCAL_PATH,
        hemisphere=hemisphere, start_year=1500.0, end_year=2025.0,
    )
    forcing = attach_atm14C(forcing, d14c_daily, years_daily)
    time_years = 1970.0 + np.array(forcing.time) / 365.25
    print(f"  Record: {len(time_years)} days ({time_years[0]:.1f}–{time_years[-1]:.1f})  "
          f"mean GPP {mean_gpp*365:.0f} gC m⁻² yr⁻¹  [{hemisphere}]")

    soc_prior_state, _c_pools_prior, ss_years, c_total_obs = build_soc_prior(
        model, forcing
    )
    # Co-located kinetic pools ⇒ per-pool stock is unobservable; the stock
    # constraint is the column total Σ_i C12_i only (ObservationData.C_total_obs).
    soc_source = "model steady state (self-referential)"
    if observation_path == "fraction":
        pool_blocks = build_fraction_14C_blocks(spec.israd_name, forcing.time, model)
        resp = jnp.full(forcing.time.shape[0], jnp.nan, dtype=jnp.float32)
        block_label = "fraction"
    elif observation_path == "combined":
        pool_blocks = (
            build_fraction_14C_blocks(spec.israd_name, forcing.time, model)
            + build_bulk_14C_blocks(spec.israd_name, forcing.time, model)
        )
        resp = build_resp_14C_obs(spec.israd_name, forcing.time)
        # Stock-source priority: a direct ISRaD measurement at the site beats a
        # SoilGrids prediction, which in turn beats the model's own steady state
        # (which is self-referential and carries no independent information — at
        # σ=0.50 it is a deliberate no-op).
        measured = build_measured_soc_total(spec.israd_name, model)
        soilgrids = build_soilgrids_soc_total(spec.israd_name, model)
        if measured is not None:
            mean_soc, sigma_soc, depth_cov = measured
            c_total_obs = (mean_soc, sigma_soc)
            soc_source = f"ISRaD MEASURED total ({depth_cov:.0%} of column)"
        elif soilgrids is not None:
            mean_soc, sigma_soc, depth_cov = soilgrids
            c_total_obs = (mean_soc, sigma_soc)
            soc_source = f"SoilGrids total ({depth_cov:.0%} of column)"
        block_label = "fraction+bulk"
    else:
        pool_blocks = build_bulk_14C_blocks(spec.israd_name, forcing.time, model)
        resp = build_resp_14C_obs(spec.israd_name, forcing.time)
        block_label = "bulk"
    n_resp = int(jnp.sum(~jnp.isnan(resp)))
    soc_total = c_total_obs[0] if c_total_obs else 0.0
    print(f"  Obs: {len(pool_blocks)} {block_label} Δ¹⁴C blocks, {n_resp} respiration Δ¹⁴C, "
          f"1 total-SOC constraint (Σ={soc_total:.0f}±"
          f"{c_total_obs[1]:.0f} gC m⁻² — {soc_source}; "
          f"{ss_years} annual-mean years)")
    if not pool_blocks or (observation_path == "bulk_resp" and n_resp == 0):
        print("  SKIP — insufficient radiocarbon obs.")
        return {"spec": spec, "skipped": True}

    T = int(forcing.time.shape[0])
    obs_full = ObservationData(
        time=forcing.time,
        NEE=jnp.full(T, jnp.nan), GPP=jnp.full(T, jnp.nan),
        ER=jnp.full(T, jnp.nan), NEE_unc=jnp.full(T, jnp.nan),
        delta14C_obs={}, deltaD14C_obs={}, C_pools_obs={}, delta14C_resp=resp,
        C_total_obs=c_total_obs,
    )

    # Whole-sample bulk sites carry no per-pool ¹⁴C split in their block names,
    # so seed the pool ICs from the observed layer profile (aged carbon → passive)
    # instead of the modern _FALLBACK_D14C. Fraction blocks still win by name.
    ic_seeds = _bulk_pool_ic_seeds(spec.israd_name)
    state0 = build_state0(model, soc_prior_state, pool_blocks, ic_seeds=ic_seeds)

    t0 = time.perf_counter()
    result = optimize_oe(
        model, forcing, obs_full, state0=state0,
        fields=OPT_FIELDS, extra_obs_blocks=pool_blocks,
    )
    ch = np.array(result.cost_history)
    print(f"  optimize_oe done [{time.perf_counter()-t0:.1f}s]  J {ch[0]:.1f}→{ch[-1]:.1f}  "
          f"({result.n_iter} iter, converged={result.converged})")

    tau_days = np.exp(np.array(result.params_opt.log_tau))
    print("  optimised τ (yr): " + ", ".join(
        f"{n}={t/365.25:.1f}" for n, t in zip(idx.pool_names, tau_days)))

    return {
        "spec": spec, "skipped": False, "model": model,
        "observation_path": observation_path,
        "config_path": config_path,
        "mean_gpp_gCm2yr": mean_gpp * 365.0,
        "soc_total_gCm2": soc_total,
        "n_cstock": 1 if c_total_obs else 0,
        "soc_source": soc_source,
        "n_pool_blocks": len(pool_blocks), "n_resp": n_resp,
        "tau_years": {n: float(t / 365.25) for n, t in zip(idx.pool_names, tau_days)},
        "cost0": float(ch[0]), "cost_final": float(ch[-1]),
        "converged": bool(result.converged), "n_iter": int(result.n_iter),
        "oe_result": result,
        # pieces needed for downstream information diagnostics (constraint ladder)
        "forcing": forcing, "state0": state0,
        "obs_full": obs_full, "pool_blocks": pool_blocks,
        "params_opt": result.params_opt,
    }


def main(names: list[str] | None = None) -> None:
    specs = select_specs(names)
    rows = []
    for spec in specs:
        try:
            r = run_site_canonical(spec, observation_path=spec.observation_path)
        except Exception as exc:  # noqa: BLE001 — keep going across sites
            print(f"  ERROR at {spec.label}: {exc}")
            continue
        if r.get("skipped"):
            continue
        rows.append({
            "site": spec.israd_name, "label": spec.label,
            "tower_id": spec.tower_id, "biome": spec.biome,
            "mean_GPP_gCm2yr": round(r["mean_gpp_gCm2yr"]),
            "SOC_gCm2": round(r["soc_total_gCm2"]), "n_cstock": r["n_cstock"],
            "n_pool_blocks": r["n_pool_blocks"], "n_resp": r["n_resp"],
            "tau_active_yr": round(r["tau_years"]["soil_active"], 2),
            "tau_slow_yr": round(r["tau_years"]["soil_slow"], 2),
            "tau_passive_yr": round(r["tau_years"]["soil_passive"], 1),
            "J0": round(r["cost0"], 1), "J_final": round(r["cost_final"], 1),
            "converged": r["converged"], "n_iter": r["n_iter"],
        })
    if rows:
        out = os.path.join(_NB_ROOT, "exports", "multisite_canonical_inversions.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSummary ({len(rows)} sites) → {os.path.relpath(out, _REPO_ROOT)}")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
