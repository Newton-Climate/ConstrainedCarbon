"""
gain_obs_metadata.py — decode the canonical OE observation vector.

Shared helpers used by both the gain/AK figure script
(``four_site_ak_gain_figures.py``) and the observation-key export
(``gain_matrix_observation_key.py``).

Each scalar entry of the OE observation vector becomes one gain-matrix column.
This module walks that vector in the exact order the gain matrix uses and,
for every scalar observation, recovers its physical identity by reading the
time index / pool column back out of each ``ObsBlock.predict`` closure:

  * resp_14C  → respired-CO₂ Δ¹⁴C sampling date (flux-weighted across pools)
  * pool_14C  → (soil pool, sampling date)
  * c_stock   → the soil pool whose time-mean stock is constrained
  * israd_*   → soil pool + measurement year of the radiocarbon block

Two public entry points:

  build_observation_key(site_diags)      → long DataFrame, one row per obs
  descriptive_labels_for_site(site)      → list[str], one label per obs in
                                           gain-matrix column order
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_EPOCH = np.datetime64("1970-01-01")


# ── short / descriptive column labels ───────────────────────────────────────

def _short_constraint_label(label: str) -> str:
    """Compact positional label (e.g. ``resp_14C[1]`` → ``resp[1]``)."""
    replacements = {
        "resp_14C": "resp",
        "pool_14C": "pool",
        "c_stock": "C",
        "er_annual": "ER",
        "israd_layer_": "layer:",
        "israd_density_": "density:",
        "israd_fraction_": "fraction:",
        "israd_macrofossil_": "macro:",
    }
    out = label
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    out = out.replace("soil_active", "active")
    out = out.replace("soil_slow", "slow")
    out = out.replace("soil_passive", "passive")
    return out


def _pool_short(pool: str | None) -> str:
    return (pool or "").replace("soil_", "")


def _descriptive_label(block_name: str, meta: dict) -> str:
    """Human-readable gain-matrix column label from decoded metadata."""
    pool = _pool_short(meta.get("pool_name"))
    year = meta.get("measurement_year")
    if block_name == "resp_14C":
        return f"resp Δ14C · {meta['sampling_date']}"
    if block_name == "pool_14C":
        return f"pool Δ14C · {pool} {year}"
    if block_name == "c_stock":
        return f"C stock · {pool}"
    if block_name.startswith(("israd_layer", "israd_bulk")):
        return f"bulk Δ14C · {pool}"
    if block_name.startswith("israd_macrofossil"):
        return f"macro Δ14C · {pool} {year}"
    if block_name.startswith("c_sum"):
        return f"C sum · {block_name.replace('c_sum_', '')}"
    # density-fraction blocks (israd_density_* / israd_*)
    return f"frac Δ14C · {pool} {year}"


# ── closure introspection ───────────────────────────────────────────────────

def _day_to_iso(day: float) -> str:
    """Days-since-1970 → ISO calendar date (observation time base)."""
    return str(_EPOCH + np.timedelta64(int(round(float(day))), "D"))


def _first_default_array(block) -> np.ndarray | None:
    """First positional default of a predict closure, as a numpy array."""
    defaults = block.predict.__defaults__ or ()
    if not defaults:
        return None
    return np.asarray(defaults[0]).ravel()


def _year_from_name(block_name: str) -> int | None:
    """Trailing 4-digit measurement year encoded in an israd block name."""
    m = re.search(r"_((?:18|19|20)\d{2})$", block_name)
    return int(m.group(1)) if m else None


def _pool_from_name(block_name: str) -> str:
    """Best-effort soil-pool name parsed from an israd block name."""
    for tok in ("soil_active", "soil_slow", "soil_passive", "active", "slow", "passive"):
        if tok in block_name:
            return tok if tok.startswith("soil_") else f"soil_{tok}"
    return ""


def _israd_description(block_name: str, pool: str | None) -> str:
    """Human description of a single-obs radiocarbon block from its name."""
    p = pool or "the mapped"
    if block_name.startswith("israd_bulk"):
        return (
            f"ISRaD whole-soil (bulk) Δ¹⁴C, depth-binned to the {p} pool "
            f"(mass-weighted mixture of all pools at that depth)"
        )
    if block_name.startswith("israd_macrofossil"):
        return f"ISRaD macrofossil Δ¹⁴C mapped to the {p} pool"
    if block_name.startswith("israd_layer"):
        return f"ISRaD bulk soil-layer Δ¹⁴C mapped to the {p} pool by depth"
    if block_name.startswith("israd_density"):
        return f"ISRaD density-fraction Δ¹⁴C isolating the {p} kinetic pool"
    if block_name.startswith("c_sum"):
        return "Summed carbon-stock constraint across multiple pools"
    if block_name.startswith("israd"):
        return f"ISRaD density-fraction Δ¹⁴C isolating the {p} kinetic pool"
    return f"Radiocarbon constraint on the {p} pool"


def _block_scalar_meta(
    block,
    n_obs: int,
    pool_names: list[str],
    time_days: np.ndarray,
    years: np.ndarray,
) -> list[dict]:
    """Per-scalar physical metadata for one ObsBlock, in OE order."""
    name = block.name
    defaults = block.predict.__defaults__ or ()
    metas: list[dict] = []

    if name == "resp_14C":
        t_arr = _first_default_array(block)
        for i in range(n_obs):
            t = int(t_arr[i])
            metas.append(
                {
                    "pool_name": "flux-weighted (all pools)",
                    "sampling_time_index": t,
                    "sampling_date": _day_to_iso(time_days[t]),
                    "sampling_year_decimal": round(float(years[t]), 3),
                    "measurement_year": int(np.floor(years[t])),
                    "model_eval_date": _day_to_iso(time_days[t]),
                    "physical_meaning": (
                        "Respired CO₂ Δ¹⁴C, flux-weighted (C/τ) mean across all "
                        "soil pools, at this sampling date"
                    ),
                    "units": "per mil (‰)",
                }
            )
        return metas

    if name == "pool_14C":
        t_arr = np.asarray(defaults[0]).ravel()
        col_arr = np.asarray(defaults[1]).ravel()
        for i in range(n_obs):
            t = int(t_arr[i])
            pool = pool_names[int(col_arr[i])]
            metas.append(
                {
                    "pool_name": pool,
                    "sampling_time_index": t,
                    "sampling_date": _day_to_iso(time_days[t]),
                    "sampling_year_decimal": round(float(years[t]), 3),
                    "measurement_year": int(np.floor(years[t])),
                    "model_eval_date": _day_to_iso(time_days[t]),
                    "physical_meaning": (
                        f"Pool-level Δ¹⁴C for the {pool} pool at this sampling date"
                    ),
                    "units": "per mil (‰)",
                }
            )
        return metas

    if name == "c_stock":
        col_arr = np.asarray(defaults[0]).ravel()
        for i in range(n_obs):
            pool = pool_names[int(col_arr[i])]
            metas.append(
                {
                    "pool_name": pool,
                    "sampling_time_index": "",
                    "sampling_date": "time-mean (full simulated record)",
                    "sampling_year_decimal": "",
                    "measurement_year": "",
                    "model_eval_date": "time-mean (full simulated record)",
                    "physical_meaning": f"Time-mean carbon stock of the {pool} pool",
                    "units": "gC m⁻²",
                }
            )
        return metas

    # Single-obs radiocarbon blocks (israd_*, c_sum_*): time window in the
    # closure's first default, pool column (if any) in the second.
    t_arr = _first_default_array(block)
    t0 = int(t_arr[0]) if t_arr is not None and t_arr.size else 0
    pool = None
    if len(defaults) >= 2 and np.ndim(defaults[1]) == 0:
        pool = pool_names[int(defaults[1])]
    if pool is None:
        pool = _pool_from_name(name) or None
    # The block name carries the true measurement year; the forward operator
    # samples the nearest available model day, which is clamped to the start of
    # the simulation window for measurements that predate it.
    name_year = _year_from_name(name)
    eval_year = int(np.floor(years[t0]))
    meas_year = name_year if name_year is not None else eval_year
    for _ in range(n_obs):
        metas.append(
            {
                "pool_name": pool or "",
                "sampling_time_index": t0,
                "sampling_date": (
                    f"{meas_year} (measurement year)"
                    if name_year is not None
                    else _day_to_iso(time_days[t0])
                ),
                "sampling_year_decimal": (
                    float(meas_year)
                    if name_year is not None
                    else round(float(years[t0]), 3)
                ),
                "measurement_year": meas_year,
                "model_eval_date": _day_to_iso(time_days[t0]),
                "physical_meaning": _israd_description(name, pool),
                "units": "per mil (‰)",
            }
        )
    return metas


def _site_years(site: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(pool_names, time_days, decimal years) for a diagnostics site dict."""
    data = site["data"]
    obs_full = data["obs_full"]
    time_days = np.asarray(obs_full.time, dtype=float)
    years = 1970.0 + time_days / 365.25
    pool_names = list(data["model"].pool_index.pool_names)
    return pool_names, time_days, years


def descriptive_labels_for_site(site: dict) -> list[str]:
    """Descriptive gain-matrix column labels, one per obs in OE order."""
    diag = site["diag"]
    pool_names, time_days, years = _site_years(site)
    labels: list[str] = []
    for block in diag["obs_blocks"]:
        n_obs = int(block.y.shape[0])
        metas = _block_scalar_meta(block, n_obs, pool_names, time_days, years)
        labels.extend(_descriptive_label(block.name, m) for m in metas)
    return labels


def build_observation_key(site_diags: list[dict]) -> pd.DataFrame:
    """One row per scalar OE observation, in gain-matrix column order."""
    rows: list[dict] = []
    for site in site_diags:
        diag = site["diag"]
        obs_blocks = diag["obs_blocks"]
        constraint_labels = diag["constraint_labels"]
        annotations = {int(a["obs_index"]): a for a in diag["obs_annotations"]}
        pool_names, time_days, years = _site_years(site)

        offset = 0
        for block in obs_blocks:
            n_obs = int(block.y.shape[0])
            y_block = np.asarray(block.y, dtype=float)
            metas = _block_scalar_meta(block, n_obs, pool_names, time_days, years)
            for i in range(n_obs):
                obs_index = offset + i
                ann = annotations[obs_index]
                # Cross-check: reconstructed scalar must be the annotated obs.
                if not np.isclose(y_block[i], ann["y_obs"], rtol=1e-4, atol=1e-4):
                    raise AssertionError(
                        f"{site['site']} obs {obs_index} ({block.name}[{i + 1}]): "
                        f"block y={y_block[i]} != annotation y={ann['y_obs']}"
                    )
                meta = metas[i]
                rows.append(
                    {
                        "site": site["site"],
                        "site_id": site["site_id"],
                        "obs_index": obs_index,
                        "plot_label_short": _short_constraint_label(
                            constraint_labels[obs_index]
                        ),
                        "plot_label_descriptive": _descriptive_label(block.name, meta),
                        "obs_label_full": ann["obs_label_full"],
                        "obs_block_name": ann["obs_block_name"],
                        "obs_family": ann["obs_family"],
                        "pool_name": meta["pool_name"],
                        "sampling_date": meta["sampling_date"],
                        "sampling_year_decimal": meta["sampling_year_decimal"],
                        "measurement_year": meta["measurement_year"],
                        "model_eval_date": meta["model_eval_date"],
                        "sampling_time_index": meta["sampling_time_index"],
                        "physical_meaning": meta["physical_meaning"],
                        "units": meta["units"],
                        "y_obs": ann["y_obs"],
                        "y_prior": ann["y_prior"],
                        "y_opt": ann["y_opt"],
                        "obs_sigma": ann["obs_sigma"],
                        "obs_variance": ann["obs_variance"],
                    }
                )
            offset += n_obs
    return pd.DataFrame(rows)
