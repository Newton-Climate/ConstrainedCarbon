"""Load user-supplied radiocarbon observations from a CSV plus YAML manifest.

The manifest records site-level metadata and the fraction-property → model-pool
mapping; the CSV remains a simple, spreadsheet-friendly laboratory data table.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

from ecosystem_complexity._oe_helpers import ObsBlock
from ecosystem_complexity.data.israd_observations import bulk_mixture_obs_block

_KINDS = {"bulk", "fraction", "respiration"}
_REQUIRED = {"sample_id", "kind", "date", "delta14c", "delta14c_sigma"}


@dataclass(frozen=True)
class Custom14CData:
    """Validated measurements and the mapping supplied in a manifest."""

    measurements: pd.DataFrame
    fraction_rules: dict[str, str]
    manifest_path: Path


def load_custom_14c_manifest(path: str | Path) -> Custom14CData:  # noqa: C901
    """Load and validate a custom ¹⁴C manifest and its measurement CSV.

    The CSV requires ``sample_id, kind, date, delta14c, delta14c_sigma``.
    ``kind`` is ``bulk``, ``fraction``, or ``respiration``. Bulk and fraction
    rows additionally require ``depth_top_cm, depth_bottom_cm``; fraction rows
    require ``fraction_property``. Dates must be ISO dates or decimal years.
    """
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as fh:
        manifest: dict[str, Any] = yaml.safe_load(fh) or {}
    csv_name = manifest.get("measurements")
    if not isinstance(csv_name, str) or not csv_name:
        raise ValueError(f"{manifest_path}: manifest needs a `measurements` CSV path")
    csv_path = (manifest_path.parent / csv_name).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Custom ¹⁴C CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns {sorted(missing)}")
    df = df.copy()
    df["kind"] = df["kind"].astype(str).str.strip().str.lower()
    bad_kinds = sorted(set(df["kind"]) - _KINDS)
    if bad_kinds:
        raise ValueError(
            f"{csv_path}: unknown `kind` values {bad_kinds}; use {sorted(_KINDS)}"
        )
    for col in ("delta14c", "delta14c_sigma"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid_measurements = (
        df[["delta14c", "delta14c_sigma"]].isna().any().any()
        or (df["delta14c_sigma"] <= 0).any()
    )
    if invalid_measurements:
        raise ValueError(
            f"{csv_path}: delta14c must be numeric and delta14c_sigma must be positive"
        )
    df["obs_year"] = _parse_years(df["date"], csv_path)
    soil_rows = df["kind"].isin(["bulk", "fraction"])
    for col in ("depth_top_cm", "depth_bottom_cm"):
        if col not in df:
            raise ValueError(
                f"{csv_path}: {col} is required for bulk and fraction rows"
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df.loc[soil_rows, col].isna().any():
            raise ValueError(
                f"{csv_path}: {col} is required for bulk and fraction rows"
            )
    if "fraction_property" not in df:
        df["fraction_property"] = ""
    df["fraction_property"] = df["fraction_property"].fillna("").astype(str).str.strip()
    if (df.loc[df["kind"].eq("fraction"), "fraction_property"] == "").any():
        raise ValueError(f"{csv_path}: fraction_property is required for fraction rows")
    rules = {
        str(prop).strip().lower(): str(pool).strip()
        for prop, pool in dict(manifest.get("fraction_rules") or {}).items()
        if pool is not None
    }
    return Custom14CData(df, rules, manifest_path)


def _parse_years(values: pd.Series, csv_path: Path) -> pd.Series:
    """Parse decimal years or ISO dates into decimal years."""
    numeric = pd.to_numeric(values, errors="coerce")
    parsed = pd.to_datetime(values, errors="coerce")
    years = numeric.astype(float)
    dates = parsed.notna() & numeric.isna()
    years.loc[dates] = (
        parsed.loc[dates].dt.year
        + (parsed.loc[dates].dt.dayofyear - 1) / 365.25
    )
    if years.isna().any():
        bad = values.loc[years.isna()].astype(str).tolist()
        raise ValueError(f"{csv_path}: invalid date values {bad}")
    return years


def _aggregate(values: pd.DataFrame, sigma_floor: float) -> tuple[float, float]:
    """Inverse-variance mean with a conservative uncertainty floor."""
    y = values["delta14c"].to_numpy(dtype=float)
    sigma = values["delta14c_sigma"].to_numpy(dtype=float)
    w = 1.0 / sigma**2
    mean = float(np.dot(w, y) / w.sum())
    propagated = float(np.sqrt(1.0 / w.sum()))
    spread = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    return mean, max(sigma_floor, propagated, spread)


def build_custom_14c_observations(
    data: Custom14CData, forcing_time, pool_index, *, sigma_floor: float = 15.0
) -> tuple[list[ObsBlock], jnp.ndarray]:
    """Build pool/bulk blocks and sparse respiration observations from lab data."""
    time_np = np.asarray(forcing_time, dtype=float)
    years_np = 1970.0 + time_np / 365.25
    blocks: list[ObsBlock] = []
    bulk = data.measurements.query("kind == 'bulk'")
    for date, group in bulk.groupby("date", sort=True):
        mean, sigma = _aggregate(group, sigma_floor)
        obs_year = float(group["obs_year"].iloc[0])
        t = jnp.array([int(np.argmin(np.abs(years_np - obs_year)))], dtype=jnp.int32)
        blocks.append(
            bulk_mixture_obs_block(
                f"custom_bulk_{_name_date(date)}", mean, sigma, t
            )
        )

    fractions = data.measurements.query("kind == 'fraction'").copy()
    fractions["pool_name"] = (
        fractions["fraction_property"].str.lower().map(data.fraction_rules)
    )
    unknown = sorted(
        fractions.loc[fractions["pool_name"].isna(), "fraction_property"].unique()
    )
    if unknown:
        raise ValueError(
            f"{data.manifest_path}: no fraction_rules mapping for {unknown}"
        )
    invalid = sorted(set(fractions["pool_name"]) - set(pool_index.pool_names))
    if invalid:
        raise ValueError(
            f"{data.manifest_path}: fraction_rules refer to unknown model pools "
            f"{invalid}"
        )
    for (pool, date), group in fractions.groupby(["pool_name", "date"], sort=True):
        mean, sigma = _aggregate(group, sigma_floor)
        pi = int(pool_index[pool])
        obs_year = float(group["obs_year"].iloc[0])
        t = jnp.array([int(np.argmin(np.abs(years_np - obs_year)))], dtype=jnp.int32)
        blocks.append(ObsBlock(
            name=f"custom_fraction_{pool}_{_name_date(date)}",
            y=jnp.array([mean], dtype=jnp.float32),
            Se=jnp.array([sigma**2], dtype=jnp.float32),
            predict=lambda out, p, t=t, pi=pi: out.delta14C[t, pi],
        ))

    resp = np.full(len(time_np), np.nan, dtype=np.float32)
    respiration = data.measurements.query("kind == 'respiration'").copy()
    for _, group in respiration.groupby("date", sort=True):
        mean, _ = _aggregate(group, sigma_floor)
        obs_year = float(group["obs_year"].iloc[0])
        resp[int(np.argmin(np.abs(years_np - obs_year)))] = mean
    return blocks, jnp.array(resp)


def _name_date(value: Any) -> str:
    """Make an observation date safe to include in an observation-block name."""
    return str(value).replace("-", "_").replace(".", "_").replace(" ", "_")
