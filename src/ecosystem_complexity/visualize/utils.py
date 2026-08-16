from __future__ import annotations

import argparse
import math
import os
from os import PathLike
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FigureConfig, load_figure_config
from .exports import append_caption, ensure_dir, save_alt_text, save_csvs, save_figure_triplet
from .io import load_tables
from .styling import PANEL_LABELS, add_panel_label, apply_publication_style


DataLike = str | PathLike[str] | pd.DataFrame | Iterable[str | PathLike[str]] | None


def coerce_table(data: DataLike, label: str) -> pd.DataFrame | None:
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (str, os.PathLike)):
        return load_tables(data, label)
    if isinstance(data, Iterable):
        items = list(data)
        if not items:
            raise ValueError(f"{label} received an empty iterable of input tables.")
        return load_tables(items, label)
    raise TypeError(
        f"{label} must be a pandas DataFrame, file path, iterable of file paths, or None."
    )


def standard_figure_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="outputs")
    return parser


def prepare_output_paths(output_dir: str, figure_id: str) -> dict[str, str]:
    fig_dir = os.path.join(output_dir, "figures")
    csv_dir = os.path.join(output_dir, "csv")
    alt_dir = os.path.join(output_dir, "alt_text")
    ensure_dir(fig_dir)
    ensure_dir(csv_dir)
    ensure_dir(alt_dir)
    return {
        "figure_base": os.path.join(fig_dir, figure_id),
        "csv_dir": os.path.join(csv_dir, figure_id),
        "alt_text": os.path.join(alt_dir, f"{figure_id}.txt"),
        "captions": os.path.join(output_dir, "figure_captions.md"),
    }


def finalize_figure(
    fig,
    figure_id: str,
    output_dir: str,
    csv_map: dict[str, pd.DataFrame],
    alt_text: str,
    caption_title: str,
    caption_text: str,
) -> None:
    paths = prepare_output_paths(output_dir, figure_id)
    save_figure_triplet(fig, paths["figure_base"])
    save_csvs(csv_map, paths["csv_dir"])
    save_alt_text(alt_text, paths["alt_text"])
    append_caption(paths["captions"], caption_title, caption_text)


def summarize_quantiles(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    probs: tuple[float, float, float] = (0.05, 0.5, 0.95),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, grp in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = grp[value_col].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        row = dict(zip(group_cols, keys))
        row["n"] = int(vals.size)
        row["q05"] = float(np.quantile(vals, probs[0]))
        row["median"] = float(np.quantile(vals, probs[1]))
        row["q95"] = float(np.quantile(vals, probs[2]))
        row["q25"] = float(np.quantile(vals, 0.25))
        row["q75"] = float(np.quantile(vals, 0.75))
        row["mean"] = float(np.mean(vals))
        row["sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    seed: int = 7,
) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (math.nan, math.nan, math.nan)
    if vals.size == 1:
        v = float(vals[0])
        return (v, v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boots = vals[idx].mean(axis=1)
    return (
        float(vals.mean()),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


def order_categories(df: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
    if col in df.columns:
        cat = pd.Categorical(df[col], categories=order, ordered=True)
        return df.assign(**{col: cat}).sort_values(col)
    return df


def maybe_add_zero_line(ax, orientation: str = "h") -> None:
    if orientation == "h":
        ax.axhline(0.0, color="0.55", linestyle=":", linewidth=0.8, zorder=0)
    else:
        ax.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8, zorder=0)


def add_one_to_one(ax) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lo = min(x0, y0)
    hi = max(x1, y1)
    ax.plot([lo, hi], [lo, hi], color="0.55", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def panelize(axes) -> None:
    for ax, label in zip(np.ravel(axes), PANEL_LABELS):
        add_panel_label(ax, label)


def setup_figure_config(config_path: str | None) -> FigureConfig:
    apply_publication_style()
    return load_figure_config(config_path)


def select_observation_subset(
    df: pd.DataFrame,
    cfg: FigureConfig,
    subset: str | None = None,
    label: str = "table",
) -> str:
    if "observation_subset" not in df.columns:
        raise ValueError(f"{label} must include an observation_subset column.")
    available = list(pd.Series(df["observation_subset"]).dropna().drop_duplicates())
    if not available:
        raise ValueError(f"{label} does not contain any observation subsets.")
    if subset is not None:
        if subset not in available:
            raise ValueError(
                f"Requested observation subset {subset!r} is not available in {label}. "
                f"Available subsets: {available}"
            )
        return subset
    preferred_aliases = [
        "full_radiocarbon",
        "all_observations",
        "stocks_fluxes_soil_respired14c",
        "stocks_soil_resp14c",
        "stocks_soil14c_resp14c",
    ]
    for name in preferred_aliases:
        if name in available:
            return name
    preferred = [name for name in reversed(cfg.observation_subset_order) if name in available]
    return preferred[0] if preferred else available[0]


def close_or_show(fig, show: bool = False) -> None:
    if show:
        plt.show()
    else:
        plt.close(fig)
