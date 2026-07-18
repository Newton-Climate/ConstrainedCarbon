from __future__ import annotations

import glob
import os
from os import PathLike
from typing import Iterable

import pandas as pd


TableSource = str | PathLike[str]


def _normalize_path(path: TableSource) -> str:
    return os.fspath(path)


def expand_table_sources(source: TableSource | Iterable[TableSource]) -> list[str]:
    if isinstance(source, (str, os.PathLike)):
        path = _normalize_path(source)
        if any(ch in path for ch in "*?["):
            matches = sorted(glob.glob(path))
            if not matches:
                raise FileNotFoundError(f"No input tables matched glob pattern: {path}")
            return matches
        if os.path.isdir(path):
            matches = sorted(
                os.path.join(path, name)
                for name in os.listdir(path)
                if os.path.splitext(name)[1].lower() in {".csv", ".parquet"}
            )
            if not matches:
                raise FileNotFoundError(
                    f"Directory does not contain any .csv or .parquet tables: {path}"
                )
            return matches
        return [path]
    return [_normalize_path(item) for item in source]


def load_table(path: TableSource) -> pd.DataFrame:
    path = _normalize_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required input table not found: {path}")
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format for {path!r}. Expected .csv or .parquet.")


def load_tables(source: TableSource | Iterable[TableSource], label: str) -> pd.DataFrame:
    paths = expand_table_sources(source)
    frames = [load_table(path) for path in paths]
    if not frames:
        raise FileNotFoundError(f"No input tables were supplied for {label}.")
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)
