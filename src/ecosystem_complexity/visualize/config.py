from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_ECOSYSTEM_ORDER = [
    "Harvard Forest",
    "Barrow",
    "Howland Forest",
    "Eight-mile Lake",
]

DEFAULT_OBSERVATION_SUBSET_ORDER = [
    "stocks",
    "stocks_fluxes",
    "stocks_fluxes_soil14c",
    "stocks_fluxes_respired14c",
    "stocks_fluxes_soil_respired14c",
    "full_radiocarbon",
]

DEFAULT_TURNOVER_MODE_LABELS = ["fast", "intermediate", "slow"]
DEFAULT_WARMING_YEARS = [10, 50, 100]
DEFAULT_DELTA_T = 4.0
DEFAULT_Q10 = 2.0
DEFAULT_Q10_SENSITIVITY = [1.5, 2.0, 2.5]
DEFAULT_REPRESENTATIVE_ECOSYSTEMS = ["Harvard Forest", "Barrow"]


@dataclass
class FigureConfig:
    ecosystem_order: list[str] = field(default_factory=lambda: list(DEFAULT_ECOSYSTEM_ORDER))
    observation_subset_order: list[str] = field(
        default_factory=lambda: list(DEFAULT_OBSERVATION_SUBSET_ORDER)
    )
    turnover_mode_labels: list[str] = field(
        default_factory=lambda: list(DEFAULT_TURNOVER_MODE_LABELS)
    )
    warming_years: list[int] = field(default_factory=lambda: list(DEFAULT_WARMING_YEARS))
    default_delta_t: float = DEFAULT_DELTA_T
    default_q10: float = DEFAULT_Q10
    q10_sensitivity: list[float] = field(default_factory=lambda: list(DEFAULT_Q10_SENSITIVITY))
    representative_ecosystems: list[str] = field(
        default_factory=lambda: list(DEFAULT_REPRESENTATIVE_ECOSYSTEMS)
    )


def _load_override_file(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Figure config file not found: {path}")
    suffix = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as fh:
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("YAML config requested but PyYAML is not installed.")
            return yaml.safe_load(fh) or {}
        if suffix == ".json":
            return json.load(fh) or {}
    raise ValueError(f"Unsupported figure config extension for {path!r}. Use .yaml, .yml, or .json.")


def load_figure_config(path: str | None = None) -> FigureConfig:
    cfg = FigureConfig()
    if path is None:
        return cfg
    overrides = _load_override_file(path)
    mapping = {
        "ECOSYSTEM_ORDER": "ecosystem_order",
        "OBSERVATION_SUBSET_ORDER": "observation_subset_order",
        "TURNOVER_MODE_LABELS": "turnover_mode_labels",
        "WARMING_YEARS": "warming_years",
        "DEFAULT_DELTA_T": "default_delta_t",
        "DEFAULT_Q10": "default_q10",
        "Q10_SENSITIVITY": "q10_sensitivity",
        "REPRESENTATIVE_ECOSYSTEMS": "representative_ecosystems",
    }
    for src_key, dst_key in mapping.items():
        if src_key in overrides:
            setattr(cfg, dst_key, overrides[src_key])
    return cfg

