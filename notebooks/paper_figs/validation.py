from __future__ import annotations

import pandas as pd


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def validate_posterior_table(df: pd.DataFrame) -> None:
    require_columns(
        df,
        [
            "ecosystem",
            "observation_subset",
            "draw",
            "mode",
            "turnover_time_years",
            "decomposition_rate_per_year",
            "carbon_stock",
            "respiration_fraction",
        ],
        "posterior table",
    )
    if (df["turnover_time_years"] <= 0).any():
        raise ValueError("posterior table contains non-positive turnover times.")
    if (df["carbon_stock"] < 0).any():
        raise ValueError("posterior table contains negative carbon stocks.")
    dup = df.duplicated(["ecosystem", "observation_subset", "draw", "mode"])
    if dup.any():
        raise ValueError("posterior table contains duplicate draw identifiers within an ecosystem/subset/mode.")


def validate_information_table(df: pd.DataFrame) -> None:
    require_columns(
        df,
        [
            "ecosystem",
            "observation_subset",
            "mode",
            "degrees_of_freedom",
            "averaging_kernel_diagonal",
            "posterior_sd",
            "prior_sd",
            "uncertainty_reduction_fraction",
            "information_gain_nats",
        ],
        "information-metric table",
    )


def validate_warming_table(df: pd.DataFrame) -> None:
    require_columns(
        df,
        [
            "ecosystem",
            "observation_subset",
            "draw",
            "q10",
            "delta_temperature_c",
            "year",
            "mode",
            "control_carbon",
            "warm_carbon",
            "control_respiration",
            "warm_respiration",
        ],
        "warming-output table",
    )

