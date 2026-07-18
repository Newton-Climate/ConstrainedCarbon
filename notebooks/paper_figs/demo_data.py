from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from notebooks.paper_figs.config import load_figure_config
else:
    from .config import load_figure_config


def build_demo_inputs(
    config_path: str | None = None,
    n_draws: int = 200,
    seed: int = 7,
) -> dict[str, pd.DataFrame]:
    cfg = load_figure_config(config_path)
    rng = np.random.default_rng(seed)
    ecosystems = list(cfg.ecosystem_order)
    subsets = list(cfg.observation_subset_order)
    modes = list(cfg.turnover_mode_labels)
    mode_tau = {"fast": 1.2, "intermediate": 18.0, "slow": 260.0}
    mode_stock = {"fast": 0.6, "intermediate": 2.2, "slow": 8.5}
    mode_resp = {"fast": 0.52, "intermediate": 0.31, "slow": 0.17}

    observations = []
    for eco_i, eco in enumerate(ecosystems):
        stored_mean = -120 + eco_i * 28
        resp_mean = stored_mean + [45, 30, 12, 70][eco_i % 4]
        for obs_type, mean_v, n_obs in [
            ("bulk_soil", stored_mean, 12),
            ("depth_resolved_soil", stored_mean - 20, 8),
            ("fraction_specific_soil", stored_mean - 10, 6),
            ("respired_carbon", resp_mean, 10),
        ]:
            for idx in range(n_obs):
                observations.append(
                    {
                        "ecosystem": eco,
                        "biome": "demo",
                        "site": f"demo-{eco_i + 1}",
                        "observation_type": obs_type,
                        "delta14c_permil": float(rng.normal(mean_v, 18)),
                        "delta14c_uncertainty_permil": 10.0,
                        "soil_depth_top_cm": float(idx * 5) if "soil" in obs_type else np.nan,
                        "soil_depth_bottom_cm": float((idx + 1) * 5) if "soil" in obs_type else np.nan,
                        "fraction_name": "bulk" if obs_type == "bulk_soil" else obs_type,
                        "sampling_date": "2025-07-01",
                    }
                )

    posterior = []
    info = []
    for eco_i, eco in enumerate(ecosystems):
        eco_shift = 1.0 + eco_i * 0.15
        for subset_i, subset in enumerate(subsets):
            completeness = 0.45 + 0.12 * subset_i
            for draw in range(n_draws):
                for mode in modes:
                    tau = mode_tau[mode] * eco_shift * np.exp(
                        rng.normal(0.0, 0.18 - 0.02 * min(subset_i, 4))
                    )
                    posterior.append(
                        {
                            "ecosystem": eco,
                            "observation_subset": subset,
                            "draw": draw,
                            "mode": mode,
                            "turnover_time_years": float(tau),
                            "decomposition_rate_per_year": float(1.0 / tau),
                            "carbon_stock": float(
                                mode_stock[mode] * eco_shift * rng.normal(1.0, 0.08)
                            ),
                            "respiration_fraction": float(
                                max(0.02, mode_resp[mode] + rng.normal(0.0, 0.02))
                            ),
                            "prior_lower_bound": mode_tau[mode] / 5.0,
                            "prior_upper_bound": mode_tau[mode] * 6.0,
                        }
                    )
            for mode_i, mode in enumerate(modes):
                prior_sd = 0.45 - 0.08 * mode_i
                ak = np.clip(completeness - 0.12 * mode_i + eco_i * 0.03, 0.05, 0.98)
                post_sd = max(0.03, prior_sd * (1.0 - 0.65 * ak))
                info.append(
                    {
                        "ecosystem": eco,
                        "observation_subset": subset,
                        "mode": mode,
                        "degrees_of_freedom": float(ak * (0.75 + 0.15 * mode_i)),
                        "averaging_kernel_diagonal": float(ak),
                        "posterior_sd": float(post_sd),
                        "prior_sd": float(prior_sd),
                        "uncertainty_reduction_fraction": float(1.0 - post_sd / prior_sd),
                        "information_gain_nats": float(
                            0.5 * np.log((prior_sd**2) / (post_sd**2))
                        ),
                    }
                )

    warming = []
    for eco_i, eco in enumerate(ecosystems):
        eco_shift = 1.0 + eco_i * 0.12
        for subset_i, subset in enumerate(subsets):
            completeness = 0.5 + 0.1 * subset_i
            for draw in range(n_draws):
                draw_scale = np.exp(rng.normal(0.0, 0.08))
                for year in cfg.warming_years:
                    for mode_i, mode in enumerate(modes):
                        base_c = (
                            mode_stock[mode]
                            * eco_shift
                            * (1.0 + 0.04 * year / 100.0)
                            * draw_scale
                        )
                        loss_frac = (
                            (0.015 + 0.01 * mode_i)
                            * (year / 10.0)
                            / (1.0 + 0.35 * completeness)
                        )
                        warm_c = base_c * max(0.2, 1.0 - loss_frac)
                        base_r = (mode_stock[mode] / mode_tau[mode]) * eco_shift * draw_scale
                        excess = (
                            base_r
                            * (0.1 + 0.08 * mode_i)
                            * (year / 100.0)
                            / (1.0 + 0.45 * completeness)
                        )
                        warming.append(
                            {
                                "ecosystem": eco,
                                "observation_subset": subset,
                                "draw": draw,
                                "q10": cfg.default_q10,
                                "delta_temperature_c": cfg.default_delta_t,
                                "year": year,
                                "mode": mode,
                                "control_carbon": float(base_c),
                                "warm_carbon": float(warm_c),
                                "control_respiration": float(base_r),
                                "warm_respiration": float(base_r + excess),
                                "cumulative_control_respiration": float(base_r * year),
                                "cumulative_warm_respiration": float((base_r + excess) * year),
                            }
                        )

    cesm = []
    for eco_i, eco in enumerate(ecosystems):
        eco_shift = 1.0 + eco_i * 0.1
        for source, source_scale in [("observation_constrained", 1.0), ("CESM", 1.3)]:
            for draw in range(80):
                for mode_i, mode in enumerate(modes):
                    cesm.append(
                        {
                            "ecosystem": eco,
                            "model_source": source,
                            "draw_or_member": draw,
                            "mode": mode,
                            "turnover_time_years": float(
                                mode_tau[mode]
                                * eco_shift
                                * source_scale
                                * np.exp(rng.normal(0.0, 0.15))
                            ),
                            "carbon_stock": float(
                                mode_stock[mode] * eco_shift * source_scale * rng.normal(1.0, 0.06)
                            ),
                            "baseline_respiration": float(
                                (mode_stock[mode] / mode_tau[mode]) * eco_shift * rng.normal(1.0, 0.05)
                            ),
                            "warm_respiration": float(
                                (mode_stock[mode] / mode_tau[mode])
                                * eco_shift
                                * source_scale
                                * rng.normal(1.2, 0.05)
                            ),
                            "cumulative_carbon_loss": float(
                                mode_stock[mode] * eco_shift * 0.2 * source_scale * rng.normal(1.0, 0.08)
                            ),
                            "slow_carbon_fraction_of_excess_respiration": float(
                                np.clip(
                                    0.35 + 0.12 * mode_i + (0.08 if source == "CESM" else 0.0),
                                    0.0,
                                    1.0,
                                )
                            ),
                        }
                    )

    topology = []
    for eco in ecosystems:
        for label, score in [("two_mode", -2.4), ("three_mode", 0.0), ("four_mode", -0.8)]:
            topology.append(
                {
                    "ecosystem": eco,
                    "topology_label": label,
                    "score_difference": score + float(rng.normal(0.0, 0.25)),
                }
            )

    kernel_rows = []
    kernel_cols = [
        "log_tau[soil_active]",
        "log_tau[soil_slow]",
        "log_tau[soil_passive]",
        "log_f_transfer[soil_active→soil_slow]",
        "log_f_transfer[soil_slow→soil_passive]",
    ]
    for eco_i, eco in enumerate(ecosystems):
        mat = np.array(
            [
                [0.62, 0.08, 0.02, 0.04, 0.01],
                [0.07, 0.46, 0.08, 0.06, 0.02],
                [0.02, 0.07, 0.24, 0.03, 0.05],
                [0.03, 0.11, 0.04, 0.35, 0.07],
                [0.01, 0.03, 0.09, 0.05, 0.28],
            ]
        ) + eco_i * 0.015
        row_names = [
            "log_tau[soil_active]",
            "log_tau[soil_slow]",
            "log_tau[soil_passive]",
            "log_f_transfer[soil_active→soil_slow]",
            "log_f_transfer[soil_slow→soil_passive]",
        ]
        for idx, row_name in enumerate(row_names):
            row = {
                "site": eco,
                "site_id": f"DEMO-{eco_i + 1}",
                "param_index": idx,
                "param_name": row_name,
                "param_group": "turnover_times"
                if "log_tau" in row_name
                else "transfer_fractions",
            }
            for j, col in enumerate(kernel_cols):
                row[col] = float(mat[idx, j])
            kernel_rows.append(row)

    return {
        "observations": pd.DataFrame(observations),
        "posterior": pd.DataFrame(posterior),
        "information_metrics": pd.DataFrame(info),
        "warming_output": pd.DataFrame(warming),
        "cesm_comparison": pd.DataFrame(cesm),
        "topology_comparison": pd.DataFrame(topology),
        "averaging_kernel_matrix": pd.DataFrame(kernel_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic demo inputs for the manuscript figure pipeline."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    demo = build_demo_inputs(config_path=args.config, n_draws=args.n_draws, seed=args.seed)
    for name, df in demo.items():
        print(f"{name}: {df.shape[0]} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
