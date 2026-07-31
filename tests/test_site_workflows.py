from __future__ import annotations

import json
import pathlib
import textwrap

import numpy as np

from ecosystem_complexity.config import load_config
from ecosystem_complexity.fetch.colocation import locate_site
from ecosystem_complexity.site_analysis import (
    compute_information_metrics,
    load_exported_analysis,
)
from ecosystem_complexity.site_config import build_site_config_dict


def _write_yaml(tmp_path: pathlib.Path, content: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(path)


def test_load_config_normalizes_analysis_and_output_defaults(tmp_path: pathlib.Path) -> None:
    cfg = load_config(
        _write_yaml(
            tmp_path,
            """
            site:
              id: TEST
              name: Test
              lat: 0.0
              lon: 0.0
            model:
              dt_days: 1.0
              solver: euler
              spinup_years: 10
              enable_14C: false
              microbial_pool_per_layer: false
              aboveground_pools: []
              soil_layers:
                - name: soil
                  depth_top_m: 0.0
                  depth_bot_m: 1.0
                  permafrost_eligible: false
                  som_pools:
                    - name: active
                      tau_prior_days: 365
                      tau_prior_std: 100
              transfer_rules: []
            parameters:
              alloc: {}
            """,
        )
    )
    assert cfg.analysis_raw["metrics"]["reduced_chi2"] is True
    assert cfg.analysis_raw["plots"]["model_results"] is True
    assert cfg.output_raw["matrices_npz"] is True
    assert cfg.output_raw["artifact_dir"] == "results/{config_stem}"


def test_locate_site_finds_harvard_forest() -> None:
    table = locate_site(flux_tower="US-Ha1", max_distance_km=5.0)
    assert not table.empty
    assert "Harvard Forest" in set(table["site_name"])


def test_build_site_config_dict_for_existing_site() -> None:
    stem, cfg = build_site_config_dict(selector="harvard_forest")
    assert stem == "harvard_forest"
    assert cfg["datasource"]["israd_name"] == "Harvard Forest"
    assert cfg["analysis"]["metrics"]["gain_matrix"] is True
    assert cfg["output"]["figure_png"] is True


def test_compute_information_metrics_from_arrays() -> None:
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    Se_diag = np.array([1.0, 1.0])
    Sx = np.array([[0.5, 0.0], [0.0, 0.5]])
    averaging_kernel = np.array([[0.5, 0.0], [0.0, 0.5]])
    y_obs = np.array([1.0, 2.0])
    y_prior = np.array([0.0, 0.0])
    y_opt = np.array([0.5, 1.5])
    k_tilde = np.array([[1.0, 0.0], [0.0, 1.0]])
    rows_by_family = {"C_stocks": [0], "resp_14C": [1]}

    metrics = compute_information_metrics(
        K=K,
        Se_diag=Se_diag,
        Sx=Sx,
        averaging_kernel=averaging_kernel,
        y_obs=y_obs,
        y_prior=y_prior,
        y_opt=y_opt,
        cost_final=2.0,
        rows_by_family=rows_by_family,
        k_tilde=k_tilde,
    )
    assert metrics["n_obs"] == 2
    assert metrics["n_params"] == 2
    assert metrics["reduced_chi2"] == 2.0
    assert metrics["dfs_total"] == 1.0
    assert len(metrics["constraint_ladder"]) > 0
    assert len(metrics["ablation"]) == 5


def test_load_exported_analysis_round_trip(tmp_path: pathlib.Path) -> None:
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    np.savez(
        export_dir / "fit_matrices.npz",
        K=np.array([[1.0, 0.0], [0.0, 1.0]]),
        Se_diag=np.array([1.0, 1.0]),
        Sx=np.array([[0.5, 0.0], [0.0, 0.5]]),
        averaging_kernel=np.array([[0.5, 0.0], [0.0, 0.5]]),
        gain_matrix=np.array([[0.5, 0.0], [0.0, 0.5]]),
        y_obs=np.array([1.0, 2.0]),
        y_prior=np.array([0.0, 0.0]),
        y_opt=np.array([0.5, 1.5]),
        sa_diag=np.array([1.0, 1.0]),
        k_tilde=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    summary = {
        "metrics": {"cost_final": 2.0},
        "rows_by_family": {"C_stocks": [0], "resp_14C": [1]},
    }
    (export_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    loaded = load_exported_analysis(str(export_dir))
    assert loaded["metrics"]["reduced_chi2"] == 2.0
    assert loaded["summary"]["rows_by_family"]["C_stocks"] == [0]

