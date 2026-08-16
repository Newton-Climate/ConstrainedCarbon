#!/usr/bin/env python3
"""Scale the realised-transit diagnostic to all optimized site ecosystems.

Uses retained MAP turnover times and the YAML transfer topology.  It therefore
provides a reproducible central estimate; posterior intervals require the OE
covariance and are calculated separately for the four-site pilot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]

from ecosystem_complexity.transit_time.realized_gradient import transit_metrics
from ecosystem_complexity.api import build_model
from ecosystem_complexity.sites.forcing import load_site_forcing, resolve_forcing_file
from ecosystem_complexity.sites.spec import load_site_spec
from ecosystem_complexity.state import make_default_params

EXPANSION_CONFIGS = {
    "La Campana": "configs/expansion/la_campana.yaml", "Biadaski": "configs/expansion/biadaski.yaml",
    "Treynor": "configs/expansion/treynor.yaml", "Dinesen": "configs/expansion/dinesen.yaml",
    "Nelson Farm": "configs/expansion/nelson_farm.yaml", "CZ_1981burn": "configs/expansion/cz_1981burn.yaml",
    "Trumbore Musick": "configs/expansion/trumbore_musick.yaml", "Trumbore Falbrook": "configs/expansion/trumbore_falbrook.yaml",
    "Trumbore Ahwahnee": "configs/expansion/trumbore_ahwahnee.yaml", "CZ_1930burn": "configs/expansion/cz_1930burn.yaml",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ROOT / "notebooks/exports/optimized_ecosystem_transit_times_20260730.csv"))
    parser.add_argument("--out", default=str(ROOT / "notebooks/exports/realized_transit_all_sites_20260731.csv"))
    parser.add_argument("--figure", default=str(ROOT / "notebooks/paper_figs/outputs/current_results/figures/figure_13_realized_transit_all_sites.png"))
    args = parser.parse_args()
    source, out, figure = Path(args.source), Path(args.out), Path(args.figure)
    df = pd.read_csv(source)
    rows, failures = [], []
    for row in df.itertuples(index=False):
        config = row.config
        try:
            spec = load_site_spec(config)
            model = build_model(config)
            forcing = load_site_forcing(spec, resolve_forcing_file(spec), model)
            params = make_default_params(model.config)
            tau = np.array([row.tau_active_yr, row.tau_slow_yr, row.tau_passive_yr])
            params = params._replace(log_tau=np.log(tau * 365.25))
            intrinsic, realised, modifier = transit_metrics(model, forcing, params)
            moisture = np.asarray(forcing.soil_moisture, dtype=float)
            rows.append({
                "site": row.site, "label": row.label, "biome": row.biome,
                "biome_group": row.biome_group, "inversion_scope": row.inversion_scope,
                "config": config, "forcing_kind": spec.forcing_kind,
                "forcing_provenance": "synthetic climate + FluxCom GPP" if spec.forcing_kind == "fluxcom" else "daily tower forcing (with loader fallbacks)",
                "intrinsic_transit_time_yr": intrinsic,
                "realised_transit_time_yr": realised,
                "realised_to_intrinsic": realised / intrinsic,
                "gpp_weighted_mean_modifier": modifier,
                "soil_moisture_sd": float(np.std(moisture)),
                "transfer_source": "configured topology; retained MAP tau",
            })
        except Exception as exc:  # noqa: BLE001
            failures.append({"site": row.site, "config": config, "error": repr(exc)})
    result = pd.DataFrame(rows).sort_values("realised_transit_time_yr")
    result.to_csv(out, index=False)
    fig, ax = plt.subplots(figsize=(9.5, 9), constrained_layout=True)
    y = np.arange(len(result))
    ax.hlines(y, result.intrinsic_transit_time_yr, result.realised_transit_time_yr, color="0.8", lw=1)
    ax.scatter(result.intrinsic_transit_time_yr, y, color="#355C7D", s=34, label="Intrinsic")
    ax.scatter(result.realised_transit_time_yr, y, color="#C06C2B", marker="s", s=34, label="Realised")
    ax.set(yticks=y, yticklabels=result.site, xlabel="Input-weighted mean transit time (years)", title="Environmental forcing reorders transit time across 34 ecosystems")
    ax.grid(axis="x", alpha=.2)
    ax.legend(frameon=False, loc="lower right")
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=250, bbox_inches="tight")
    if failures:
        pd.DataFrame(failures).to_csv(out.with_name("realized_transit_all_sites_failures_20260731.csv"), index=False)
    print(f"Wrote {len(rows)} sites to {out} and {figure}; failures={len(failures)}")
    return 0
