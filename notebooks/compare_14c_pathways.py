"""Compare bulk+respired and fraction-only radiocarbon information by site."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_NB_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_NB_ROOT)
for _p in (os.path.join(_REPO_ROOT, "src"), _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ecosystem_complexity._oe_helpers import build_oe_prior_sigma
from ecosystem_complexity.state import make_default_params
from sites.multisite_canonical import (
    OPT_FIELDS,
    discover_site_specs,
    run_site_canonical,
)

# Config specs keyed by ISRaD site_name (the driver keys configs by file stem).
_SPEC_BY_ISRAD = {s.israd_name: s for s in discover_site_specs().values()}

PAIRED_SITES = ("Howland Forest", "Harvard Forest", "Solling", "Appi forest", "EML")
BULK_ONLY_SITES = (
    "FLONA", "ZF2", "CZ_Old_Black_Spruce", "CZ_1964burn_NSA",
    "Auchencorth Moss", "Baram Basin",
)
# New fraction-only ISRaD sites (no respired Δ¹⁴C → no bulk_resp pathway); we
# compare their fraction pathway against the richer combined pathway.
NEW_SITES = ("MI-Coarse UMBS", "MO Ozark BREA", "NH Bartlett Forest")
BIOMES = {
    "Howland Forest": "temperate evergreen forest",
    "Harvard Forest": "temperate deciduous forest",
    "Solling": "temperate forest",
    "Appi forest": "cool-temperate forest",
    "EML": "tundra/permafrost",
    "FLONA": "tropical moist forest",
    "ZF2": "tropical moist forest",
    "CZ_Old_Black_Spruce": "boreal forest",
    "CZ_1964burn_NSA": "boreal wet forest/fire",
    "Auchencorth Moss": "temperate peatland",
    "Baram Basin": "tropical peat forest",
    "MI-Coarse UMBS": "temperate mixed forest",
    "MO Ozark BREA": "temperate deciduous forest",
    "NH Bartlett Forest": "temperate mixed forest",
}

OUT = os.path.join(_NB_ROOT, "exports", "israd_14c_pathway_information.csv")
OUT_PAIRED = os.path.join(_NB_ROOT, "exports", "israd_14c_pathway_paired_comparison.csv")
OUT_BIOME = os.path.join(_NB_ROOT, "exports", "israd_14c_pathway_biome_summary.csv")


def summarize(result: dict) -> dict:
    oe = result["oe_result"]
    names = list(oe.state_names)
    A = np.array(oe.averaging_kernel, dtype=float)
    prior = np.array(build_oe_prior_sigma(
        result["model"].config,
        make_default_params(result["model"].config),
        tuple(OPT_FIELDS),
    ), dtype=float)
    post = np.sqrt(np.clip(np.diag(np.array(oe.Sx, dtype=float)), 0.0, None))
    ur = 1.0 - post / prior
    diag = np.diag(A)
    tau = [i for i, name in enumerate(names) if name.startswith("log_tau[")]
    return {
        "site": result["spec"].israd_name,
        "biome": BIOMES[result["spec"].israd_name],
        "pathway": result["observation_path"],
        "converged": result["converged"],
        "n_obs": int(np.array(oe.y_obs).shape[0]),
        "n_14c_blocks": result["n_pool_blocks"] + result["n_resp"],
        "dfs_total": float(np.trace(A)),
        "dfs_tau": float(diag[tau].sum()),
        "ur_tau_mean": float(ur[tau].mean()),
        "ur_tau_active": float(ur[tau[0]]),
        "ur_tau_slow": float(ur[tau[1]]),
        "ur_tau_passive": float(ur[tau[2]]),
        "J_final": result["cost_final"],
    }


def write_summaries(df: pd.DataFrame) -> None:
    metrics = ["dfs_total", "dfs_tau", "ur_tau_mean"]
    paired = df[df["site"].isin(PAIRED_SITES)].pivot(
        index=["site", "biome"], columns="pathway", values=metrics,
    )
    paired.columns = [f"{metric}_{path}" for metric, path in paired.columns]
    paired = paired.reset_index()
    paired["dfs_total_ratio_fraction_to_bulk_resp"] = (
        paired["dfs_total_fraction"] / paired["dfs_total_bulk_resp"]
    )
    paired["dfs_tau_ratio_fraction_to_bulk_resp"] = (
        paired["dfs_tau_fraction"] / paired["dfs_tau_bulk_resp"]
    )
    paired["dfs_tau_difference_fraction_minus_bulk_resp"] = (
        paired["dfs_tau_fraction"] - paired["dfs_tau_bulk_resp"]
    )
    paired["ur_tau_difference_fraction_minus_bulk_resp"] = (
        paired["ur_tau_mean_fraction"] - paired["ur_tau_mean_bulk_resp"]
    )
    paired["equivalent_tau_dfs_within_10pct"] = paired[
        "dfs_tau_ratio_fraction_to_bulk_resp"
    ].between(0.9, 1.1)
    paired.to_csv(OUT_PAIRED, index=False)

    biome = (
        df.groupby(["biome", "pathway"], as_index=False)
        .agg(
            n_sites=("site", "nunique"),
            dfs_total_mean=("dfs_total", "mean"),
            dfs_tau_mean=("dfs_tau", "mean"),
            ur_tau_mean=("ur_tau_mean", "mean"),
            ur_tau_active_mean=("ur_tau_active", "mean"),
            ur_tau_slow_mean=("ur_tau_slow", "mean"),
            ur_tau_passive_mean=("ur_tau_passive", "mean"),
        )
    )
    biome.to_csv(OUT_BIOME, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument(
        "--pathways", nargs="*",
        choices=("bulk_resp", "fraction", "combined"), default=None,
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)

    if args.summarize_only:
        write_summaries(pd.read_csv(OUT))
        print(f"Saved {os.path.relpath(OUT_PAIRED, _REPO_ROOT)}")
        print(f"Saved {os.path.relpath(OUT_BIOME, _REPO_ROOT)}")
        return 0

    selected = args.sites or list(PAIRED_SITES + BULK_ONLY_SITES + NEW_SITES)
    rows = []
    for site in selected:
        if args.pathways:
            paths = list(args.pathways)
        elif site in PAIRED_SITES:
            paths = ["bulk_resp", "fraction", "combined"]
        elif site in NEW_SITES:
            paths = ["fraction", "combined"]
        else:  # BULK_ONLY_SITES
            paths = ["bulk_resp", "combined"]
        spec = _SPEC_BY_ISRAD.get(site)
        if spec is None:
            print(f"  SKIP {site!r}: no config in configs/multisite/")
            continue
        for path in paths:
            try:
                result = run_site_canonical(spec, observation_path=path)
            except Exception as exc:  # noqa: BLE001 — keep going across sites
                print(f"  ERROR {site} [{path}]: {exc}")
                continue
            if not result.get("skipped"):
                rows.append(summarize(result))

    df = pd.DataFrame(rows).sort_values(["biome", "site", "pathway"])
    df.to_csv(OUT, index=False)
    write_summaries(df)
    print(f"Saved {os.path.relpath(OUT, _REPO_ROOT)}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
