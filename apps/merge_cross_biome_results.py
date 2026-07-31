#!/usr/bin/env python3
"""Append a site-set run to the canonical network, warming, and transit inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _append(base: Path, addition: Path) -> pd.DataFrame:
    out = pd.concat([pd.read_csv(base), pd.read_csv(addition)], ignore_index=True, sort=False)
    return out.drop_duplicates("config", keep="last")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-addition", required=True)
    parser.add_argument("--warming-addition", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--base-network")
    parser.add_argument("--base-warming")
    parser.add_argument("--base-transit")
    parser.add_argument("--sites", nargs="+", help="optional ISRaD site names to retain from the additions")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_network = Path(args.base_network) if args.base_network else ROOT / "notebooks/exports/network_inversion_fluxcom_er_20260719/site_summary.csv"
    base_warming = Path(args.base_warming) if args.base_warming else ROOT / "notebooks/exports/warming_vulnerability_fluxcom_er_20260719/site_warming_summary.csv"
    base_transit = Path(args.base_transit) if args.base_transit else ROOT / "notebooks/exports/optimized_ecosystem_transit_times_20260730.csv"
    network_addition = pd.read_csv(args.network_addition)
    warming_addition = pd.read_csv(args.warming_addition)
    if args.sites:
        network_addition = network_addition[network_addition["site"].isin(args.sites)]
        warming_addition = warming_addition[warming_addition["site"].isin(args.sites)]
    network = _append(base_network, Path(args.network_addition)) if not args.sites else pd.concat([pd.read_csv(base_network), network_addition], ignore_index=True, sort=False).drop_duplicates("config", keep="last")
    warming = _append(base_warming, Path(args.warming_addition)) if not args.sites else pd.concat([pd.read_csv(base_warming), warming_addition], ignore_index=True, sort=False).drop_duplicates("config", keep="last")
    transit = pd.read_csv(base_transit)
    additions = network_addition.copy()
    additions = additions.rename(columns={
        "tau_soil_active": "tau_active_yr",
        "tau_soil_slow": "tau_slow_yr",
        "tau_soil_passive": "tau_passive_yr",
    })
    additions["status"] = "ok"
    additions["inversion_scope"] = "direct warming"
    additions["transfer_source"] = "configured topology; MAP tau"
    keep = list(transit.columns)
    transit = pd.concat([transit, additions.reindex(columns=keep)], ignore_index=True, sort=False)
    transit = transit.drop_duplicates("config", keep="last")

    network.to_csv(outdir / "site_summary.csv", index=False)
    warming.to_csv(outdir / "site_warming_summary.csv", index=False)
    transit.to_csv(outdir / "optimized_transit_input.csv", index=False)
    print(f"network={len(network)} warming={len(warming)} transit_input={len(transit)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
