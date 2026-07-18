"""
export_multisite_constraint_ladder.py — OE constraint ladder for the multisite
canonical inversions (every configs/multisite/*.yaml except Harvard & Barrow,
which are already covered by export_four_site_information_tables.py).

For each converged site it runs the canonical inversion, then computes the
one-constraint-at-a-time OE ladder at the MAP estimate: the standalone DFS
(degrees of freedom for signal) that each observation block would contribute
on its own, given the prior. Writes:

    notebooks/exports/multisite_constraint_ladder.csv

Run from the repository root:
    python notebooks/export_multisite_constraint_ladder.py
    python notebooks/export_multisite_constraint_ladder.py flona zf2
"""
from __future__ import annotations

import os
import sys

import pandas as pd

_NB_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_NB_ROOT)
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_REPO_ROOT)

from ecosystem_complexity.oe_diagnostics import oe_constraint_ladder
from ecosystem_complexity.sites import (
    OPT_FIELDS,
    discover_site_specs,
    run_site_canonical,
)

# Skip the two sites already diagnosed in the four-site tables.
SKIP_STEMS = {"harvard_forest"}

OUT = os.path.join(_NB_ROOT, "exports", "multisite_constraint_ladder.csv")


def main(names: list[str] | None = None) -> None:
    specs = discover_site_specs()
    if names:
        wanted = {n.lower() for n in names}
        specs = {k: v for k, v in specs.items() if k.lower() in wanted}

    rows: list[dict] = []
    for stem, spec in sorted(specs.items()):
        if stem in SKIP_STEMS:
            print(f"\n══ SKIP {spec.label} (already in four-site tables)")
            continue
        try:
            data = run_site_canonical(spec, observation_path=spec.observation_path)
        except Exception as exc:  # noqa: BLE001 — keep going across sites
            print(f"  ERROR at {spec.label}: {exc}")
            continue
        if data.get("skipped"):
            print(f"  SKIP {spec.label} — insufficient obs.")
            continue
        if not data.get("converged"):
            print(f"  NON-CONVERGED {spec.label} — ladder skipped.")
            continue

        try:
            ladder = oe_constraint_ladder(
                data["model"], data["forcing"], data["state0"],
                data["params_opt"], data["obs_full"],
                opt_fields=OPT_FIELDS,
                extra_obs_blocks=data["pool_blocks"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  LADDER ERROR at {spec.label}: {exc}")
            continue

        ranked = sorted(ladder, key=lambda r: (-float(r["dfs"]), r["label"]))
        dfs_sum = sum(float(r["dfs"]) for r in ranked)
        for rank, rung in enumerate(ranked, start=1):
            rows.append({
                "site": spec.label,
                "site_id": spec.tower_id,
                "biome": spec.biome,
                "mean_GPP_gCm2yr": round(data["mean_gpp_gCm2yr"], 1),
                "rank_within_site": rank,
                "constraint_label": rung["label"],
                "obs_type": rung["obs_type"],
                "n_obs": int(rung["n_obs"]),
                "dfs": float(rung["dfs"]),
                "dfs_share_of_site": float(rung["dfs"]) / dfs_sum if dfs_sum else float("nan"),
            })
        print(f"  ladder: {len(ranked)} constraints, Σdfs={dfs_sum:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nSaved ladder: {OUT}  ({len(df)} rows, "
          f"{df['site'].nunique() if not df.empty else 0} sites)")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
