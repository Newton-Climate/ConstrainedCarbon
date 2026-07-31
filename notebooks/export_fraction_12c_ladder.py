"""OE cumulative-constraint ladder including the ISRaD density-fraction ¹²C rung.

Reports DFS added by each family for every site that carries density fractions.
The final rung is `fraction_12C` — the new observation family this diagnostic
was written to quantify. Writes:

    notebooks/exports/multisite_ladder_with_frac12c_20260731.csv

Run:
    python notebooks/export_fraction_12c_ladder.py                # all sites
    python notebooks/export_fraction_12c_ladder.py harvard_forest # a subset
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

from ecosystem_complexity.oe_diagnostics import (
    LADDER_STEPS,
    cumulative_ladder_from_context,
    oe_ladder_context,
)
from ecosystem_complexity.sites import (
    OPT_FIELDS,
    discover_site_specs,
    run_site_canonical,
)

OUT = os.path.join(
    _NB_ROOT, "exports", "multisite_ladder_with_frac12c_20260731.csv"
)


def _run_one(stem: str, spec) -> list[dict]:
    print(f"── {spec.label} ({stem}) ──")
    res = run_site_canonical(
        spec, observation_path="combined", include_fraction_12c_constraint=True,
    )
    if res.get("skipped") or not res.get("converged"):
        print("  skipped or non-converged — no ladder row.")
        return []
    ctx = oe_ladder_context(
        res["model"], res["forcing"], res["state0"], res["params_opt"],
        res["obs_full"], OPT_FIELDS, extra_obs_blocks=res["pool_blocks"],
    )
    rungs = cumulative_ladder_from_context(ctx, steps=LADDER_STEPS)
    prev = 0.0
    rows: list[dict] = []
    for r in rungs:
        rows.append({
            "site": spec.israd_name,
            "config_stem": stem,
            "biome": spec.biome,
            "rung": r["rung"],
            "label": r["label"],
            "added_family": r["added_family"],
            "n_obs_cumulative": r["n_obs_cumulative"],
            "n_obs_added_family": r["n_obs_added_family"],
            "dfs_cumulative": round(r["dfs_cumulative"], 4),
            "dfs_increment": round(r["dfs_cumulative"] - prev, 4),
        })
        prev = r["dfs_cumulative"]
    for row in rows:
        print(
            f"  {row['label']:<70s} "
            f"n={row['n_obs_cumulative']:>2d} "
            f"DFS={row['dfs_cumulative']:>5.2f} "
            f"(+{row['dfs_increment']:>+.2f})"
        )
    return rows


def main(names: list[str] | None = None) -> None:
    specs = discover_site_specs()
    if names:
        wanted = {n.lower() for n in names}
        specs = {k: v for k, v in specs.items() if k.lower() in wanted}

    rows: list[dict] = []
    for stem, spec in specs.items():
        try:
            rows.extend(_run_one(stem, spec))
        except Exception as exc:  # noqa: BLE001 — one bad site must not stop the rest
            print(f"  ERROR at {spec.label}: {exc}")
    if not rows:
        print("No ladder rows produced.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(df)} rows, {df['site'].nunique()} sites)")

    # Sites where the 12C rung actually fired.
    fired = df[(df["added_family"] == "fraction_12C") & (df["n_obs_added_family"] > 0)]
    if not fired.empty:
        summary = fired[[
            "site", "biome", "n_obs_added_family", "dfs_cumulative", "dfs_increment",
        ]].rename(columns={
            "n_obs_added_family": "n_frac12c_obs",
            "dfs_cumulative": "dfs_total",
            "dfs_increment": "dfs_from_frac12c",
        })
        print("\n── ΔDFS contributed by fraction_12C ──")
        print(summary.sort_values("dfs_from_frac12c", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
