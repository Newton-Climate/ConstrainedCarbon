"""
export_oe_constraint_ladder.py — cumulative OE constraint ladders across sites,
with and without the ¹²C-stock rung, plus ¹²C-vs-Δ¹⁴C orthogonality and an
order-independent Shapley attribution of DFS to each observation family.

For each site it runs the canonical inversion in the ``combined`` observation
path (so every constraint family is present: ¹²C stocks, bulk-layer Δ¹⁴C,
density-fraction Δ¹⁴C, respired-CO₂ Δ¹⁴C), linearises once at the MAP, and from
that single Jacobian computes four views:

  1. ``with_12C_stocks``  — ¹²C stocks → +bulk → +fraction → +respired
  2. ``no_12C_stocks``    — bulk → +fraction → +respired (stocks withheld)
  3. orthogonality        — how much of the ¹²C-stock constraint is information
                            the Δ¹⁴C data does not already carry
  4. Shapley attribution  — each family's average marginal DFS over all orderings

Ladders (1) and (2) are order-dependent by construction: whichever family goes
first collects credit for information the later ones would have supplied anyway.
View (4) is the order-independent answer, and (3) is the sharper pairwise test —
what do stocks add *last*, once every Δ¹⁴C constraint is already in hand?

Sites are independent (one inversion + one Jacobian each), so they are farmed out
to a process pool. Each site peaks around 2.3 GB resident, which — not core count
— is what caps the worker count on a 16 GB machine; see ``_DEFAULT_WORKERS``.

Writes:
    notebooks/exports/oe_constraint_ladder.csv        (both ladders, long form)
    notebooks/exports/oe_stocks_orthogonality.csv     (one row per site)
    notebooks/exports/oe_family_shapley.csv           (one row per site × family)

Run from the repository root:
    python notebooks/export_oe_constraint_ladder.py                    # all sites
    python notebooks/export_oe_constraint_ladder.py --workers 6
    python notebooks/export_oe_constraint_ladder.py harvard_forest solling
    python notebooks/export_oe_constraint_ladder.py --workers 1        # serial
"""
from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

_NB_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_NB_ROOT)
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_REPO_ROOT)

# Each worker otherwise grabs every core for its own XLA/BLAS pool and the pool
# members thrash each other. A single site is ~1.1x threaded (measured), so
# pinning workers to one thread costs almost nothing and process-level fan-out
# scales nearly linearly instead. This must run before the transitive `import jax`
# below, which is why it keys off an env var the parent sets before spawning
# rather than a function argument — spawned children inherit os.environ, but a
# ProcessPoolExecutor `initializer` would run only after this module is imported.
_WORKER_ENV = "_OE_LADDER_WORKER"
if os.environ.get(_WORKER_ENV) == "1":
    for _var in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_var, "1")
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    )

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

from ecosystem_complexity.oe_diagnostics import (  # noqa: E402
    LADDER_STEPS,
    LADDER_STEPS_NO_STOCKS,
    constraint_orthogonality_from_context,
    cumulative_ladder_from_context,
    oe_ladder_context,
    shapley_dfs_attribution_from_context,
)
from sites.multisite_canonical import (  # noqa: E402
    OPT_FIELDS,
    select_specs,
    run_site_canonical,
)

OUT_LADDER = os.path.join(_NB_ROOT, "exports", "oe_constraint_ladder.csv")
OUT_ORTHO = os.path.join(_NB_ROOT, "exports", "oe_stocks_orthogonality.csv")
OUT_SHAPLEY = os.path.join(_NB_ROOT, "exports", "oe_family_shapley.csv")

_LADDERS = (
    ("with_12C_stocks", LADDER_STEPS),
    ("no_12C_stocks", LADDER_STEPS_NO_STOCKS),
)

# Peak RSS is ~2.3 GB per site, so on a 16 GB box the memory budget binds well
# before the 8 cores do: 4 workers ≈ 9 GB and leaves headroom for the OS.
_DEFAULT_WORKERS = 4


def _run_site(stem: str) -> dict:
    """Run one site end-to-end. Returns only picklable plain-Python rows.

    Runs in a worker process, so nothing here may return JAX/model objects, and
    every exception must be caught and reported rather than killing the pool.
    Site stdout is captured and returned so the parent can print each site's log
    in one contiguous block instead of interleaving it with other workers'.
    """
    buf = io.StringIO()
    out: dict = {
        "stem": stem, "status": "ok", "error": None,
        "ladder_rows": [], "ortho_row": None, "shapley_rows": [],
    }
    try:
        with contextlib.redirect_stdout(buf):
            spec = select_specs([stem])[0]
            data = run_site_canonical(spec, observation_path="combined")
            if data.get("skipped"):
                out["status"] = "skipped (insufficient obs)"
            elif not data.get("converged"):
                out["status"] = "non-converged"
            else:
                ctx = oe_ladder_context(
                    data["model"], data["forcing"], data["state0"],
                    data["params_opt"], data["obs_full"],
                    opt_fields=OPT_FIELDS,
                    extra_obs_blocks=data["pool_blocks"],
                )
                site_meta = {
                    "site": spec.label,
                    "site_id": spec.tower_id,
                    "biome": spec.biome,
                    "mean_GPP_gCm2yr": round(data["mean_gpp_gCm2yr"], 1),
                }
                for ladder_name, steps in _LADDERS:
                    ladder = cumulative_ladder_from_context(ctx, steps=steps)
                    for rung in ladder:
                        out["ladder_rows"].append({
                            **site_meta,
                            "ladder": ladder_name,
                            "rung": rung["rung"],
                            "constraint_label": rung["label"],
                            "added_family": rung["added_family"],
                            "n_obs_added_family": rung["n_obs_added_family"],
                            "n_obs_cumulative": rung["n_obs_cumulative"],
                            "dfs_cumulative": round(rung["dfs_cumulative"], 4),
                            "dfs_increment": round(rung["dfs_increment"], 4),
                        })
                    print(f"  {ladder_name:16s} DFS: " + " → ".join(
                        f"{r['dfs_cumulative']:.2f}" for r in ladder))

                ortho = constraint_orthogonality_from_context(ctx)
                angles = np.atleast_1d(ortho["principal_angles_deg"])
                out["ortho_row"] = {
                    **site_meta,
                    "n_obs_12C": ortho["n_obs_a"],
                    "n_obs_14C": ortho["n_obs_b"],
                    "dfs_12C_alone": round(ortho["dfs_a_alone"], 4),
                    "dfs_14C_alone": round(ortho["dfs_b_alone"], 4),
                    "dfs_joint": round(ortho["dfs_joint"], 4),
                    "redundancy": round(ortho["redundancy"], 4),
                    "unique_12C_added_last": round(ortho["unique_a"], 4),
                    "unique_14C_added_last": round(ortho["unique_b"], 4),
                    "uniqueness_12C": round(ortho["uniqueness_a"], 4),
                    "uniqueness_14C": round(ortho["uniqueness_b"], 4),
                    "min_principal_angle_deg": round(
                        ortho["min_principal_angle_deg"], 2),
                    "mean_principal_angle_deg": round(
                        ortho["mean_principal_angle_deg"], 2),
                    "principal_angles_deg": ";".join(f"{a:.1f}" for a in angles),
                }
                print(f"  orthogonality: DFS 12C={ortho['dfs_a_alone']:.2f} "
                      f"14C={ortho['dfs_b_alone']:.2f} "
                      f"joint={ortho['dfs_joint']:.2f}  "
                      f"redundancy={ortho['redundancy']:.2f}  "
                      f"unique_12C={ortho['unique_a']:.2f}")

                for att in shapley_dfs_attribution_from_context(ctx):
                    out["shapley_rows"].append({
                        **site_meta,
                        "family": att["family"],
                        "n_obs": att["n_obs"],
                        "present": att["n_obs"] > 0,
                        "shapley_dfs": round(att["shapley_dfs"], 4),
                        "shapley_share": round(att["shapley_share"], 4),
                        "dfs_standalone": round(att["dfs_standalone"], 4),
                        "dfs_unique_last": round(att["dfs_unique_last"], 4),
                        "dfs_joint_total": round(att["dfs_joint_total"], 4),
                    })
                print("  shapley: " + "  ".join(
                    f"{r['family']}={r['shapley_dfs']:.2f}"
                    for r in out["shapley_rows"]))
    except Exception as exc:  # noqa: BLE001 — one bad site must not kill the pool
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["log"] = buf.getvalue()
    return out


def _collect(results: list[dict]) -> None:
    """Sort, write, and summarise the per-site results."""
    ladder_rows = [r for res in results for r in res["ladder_rows"]]
    ortho_rows = [res["ortho_row"] for res in results if res["ortho_row"]]
    shapley_rows = [r for res in results for r in res["shapley_rows"]]

    ladder_df = pd.DataFrame(ladder_rows)
    if not ladder_df.empty:
        ladder_df = ladder_df.sort_values(["site", "ladder", "rung"])
    ortho_df = pd.DataFrame(ortho_rows)
    if not ortho_df.empty:
        ortho_df = ortho_df.sort_values("site")
    shapley_df = pd.DataFrame(shapley_rows)
    if not shapley_df.empty:
        shapley_df = shapley_df.sort_values(["site", "family"])

    ladder_df.to_csv(OUT_LADDER, index=False)
    ortho_df.to_csv(OUT_ORTHO, index=False)
    shapley_df.to_csv(OUT_SHAPLEY, index=False)

    n_sites = ladder_df["site"].nunique() if not ladder_df.empty else 0
    print(f"\nSaved ladders: {os.path.relpath(OUT_LADDER, _REPO_ROOT)}  "
          f"({len(ladder_df)} rows, {n_sites} sites)")
    print(f"Saved orthogonality: {os.path.relpath(OUT_ORTHO, _REPO_ROOT)}  "
          f"({len(ortho_df)} rows)")
    print(f"Saved Shapley attribution: {os.path.relpath(OUT_SHAPLEY, _REPO_ROOT)}  "
          f"({len(shapley_df)} rows)")

    bad = [r for r in results if r["status"] != "ok"]
    if bad:
        print(f"\n{len(bad)} site(s) produced no ladder:")
        for r in bad:
            detail = r["error"] or r["status"]
            print(f"  {r['stem']:22s} {detail}")


def main(names: list[str] | None = None, workers: int = _DEFAULT_WORKERS) -> None:
    specs = select_specs(names)
    stems = [s.config_stem for s in specs]
    workers = max(1, min(workers, len(stems)))
    print(f"Running {len(stems)} site(s) on {workers} worker(s)\n")

    results: list[dict] = []
    if workers == 1:
        for i, stem in enumerate(stems, 1):
            res = _run_site(stem)
            results.append(res)
            print(f"[{i}/{len(stems)}] {stem} — {res['status']}")
            print(res["log"], end="")
    else:
        # Children inherit this and pin themselves to one compute thread.
        os.environ[_WORKER_ENV] = "1"
        try:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=mp.get_context("spawn")
            ) as pool:
                futures = {pool.submit(_run_site, s): s for s in stems}
                for i, fut in enumerate(as_completed(futures), 1):
                    res = fut.result()
                    results.append(res)
                    print(f"[{i}/{len(stems)}] {res['stem']} — {res['status']}")
                    print(res["log"], end="")
        finally:
            os.environ.pop(_WORKER_ENV, None)

    _collect(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sites", nargs="*", help="config stems (default: all)")
    parser.add_argument(
        "--workers", type=int, default=_DEFAULT_WORKERS,
        help=f"parallel site workers (default {_DEFAULT_WORKERS}; "
             "~2.3 GB resident each)",
    )
    args = parser.parse_args()
    main(args.sites or None, workers=args.workers)
