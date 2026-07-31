"""Compare bulk+respired and fraction-only radiocarbon information by site."""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

_NB_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_NB_ROOT)
for _p in (os.path.join(_REPO_ROOT, "src"), _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ecosystem_complexity._oe_helpers import build_oe_prior_sigma
from ecosystem_complexity.state import make_default_params
from ecosystem_complexity.sites import (
    OPT_FIELDS,
    run_site_canonical,
)
from ecosystem_complexity.sites.spec import load_site_spec


def _discover_all_specs() -> dict[str, object]:
    specs: dict[str, object] = {}
    for subdir in ("configs/multisite", "configs/expansion"):
        for path in sorted(os.listdir(os.path.join(_REPO_ROOT, subdir))):
            if not path.endswith(".yaml"):
                continue
            spec = load_site_spec(os.path.join(_REPO_ROOT, subdir, path))
            specs[spec.israd_name] = spec
    return specs


# Config specs keyed by ISRaD site_name.
_SPEC_BY_ISRAD = _discover_all_specs()

OUT = os.path.join(_NB_ROOT, "exports", "israd_14c_pathway_information.csv")
OUT_PAIRED = os.path.join(_NB_ROOT, "exports", "israd_14c_pathway_paired_comparison.csv")
OUT_BIOME = os.path.join(_NB_ROOT, "exports", "israd_14c_pathway_biome_summary.csv")
DEFAULT_SITE_SUMMARY = os.path.join(
    _NB_ROOT, "exports", "network_inversion_fluxcom_er_20260719", "site_summary.csv"
)


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
        "biome": result["spec"].biome,
        "pathway": result["observation_path"],
        "converged": result["converged"],
        "n_obs": int(np.array(oe.y_obs).shape[0]),
        "n_14c_blocks": (
            result["n_pool_blocks"]
            + result["n_resp"]
            + result["n_incubation_14c"]
        ),
        "dfs_total": float(np.trace(A)),
        "dfs_tau": float(diag[tau].sum()),
        "ur_tau_mean": float(ur[tau].mean()),
        "ur_tau_active": float(ur[tau[0]]),
        "ur_tau_slow": float(ur[tau[1]]),
        "ur_tau_passive": float(ur[tau[2]]),
        "J_final": result["cost_final"],
    }


def write_summaries(
    df: pd.DataFrame,
    paired_out: str = OUT_PAIRED,
    biome_out: str = OUT_BIOME,
) -> None:
    metrics = ["dfs_total", "dfs_tau", "ur_tau_mean"]
    paired_sites = (
        df[df["pathway"].isin(["fraction", "bulk_resp"])]
        .groupby("site")["pathway"]
        .nunique()
    )
    paired = df[df["site"].isin(paired_sites[paired_sites == 2].index)].pivot(
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
    paired.to_csv(paired_out, index=False)

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
    biome.to_csv(biome_out, index=False)


def _pathways_for_site(site: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        return list(explicit)
    # Try every pathway for every site; run_site_canonical will skip invalid ones.
    return ["bulk_resp", "fraction", "combined"]


def _run_one(
    site: str,
    pathway: str,
    include_er_constraint: bool,
    include_incubation_14c_constraint: bool,
) -> dict:
    spec = _SPEC_BY_ISRAD.get(site)
    if spec is None:
        return {"status": "missing", "site": site, "pathway": pathway}
    try:
        result = run_site_canonical(
            spec,
            observation_path=pathway,
            include_er_constraint=include_er_constraint,
            include_incubation_14c_constraint=include_incubation_14c_constraint,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "site": site,
            "pathway": pathway,
            "error": str(exc),
        }
    if result.get("skipped"):
        return {"status": "skipped", "site": site, "pathway": pathway}
    return {"status": "ok", "row": summarize(result)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument(
        "--pathways", nargs="*",
        choices=("bulk_resp", "fraction", "combined"), default=None,
    )
    parser.add_argument(
        "--site-summary",
        default=DEFAULT_SITE_SUMMARY,
        help="Latest network site_summary.csv used to define the default all-site universe.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of worker processes for site-pathway runs.",
    )
    parser.add_argument(
        "--no-er-constraint",
        action="store_false",
        dest="include_er_constraint",
        help="Disable annual ER constraints. Default uses the latest ER-backed setup.",
    )
    parser.add_argument(
        "--no-incubation-14c-constraint",
        action="store_false",
        dest="include_incubation_14c_constraint",
        help="Disable dated ISRaD incubation-Δ14C constraints (enabled by default).",
    )
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--out", default=OUT, help="Path for the pathway-information CSV.")
    parser.add_argument("--paired-out", default=None, help="Path for the paired-pathway summary CSV.")
    parser.add_argument("--biome-out", default=None, help="Path for the biome pathway-summary CSV.")
    parser.add_argument(
        "--reference-pathway-information",
        default=None,
        help=(
            "Reuse the site/pathway plan from an existing pathway-information CSV. "
            "This makes it possible to regenerate a legacy comparison under the current model."
        ),
    )
    parser.add_argument(
        "--add-combined-sites",
        nargs="*",
        default=[],
        help="Additional sites to run with the combined pathway alongside a reference plan.",
    )
    parser.set_defaults(include_er_constraint=True, include_incubation_14c_constraint=True)
    args = parser.parse_args(argv)

    if args.summarize_only:
        paired_out = args.paired_out or OUT_PAIRED
        biome_out = args.biome_out or OUT_BIOME
        write_summaries(pd.read_csv(args.out), paired_out, biome_out)
        print(f"Saved {os.path.relpath(paired_out, _REPO_ROOT)}")
        print(f"Saved {os.path.relpath(biome_out, _REPO_ROOT)}")
        return 0

    if args.reference_pathway_information:
        reference = pd.read_csv(args.reference_pathway_information)
        required_plan_cols = {"site", "pathway"}
        missing_plan_cols = required_plan_cols - set(reference.columns)
        if missing_plan_cols:
            raise ValueError(
                "reference pathway information is missing columns: "
                f"{sorted(missing_plan_cols)}"
            )
        jobs = list(
            reference[["site", "pathway"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        jobs.extend((site, "combined") for site in args.add_combined_sites)
        jobs = list(dict.fromkeys(jobs))
    elif args.sites:
        selected = args.sites
        jobs = [(site, path) for site in selected for path in _pathways_for_site(site, args.pathways)]
    elif os.path.isfile(args.site_summary):
        selected = sorted(pd.read_csv(args.site_summary)["site"].dropna().unique().tolist())
        jobs = [(site, path) for site in selected for path in _pathways_for_site(site, args.pathways)]
    else:
        selected = sorted(_SPEC_BY_ISRAD)
        jobs = [(site, path) for site in selected for path in _pathways_for_site(site, args.pathways)]
    rows = []
    failures = []
    max_workers = max(1, min(args.workers, len(jobs)))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(
                _run_one,
                site,
                path,
                args.include_er_constraint,
                args.include_incubation_14c_constraint,
            ): (site, path)
            for site, path in jobs
        }
        for i, fut in enumerate(as_completed(future_map), start=1):
            site, path = future_map[fut]
            payload = fut.result()
            status = payload["status"]
            if status == "ok":
                rows.append(payload["row"])
                print(f"[{i}/{len(future_map)}] {site} [{path}] :: kept", flush=True)
            elif status == "skipped":
                print(f"[{i}/{len(future_map)}] {site} [{path}] :: skipped", flush=True)
            elif status == "missing":
                print(f"[{i}/{len(future_map)}] {site} [{path}] :: missing spec", flush=True)
            else:
                failures.append(payload)
                print(f"[{i}/{len(future_map)}] {site} [{path}] :: ERROR {payload['error']}", flush=True)

    df = pd.DataFrame(rows).sort_values(["biome", "site", "pathway"])
    df.to_csv(args.out, index=False)
    paired_out = args.paired_out or OUT_PAIRED
    biome_out = args.biome_out or OUT_BIOME
    write_summaries(df, paired_out, biome_out)
    print(f"Saved {os.path.relpath(args.out, _REPO_ROOT)}")
    print(df.to_string(index=False))
    if failures:
        print(f"{len(failures)} pathway runs failed.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
