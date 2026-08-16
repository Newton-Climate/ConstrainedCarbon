#!/usr/bin/env python3
"""``ecosys information`` — information-content diagnostics dispatcher.

Subverbs
    shapley  per-parameter Shapley DFS + AK/gain matrices across the site set
    dfs      per-parameter degrees of freedom for signal (subset AK diagonal)
    ak       averaging-kernel diagnostics at the MAP
    gain     gain matrix G = C_post K^T Se^-1 at the MAP
    ose      observation system experiments (scenario ranking)

All subverbs share the same forward pipeline: run the canonical OE
inversion, then compute the requested diagnostic from ``oe_diagnostics``.
Subverb-specific artifacts land under
``outputs/{name}/information/{subverb}/`` with the standard contract.

The ``information.shapley.sigma_rule`` YAML block (or ``--sigma-rule
REL:ABS``) folds the old ``analyze_shapley_per_parameter_tight_c.py``
into a flag on this app — same monkey-patch on ``build_measured_soc_total``,
now applied only when the rule is set.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logger = logging.getLogger("ecosys.information")

DEFAULT_FAMILIES = (
    "C_stocks", "bulk_14C", "fraction_14C", "resp_14C", "ER_annual", "fraction_12C",
)

_STATE_SHORT = {
    "log_tau[soil_active]": "tau_active",
    "log_tau[soil_slow]": "tau_slow",
    "log_tau[soil_passive]": "tau_passive",
    "log_f_transfer[soil_active→soil_slow]": "f_a_to_s",
    "log_f_transfer[soil_slow→soil_passive]": "f_s_to_p",
}


# ---------------------------------------------------------------------------
# Optional σ_carbon rule (folded from analyze_shapley_per_parameter_tight_c)
# ---------------------------------------------------------------------------


def _install_carbon_sigma_rule(rule: str | None) -> None:
    """Apply a REL:ABS floor to measured-SOC σ via a monkey-patch.

    The default σ inside ``ecosystem_complexity.data.soc_stocks.
    build_measured_soc_total`` is ``max(profile_SD, rel * mean_SOC)``.
    This helper replaces that call so ``rel`` is the given fraction and
    the returned σ is additionally floored at ``abs`` (gC/m²) when
    positive. Called once per process, before the site drivers import
    the function.
    """
    if not rule:
        return
    try:
        rel_str, abs_str = rule.split(":")
        rel = float(rel_str)
        abs_floor = float(abs_str)
    except ValueError as exc:
        raise SystemExit(f"bad --sigma-rule {rule!r} (want REL:ABS)") from exc

    import ecosystem_complexity.data.soc_stocks as _soc
    _orig = _soc.build_measured_soc_total

    def _measured(israd_name, model, sigma_floor_frac=rel):
        res = _orig(israd_name, model, sigma_floor_frac=rel)
        if res is None:
            return None
        mean_val, sigma, depth_cov = res
        if abs_floor > 0 and sigma < abs_floor:
            sigma = abs_floor
        return mean_val, sigma, depth_cov

    _soc.build_measured_soc_total = _measured
    import ecosystem_complexity.sites.driver as _drv
    _drv.build_measured_soc_total = _measured
    print(f"[σ_carbon rule] rel_floor={rel:.3f} abs_floor={abs_floor:.0f}", flush=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _biome_group(biome: str) -> str:
    b = biome.lower()
    if any(k in b for k in ("arctic", "tundra", "permafrost")):
        return "arctic_permafrost"
    if "boreal" in b:
        return "boreal"
    if any(k in b for k in ("peatland", "moss")):
        return "peatland"
    if "tropical" in b:
        return "tropical"
    if any(k in b for k in ("grassland", "mollisol", "mediterranean")):
        return "grassland_mediterranean"
    if "temperate" in b:
        return "temperate_forest"
    return "other"


def _load_site_set(path: str) -> tuple[str, list[str], dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    name = str(payload.get("name") or Path(path).stem)
    configs = payload.get("configs")
    if not isinstance(configs, list) or not configs or not all(isinstance(p, str) for p in configs):
        raise SystemExit(f"error: {path}: expected non-empty string list at 'configs'")
    resolved = [p if os.path.isabs(p) else str(_REPO_ROOT / p) for p in configs]
    missing = [p for p in resolved if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"error: {path}: missing config(s): {', '.join(missing)}")
    return name, resolved, dict(payload.get("run_options") or {})


def _resolve_specs(selectors: list[str]) -> list[str]:
    from ecosystem_complexity.sites import discover_site_specs
    known = discover_site_specs()
    by_key = {}
    for spec in known.values():
        by_key[spec.config_stem] = spec.config_path
        by_key[spec.israd_name] = spec.config_path
        by_key[spec.label] = spec.config_path
    out: list[str] = []
    for sel in selectors:
        if sel.endswith((".yaml", ".yml")) or os.path.sep in sel:
            p = os.path.abspath(sel)
            if not os.path.isfile(p):
                raise SystemExit(f"error: no such config file: {sel}")
            out.append(p)
        elif sel in by_key:
            out.append(str(by_key[sel]))
        else:
            raise SystemExit(f"error: unknown site {sel!r}")
    return out


def _read_sigma_rule_from_yaml(config_path: str) -> str | None:
    """Return ``information.shapley.sigma_rule`` when set, else None."""
    from ecosystem_complexity.config import load_config
    block = load_config(config_path).information_raw or {}
    return (block.get("shapley") or {}).get("sigma_rule")


# ---------------------------------------------------------------------------
# Worker: shapley + AK/gain per site
# ---------------------------------------------------------------------------


def _shapley_one(
    config_path: str,
    include_er: bool,
    include_incubation: bool,
    include_incubation_14c: bool,
    sigma_rule: str | None,
    outdir_root: str | None,
) -> dict[str, Any]:
    _install_carbon_sigma_rule(sigma_rule)

    from ecosystem_complexity.config import load_config
    from ecosystem_complexity.oe_diagnostics import (
        fit_param_subset_indices,
        oe_gain_matrix_diagnostics,
        oe_ladder_context,
        shapley_dfs_per_parameter_from_context,
    )
    from ecosystem_complexity.outputs import (
        open_run_dir, resolve_run_name, write_npz, write_parquet,
    )
    from ecosystem_complexity.sites.driver import OPT_FIELDS, run_site_canonical
    from ecosystem_complexity.sites.spec import load_site_spec

    spec = load_site_spec(config_path)
    t0 = time.perf_counter()
    result = run_site_canonical(
        spec,
        observation_path=spec.observation_path,
        include_er_constraint=include_er,
        include_incubation_constraint=include_incubation,
        include_incubation_14c_constraint=include_incubation_14c,
    )
    if result.get("skipped"):
        return {"status": "skipped", "config": config_path, "label": spec.label,
                "runtime_s": time.perf_counter() - t0}

    ctx = oe_ladder_context(
        result["model"], result["forcing"], result["state0"],
        result["params_opt"], result["obs_full"], OPT_FIELDS,
        extra_obs_blocks=result["pool_blocks"],
    )
    shap = shapley_dfs_per_parameter_from_context(ctx, families=DEFAULT_FAMILIES)
    ak_bundle = oe_gain_matrix_diagnostics(
        result["model"], result["forcing"], result["state0"],
        result["params_opt"], result["obs_full"], OPT_FIELDS,
        extra_obs_blocks=result["pool_blocks"],
    )

    site = spec.israd_name
    biome_group = _biome_group(spec.biome)

    # Long-form per-(param, family) rows for the network table
    shap_rows: list[dict] = []
    subset_idx = fit_param_subset_indices(shap["state_names"])
    for fi, fam in enumerate(shap["families"]):
        for pi, pname in enumerate(shap["state_names"]):
            if pi not in subset_idx:
                continue
            shap_rows.append({
                "site": site, "label": spec.label,
                "biome": spec.biome, "biome_group": biome_group,
                "family": fam,
                "n_obs_family": int(shap["n_obs_per_family"].get(fam, 0)),
                "param": _STATE_SHORT.get(pname, pname),
                "shapley_dfs": float(shap["shapley"][fi, pi]),
            })

    sub_ak = np.asarray(ak_bundle["subset_averaging_kernel"], dtype=float)
    subset_state_names = list(ak_bundle["subset_state_names"])
    site_row = {
        "site": site, "label": spec.label,
        "biome": spec.biome, "biome_group": biome_group,
        "n_obs_total": int(ak_bundle["y_obs"].size),
        "dfs_total_subset": float(np.trace(sub_ak)),
        "runtime_s": time.perf_counter() - t0,
    }
    for i, name in enumerate(subset_state_names):
        short = _STATE_SHORT.get(name, name)
        site_row[f"dfs_{short}"] = float(sub_ak[i, i])

    # Per-site artifacts under outputs/{site_id}/information/shapley/
    config = load_config(config_path)
    name = resolve_run_name(config=config)
    run = open_run_dir(
        verb="information",
        subverb="shapley",
        name=name,
        outdir=(Path(outdir_root) / name / "information" / "shapley") if outdir_root else None,
        inputs={
            "config_path": config_path,
            "families": list(DEFAULT_FAMILIES),
            "sigma_rule": sigma_rule,
            "include_er": include_er,
            "include_incubation": include_incubation,
            "include_incubation_14c": include_incubation_14c,
        },
    )
    run.snapshot_config(config)
    write_npz(
        run, "matrices.npz",
        subset_averaging_kernel=ak_bundle["subset_averaging_kernel"],
        subset_gain_matrix=ak_bundle["subset_gain_matrix"],
        averaging_kernel=ak_bundle["averaging_kernel"],
        gain_matrix=ak_bundle["gain_matrix"],
        state_names=np.array(subset_state_names),
        subset_indices=np.array(ak_bundle["subset_indices"]),
        constraint_labels=np.array(list(ak_bundle["constraint_labels"])),
        Se_diag=ak_bundle["Se_diag"],
        y_obs=ak_bundle["y_obs"],
        y_prior=ak_bundle["y_prior"],
        y_opt=ak_bundle["y_opt"],
    )
    write_parquet(run, "shapley_by_parameter.parquet", shap_rows)
    write_parquet(run, "metrics.parquet", [site_row])
    run.add_manifest_field("site_id", site)
    run.add_manifest_field("biome_group", biome_group)
    run.finalize()

    return {
        "status": "ok",
        "config": config_path,
        "label": spec.label,
        "site_row": site_row,
        "shap_rows": shap_rows,
        "run_dir": str(run.root),
    }


# ---------------------------------------------------------------------------
# Site-set aggregation
# ---------------------------------------------------------------------------


def _write_site_set_shapley(
    site_set_name: str, site_rows: list[dict], shap_rows: list[dict],
    outdir_root: str | None, options: dict[str, Any], plot_by: str | None,
) -> Path:
    from ecosystem_complexity.outputs import open_run_dir, write_parquet
    run = open_run_dir(
        verb="information", subverb="shapley",
        name=site_set_name,
        outdir=(Path(outdir_root) / site_set_name / "information" / "shapley") if outdir_root else None,
        inputs={"n_members": len(site_rows), **options},
    )
    site_df = pd.DataFrame(site_rows).sort_values("site").reset_index(drop=True)
    shap_df = pd.DataFrame(shap_rows).sort_values(["site", "family", "param"]).reset_index(drop=True)
    write_parquet(run, "network_site_summary.parquet", site_df)
    write_parquet(run, "shapley_per_param_long.parquet", shap_df)
    if not shap_df.empty:
        pivot = (
            shap_df.groupby(["biome_group", "family", "param"])["shapley_dfs"]
            .mean().unstack("param").reset_index()
        )
        write_parquet(run, "biome_shapley_pivot.parquet", pivot)
        best = (
            shap_df.groupby(["biome_group", "param", "family"])["shapley_dfs"]
            .mean().reset_index()
            .sort_values(["biome_group", "param", "shapley_dfs"], ascending=[True, True, False])
        )
        write_parquet(run, "biome_param_family_ranking.parquet", best)
    run.add_manifest_field("site_set", site_set_name)
    run.add_manifest_field("plot_by", plot_by)
    run.finalize()
    return run.root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_SUBVERBS = ("shapley", "dfs", "ak", "gain", "ose")


def _build_shapley_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ecosys information shapley")
    p.add_argument("sites", nargs="*")
    p.add_argument("--site-set", metavar="YAML")
    p.add_argument("--configs-dir", metavar="DIR", default=None,
                   help="glob every *.yaml under DIR as members "
                        "(replaces the old apps/hf_pool_sweep entry point).")
    p.add_argument("--outdir", default=None)
    p.add_argument("--sigma-rule", default=None,
                   help="σ_carbon rule REL:ABS (e.g. 0.20:500); overrides "
                        "information.shapley.sigma_rule from YAML.")
    p.add_argument("--plot-by", choices=("biome", "pool_count"), default=None,
                   help="record grouping for downstream plotters (recorded in manifest).")
    p.add_argument("--include-er", action="store_true", default=True)
    p.add_argument("--no-include-er", dest="include_er", action="store_false")
    p.add_argument("--include-incubation", action="store_true")
    p.add_argument("--include-incubation-14c", action="store_true")
    p.add_argument("-j", "--workers", type=int, default=1)
    return p


def _dispatch_shapley(argv: list[str]) -> int:
    args = _build_shapley_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    exclusive = sum(bool(v) for v in (args.sites, args.site_set, args.configs_dir))
    if exclusive == 0:
        _build_shapley_parser().error("give at least one site, --site-set, or --configs-dir")
    if exclusive > 1:
        _build_shapley_parser().error("sites, --site-set, and --configs-dir are mutually exclusive")

    site_set_name: str | None = None
    if args.site_set:
        site_set_name, configs, run_options = _load_site_set(args.site_set)
    elif args.configs_dir:
        d = Path(args.configs_dir)
        if not d.is_absolute():
            d = _REPO_ROOT / d
        if not d.is_dir():
            _build_shapley_parser().error(f"no such directory: {args.configs_dir}")
        configs = sorted(str(p) for p in d.glob("*.yaml"))
        if not configs:
            _build_shapley_parser().error(f"no *.yaml under {args.configs_dir}")
        site_set_name = d.name
        run_options = {}
    else:
        configs = _resolve_specs(args.sites)
        run_options = {}

    include_er = args.include_er or bool(run_options.get("include_er"))
    include_incub = args.include_incubation or bool(run_options.get("include_incubation"))
    include_incub14 = args.include_incubation_14c or bool(run_options.get("include_incubation_14c"))

    # CLI --sigma-rule wins; else read the anchor config's YAML.
    sigma_rule = args.sigma_rule or _read_sigma_rule_from_yaml(configs[0])

    site_rows: list[dict] = []
    shap_rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    workers = min(args.workers, len(configs)) if args.workers > 1 else 1

    def _do_one(cfg: str) -> dict[str, Any]:
        return _shapley_one(cfg, include_er, include_incub, include_incub14,
                            sigma_rule, args.outdir)

    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            fut_to_cfg = {pool.submit(_do_one, c): c for c in configs}
            for fut in as_completed(fut_to_cfg):
                cfg = fut_to_cfg[fut]
                try:
                    payload = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append((cfg, repr(exc)))
                    print(f"  FAILED  {cfg}: {exc}")
                    continue
                if payload["status"] == "ok":
                    site_rows.append(payload["site_row"])
                    shap_rows.extend(payload["shap_rows"])
                    r = payload["site_row"]
                    print(f"  {r['label']}  DFS={r['dfs_total_subset']:.2f}")
                else:
                    print(f"  {payload['label']}: skipped")
    else:
        for cfg in configs:
            try:
                payload = _do_one(cfg)
            except Exception as exc:  # noqa: BLE001
                failures.append((cfg, repr(exc)))
                print(f"  FAILED  {cfg}: {exc}")
                continue
            if payload["status"] == "ok":
                site_rows.append(payload["site_row"])
                shap_rows.extend(payload["shap_rows"])
                r = payload["site_row"]
                print(f"  {r['label']}  DFS={r['dfs_total_subset']:.2f}")
            else:
                print(f"  {payload['label']}: skipped")

    if site_set_name and site_rows:
        options = {
            "sigma_rule": sigma_rule,
            "include_er": include_er,
            "include_incubation": include_incub,
            "include_incubation_14c": include_incub14,
        }
        agg_dir = _write_site_set_shapley(
            site_set_name, site_rows, shap_rows, args.outdir, options, args.plot_by,
        )
        print(f"  wrote site-set summary {agg_dir}")

    print(f"\n{len(site_rows)}/{len(configs)} sites analyzed"
          + (f", {len(failures)} failed" if failures else ""))
    return 1 if failures else 0


def _not_yet(subverb: str) -> int:
    """Subverbs that reuse the same forward pipeline but ship with the
    information module only after Session B / C. Kept as explicit stubs
    so the CLI surface is stable — implementation lands with the same
    output-contract layout under outputs/{name}/information/{subverb}/."""
    print(
        f"ecosys information {subverb}: not yet implemented in this build.\n"
        f"  The forward pipeline (oe_ladder_context, oe_gain_matrix_diagnostics)\n"
        f"  is in ecosystem_complexity.oe_diagnostics; wiring it into the\n"
        f"  standard output contract is Session D.\n"
        f"  Meanwhile the AK, gain, and DFS matrices are already written by\n"
        f"  `ecosys information shapley` (matrices.npz has averaging_kernel,\n"
        f"  gain_matrix, subset_averaging_kernel, subset_gain_matrix).",
        file=sys.stderr,
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: ecosys information {shapley|dfs|ak|gain|ose} [args...]",
            file=sys.stderr,
        )
        return 0 if argv else 2
    sub, rest = argv[0], argv[1:]
    if sub not in _SUBVERBS:
        print(f"unknown information subverb: {sub!r}", file=sys.stderr)
        return 2
    if sub == "shapley":
        return _dispatch_shapley(rest)
    return _not_yet(sub)


if __name__ == "__main__":
    raise SystemExit(main())
