#!/usr/bin/env python3
"""``ecosys fetch`` — download / stage forcing and observation data.

Subverbs
    flux     AmeriFlux/FLUXNET tower CSV for one site (requires .env credentials)
    fluxcom  FLUXCOM-X 2021 global GPP/NEE grids → per-site CSV extraction
    clm      Stage Community Land Model NetCDF(s) and extract site series
    israd    Download the versioned ISRaD compiled tables
    atm14c   Download atmospheric ¹⁴C source records

Every subverb writes a small manifest under
``outputs/{name}/fetch/{source}/`` recording what was retrieved, so
downstream apps can trust the data cache is fresh without re-running.
The bulk data itself lands in ``data/`` (raw archives + per-site CSVs),
not under ``outputs/`` — that layout is what the site drivers already
consume and moving it would break configs' relative ``forcing_glob`` paths.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ecosystem_complexity.outputs import open_run_dir  # noqa: E402

logger = logging.getLogger("ecosys.fetch")

_SUBVERBS = ("flux", "fluxcom", "clm", "israd", "atm14c")


# ---------------------------------------------------------------------------
# flux — AmeriFlux tower CSV for one site
# ---------------------------------------------------------------------------


def _cmd_flux(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="ecosys fetch flux")
    p.add_argument("site", help="Configured site selector or tower id.")
    p.add_argument("--out-dir", default=str(_REPO_ROOT / "data"))
    p.add_argument("--env-file", default=str(_REPO_ROOT / ".env"))
    p.add_argument("--user-id", help="AmeriFlux user id.")
    p.add_argument("--email", help="AmeriFlux account email.")
    p.add_argument("--accept-policy", action="store_true")
    p.add_argument("--accept-license", action="store_true")
    p.add_argument("--keep-archive", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--outdir", default=None,
                   help="root for the fetch manifest (default ./outputs/)")
    args = p.parse_args(argv)

    from ecosystem_complexity.fetch import download_flux_data, resolve_flux_download_plan
    plan = resolve_flux_download_plan(args.site)
    run = open_run_dir(
        verb="fetch", subverb="flux",
        name=plan.tower_id or "flux_download",
        outdir=(Path(args.outdir) / (plan.tower_id or "flux_download") / "fetch" / "flux")
        if args.outdir else None,
        inputs={"selector": plan.selector, "tower_id": plan.tower_id,
                "source": plan.source, "remote": plan.remote_label,
                "local_path": str(plan.local_path), "dry_run": args.dry_run},
    )
    print(
        f"{plan.selector} -> {plan.source} ({plan.tower_id})\n"
        f"target: {plan.remote_label}\n"
        f"local : {plan.local_path}"
    )
    outputs = download_flux_data(
        args.site, out_dir=args.out_dir,
        accept_policy=args.accept_policy, accept_license=args.accept_license,
        user_id=args.user_id, email=args.email,
        env_file=args.env_file, keep_archive=args.keep_archive,
        dry_run=args.dry_run,
    )
    for item in outputs:
        print(item)
    run.add_manifest_field("downloaded_paths", [str(x) for x in outputs])
    run.finalize()
    return 0


# ---------------------------------------------------------------------------
# fluxcom — FLUXCOM-X 2021 site extraction
# ---------------------------------------------------------------------------


def _cmd_fluxcom(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="ecosys fetch fluxcom")
    p.add_argument("configs", nargs="*",
                   help="site config paths; defaults to configs/expansion/*.yaml")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--outdir", default=None)
    args = p.parse_args(argv)

    from ecosystem_complexity.data.fetch_fluxcom import (
        FLUXCOM_X_2021, FLUXCOM_X_2021_NEE, fetch_fluxcom_x_for_configs,
    )
    config_paths = (
        [Path(p) for p in args.configs] if args.configs
        else sorted((_REPO_ROOT / "configs" / "expansion").glob("*.yaml"))
    )
    run = open_run_dir(
        verb="fetch", subverb="fluxcom", name="fluxcom_x_2021",
        outdir=(Path(args.outdir) / "fluxcom_x_2021" / "fetch" / "fluxcom") if args.outdir else None,
        inputs={"n_configs": len(config_paths), "overwrite": args.overwrite,
                "gpp_source": FLUXCOM_X_2021["landing_page"],
                "nee_source": FLUXCOM_X_2021_NEE["landing_page"]},
    )
    rows = fetch_fluxcom_x_for_configs(config_paths, overwrite=args.overwrite)
    if not rows:
        print("No configs with forcing_kind: fluxcom")
        run.finalize()
        return 0
    print("GPP source:", FLUXCOM_X_2021["landing_page"])
    print("NEE source:", FLUXCOM_X_2021_NEE["landing_page"])
    for row in rows:
        status = str(row["status"])
        if status == "skipped":
            print(f"- {row['site_id']}: kept existing {row['forcing_output_path']}")
        else:
            print(f"- {row['site_id']}: wrote {row['forcing_output_path']} "
                  f"({row['n_days']} days, mean GPP {row['mean_gpp_gCm2day']:.2f})")
    run.add_manifest_field("site_rows", rows)
    run.finalize()
    return 0


# ---------------------------------------------------------------------------
# clm — Community Land Model site extraction
# ---------------------------------------------------------------------------


def _cmd_clm(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="ecosys fetch clm",
        description="Stage CLM NetCDFs or extract site series from Pangeo CMIP6.",
    )
    p.add_argument("configs", nargs="*",
                   help="site config paths; defaults to configs/multisite/*.yaml "
                        "filtered to forcing_kind: clm")
    p.add_argument("--source-dir", default=str(_REPO_ROOT / "data" / "raw" / "clm"),
                   help="directory containing CLM history NetCDFs (one shared file "
                        "for the domain, or per-site files with the tower id in the name)")
    p.add_argument("--url", action="append", default=[],
                   help="direct HTTP(S) URL of a CLM NetCDF; repeat for multiple files")
    p.add_argument("--pangeo", action="store_true",
                   help="read public CMIP6 Zarr stores through Pangeo instead of local NetCDFs")
    p.add_argument("--pangeo-model", default="CESM2", help="Pangeo CMIP6 source_id")
    p.add_argument("--pangeo-experiment", default="historical", help="Pangeo CMIP6 experiment_id")
    p.add_argument("--pangeo-member", default="r1i1p1f1", help="Pangeo CMIP6 member_id")
    p.add_argument("--pangeo-ssp", action="append", default=[],
                   help="also retrieve this CMIP6 SSP experiment; repeat as needed")
    p.add_argument("--pangeo-include-14c", action="store_true",
                   help="also request c14Soil; errors if the selected Pangeo data lack it")
    p.add_argument("--variables", nargs="+", default=None,
                   help="CLM variables to extract (aliases are tried in order)")
    p.add_argument("--out-root", default=None,
                   help="site CSV output directory (default data/shared/clm/)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--outdir", default=None,
                   help="root for the fetch manifest (default ./outputs/)")
    args = p.parse_args(argv)
    variables = args.variables or (
        ["CSOILFAST", "CSOILMEDIUM", "CSOILSLOW", "CSOIL", "CLITTER",
         "GPP", "NPP", "HR"] if args.pangeo else ["GPP", "NEE", "HR"]
    )
    pangeo_experiments = [args.pangeo_experiment, *args.pangeo_ssp]
    if args.pangeo_include_14c:
        if not args.pangeo:
            p.error("--pangeo-include-14c requires --pangeo")
        variables.append("C14SOIL")

    from ecosystem_complexity.data.fetch_clm import (
        _CLM_ROOT, fetch_clm_for_configs,
    )
    config_paths = (
        [Path(p) for p in args.configs] if args.configs
        else sorted((_REPO_ROOT / "configs" / "multisite").glob("*.yaml"))
    )
    out_root = Path(args.out_root) if args.out_root else _CLM_ROOT
    run = open_run_dir(
        verb="fetch", subverb="clm", name="clm_site_extract",
        outdir=(Path(args.outdir) / "clm_site_extract" / "fetch" / "clm") if args.outdir else None,
        inputs={"source_dir": args.source_dir, "urls": args.url, "pangeo": args.pangeo,
                "pangeo_model": args.pangeo_model, "pangeo_experiment": args.pangeo_experiment,
                "pangeo_ssp": args.pangeo_ssp, "pangeo_member": args.pangeo_member,
                "pangeo_include_14c": args.pangeo_include_14c, "variables": variables,
                "out_root": str(out_root), "overwrite": args.overwrite,
                "n_configs": len(config_paths)},
    )
    try:
        if args.pangeo and args.url:
            raise ValueError("--pangeo and --url cannot be used together")
        if args.pangeo:
            from ecosystem_complexity.data.fetch_clm import fetch_pangeo_clm_for_configs
            rows = fetch_pangeo_clm_for_configs(
                config_paths, source_id=args.pangeo_model,
                experiment_ids=pangeo_experiments,
                member_id=args.pangeo_member, variables=variables,
                overwrite=args.overwrite, out_root=out_root,
            )
        else:
            if args.url:
                from ecosystem_complexity.data.fetch_clm import download_clm_sources
                staged = download_clm_sources(
                    args.url, source_dir=args.source_dir, overwrite=args.overwrite,
                )
                run.add_manifest_field("staged_netcdf_paths", [str(path) for path in staged])
            rows = fetch_clm_for_configs(
                config_paths, source_dir=args.source_dir,
                variables=variables, overwrite=args.overwrite, out_root=out_root,
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        run.add_manifest_field("error", str(exc))
        run.finalize()
        return 2

    if not rows:
        print("No configs with forcing_kind: clm")
        run.finalize()
        return 0

    for row in rows:
        status = str(row["status"])
        if status == "skipped":
            path = row.get("archive_output_path", row.get("forcing_output_path"))
            print(f"- {row['site_id']}: kept existing {path}")
        else:
            if args.pangeo:
                print(f"- {row['site_id']} ({row['experiment_id']}): wrote "
                      f"{row['archive_output_path']}")
            else:
                print(f"- {row['site_id']}: wrote {row['forcing_output_path']} "
                      f"({row['n_days']} days)")
    run.add_manifest_field("site_rows", rows)
    run.finalize()
    return 0


# ---------------------------------------------------------------------------
# israd / atm14c — versioned observation inputs
# ---------------------------------------------------------------------------


def _cmd_israd(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="ecosys fetch israd")
    p.add_argument("--url", default=None, help="official ISRaD compiled-database zip URL")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    from ecosystem_complexity.data.paths import (
        ISRAD_COMPILED_ARCHIVE_URL,
        ISRAD_DIR,
        ISRAD_FRACTION,
        ISRAD_FLUX,
        ISRAD_INCUBATION,
        ISRAD_LAYER,
    )
    from ecosystem_complexity.fetch.external import download_file, extract_named_zip_members
    required = [Path(p).name for p in (ISRAD_LAYER, ISRAD_FRACTION, ISRAD_FLUX, ISRAD_INCUBATION)]
    missing = [name for name in required if not (Path(ISRAD_DIR) / name).is_file()]
    if not missing and not args.overwrite:
        print(f"ISRaD {Path(ISRAD_LAYER).name.split('_v ')[-1].removesuffix('.csv')} is already staged: {ISRAD_DIR}")
        return 0
    archive = download_file(
        args.url or ISRAD_COMPILED_ARCHIVE_URL,
        Path(ISRAD_DIR) / ".israd-download.zip",
        overwrite=True,
    )
    outputs = extract_named_zip_members(archive, ISRAD_DIR, required, overwrite=args.overwrite)
    archive.unlink(missing_ok=True)
    print("Downloaded ISRaD tables:")
    for output in outputs:
        print(output)
    return 0


def _cmd_atm14c(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="ecosys fetch atm14c")
    p.add_argument("--hua-url", help="direct URL for Hua_2021.csv")
    p.add_argument("--graven-url", help="direct URL for Graven_2017.csv")
    p.add_argument("--intcal-url", default="https://intcal.org/curves/intcal20.14c",
                   help="direct URL for intcal20.14c")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    from ecosystem_complexity.data.paths import GRAVEN_PATH, HUA_PATH, INTCAL_PATH
    from ecosystem_complexity.fetch.external import download_file
    sources = (("HUA", HUA_PATH, args.hua_url), ("GRAVEN", GRAVEN_PATH, args.graven_url),
               ("INTCAL", INTCAL_PATH, args.intcal_url))
    for name, path, url in sources:
        if url and (args.overwrite or not Path(path).is_file()):
            download_file(url, path, overwrite=args.overwrite)
        exists = "OK" if Path(path).is_file() else "MISSING"
        detail = f" from {url}" if url else " (no URL supplied)"
        print(f"  [{exists}] {name}: {path}{detail}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_HANDLERS = {
    "flux": _cmd_flux,
    "fluxcom": _cmd_fluxcom,
    "clm": _cmd_clm,
    "israd": _cmd_israd,
    "atm14c": _cmd_atm14c,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(f"usage: ecosys fetch {{{'|'.join(_SUBVERBS)}}} [args...]", file=sys.stderr)
        return 0 if argv else 2
    sub, rest = argv[0], argv[1:]
    handler = _HANDLERS.get(sub)
    if handler is None:
        print(f"unknown fetch source: {sub!r}", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    try:
        return handler(rest)
    except (OSError, RuntimeError, ValueError, PermissionError, KeyError) as exc:
        logger.error("error: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
