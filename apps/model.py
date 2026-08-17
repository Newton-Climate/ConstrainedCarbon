#!/usr/bin/env python3
"""``ecosys model`` — validate inputs or run the forward carbon model."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from ecosystem_complexity.api import build_model, run_model, spinup
from ecosystem_complexity.data.custom_14c import (
    build_custom_14c_observations,
    load_custom_14c_manifest,
)
from ecosystem_complexity.data.parsers import attach_atm14C
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.paths import GRAVEN_PATH, HUA_PATH, INTCAL_PATH
from ecosystem_complexity.outputs import (
    attach_file_logger,
    open_run_dir,
    resolve_run_name,
    write_json,
    write_npz,
)
from ecosystem_complexity.sites.forcing import load_site_forcing, resolve_forcing_file
from ecosystem_complexity.sites.spec import load_site_spec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecosys model",
        description="Validate a site input set or run the forward carbon model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="validate config, forcing, and custom ¹⁴C input"
    )
    validate.add_argument("config", help="path to a site YAML config")
    run = commands.add_parser("run", help="run the forward model and write outputs")
    run.add_argument("config", help="path to a site YAML config")
    run.add_argument("--outdir", metavar="DIR", default=None)
    run.add_argument(
        "--spinup-years", type=int, default=0, metavar="N",
        help="spin up C pools for up to N years before the run (default: 0)",
    )
    return parser


def _load_inputs(config_path: str):
    """Load the config, forcing, and optional custom observations once."""
    spec = load_site_spec(config_path)
    model = build_model(config_path)
    forcing_path = resolve_forcing_file(spec)
    forcing = load_site_forcing(spec, forcing_path, model)
    custom_blocks = []
    custom_resp_n = 0
    if spec.radiocarbon_manifest:
        manifest = Path(config_path).parent / spec.radiocarbon_manifest
        custom = load_custom_14c_manifest(manifest)
        custom_blocks, resp = build_custom_14c_observations(
            custom, forcing.time, model.pool_index
        )
        custom_resp_n = int(np.isfinite(np.asarray(resp)).sum())
    return spec, model, forcing_path, forcing, custom_blocks, custom_resp_n


def _attach_atmospheric_14c(forcing, latitude: float):
    hemisphere = "NH" if latitude >= 0 else "SH"
    years, values = load_full_14C_record(
        hua_path=HUA_PATH,
        graven_path=GRAVEN_PATH,
        intcal_path=INTCAL_PATH,
        hemisphere=hemisphere,
        start_year=1500.0,
        end_year=2025.0,
    )
    return attach_atm14C(forcing, values, years), hemisphere


def _validate(config_path: str) -> int:
    spec, model, forcing_path, forcing, blocks, n_resp = _load_inputs(config_path)
    years = 1970.0 + np.asarray(forcing.time) / 365.25
    print(f"VALID  {spec.label}")
    print(
        f"  forcing: {forcing_path} ({len(forcing.time)} days, "
        f"{years[0]:.1f}–{years[-1]:.1f})"
    )
    print(f"  pools: {', '.join(model.pool_index.pool_names)}")
    if spec.radiocarbon_manifest:
        print(f"  custom ¹⁴C: {len(blocks)} blocks, {n_resp} respiration date(s)")
    else:
        print("  custom ¹⁴C: not configured (ISRaD observation path remains active)")
    return 0


def _run(config_path: str, outdir: str | None, spinup_years: int) -> int:
    if spinup_years < 0:
        raise ValueError("--spinup-years must be non-negative")
    spec, model, forcing_path, forcing, blocks, n_resp = _load_inputs(config_path)
    forcing, hemisphere = _attach_atmospheric_14c(forcing, spec.lat)
    run = open_run_dir(
        verb="model",
        subverb="run",
        name=resolve_run_name(config=model.config),
        outdir=outdir,
        inputs={
            "config_path": str(Path(config_path).resolve()),
            "forcing_path": str(forcing_path),
            "spinup_years": spinup_years,
            "hemisphere": hemisphere,
        },
    )
    handler = attach_file_logger(run)
    try:
        state0 = spinup(model, forcing, n_years=spinup_years) if spinup_years else None
        output = run_model(model, forcing, state0=state0)
        write_npz(
            run,
            "forward_output.npz",
            time=np.asarray(forcing.time),
            C12=np.asarray(output.C12),
            delta14C=np.asarray(output.delta14C),
            NEE=np.asarray(output.NEE),
            GPP=np.asarray(output.GPP),
            ER=np.asarray(output.ER),
            Rh=np.asarray(output.Rh),
        )
        write_json(
            run,
            "diagnostics.json",
            {
                "n_days": int(len(forcing.time)),
                "n_pools": len(model.pool_index.pool_names),
                "pool_names": list(model.pool_index.pool_names),
                "custom_14c_blocks_validated": len(blocks),
                "custom_respiration_dates_validated": n_resp,
                "final_c12_total_gCm2": float(
                    np.sum(np.asarray(output.final_state.C12))
                ),
            },
        )
        run.snapshot_config(model.config)
        run.finalize()
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    print(f"wrote {run.root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        return _validate(args.config)
    return _run(args.config, args.outdir, args.spinup_years)


if __name__ == "__main__":
    raise SystemExit(main())
