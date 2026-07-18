#!/usr/bin/env python
"""Download flux-tower data for a configured site."""

from __future__ import annotations

import argparse
import os
import sys

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_APP_DIR)
_SRC = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ecosystem_complexity.fetch import download_flux_data, resolve_flux_download_plan  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", help="Configured site selector or tower id.")
    parser.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "data"))
    parser.add_argument("--env-file", default=os.path.join(_REPO_ROOT, ".env"))
    parser.add_argument("--user-id", help="AmeriFlux user id.")
    parser.add_argument("--email", help="AmeriFlux account email.")
    parser.add_argument("--accept-policy", action="store_true")
    parser.add_argument("--accept-license", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    plan = resolve_flux_download_plan(args.site)
    print(
        f"{plan.selector} -> {plan.source} ({plan.tower_id})\n"
        f"target: {plan.remote_label}\n"
        f"local : {plan.local_path}"
    )
    outputs = download_flux_data(
        args.site,
        out_dir=args.out_dir,
        accept_policy=args.accept_policy,
        accept_license=args.accept_license,
        user_id=args.user_id,
        email=args.email,
        env_file=args.env_file,
        keep_archive=args.keep_archive,
        dry_run=args.dry_run,
    )
    for item in outputs:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

