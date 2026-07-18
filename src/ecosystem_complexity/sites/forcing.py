"""Locating a configured site's forcing file.

This is the site-shaped half of forcing handling: turning a ``SiteSpec`` into a
path, and validating the ``forcing_kind`` it declares. Reading and sanitising
that file is generic and lives in :mod:`ecosystem_complexity.data.forcing`.
"""
from __future__ import annotations

import glob
import os

from ecosystem_complexity.data.forcing import (
    build_annual_mean_forcing,
    load_daily_forcing,
    resolve_dd_file,
)
from ecosystem_complexity.data.paths import REPO_ROOT as _REPO_ROOT
from ecosystem_complexity.sites.spec import SiteSpec

# Re-exported so the drivers keep importing their forcing entry points from one
# place; the implementations are in data.forcing.
__all__ = [
    "build_annual_mean_forcing",
    "load_site_forcing",
    "resolve_forcing_file",
]


def resolve_forcing_file(spec: SiteSpec) -> str:
    """Resolve a site's configured ``forcing_glob`` to a concrete file."""
    if spec.forcing_kind == "daily":
        return resolve_dd_file(spec.forcing_glob)
    matches = glob.glob(os.path.join(_REPO_ROOT, "data", spec.forcing_glob))
    if not matches:
        raise FileNotFoundError(f"No forcing file matching data/{spec.forcing_glob}")
    return matches[0]


def load_site_forcing(spec: SiteSpec, path: str, model):
    """Load a site's forcing from its FLUXNET daily product.

    Every configured site now uses the same ``daily`` product, so this no longer
    dispatches. The two site-specific branches it used to carry —
    ``harvard_hr`` (FULLSET HR) and ``eml_hh`` (BASE HH) — went away when
    Harvard and EML were migrated onto their FLUXNET daily releases.

    Dropping Harvard's HR branch also fixed a silent halving of its fluxes:
    AmeriFlux "HR" is an *hourly* product (24 records/day), but
    ``load_harvard_forest`` applies the half-hourly 1800 s conversion — correct
    for genuine half-hourly input, which is what its tests cover, and wrong by
    2x for that file. Harvard's mean GPP was 735 gC m⁻² yr⁻¹ against a
    literature value near 1400; on the daily product it is 1500.

    An unrecognised ``forcing_kind`` raises rather than falling through to the
    daily reader, because that silent fallthrough is exactly how a product
    mismatch like the above goes unnoticed.
    """
    if spec.forcing_kind != "daily":
        raise ValueError(
            f"{spec.config_stem}: unsupported forcing_kind {spec.forcing_kind!r}; "
            "the multisite recipe reads FLUXNET daily products only. Load a "
            "one-off half-hourly product with the site loaders in "
            "ecosystem_complexity.data.loaders instead."
        )
    return load_daily_forcing(path, model)
