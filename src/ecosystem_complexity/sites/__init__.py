"""Site inversion drivers — the shared, config-driven OE recipe.

This package holds the code that used to live under ``notebooks/sites/`` and was
importable only via ``sys.path`` manipulation. The analysis scripts in
``notebooks/`` and the CLI in ``apps/optim_site_main.py`` both import from here.

The OE ablation and constraint-ladder diagnostics are *not* re-exported here;
they live in :mod:`ecosystem_complexity.oe_diagnostics` alongside the other
information-content diagnostics.
"""
from __future__ import annotations

from ecosystem_complexity.sites.driver import run_oe_canonical
from ecosystem_complexity.sites.multisite import (
    OPT_FIELDS,
    SiteSpec,
    build_annual_mean_forcing,
    build_bulk_14C_blocks,
    build_fraction_14C_blocks,
    build_measured_soc_total,
    build_resp_14C_obs,
    build_soc_prior,
    build_soilgrids_soc_total,
    build_state0,
    discover_site_specs,
    load_site_spec,
    run_site_canonical,
    run_sites,
    select_specs,
    summary_row,
)
from ecosystem_complexity.sites.paths import CONFIG_DIR, REPO_ROOT

__all__ = [
    "CONFIG_DIR",
    "OPT_FIELDS",
    "REPO_ROOT",
    "SiteSpec",
    "build_annual_mean_forcing",
    "build_bulk_14C_blocks",
    "build_fraction_14C_blocks",
    "build_measured_soc_total",
    "build_resp_14C_obs",
    "build_soc_prior",
    "build_soilgrids_soc_total",
    "build_state0",
    "discover_site_specs",
    "load_site_spec",
    "run_oe_canonical",
    "run_site_canonical",
    "run_sites",
    "select_specs",
    "summary_row",
]
