"""Site inversion drivers — the shared, config-driven OE recipe.

This package holds the code that used to live under ``notebooks/sites/`` and was
importable only via ``sys.path`` manipulation. The analysis scripts in
``notebooks/`` and the CLI in ``apps/optim_site_main.py`` both import from here.

Modules follow the pipeline order:

``spec``             per-site config: the SiteSpec record and config discovery
``forcing``          flux-tower forcing: resolution, loading, annual-mean collapse
``fraction_mapping`` ISRaD ``frc_property`` → kinetic pool, keyed on the
                     fractionation vocabulary rather than on the site
``israd_14c``        bulk / fraction / respired Δ¹⁴C observation blocks
``soc``              steady-state SOC prior and total-column stock constraints
``driver``           ``run_oe_canonical`` and the config-driven site runner

The OE ablation and constraint-ladder diagnostics are *not* re-exported here;
they live in :mod:`ecosystem_complexity.oe_diagnostics` alongside the other
information-content diagnostics.
"""
from __future__ import annotations

from ecosystem_complexity.sites.driver import (
    OPT_FIELDS,
    build_state0,
    run_oe_canonical,
    run_site_canonical,
    run_sites,
    summary_row,
)
from ecosystem_complexity.sites.forcing import build_annual_mean_forcing
from ecosystem_complexity.sites.fraction_mapping import (
    BULK_PROPERTIES,
    PROPERTY_POLICY,
    PROPERTY_ROLES,
    SCHEME_POLICY,
    FractionMapping,
    build_fraction_mapping,
)
from ecosystem_complexity.sites.israd_14c import (
    build_bulk_14C_blocks,
    build_fraction_14C_blocks,
    build_resp_14C_obs,
)
from ecosystem_complexity.sites.paths import CONFIG_DIR, REPO_ROOT
from ecosystem_complexity.sites.soc import (
    build_measured_soc_total,
    build_soc_prior,
    build_soilgrids_soc_total,
)
from ecosystem_complexity.sites.spec import (
    SiteSpec,
    discover_site_specs,
    load_site_spec,
    select_specs,
)

__all__ = [
    "BULK_PROPERTIES",
    "PROPERTY_POLICY",
    "CONFIG_DIR",
    "PROPERTY_ROLES",
    "OPT_FIELDS",
    "SCHEME_POLICY",
    "REPO_ROOT",
    "FractionMapping",
    "SiteSpec",
    "build_annual_mean_forcing",
    "build_bulk_14C_blocks",
    "build_fraction_14C_blocks",
    "build_fraction_mapping",
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
