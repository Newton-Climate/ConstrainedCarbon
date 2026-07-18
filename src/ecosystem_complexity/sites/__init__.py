"""Site inversion drivers — the shared, config-driven OE recipe.

This package holds the code that used to live under ``notebooks/sites/`` and was
importable only via ``sys.path`` manipulation. The analysis scripts in
``notebooks/`` and the CLI in ``apps/optim_site_main.py`` both import from here.

What belongs here is what needs a *site* — a ``SiteSpec``, a config, or the
inversion itself:

``spec``     per-site config: the SiteSpec record and config discovery
``paths``    the per-site config directory
``forcing``  resolving a spec's ``forcing_glob`` to a concrete file
``soc``      the model's own steady-state SOC prior (runs the forward model)
``driver``   ``run_oe_canonical`` and the config-driven site runner

Reading and interpreting observations is *data*, not site, and lives in
:mod:`ecosystem_complexity.data`: ``israd_14c`` (bulk / fraction / respired
Δ¹⁴C blocks), ``fraction_mapping`` (ISRaD ``frc_scheme`` → kinetic pool),
``soc_stocks`` (measured ISRaD / SoilGrids stocks), ``forcing`` (reading and
sanitising a tower file), and ``paths`` (where the input tables live). The
dependency runs one way: ``sites`` imports ``data``, never the reverse.
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
from ecosystem_complexity.sites.forcing import (
    build_annual_mean_forcing,
    load_site_forcing,
    resolve_forcing_file,
)
from ecosystem_complexity.sites.paths import CONFIG_DIR, REPO_ROOT
from ecosystem_complexity.sites.soc import build_soc_prior
from ecosystem_complexity.sites.spec import (
    SiteSpec,
    discover_site_specs,
    load_site_spec,
    select_specs,
)

__all__ = [
    "CONFIG_DIR",
    "OPT_FIELDS",
    "REPO_ROOT",
    "SiteSpec",
    "build_annual_mean_forcing",
    "build_soc_prior",
    "build_state0",
    "discover_site_specs",
    "load_site_forcing",
    "load_site_spec",
    "resolve_forcing_file",
    "run_oe_canonical",
    "run_site_canonical",
    "run_sites",
    "select_specs",
    "summary_row",
]
