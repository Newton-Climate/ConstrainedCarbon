"""MCMC / Gaussian-posterior sampling pipeline.

Package-level constants are the CLI defaults that the ``ecosys mcmc``
dispatcher can override from a per-site YAML ``mcmc:`` block or a CLI
flag. Downstream modules read them via attribute access on this package
(``from ecosystem_complexity import mcmc; mcmc.RNG_SEED``) so that
parent-process overrides propagate to the pipeline entry point.
"""
from __future__ import annotations

RNG_SEED = 7
POSTERIOR_DRAW_COUNT = 64
PRIOR_DRAW_COUNT = 64
MC_ITERATIONS = 2000
NULL_ITERATIONS = 1000
WARMING_HORIZON_YEARS = 100.0
WARMING_DELTA_C = 4.0
OLD_POOLS: tuple[str, ...] = ("soil_slow", "soil_passive")
