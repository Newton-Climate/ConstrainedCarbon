"""Back-compat shim: ``ecosystem_complexity.fluxes`` was renamed to
``ecosystem_complexity.climate``.  Re-export the public symbols so legacy
notebooks (notebook_utils.py, compare_sites.py, etc.) continue to work.
"""
from ecosystem_complexity.climate import (  # noqa: F401
    f_temp,
    f_moisture,
    thawed_frac,
)
