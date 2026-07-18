"""Repo-root-relative paths to the versioned input data.

The loaders read inputs that are versioned alongside the code but *not*
packaged with it: the ISRaD tables and atmospheric ¹⁴C records under
``data/shared/``, and the exported SoilGrids table under ``notebooks/exports/``.
When this code lived in ``notebooks/sites/`` it reached those by walking up from
``__file__``; that breaks once the same code is imported from an installed
package.

``configs/multisite`` is deliberately *not* here — it locates per-site inversion
configs, which is a ``sites`` concern; see :mod:`ecosystem_complexity.sites.paths`.

``repo_root()`` therefore resolves the root by walking up from this file looking
for the marker directories, and honours ``ECOSYSTEM_COMPLEXITY_ROOT`` for
checkouts where the data lives elsewhere (or where the package is installed
non-editable and the data tree is external).
"""
from __future__ import annotations

import os

_ENV_VAR = "ECOSYSTEM_COMPLEXITY_ROOT"
# A directory is the repo root only if it has both — `configs` alone matches
# too many unrelated parents.
_MARKERS = ("configs", "data")


def repo_root() -> str:
    """Absolute path to the repo root holding ``configs/`` and ``data/``.

    Raises ``FileNotFoundError`` rather than silently returning a wrong root,
    so a misconfigured checkout fails at import of a path constant instead of
    much later as a confusing "no forcing file matching data/..." error.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        if not os.path.isdir(override):
            raise FileNotFoundError(
                f"{_ENV_VAR}={override!r} is not a directory"
            )
        return os.path.abspath(override)

    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    while True:
        if all(os.path.isdir(os.path.join(cur, m)) for m in _MARKERS):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise FileNotFoundError(
                "Could not locate the repo root (a directory containing both "
                f"{' and '.join(_MARKERS)}/) above {here}. Set {_ENV_VAR} to "
                "point at the checkout holding the config and data trees."
            )
        cur = parent


REPO_ROOT = repo_root()

ISRAD_DIR = os.path.join(REPO_ROOT, "data", "shared", "israd")
ISRAD_VERSION = "2.6.6.2024-01-25"
ISRAD_LAYER = os.path.join(ISRAD_DIR, f"ISRaD_data_flat_layer_v {ISRAD_VERSION}.csv")
ISRAD_FRACTION = os.path.join(
    ISRAD_DIR, f"ISRaD_data_flat_fraction_v {ISRAD_VERSION}.csv"
)
ISRAD_FLUX = os.path.join(ISRAD_DIR, f"ISRaD_data_flat_flux_v {ISRAD_VERSION}.csv")

_ATM_14C_DIR = os.path.join(REPO_ROOT, "data", "shared", "atm_14C")
HUA_PATH = os.path.join(_ATM_14C_DIR, "Hua_2021.csv")
GRAVEN_PATH = os.path.join(_ATM_14C_DIR, "Graven_2017.csv")
INTCAL_PATH = os.path.join(_ATM_14C_DIR, "intcal20.14c")

EXPORTS_DIR = os.path.join(REPO_ROOT, "notebooks", "exports")
SOILGRIDS_CSV = os.path.join(EXPORTS_DIR, "soilgrids_soc_pools.csv")
