"""Repo-root-relative paths owned by the site inversion drivers.

Only the per-site config directory lives here. Everything else the drivers read
— ISRaD tables, atmospheric ¹⁴C records, the SoilGrids export — is input data
and lives in :mod:`ecosystem_complexity.data.paths`, so that ``data`` never has
to import ``sites``.
"""
from __future__ import annotations

import os

from ecosystem_complexity.data.paths import REPO_ROOT, repo_root

CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "multisite")

__all__ = ["CONFIG_DIR", "REPO_ROOT", "repo_root"]
