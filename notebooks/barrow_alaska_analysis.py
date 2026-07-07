"""
barrow_alaska_analysis.py — Information-content analysis for Barrow, Alaska.

Thin wrapper: runs the canonical 3-pool OE inversion from
``sites.canonical.run_barrow_canonical`` and produces the standard 6-panel
information-content figure via ``sites.canonical.site_figure``.

Run from the repository root:
    python notebooks/barrow_alaska_analysis.py

Output
------
  notebooks/barrow_canonical_information_content_analysis.png
"""
from __future__ import annotations

import os
import sys

# ── Paths ───────────────────────────────────────────────────────────────────
_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NB_ROOT = os.path.join(_SCRIPT_ROOT, "notebooks")
_SRC_ROOT = os.path.join(_SCRIPT_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_SCRIPT_ROOT)

from sites.canonical import (
    run_barrow_canonical, site_information_analysis, site_figure,
)

# Barrow 3-pool pool styles for Δ¹⁴C overlay scatter
_POOL_STYLES = {
    "soil_active":  ("Active (0–14 cm)",   "tab:green",  "o"),
    "soil_slow":    ("Slow (14–30 cm)",    "tab:orange", "s"),
    "soil_passive": ("Passive (30–100 cm)","tab:purple", "D"),
}

_TITLE = (
    "Barrow, Alaska (US-A10) 3-Pool Permafrost Model — "
    "Canonical OE Inversion + Information-Content Analysis"
)


if __name__ == "__main__":
    data = run_barrow_canonical()
    analysis = site_information_analysis(data)
    out_path = os.path.join(_NB_ROOT, "barrow_canonical_information_content_analysis.png")
    site_figure(data, analysis, out_path,
                pool_styles=_POOL_STYLES, site_title=_TITLE)
