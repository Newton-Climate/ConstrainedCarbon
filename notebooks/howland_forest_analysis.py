"""
howland_forest_analysis.py — Information-content analysis for Howland Forest.

This `US-Ho1` simulation uses the FLUXNET FULLSET daily package, including
observed GPP, together with nearby control ISRaD Howland records for bulk
layer 14C, density fractions, soil-respiration 14C, and soil C stocks.

Run from the repository root:
    python notebooks/howland_forest_analysis.py

Outputs
-------
  notebooks/howland_forest_canonical_information_content_analysis.png
  notebooks/howland_forest_canonical_information_content_analysis.summary.json
"""
from __future__ import annotations

import json
import os
import sys

_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NB_ROOT = os.path.join(_SCRIPT_ROOT, "notebooks")
_SRC_ROOT = os.path.join(_SCRIPT_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_SCRIPT_ROOT)

from sites.canonical import site_information_analysis, site_figure
from sites.howland_forest import run_howland_canonical, build_summary

_POOL_STYLES = {
    "soil_active": ("Active", "tab:green", "o"),
    "soil_slow": ("Slow", "tab:orange", "s"),
    "soil_passive": ("Passive", "tab:purple", "D"),
}

_TITLE = (
    "Howland Forest (US-Ho1) 3-Pool Model — "
    "Canonical OE Inversion + Information-Content Analysis"
)


if __name__ == "__main__":
    data = run_howland_canonical()
    analysis = site_information_analysis(data)
    out_path = os.path.join(_NB_ROOT, "howland_forest_canonical_information_content_analysis.png")
    site_figure(data, analysis, out_path, pool_styles=_POOL_STYLES, site_title=_TITLE)
    summary_path = out_path.replace(".png", ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(build_summary(data, analysis), fh, indent=2)
    print(f"\nSaved figure:   {out_path}")
    print(f"Saved summary:  {summary_path}")
