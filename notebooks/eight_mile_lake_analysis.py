"""
eight_mile_lake_analysis.py — Information-content analysis for Eight-mile Lake.

This `US-EML` simulation is constrained with the nearest ISRaD EML records
at the site coordinates. EML “fraction” 14C observations are macrofossil
measurements, not density fractions, so pool constraints use a depth-based
observational mapping rather than the Harvard-style density-fraction mapping.
The optimized run combines bulk soil layer 14C, macrofossil 14C,
soil-respiration 14C, and soil C stocks.

Run from the repository root:
    python notebooks/eight_mile_lake_analysis.py

Outputs
-------
  notebooks/eight_mile_lake_canonical_information_content_analysis.png
  notebooks/eight_mile_lake_canonical_information_content_analysis.summary.json
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
from sites.eight_mile_lake import run_eml_canonical, build_summary

_POOL_STYLES = {
    "soil_active": ("Active", "tab:green", "o"),
    "soil_slow": ("Slow", "tab:orange", "s"),
    "soil_passive": ("Passive", "tab:purple", "D"),
}

_TITLE = (
    "Eight-mile Lake (US-EML) 3-Pool Model — "
    "Canonical OE Inversion + Information-Content Analysis"
)


if __name__ == "__main__":
    data = run_eml_canonical()
    analysis = site_information_analysis(data)
    out_path = os.path.join(_NB_ROOT, "eight_mile_lake_canonical_information_content_analysis.png")
    site_figure(data, analysis, out_path, pool_styles=_POOL_STYLES, site_title=_TITLE)
    summary_path = out_path.replace(".png", ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(build_summary(data, analysis), fh, indent=2)
    print(f"\nSaved figure:   {out_path}")
    print(f"Saved summary:  {summary_path}")
