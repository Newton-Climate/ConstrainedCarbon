"""
barrow_model_israd.py — compatibility wrapper for the canonical 3-pool
Barrow OE inversion.

It preserves a compatibility-only output path, but now runs
the same all-profile ISRaD bulk-layer pool-Δ¹⁴C workflow as
``notebooks/barrow_model.py``.

Run
---
  python notebooks/barrow_model_israd.py

Output
------
  notebooks/barrow_three_pool_canonical_oe_summary_compatibility_alias.png
      8-panel OE comparison figure
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sites.barrow import run_optimal_inversion_israd, make_figure_israd

if __name__ == "__main__":
    results = run_optimal_inversion_israd()
    make_figure_israd(results)
