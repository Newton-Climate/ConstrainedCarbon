"""
harvard_optimal_model.py — 3-pool OE inversion for Harvard Forest (US-Ha1).

All data helpers, obs builders, and the full OE5 workflow live in
``sites/harvard_forest.py``.  This script is the entry point only.

Run
---
  python notebooks/harvard_optimal_model.py

Output
------
  notebooks/harvard_optimal_model.png  — 8-panel OE comparison figure
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sites.harvard_forest import run_optimal_inversion, make_figure

if __name__ == "__main__":
    results = run_optimal_inversion()
    make_figure(results)
