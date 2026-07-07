"""
barrow_model.py — canonical 3-pool OE inversion for Barrow, Alaska (US-A10).

Pool Δ¹⁴C comes from ISRaD bulk soil layers (Vaughn 2018 + Nave 2021).
All data helpers, obs builders, and the full OE5 workflow live in
``sites/barrow.py``.  This script is the entry point only.

Run
---
  python notebooks/barrow_model.py

Output
------
  notebooks/barrow_three_pool_canonical_oe_summary.png  — 8-panel OE comparison figure
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sites.barrow import run_optimal_inversion, make_figure

if __name__ == "__main__":
    results = run_optimal_inversion()
    make_figure(results)
