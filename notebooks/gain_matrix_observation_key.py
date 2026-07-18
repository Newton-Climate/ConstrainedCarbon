"""
gain_matrix_observation_key.py — decode the gain-matrix column labels.

The 4-site gain figure (``four_site_ak_gain_figures.py``) labels its columns
with short constraint names like ``resp[1]``, ``resp[2]``, ``C:active``,
``density:slow`` … .  Those are just positional labels into the canonical OE
observation vector; on their own they do not say *which* physical measurement
each column is.

This script runs the same four canonical site inversions, walks the OE
observation vector in the exact order the gain matrix uses (see
``gain_obs_metadata``), and writes one row per scalar observation with:

  * the plotted short + descriptive labels and the full ``block[index]`` label
  * the observation block / family
  * the physical identity of the measurement (soil pool, sampling date /
    measurement year, human description)
  * the observed value and its OE error, plus the prior/posterior model values

The per-observation time index and pool column are read back out of each
ObsBlock's ``predict`` closure and the observed values are cross-checked against
the diagnostics annotations, so the decoded identity is guaranteed to line up
with the plotted column.

Run from the repository root:
    python notebooks/gain_matrix_observation_key.py
"""
from __future__ import annotations

import os
import sys

_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NB_ROOT = os.path.join(_SCRIPT_ROOT, "notebooks")
_SRC_ROOT = os.path.join(_SCRIPT_ROOT, "src")
for _p in (_SRC_ROOT, _NB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_SCRIPT_ROOT)

from four_site_ak_gain_figures import _compute_site_diagnostics  # noqa: E402
from gain_obs_metadata import build_observation_key  # noqa: E402


def main() -> None:
    export_dir = os.path.join(_NB_ROOT, "exports")
    os.makedirs(export_dir, exist_ok=True)

    from sites.canonical import run_barrow_canonical, run_hf_canonical
    from sites.eight_mile_lake import run_eml_canonical
    from sites.howland_forest import run_howland_canonical

    site_runs = [
        ("Harvard Forest", "US-Ha1", run_hf_canonical),
        ("Barrow", "US-A10", run_barrow_canonical),
        ("Howland Forest", "US-Ho1", run_howland_canonical),
        ("Eight-mile Lake", "US-EML", run_eml_canonical),
    ]
    print("Running canonical site inversions to decode gain-matrix columns…")
    site_diags = _compute_site_diagnostics(site_runs)

    df = build_observation_key(site_diags)
    out_path = os.path.join(export_dir, "gain_matrix_observation_key.csv")
    df.to_csv(out_path, index=False)
    print(f"\nsaved {out_path}  ({len(df)} scalar observations)")

    # Quick console preview of the respired-CO₂ columns (resp[1], resp[2], …).
    resp = df[df["obs_block_name"] == "resp_14C"]
    if not resp.empty:
        print("\nRespired-CO₂ Δ¹⁴C columns (resp[i] → sampling date):")
        for _, r in resp.iterrows():
            print(
                f"  {r['site']:<16s} {r['plot_label_short']:<10s} "
                f"{r['sampling_date']}  Δ¹⁴C={r['y_obs']:+7.1f}‰  "
                f"σ={r['obs_sigma']:.1f}‰"
            )


if __name__ == "__main__":
    main()
