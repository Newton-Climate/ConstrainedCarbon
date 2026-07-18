from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from notebooks.paper_figs.utils import (
        finalize_figure,
        panelize,
        setup_figure_config,
        standard_figure_parser,
    )
else:
    from .utils import finalize_figure, panelize, setup_figure_config, standard_figure_parser


def _arrow(ax, p0, p1, linestyle="-", lw=1.6):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="->", mutation_scale=12, linewidth=lw, linestyle=linestyle, color="0.2"
        )
    )


def make_figure_01(output_dir: str = "outputs", config_path: str | None = None):
    setup_figure_config(config_path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panelize(axes)

    ax = axes[0]
    ax.set_title("Model Topology")
    boxes = {
        "inputs": (0.05, 0.45, 0.18, 0.12, "Carbon\ninputs"),
        "fast": (0.33, 0.68, 0.2, 0.14, "Fast mode"),
        "intermediate": (0.33, 0.43, 0.2, 0.14, "Intermediate mode"),
        "slow": (0.33, 0.18, 0.2, 0.14, "Slow mode"),
        "rf": (0.73, 0.68, 0.2, 0.12, "Respiration"),
        "ri": (0.73, 0.43, 0.2, 0.12, "Respiration"),
        "rs": (0.73, 0.18, 0.2, 0.12, "Respiration"),
    }
    topo_rows = []
    for key, (x, y, w, h, label) in boxes.items():
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="0.2", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center")
        topo_rows.append({"node": key, "x": x, "y": y, "width": w, "height": h, "label": label})
    _arrow(ax, (0.23, 0.51), (0.33, 0.75))
    _arrow(ax, (0.23, 0.51), (0.33, 0.50))
    _arrow(ax, (0.23, 0.51), (0.33, 0.25))
    _arrow(ax, (0.53, 0.75), (0.73, 0.75))
    _arrow(ax, (0.53, 0.50), (0.73, 0.50))
    _arrow(ax, (0.53, 0.25), (0.73, 0.25))
    _arrow(ax, (0.43, 0.68), (0.43, 0.57), linestyle="--")
    _arrow(ax, (0.43, 0.43), (0.43, 0.32), linestyle="--")
    _arrow(ax, (0.43, 0.43), (0.43, 0.75), linestyle="--")
    _arrow(ax, (0.43, 0.18), (0.43, 0.50), linestyle="--")
    ax.set_axis_off()

    ax = axes[1]
    ax.set_title("Observation Operators")
    obs_boxes = {
        "stocks": (0.04, 0.76, 0.28, 0.12, "Carbon stocks"),
        "fluxes": (0.04, 0.57, 0.28, 0.12, "Flux measurements"),
        "soil14c": (0.04, 0.38, 0.28, 0.12, "Soil radiocarbon"),
        "resp14c": (0.04, 0.19, 0.28, 0.12, "Respired radiocarbon"),
        "stored": (0.54, 0.66, 0.32, 0.14, "Stored-carbon\namount"),
        "stored_age": (0.54, 0.43, 0.32, 0.14, "Stored-carbon\nage structure"),
        "resp_age": (0.54, 0.20, 0.32, 0.14, "Respiring-carbon\nage structure"),
    }
    obs_rows = []
    for key, (x, y, w, h, label) in obs_boxes.items():
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="0.2", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center")
        obs_rows.append({"node": key, "x": x, "y": y, "width": w, "height": h, "label": label})
    _arrow(ax, (0.32, 0.82), (0.54, 0.73))
    _arrow(ax, (0.32, 0.63), (0.54, 0.73))
    _arrow(ax, (0.32, 0.44), (0.54, 0.50))
    _arrow(ax, (0.32, 0.25), (0.54, 0.27))
    _arrow(ax, (0.32, 0.44), (0.54, 0.27), linestyle="--")
    _arrow(ax, (0.32, 0.63), (0.54, 0.50), linestyle="--")
    ax.set_axis_off()

    ax = axes[2]
    ax.set_title("Analysis Workflow")
    steps = [
        (0.08, 0.82, "Prior ensemble"),
        (0.08, 0.66, "Observation subsets"),
        (0.08, 0.50, "Bayesian inversion"),
        (0.08, 0.34, "Posterior turnover modes"),
        (0.08, 0.18, "Information metrics"),
        (0.56, 0.58, "Standardized\nwarming experiment"),
        (0.56, 0.30, "Projected carbon-loss\nuncertainty"),
    ]
    flow_rows = []
    for x, y, label in steps:
        ax.add_patch(Rectangle((x, y), 0.30, 0.11, facecolor="white", edgecolor="0.2", linewidth=1.2))
        ax.text(x + 0.15, y + 0.055, label, ha="center", va="center")
        flow_rows.append({"x": x, "y": y, "label": label})
    for y0, y1 in [(0.82, 0.77), (0.66, 0.61), (0.50, 0.45), (0.34, 0.29)]:
        _arrow(ax, (0.23, y0), (0.23, y1))
    _arrow(ax, (0.38, 0.395), (0.56, 0.635))
    _arrow(ax, (0.38, 0.235), (0.56, 0.355))
    _arrow(ax, (0.71, 0.58), (0.71, 0.41))
    ax.set_axis_off()

    alt_text = (
        "Figure 1 has three schematic panels. Panel A shows a three-mode soil-carbon model with carbon inputs "
        "entering fast, intermediate, and slow modes, dashed transfer arrows among modes, and separate respiratory "
        "losses from each mode. Panel B shows observation operators: carbon stocks point directly to stored-carbon "
        "amount, soil radiocarbon points directly to stored-carbon age structure, respired radiocarbon points directly "
        "to the age structure of currently respiring carbon, and flux measurements point most directly to total inputs "
        "and losses. Dashed arrows indicate indirect constraints. Panel C shows the workflow from a prior ensemble to "
        "observation subsets, Bayesian inversion, posterior turnover modes, information metrics, a standardized warming "
        "experiment, and projected carbon-loss uncertainty. The principal conclusion is that the manuscript links "
        "present-day observability to future warming-vulnerability uncertainty."
    )
    caption = (
        "Conceptual framework for the manuscript. (A) A flexible three-mode soil-carbon model receives carbon inputs, "
        "routes carbon among fast, intermediate, and slow modes, and loses carbon through respiration from each mode. "
        "(B) Observation operators show the inferential role of carbon stocks, flux measurements, soil radiocarbon, "
        "and respired radiocarbon. Solid arrows denote direct constraints and dashed arrows denote indirect constraints. "
        "(C) Analysis workflow from a prior ensemble through observation subsets, inversion, posterior turnover modes, "
        "information metrics, and a standardized warming experiment to projected carbon-loss uncertainty."
    )
    import pandas as pd

    finalize_figure(
        fig,
        "figure_01",
        output_dir,
        {
            "panel_a_topology": pd.DataFrame(topo_rows),
            "panel_b_operators": pd.DataFrame(obs_rows),
            "panel_c_workflow": pd.DataFrame(flow_rows),
        },
        alt_text,
        "Figure 1",
        caption,
    )
    return fig, axes


def main() -> None:
    parser = standard_figure_parser(__doc__ or "Figure 1")
    args = parser.parse_args()
    make_figure_01(output_dir=args.output_dir, config_path=args.config)


if __name__ == "__main__":
    main()
