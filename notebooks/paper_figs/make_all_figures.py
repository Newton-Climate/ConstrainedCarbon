from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from notebooks.paper_figs.demo_data import build_demo_inputs
    from notebooks.paper_figs.exports import reset_caption_file
    from notebooks.paper_figs.fig_01 import make_figure_01
    from notebooks.paper_figs.fig_02 import make_figure_02
    from notebooks.paper_figs.fig_03 import make_figure_03
    from notebooks.paper_figs.fig_04 import make_figure_04
    from notebooks.paper_figs.fig_05 import make_figure_05
    from notebooks.paper_figs.fig_06 import make_figure_06
    from notebooks.paper_figs.fig_07 import make_figure_07
    from notebooks.paper_figs.fig_08 import make_figure_08
    from notebooks.paper_figs.utils import close_or_show
else:
    from .demo_data import build_demo_inputs
    from .exports import reset_caption_file
    from .fig_01 import make_figure_01
    from .fig_02 import make_figure_02
    from .fig_03 import make_figure_03
    from .fig_04 import make_figure_04
    from .fig_05 import make_figure_05
    from .fig_06 import make_figure_06
    from .fig_07 import make_figure_07
    from .fig_08 import make_figure_08
    from .utils import close_or_show

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


FIGURE_BUILDERS = {
    "figure_01": {"builder": make_figure_01, "required": [], "optional": []},
    "figure_02": {"builder": make_figure_02, "required": ["observations"], "optional": []},
    "figure_03": {
        "builder": make_figure_03,
        "required": ["posterior", "information_metrics"],
        "optional": ["topology_comparison", "summary_subset"],
    },
    "figure_04": {
        "builder": make_figure_04,
        "required": ["information_metrics"],
        "optional": ["averaging_kernel_matrix"],
    },
    "figure_05": {
        "builder": make_figure_05,
        "required": ["observations", "posterior", "information_metrics", "warming_output"],
        "optional": ["summary_subset"],
    },
    "figure_06": {
        "builder": make_figure_06,
        "required": ["warming_output"],
        "optional": ["horizon_year"],
    },
    "figure_07": {"builder": make_figure_07, "required": ["cesm_comparison"], "optional": []},
    "figure_08": {"builder": make_figure_08, "required": ["pathway_information"], "optional": []},
}

CLI_TO_INPUT_KEY = {
    "posterior": "posterior",
    "observations": "observations",
    "information_metrics": "information_metrics",
    "warming_output": "warming_output",
    "cesm_output": "cesm_comparison",
    "topology_comparison": "topology_comparison",
    "averaging_kernel_matrix": "averaging_kernel_matrix",
    "pathway_information": "pathway_information",
}


def _load_manifest(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Manifest file not found: {path}")
    suffix = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as fh:
        if suffix == ".json":
            return json.load(fh) or {}
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("YAML manifest requested but PyYAML is not installed.")
            return yaml.safe_load(fh) or {}
    raise ValueError(f"Unsupported manifest extension for {path!r}. Use .json, .yaml, or .yml.")


def _normalize_figures(raw: str | None) -> list[str]:
    if not raw:
        return list(FIGURE_BUILDERS)
    out = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token.isdigit():
            token = f"figure_{int(token):02d}"
        if token not in FIGURE_BUILDERS:
            raise ValueError(f"Unknown figure identifier: {item!r}")
        out.append(token)
    if not out:
        raise ValueError("No valid figures were requested.")
    return out


def _resolve_inputs(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    inputs = dict(manifest.get("inputs", {}))
    if "cesm_output" in inputs and "cesm_comparison" not in inputs:
        inputs["cesm_comparison"] = inputs["cesm_output"]
    for arg_name, key in CLI_TO_INPUT_KEY.items():
        value = getattr(args, arg_name)
        if value is not None:
            inputs[key] = value
    return inputs


def _resolve_figure_options(
    figure_id: str,
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    figure_opts = dict(manifest.get("figures", {}).get(figure_id, {}))
    if args.summary_subset is not None and "summary_subset" in FIGURE_BUILDERS[figure_id]["optional"]:
        figure_opts["summary_subset"] = args.summary_subset
    if args.horizon_year is not None and "horizon_year" in FIGURE_BUILDERS[figure_id]["optional"]:
        figure_opts["horizon_year"] = args.horizon_year
    return figure_opts


def _build_one(
    figure_id: str,
    args: argparse.Namespace,
    inputs: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    spec = FIGURE_BUILDERS[figure_id]
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "config_path": args.config,
    }
    for key in spec["required"]:
        if key not in inputs:
            raise ValueError(
                f"{figure_id} requires input {key!r}. Supply it via CLI, manifest, or --use-demo-data."
            )
        kwargs[key] = inputs[key]
    for key in spec["optional"]:
        if key in inputs:
            kwargs[key] = inputs[key]
    kwargs.update(_resolve_figure_options(figure_id, args, manifest))
    fig, _ = spec["builder"](**kwargs)
    close_or_show(fig, show=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the manuscript figures from tidy input tables."
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="YAML/JSON manifest describing shared inputs and per-figure options.",
    )
    parser.add_argument("--posterior", nargs="+", default=None)
    parser.add_argument("--observations", nargs="+", default=None)
    parser.add_argument("--information-metrics", nargs="+", default=None)
    parser.add_argument("--warming-output", nargs="+", default=None)
    parser.add_argument("--cesm-output", nargs="+", default=None)
    parser.add_argument("--topology-comparison", nargs="+", default=None)
    parser.add_argument("--averaging-kernel-matrix", nargs="+", default=None)
    parser.add_argument("--pathway-information", nargs="+", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--figures",
        default=None,
        help="Comma-separated list like 1,2,6 or figure_01,figure_06.",
    )
    parser.add_argument(
        "--summary-subset",
        default=None,
        help="Subset to use for figures that summarize a single observation subset.",
    )
    parser.add_argument(
        "--horizon-year",
        type=int,
        default=None,
        help="Warming horizon for Figure 6.",
    )
    parser.add_argument(
        "--use-demo-data",
        action="store_true",
        help="Run the pipeline with explicitly synthetic demonstration data.",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    inputs = _resolve_inputs(args, manifest)
    if args.use_demo_data:
        inputs = {**build_demo_inputs(config_path=args.config), **inputs}

    reset_caption_file(os.path.join(args.output_dir, "figure_captions.md"))
    for figure_id in _normalize_figures(args.figures):
        _build_one(figure_id, args, inputs, manifest)


if __name__ == "__main__":
    main()
