from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from notebooks.paper_figs.utils import (
        bootstrap_mean_ci,
        coerce_table,
        finalize_figure,
        maybe_add_zero_line,
        panelize,
        setup_figure_config,
        standard_figure_parser,
    )
    from notebooks.paper_figs.validation import require_columns
else:
    from .utils import (
        bootstrap_mean_ci,
        coerce_table,
        finalize_figure,
        maybe_add_zero_line,
        panelize,
        setup_figure_config,
        standard_figure_parser,
    )
    from .validation import require_columns


OBS_TYPE_LABELS = {
    "bulk_soil": "Bulk soil",
    "depth_resolved_soil": "Depth-resolved soil",
    "fraction_specific_soil": "Fraction-specific soil",
    "respired_carbon": "Respired carbon",
}


def _summary(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_columns(
        obs,
        ["ecosystem", "observation_type", "delta14c_permil"],
        "observation table",
    )
    detail = obs.copy()
    rows = []
    for (ecosystem, obs_type), grp in detail.groupby(["ecosystem", "observation_type"], sort=False):
        mean_v, lo, hi = bootstrap_mean_ci(grp["delta14c_permil"].to_numpy(dtype=float))
        rows.append(
            {
                "ecosystem": ecosystem,
                "observation_type": obs_type,
                "mean_delta14c_permil": mean_v,
                "ci95_low_permil": lo,
                "ci95_high_permil": hi,
                "n_obs": int(grp.shape[0]),
            }
        )
    summary = pd.DataFrame(rows)
    stored_pref = ["bulk_soil", "depth_resolved_soil", "fraction_specific_soil"]
    stored_rows = []
    for ecosystem, grp in summary.groupby("ecosystem", sort=False):
        chosen = None
        for obs_type in stored_pref:
            sub = grp[grp["observation_type"] == obs_type]
            if not sub.empty:
                chosen = sub.iloc[0]
                break
        if chosen is None:
            continue
        stored_rows.append(
            {
                "ecosystem": ecosystem,
                "stored_observation_type": chosen["observation_type"],
                "stored_mean": chosen["mean_delta14c_permil"],
                "stored_ci95_low": chosen["ci95_low_permil"],
                "stored_ci95_high": chosen["ci95_high_permil"],
                "stored_n": chosen["n_obs"],
            }
        )
    bulk = pd.DataFrame(stored_rows)
    resp = summary[summary["observation_type"] == "respired_carbon"].rename(
        columns={
            "mean_delta14c_permil": "respired_mean",
            "ci95_low_permil": "respired_ci95_low",
            "ci95_high_permil": "respired_ci95_high",
            "n_obs": "respired_n",
        }
    )
    merged = bulk.merge(resp, on="ecosystem", how="inner")
    if merged.empty:
        raise ValueError(
            "Figure 2 requires respired_carbon observations and at least one stored-soil "
            "observation type among bulk_soil, depth_resolved_soil, or fraction_specific_soil."
        )
    merged["delta14c_offset"] = merged["respired_mean"] - merged["stored_mean"]
    merged["offset_ci95_low"] = merged["respired_ci95_low"] - merged["stored_ci95_high"]
    merged["offset_ci95_high"] = merged["respired_ci95_high"] - merged["stored_ci95_low"]
    availability = (
        detail.groupby(["ecosystem", "observation_type"]).size().rename("n_obs").reset_index()
    )
    return detail, summary, merged, availability


def make_figure_02(
    observations: str | pd.DataFrame,
    output_dir: str = "outputs",
    config_path: str | None = None,
):
    cfg = setup_figure_config(config_path)
    obs = coerce_table(observations, "observations")
    assert obs is not None
    detail, summary, offsets, availability = _summary(obs)

    ecosystems = [e for e in cfg.ecosystem_order if e in detail["ecosystem"].unique()]
    if not ecosystems:
        ecosystems = list(detail["ecosystem"].drop_duplicates())
    ypos = np.arange(len(ecosystems))

    fig = plt.figure(figsize=(16, 9))
    axes = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.35, 1.0, 1.05]})
    panelize(axes)

    ax = axes[0]
    stored = offsets.set_index("ecosystem").loc[ecosystems]
    ax.hlines(ypos, stored["stored_ci95_low"], stored["stored_ci95_high"], color="0.35", linewidth=1.3)
    ax.scatter(stored["stored_mean"], ypos, marker="s", s=55, facecolors="white", edgecolors="0.1", label="Stored soil")
    ax.hlines(ypos, stored["respired_ci95_low"], stored["respired_ci95_high"], color="0.35", linewidth=1.3)
    ax.scatter(stored["respired_mean"], ypos, marker="o", s=50, facecolors="0.2", edgecolors="0.1", label="Respired")
    for i, eco in enumerate(ecosystems):
        ax.plot([stored.loc[eco, "stored_mean"], stored.loc[eco, "respired_mean"]], [i, i], color="0.7", lw=0.8)
    if detail.shape[0] > 0:
        bulk_pts = detail[detail["observation_type"] == "bulk_soil"]
        resp_pts = detail[detail["observation_type"] == "respired_carbon"]
        for i, eco in enumerate(ecosystems):
            bv = bulk_pts[bulk_pts["ecosystem"] == eco]["delta14c_permil"].to_numpy(dtype=float)
            rv = resp_pts[resp_pts["ecosystem"] == eco]["delta14c_permil"].to_numpy(dtype=float)
            if bv.size:
                ax.scatter(bv, np.full(bv.size, i) - 0.08, marker="|", s=55, color="0.55", alpha=0.35)
            if rv.size:
                ax.scatter(rv, np.full(rv.size, i) + 0.08, marker="_", s=55, color="0.4", alpha=0.35)
    maybe_add_zero_line(ax)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ecosystems)
    ax.set_xlabel("Δ$^{14}$C (‰)")
    ax.set_title("Stored-Soil and Respired Δ$^{14}$C by Ecosystem")
    ax.invert_yaxis()
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.hlines(ypos, stored["offset_ci95_low"], stored["offset_ci95_high"], color="0.35", linewidth=1.3)
    ax.scatter(stored["delta14c_offset"], ypos, marker="D", s=46, facecolors="white", edgecolors="0.1")
    maybe_add_zero_line(ax)
    ax.set_yticks(ypos)
    ax.set_yticklabels([])
    ax.set_xlabel("Respired Δ$^{14}$C - stored Δ$^{14}$C (‰)")
    ax.set_title("Stored-Respired Offset")
    ax.invert_yaxis()

    ax = axes[2]
    avail = availability.copy()
    mat = []
    cols = list(OBS_TYPE_LABELS.keys())
    for eco in ecosystems:
        row = []
        for obs_type in cols:
            sub = avail[(avail["ecosystem"] == eco) & (avail["observation_type"] == obs_type)]
            row.append(int(sub["n_obs"].iloc[0]) if not sub.empty else 0)
        mat.append(row)
    arr = np.array(mat, dtype=float)
    im = ax.imshow(arr, cmap="Greys", aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([OBS_TYPE_LABELS[c] for c in cols], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ecosystems)))
    ax.set_yticklabels(ecosystems)
    ax.set_title("Observation Availability")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{int(arr[i, j])}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Sample size")

    top_offset = stored["delta14c_offset"].sort_values(ascending=False)
    alt_text = (
        f"Figure 2 has three panels. Panel A compares stored-soil and respired radiocarbon for {len(ecosystems)} ecosystems. "
        f"The largest positive stored-respired offset occurs at {top_offset.index[0]} at about {top_offset.iloc[0]:.1f} per mil, "
        f"while the smallest occurs at {top_offset.index[-1]} at about {top_offset.iloc[-1]:.1f} per mil. "
        "Panel B shows the offset defined as respired minus stored Δ14C with a zero reference line and 95 percent bootstrap intervals. "
        "Panel C shows the availability matrix with printed sample sizes for bulk soil, depth-resolved soil, fraction-specific soil, and respired carbon. "
        "The principal conclusion is that the stored-respired radiocarbon offset is ecosystem-dependent and the data coverage differs strongly among observation types."
    )
    caption = (
        "Stored-soil and respired radiocarbon across ecosystems. (A) Ecosystem-level means and 95% bootstrap intervals for the preferred stored-soil "
        "observation type available in each ecosystem and for respired "
        "Δ14C, with individual observations shown lightly behind the summaries where available. (B) The stored-respired radiocarbon offset, "
        "defined as respired Δ14C minus stored Δ14C, with 95% bootstrap intervals and a zero reference line. (C) Observation availability by "
        "ecosystem and observation type, with sample sizes printed in each cell. Bootstrap intervals summarize uncertainty in the ecosystem-level means."
    )
    finalize_figure(
        fig,
        "figure_02",
        output_dir,
        {
            "observations_detail": detail,
            "ecosystem_observation_summary": summary,
            "ecosystem_offset_summary": stored.reset_index(),
            "availability_matrix": availability,
        },
        alt_text,
        "Figure 2",
        caption,
    )
    return fig, axes


def main() -> None:
    parser = standard_figure_parser(__doc__ or "Figure 2")
    parser.add_argument("--observations", required=True)
    args = parser.parse_args()
    make_figure_02(args.observations, output_dir=args.output_dir, config_path=args.config)


if __name__ == "__main__":
    main()
