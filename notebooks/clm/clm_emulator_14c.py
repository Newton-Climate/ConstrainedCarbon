"""
clm_emulator_14c.py — Compare CLM-emulated τ against observed respired Δ14C.

For each of the four canonical analysis sites, fit our model to CESM2 C + Rh
only, run the resulting τ vector forward, and compare that respired Δ14C
trajectory against the observed respired Δ14C and the fully 14C-constrained OE
posterior.

Output: notebooks/clm/clm_emulator_14c_mismatch.png
"""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_SCRIPT_ROOT, "src"))
sys.path.insert(0, os.path.join(_SCRIPT_ROOT, "notebooks"))
os.chdir(_SCRIPT_ROOT)

from ecosystem_complexity.analysis import compute_resp_delta14C
from ecosystem_complexity.api import build_model, run_model

from clm.cmip_global import SITE_SPECS, load_clm_targets
from clm.fit_clm import SITE_RUNTIME, fit_one_site

OUT = os.path.join(_SCRIPT_ROOT, "notebooks", "clm", "clm_emulator_14c_mismatch.png")

C_PRIMARY = "#1F3A2E"
C_CESM = "#B85042"
C_OE = "#1F3A2E"
C_OBS = "#D4A574"


def _replace_log_tau(params, new_log_tau):
    return params._replace(log_tau=new_log_tau)


def _stats(sim_curve, obs_arr):
    mask = np.isfinite(obs_arr)
    if not mask.any():
        return float("nan"), float("nan"), mask
    resid = sim_curve[mask] - obs_arr[mask]
    return float(np.sqrt(np.mean(resid**2))), float(np.mean(resid)), mask


def make_figure(site_packages: list[dict], out_path: str) -> None:
    fig = plt.figure(figsize=(13.5, 10.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.24, hspace=0.28, left=0.07, right=0.97, top=0.90, bottom=0.10)

    for ax_i, package in enumerate(site_packages):
        ax = fig.add_subplot(gs[ax_i // 2, ax_i % 2])
        ax.axhline(0.0, lw=0.4, color="0.4", linestyle=":")

        mask = package["mask"]
        if mask.any():
            ax.scatter(package["time"][mask], package["obs"][mask], s=42, color=C_OBS, edgecolors=C_PRIMARY, linewidth=0.8, zorder=4, label="Observed")

        ax.plot(
            package["time"],
            package["sim_oe"],
            color=C_OE,
            lw=1.8,
            zorder=3,
            label=f"OE posterior\nRMSE {package['rmse_oe']:.0f} ‰  bias {package['bias_oe']:+.0f} ‰",
        )
        ax.plot(
            package["time"],
            package["sim_clm"],
            color=C_CESM,
            lw=1.8,
            linestyle="--",
            zorder=2,
            label=f"CESM2-emulated τ\nRMSE {package['rmse_clm']:.0f} ‰  bias {package['bias_clm']:+.0f} ‰",
        )

        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("Respired CO₂ Δ14C (‰)", fontsize=10)
        ax.set_title(
            f"{package['label']} ({package['code']})",
            fontsize=11,
            color=C_PRIMARY,
            fontweight="bold",
            loc="left",
            pad=10,
        )
        ax.grid(lw=0.3, alpha=0.35)
        ax.legend(fontsize=8, framealpha=0.92, loc="upper right", handlelength=2.0)

    fig.suptitle(
        "CESM2 C+Rh fits systematically mis-fit observed respired Δ14C",
        fontsize=13,
        color=C_PRIMARY,
        fontweight="bold",
    )
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"\nsaved {out_path}")


def main():
    print("Running CESM2-emulator fits and 14C mismatch diagnostics…")
    site_packages: list[dict] = []

    print("Loading CESM2 targets and forcing…")
    clm_targets = {site.key: load_clm_targets(site.key) for site in SITE_SPECS}
    forcing = {site.key: SITE_RUNTIME[site.key]["forcing_builder"]() for site in SITE_SPECS}

    for site in SITE_SPECS:
        runtime = SITE_RUNTIME[site.key]
        model = build_model(runtime["config"])
        clm_post = fit_one_site(site.key, runtime["config"], forcing[site.key], clm_targets[site.key])

        print(f"\nRunning 14C-constrained reference inversion for {site.label}…")
        oe = runtime["canonical_runner"]()

        params_oe = oe["params_opt"]
        params_clm = _replace_log_tau(params_oe, jnp.log(jnp.array(clm_post["tau"] * 365.0)))

        out_oe = oe["out_opt"]
        resp_oe = np.array(compute_resp_delta14C(out_oe, params_oe))

        out_clm = run_model(model, oe["forcing"], state0=oe["state_at_map"], params=params_clm)
        jax.block_until_ready(out_clm.delta14C)
        resp_clm = np.array(compute_resp_delta14C(out_clm, params_clm))

        obs = np.array(oe["delta14C_resp"])
        years = 1970.0 + np.array(oe["forcing"].time) / 365.25

        rmse_oe, bias_oe, mask = _stats(resp_oe, obs)
        rmse_clm, bias_clm, _ = _stats(resp_clm, obs)

        site_packages.append(
            {
                "label": site.label,
                "code": site.code,
                "time": years,
                "obs": obs,
                "mask": mask,
                "sim_oe": resp_oe,
                "sim_clm": resp_clm,
                "rmse_oe": rmse_oe,
                "bias_oe": bias_oe,
                "rmse_clm": rmse_clm,
                "bias_clm": bias_clm,
            }
        )

    print("\n┌─ Respired Δ14C fit vs observations ──────────────────────────────────────┐")
    print(f"  {'Site':<10s}  {'Model':<18s}  {'RMSE (‰)':>9s}  {'bias (‰)':>9s}  {'n_obs':>6s}")
    print("  " + "─" * 74)
    for package in site_packages:
        n_obs = int(package["mask"].sum())
        print(f"  {package['code']:<10s}  {'OE posterior':<18s}  {package['rmse_oe']:>9.1f}  {package['bias_oe']:+9.1f}  {n_obs:>6d}")
        print(f"  {package['code']:<10s}  {'CESM2-emulated τ':<18s}  {package['rmse_clm']:>9.1f}  {package['bias_clm']:+9.1f}  {n_obs:>6d}")
    print("  └─" + "─" * 72 + "┘")

    make_figure(site_packages, OUT)


if __name__ == "__main__":
    main()
