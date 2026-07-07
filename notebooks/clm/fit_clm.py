"""
fit_clm.py — CLM5 emulator inversion against global CESM2 fields.

This uses the global CESM2 historical files in ``data/cmip`` directly and
fits our 3-pool model to the nearest CESM2 grid cell for the four canonical
analysis sites:

  - Harvard Forest (US-Ha1)
  - Barrow, Alaska (US-A10)
  - Howland Forest (US-Ho1)
  - Eight-mile Lake (US-EML)

Output: notebooks/clm/clm_emulator_posterior.png
"""
from __future__ import annotations

import os
import sys
import time

import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_SCRIPT_ROOT, "src"))
sys.path.insert(0, os.path.join(_SCRIPT_ROOT, "notebooks"))
os.chdir(_SCRIPT_ROOT)

from ecosystem_complexity.api import ObsBlock, build_model, optimize_oe
from ecosystem_complexity.data.loaders import (
    load_barrow_alaska,
    load_eight_mile_lake,
    load_harvard_forest,
    load_howland_forest,
)
from ecosystem_complexity.data.parsers import attach_atm14C, slice_forcing
from ecosystem_complexity.data.parsers_14C import load_full_14C_record
from ecosystem_complexity.data.schemas import ObservationData
from ecosystem_complexity.state import make_initial_state

from clm.cmip_global import SITE_SPECS, get_site_spec, load_clm_targets
from sites.barrow import BARROW_ERA5_PATH, BARROW_FLUXMET_PATH
from sites.canonical import (
    BARROW_CONFIG,
    GRAVEN_PATH,
    HF_CONFIG,
    HF_HR_PATH,
    HUA_PATH,
    INTCAL_PATH,
    run_barrow_canonical,
    run_hf_canonical,
)
from sites.eight_mile_lake import EML_HH_PATH, OPT_CONFIG as EML_CONFIG, run_eml_canonical
from sites.howland_forest import HO1_DD_PATH, OPT_CONFIG as HOWLAND_CONFIG, run_howland_canonical

OUT_PATH = os.path.join(_SCRIPT_ROOT, "notebooks", "clm", "clm_emulator_posterior.png")

C_PRIMARY = "#1F3A2E"
C_CESM = "#B85042"
C_OE = "#1F3A2E"


def build_clm_obs(model, forcing, clm: dict, sigma_rel_c: float = 0.20, sigma_rel_rh: float = 0.25) -> tuple:
    """Map CESM2 pool stocks + Rh into ``ObservationData`` plus extra blocks."""
    t_steps = int(forcing.time.shape[0])
    pool_names = set(model.pool_index.pool_names)

    c_obs: dict[str, tuple[float, float]] = {}
    mapping = [
        ("soil_active", clm["cLitter"] + clm["cFast"]),
        ("soil_slow", clm["cMed"]),
        ("soil_passive", clm["cSlow"]),
    ]
    for pool_name, value in mapping:
        if pool_name not in pool_names:
            continue
        sigma = max(sigma_rel_c * value, 50.0)
        c_obs[pool_name] = (float(value), float(sigma))

    obs = ObservationData(
        time=forcing.time,
        NEE=jnp.full(t_steps, jnp.nan),
        GPP=jnp.full(t_steps, jnp.nan),
        ER=jnp.full(t_steps, jnp.nan),
        NEE_unc=jnp.full(t_steps, jnp.nan),
        delta14C_obs={},
        deltaD14C_obs={},
        C_pools_obs=c_obs,
        delta14C_resp=None,
    )

    rh_target = float(clm["rh"])
    rh_sigma = max(sigma_rel_rh * rh_target, 20.0)

    def _predict_rh(out, params):
        del params
        return jnp.array([jnp.mean(out.Rh) * 365.0])

    rh_block = ObsBlock(
        name="clm_rh_total",
        y=jnp.array([rh_target], dtype=jnp.float32),
        Se=jnp.array([rh_sigma**2], dtype=jnp.float32),
        predict=_predict_rh,
    )
    return obs, [rh_block]


def build_clm_state0(model, clm_targets: dict):
    """Initialise C pools from the CESM2 targets at the matched site cell."""
    base = make_initial_state(model.config, getattr(model, "_site_config", {}))
    c12_arr = np.zeros(len(model.pool_index), dtype=np.float32)
    c14_arr = np.zeros(len(model.pool_index), dtype=np.float32)
    fallback_d14c = {
        "soil_active": 75.0,
        "soil_slow": 0.0,
        "soil_passive": -150.0,
    }
    mapped_c = {
        "soil_active": clm_targets["cLitter"] + clm_targets["cFast"],
        "soil_slow": clm_targets["cMed"],
        "soil_passive": clm_targets["cSlow"],
    }
    r_std = 1.176e-12
    for pool_name in model.pool_index.pool_names:
        c_mean = float(mapped_c.get(pool_name, 0.0))
        d14c_val = fallback_d14c.get(pool_name, 0.0)
        idx = model.pool_index[pool_name]
        c12_arr[idx] = max(c_mean, 0.0)
        c14_arr[idx] = c12_arr[idx] * r_std * (1.0 + d14c_val / 1000.0)
    return base._replace(C12=jnp.array(c12_arr), C14=jnp.array(c14_arr))


def _attach_atm14c_and_slice(forcing_full, start_year: float):
    atm14c = load_full_14C_record(
        hua_path=HUA_PATH,
        graven_path=GRAVEN_PATH,
        intcal_path=INTCAL_PATH,
        hemisphere="NH",
        start_year=1500.0,
        end_year=2025.0,
    )
    years_daily, d14c_daily = atm14c
    forcing_full = attach_atm14C(forcing_full, d14c_daily, years_daily)
    years_all = 1970.0 + np.array(forcing_full.time) / 365.25
    start_idx = int(np.searchsorted(years_all, start_year))
    return slice_forcing(forcing_full, start_idx, len(years_all))


def _forcing_hf():
    forcing_full, _ = load_harvard_forest(
        hr_path=HF_HR_PATH,
        config=build_model(HF_CONFIG).config,
        qc_threshold=2,
        include_gpp_forcing=True,
    )
    return _attach_atm14c_and_slice(forcing_full, 1996.0)


def _forcing_barrow():
    forcing_full, _ = load_barrow_alaska(
        era5_path=BARROW_ERA5_PATH,
        fluxmet_path=BARROW_FLUXMET_PATH,
        config=build_model(BARROW_CONFIG).config,
        qc_threshold=0.0,
        include_gpp_forcing=True,
    )
    return _attach_atm14c_and_slice(forcing_full, 2011.0)


def _forcing_howland():
    forcing_full, _ = load_howland_forest(
        HO1_DD_PATH,
        config=build_model(HOWLAND_CONFIG).config,
        include_gpp_forcing=True,
    )
    return _attach_atm14c_and_slice(forcing_full, 1996.0)


def _forcing_eml():
    forcing_full, _ = load_eight_mile_lake(
        hh_path=EML_HH_PATH,
        config=build_model(EML_CONFIG).config,
        include_gpp_forcing=True,
    )
    return _attach_atm14c_and_slice(forcing_full, 2008.0)


SITE_RUNTIME = {
    "harvard_forest": {
        "config": HF_CONFIG,
        "forcing_builder": _forcing_hf,
        "canonical_runner": run_hf_canonical,
    },
    "barrow": {
        "config": BARROW_CONFIG,
        "forcing_builder": _forcing_barrow,
        "canonical_runner": run_barrow_canonical,
    },
    "howland_forest": {
        "config": HOWLAND_CONFIG,
        "forcing_builder": _forcing_howland,
        "canonical_runner": run_howland_canonical,
    },
    "eight_mile_lake": {
        "config": EML_CONFIG,
        "forcing_builder": _forcing_eml,
        "canonical_runner": run_eml_canonical,
    },
}


def fit_one_site(site_key: str, config_path: str, forcing, clm_targets: dict) -> dict:
    site = get_site_spec(site_key)
    print(f"\n[{site.label}] fitting our 3-pool model to CESM2 C + Rh…")
    model = build_model(config_path)
    state0 = build_clm_state0(model, clm_targets)
    obs, extras = build_clm_obs(model, forcing, clm_targets)

    print(f"  CESM2 cell: ({clm_targets['cell_lat']:.2f}, {clm_targets['cell_lon']:.2f})"
          f"  dist={clm_targets['dist_km']:.0f} km")
    print(f"  Targets: active={clm_targets['cLitter'] + clm_targets['cFast']:.0f}"
          f"  slow={clm_targets['cMed']:.0f}  passive={clm_targets['cSlow']:.0f}"
          f"  Rh={clm_targets['rh']:.1f}")

    t0 = time.perf_counter()
    result = optimize_oe(
        model,
        forcing,
        obs,
        state0=state0,
        fields=("log_tau",),
        extra_obs_blocks=extras,
    )
    ch = np.array(result.cost_history)
    print(f"  Done [{time.perf_counter() - t0:.1f}s]  J {ch[0]:.2f} → {ch[-1]:.2f}"
          f"  ({result.n_iter} iter, converged={result.converged})")

    tau = np.exp(np.array(result.params_opt.log_tau)) / 365.0
    s_diag = np.array(jnp.diag(result.Sx))
    sigma_log = np.sqrt(np.abs(s_diag))
    tau_lo = tau * np.exp(-sigma_log)
    tau_hi = tau * np.exp(+sigma_log)

    return {
        "site_key": site_key,
        "site_label": site.label,
        "pool_names": model.pool_index.pool_names,
        "tau": tau,
        "tau_lo": tau_lo,
        "tau_hi": tau_hi,
        "converged": result.converged,
    }


def _fetch_oe(data: dict) -> dict:
    tau = np.exp(np.array(data["params_opt"].log_tau)) / 365.0
    s_diag = np.array(jnp.diag(data["oe_result"].Sx))[: len(tau)]
    sigma_log = np.sqrt(np.abs(s_diag))
    return {
        "pool_names": data["idx"].pool_names,
        "tau": tau,
        "tau_lo": tau * np.exp(-sigma_log),
        "tau_hi": tau * np.exp(+sigma_log),
    }


def fetch_oe_posteriors() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for site_key, runtime in SITE_RUNTIME.items():
        out[site_key] = _fetch_oe(runtime["canonical_runner"]())
    return out


def make_figure(clm_posts: dict[str, dict], oe_posts: dict[str, dict], clm_targets: dict[str, dict], out_path: str) -> None:
    fig = plt.figure(figsize=(13.5, 10.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.32, left=0.07, right=0.97, top=0.90, bottom=0.10)

    for ax_i, site in enumerate(SITE_SPECS):
        ax = fig.add_subplot(gs[ax_i // 2, ax_i % 2])
        clm_post = clm_posts[site.key]
        oe_post = oe_posts[site.key]
        tgt = clm_targets[site.key]
        x = np.arange(len(clm_post["pool_names"]))
        w = 0.36

        ax.bar(
            x - w / 2,
            clm_post["tau"],
            w,
            color=C_CESM,
            alpha=0.85,
            edgecolor=C_PRIMARY,
            lw=0.5,
            label="CESM2-emulated τ ± 1σ",
        )
        ax.errorbar(
            x - w / 2,
            clm_post["tau"],
            yerr=[clm_post["tau"] - clm_post["tau_lo"], clm_post["tau_hi"] - clm_post["tau"]],
            fmt="none",
            color=C_PRIMARY,
            capsize=4,
            lw=1.4,
        )

        ax.bar(
            x + w / 2,
            oe_post["tau"],
            w,
            color=C_OE,
            alpha=0.85,
            edgecolor=C_PRIMARY,
            lw=0.5,
            label="OE posterior τ ± 1σ",
        )
        ax.errorbar(
            x + w / 2,
            oe_post["tau"],
            yerr=[oe_post["tau"] - oe_post["tau_lo"], oe_post["tau_hi"] - oe_post["tau"]],
            fmt="none",
            color=C_PRIMARY,
            capsize=4,
            lw=1.4,
        )

        for i in range(len(x)):
            if oe_post["tau"][i] > 0.0 and clm_post["tau"][i] > 0.0:
                ratio = clm_post["tau"][i] / oe_post["tau"][i]
                top = max(clm_post["tau_hi"][i], oe_post["tau_hi"][i])
                ax.text(x[i], top * 1.45, f"{ratio:.2f}×", ha="center", va="bottom", fontsize=9, color=C_PRIMARY, fontweight="bold")

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace("soil_", "") for name in clm_post["pool_names"]], fontsize=10, fontweight="bold")
        ax.set_ylabel("Turnover time τ (years, log)", fontsize=10)
        ax.set_title(
            f"{site.label} ({site.code})\ncell=({tgt['cell_lat']:.2f}, {tgt['cell_lon']:.2f})"
            f"  dist={tgt['dist_km']:.0f} km  Rh={tgt['rh']:.0f}",
            fontsize=10,
            color=C_PRIMARY,
            fontweight="bold",
            loc="left",
            pad=9,
        )
        ax.grid(axis="y", lw=0.3, alpha=0.4)
        ax.legend(fontsize=8, framealpha=0.92, loc="upper left")

    fig.suptitle(
        "CESM2 / CLM5 global-grid emulation vs 14C-constrained posterior",
        fontsize=13,
        color=C_PRIMARY,
        fontweight="bold",
    )
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"\nsaved {out_path}")


def main():
    print("═══ CLM5 emulator inversion from global CESM2 files ══════════════")

    print("Loading CESM2 targets…")
    clm_targets = {site.key: load_clm_targets(site.key) for site in SITE_SPECS}

    print("\nLoading site forcing…")
    forcing = {site_key: runtime["forcing_builder"]() for site_key, runtime in SITE_RUNTIME.items()}

    clm_posts: dict[str, dict] = {}
    for site in SITE_SPECS:
        runtime = SITE_RUNTIME[site.key]
        clm_posts[site.key] = fit_one_site(site.key, runtime["config"], forcing[site.key], clm_targets[site.key])

    print("\nRunning canonical OE inversions for reference…")
    oe_posts = fetch_oe_posteriors()

    print("\n┌─ Comparison table ─────────────────────────────────────────────────────────────┐")
    print(f"  {'Site':<10s}  {'Pool':<14s}  {'CESM2 τ (yr)':>12s}  {'14C τ (yr)':>11s}  {'CESM2/14C':>10s}")
    print("  " + "─" * 76)
    for site in SITE_SPECS:
        clm_post = clm_posts[site.key]
        oe_post = oe_posts[site.key]
        for i, pool_name in enumerate(clm_post["pool_names"]):
            ratio = clm_post["tau"][i] / oe_post["tau"][i] if oe_post["tau"][i] > 0.0 else float("nan")
            print(f"  {site.short_label:<10s}  {pool_name:<14s}  {clm_post['tau'][i]:>12.1f}  {oe_post['tau'][i]:>11.1f}  {ratio:>9.2f}×")
    print("  └─" + "─" * 74 + "┘")

    make_figure(clm_posts, oe_posts, clm_targets, OUT_PATH)


if __name__ == "__main__":
    main()
