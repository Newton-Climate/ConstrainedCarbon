# Outputs and scientific interpretation

This page is the output contract for `ecosys` workflows. Read it before
combining runs: a numerical result is conditional on the resolved config,
observation set, forcing, and command options recorded with it.

## Run-directory contract

Contract-aware commands write below:

```
outputs/<site-id-or-site-set>/<command>/<optional-subcommand>/
```

Passing `--outdir ROOT` changes the root to `ROOT`; it does not create a
timestamped run. A rerun with the same destination overwrites same-named
artifacts. Copy or choose a distinct output root for results that must be kept
separate.

Every completed contract-aware run has `manifest.json` (contract version,
software version and Git revision, start/end times, inputs, and the authoritative
relative file list). Site runs also have `config.snapshot.yaml`, the fully
resolved model configuration. `logs/run.log` is present for commands that
attach the shared logger. Treat a missing `manifest.json` as an incomplete or
failed run, and read its `outputs` list rather than assuming that every
possible payload exists. `*.parquet` is the analysis-ready table; its small
`*.csv` mirror is for inspection. `*.npz` preserves named NumPy arrays and is
not a stable columnar table.

| Workflow | Contract payloads | Primary scientific use |
|---|---|---|
| `optimize` | `posterior`, `summary`, `diagnostics`, raw `posterior.npz` | fitted turnover and fit checks |
| `warming` | `summary`, `diagnostics`; site-set `network_warming_summary`, `by_biome_group` | standardized counterfactual response |
| `information shapley` | `shapley_by_parameter`, `metrics`, `matrices.npz`; site-set network and biome tables | which observation families identify parameters |
| `model run` | `forward_output.npz`, `diagnostics` | inspect a prescribed forward simulation |
| `mcmc` | manifest plus the Figure-10 CSV/figure bundle written by the sampler | propagated cross-site uncertainty |

`fetch` writes a contract manifest for `flux`, `fluxcom`, and `clm`, while the
downloaded forcing remains under `data/`. `config` and `model validate` print
validation results and do not make a scientific-result directory. `analyze`
and `report` currently forward to legacy modules: their output paths and file
names are set by their flags and are **not** yet covered by this contract.

## What the core outputs mean

### Fitted model: `optimize`

`posterior.parquet` has one row per fitted pool turnover parameter.
`value_tau_days` and `value_tau_years` are the fitted e-folding turnover times
for that configured pool; they are model parameters, not direct radiocarbon
ages and not necessarily mean transit times through the full network.
`posterior.npz` additionally contains `log_f_transfer`, the fitted transfer
logits needed to reconstruct routing between pools.

`summary.parquet` is one row per site. Its `tau_<pool>_yr` columns are the same
fitted turnover times; `SOC_gCm2` and `mean_GPP_gCm2yr` provide the fitted
system context; and the `n_*` columns say which constraint types were actually
used. `J0` and `J_final` are the initial and final weighted least-squares
objective values. A lower `J_final` shows improvement over the starting point,
not proof that the model is adequate or that sites are comparable.

Check `diagnostics.json` before interpretation: `converged` should be true,
`n_iter` should be plausible, and the `n_*` counts and `soc_source` must match
the intended evidence. In particular, a model-derived SOC source is not an
independent stock constraint. Compare turnover estimates only among runs with
compatible pool definitions, forcing, priors, and observation choices; use the
snapshot to establish that compatibility.

### Warming response: `warming`

The command repeats baseline forcing for the requested horizon and compares it
with the same forcing warmed by `warming_delta_c`. Thus all warming metrics are
**model counterfactuals under that scenario**, not observations and not a
forecast including changing vegetation, inputs, moisture, disturbance, or
adaptation.

| Column | Meaning | Reading it scientifically |
|---|---|---|
| `frac_c_loss` | `(baseline final C − warmed final C) / initial C` | Positive values mean less modeled C remains under warming. Compare only when horizon and warming increment match. |
| `abs_c_loss_gCm2` | Baseline-final minus warmed-final total C | Absolute pool-system loss; it depends on initial stock size. |
| `delta_rh_annual_mean_gCm2yr` | Cumulative excess heterotrophic respiration divided by horizon | Mean annual excess decomposition over the experiment. |
| `old_fraction_of_excess_rh` | Fraction of excess Rh attributed to configured slow/passive pools | A source attribution within this model, not a direct measured age fraction. It can be undefined when total excess Rh is near zero. |

`c_initial_gCm2`, `c_base_final_gCm2`, and `c_warm_final_gCm2` make the carbon
balance auditable. Do not call a difference across sites a biome effect without
quantifying uncertainty and accounting for sparse, non-random site coverage.

### Information diagnostics: `information shapley`

`shapley_by_parameter.parquet` attributes degrees of freedom for signal (DFS)
to each observation family and fitted parameter. A larger `shapley_dfs` means
that family makes a larger marginal contribution to local parameter resolution
across all inclusion orders; it is not a causal contribution and does not mean
the observation is more accurate. Its rows include `n_obs_family`, which must
be considered alongside the attribution.

`metrics.parquet` contains `dfs_total_subset` and per-parameter `dfs_*` values.
DFS near zero means a parameter remains prior-dominated locally; values nearer
one indicate stronger local resolution. Values should not be treated as a
universal threshold, and correlations between DFS and an ecological outcome
can reflect observation design. `matrices.npz` is for technical audit:
`averaging_kernel` is the local sensitivity of retrieved to true state, and
its trace is DFS; `gain_matrix` maps observation perturbations into the
retrieved state. These are local, linearized OE diagnostics around the fit.

### Forward model: `model run`

`forward_output.npz` uses daily rows. `C12` and `delta14C` have shape
`(day, pool)` in configured pool order (listed in `diagnostics.json`); `NEE`,
`GPP`, `ER`, and `Rh` are daily modeled flux series. This is a simulation from
the configured/default initial state (or the requested spinup), not a fitted
posterior unless you separately supplied fitted parameters through the model
workflow. Check `spinup_years`, forcing provenance, and the final total C in
the manifest and diagnostics before using it for a mechanism claim.

### Cross-site uncertainty: `mcmc`

The MCMC bundle contains draw-level tables, regression summaries, leave-one-out
results, predictor comparisons, a prior-driven structural null, and figures.
Use intervals from `posterior_regression_summary.csv`, not only median slopes
or correlations. A relationship is more robust when its interval is stable in
`leave_one_out.csv` and remains distinct from the structural-null distribution.
It remains conditional on the site set, priors, fitted-model approximation, and
shared warming experiment; it does not establish a causal cross-ecosystem
mechanism.

## Minimal reporting checklist

For each reported result, state the config/site set, Git revision from the
manifest, observation families and optional constraints, forcing source,
warming horizon and increment when applicable, convergence status, and whether
the quantity is a fitted parameter, a diagnostic, or a counterfactual model
projection. Report uncertainty for cross-site claims and keep raw tables,
manifest, and config snapshot together with any figure.
