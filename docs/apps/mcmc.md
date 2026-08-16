# `ecosys mcmc`

MCMC / Gaussian-posterior sampling driver. Draws posterior + prior samples for
each site in the network, propagates each draw through the standardized
`+4 °C / 100-year` warming experiment, and rolls the site-level draws up into
cross-site regressions and a prior-structural null test.

Saved MCMC chains are reused for the four sites where they exist
(`US-Ha1`, `US-A10`, `US-Ho1`, `US-EML`); every other site draws Gaussian
samples from its OE posterior mean and covariance.

## Synopsis

```bash
ecosys mcmc [sites ...] [options]
```

`sites` is the anchor config whose `mcmc:` block sets defaults. When
`--site-set` is passed the anchor becomes the site-set's first entry.

## Common invocations

Sample defaults for one anchor site, letting its `mcmc:` block drive the seed
/ draw counts:

```bash
ecosys mcmc configs/multisite/harvard_forest.yaml
```

Run the full 41-site network:

```bash
ecosys mcmc --site-set configs/site_sets/full_network_41.yaml -j 8
```

Cheap smoke test — tiny counts, purely to prove the call graph:

```bash
ecosys mcmc harvard_forest \
    --posterior-draws 4 --prior-draws 4 \
    --mc-iterations 50 --null-iterations 20
```

## Flags

| Flag | Purpose |
|---|---|
| `sites` | anchor site config path (defaults from its `mcmc:` block) |
| `--site-set YAML` | use the site-set's first entry as the anchor config |
| `--outdir OUTDIR` | root under which outputs land (default `./outputs/`) |
| `--rng-seed N` | override `mcmc.rng_seed` |
| `--posterior-draws N` | override `mcmc.posterior_draw_count` |
| `--prior-draws N` | override `mcmc.prior_draw_count` |
| `--mc-iterations N` | cross-site iterations for the regression rollups |
| `--null-iterations N` | prior-structural null iterations |
| `--workers N` | per-site worker process count |
| `--network-summary PATH` | override the network-inversion summary CSV the site list is derived from |
| `--warming-summary PATH` | override the paired warming-vulnerability CSV |
| `--new-sites A B ...` | extra expansion-site tables to fold in |

## Outputs

Runs land under `./outputs/{site_id or site_set_name}/mcmc/`.

```
manifest.json
config.snapshot.yaml
csv/figure_10/
  posterior_site_metrics.csv           per-site posterior medians + 95% intervals
  posterior_site_draws.csv             every draw (for regression rollups)
  prior_null_draws.csv                 prior draws used in the null test
  posterior_regression_samples.csv     resampled cross-site slopes / correlations
  posterior_regression_summary.csv     medians + 95% intervals per relationship
  leave_one_out.csv                    leave-one-site-out robustness
  predictor_comparison.csv             turnover-sep vs DFS vs bulk-resp-offset
  structural_null_samples.csv          Pearson r + Spearman rho under the prior
  structural_null_summary.csv          observed vs null percentile / empirical p
  predicted_percentiles.csv            p20/p80 turnover-sep -> vulnerability
site_cache/{site}.csv                  per-site draw cache (skipped on re-run)
figures/                               Figure 10 + structural-null supplement
README.md                              methods / main-results summary
logs/run.log
```

The `site_cache/` CSVs are read on re-run; delete a site's cache file to
force it to be re-sampled.

## YAML block consulted

```yaml
mcmc:
  rng_seed: 7
  mc_iterations: 2000
  null_iterations: 500
  posterior_draw_count: 200
  prior_draw_count: 200
  old_pools: [soil_slow, soil_passive]
  warming_horizon_years: 100
  warming_delta_c: 4.0
```

Any of these can be overridden by the corresponding CLI flag. `old_pools`
names the pools whose Rh response defines the "old-fraction of excess Rh"
metric.

## Notes

- Constant overrides applied by the dispatcher (via `setattr` on the
  `ecosystem_complexity.mcmc` package) take effect in the parent process. On
  macOS the process-pool workers use `spawn`, so subprocess reads see the
  source-code defaults — configure via the YAML `mcmc:` block for values that
  need to reach every worker.

## Related verbs

- [`optimize`](optimize.md) / [`warming`](warming.md) — write the inputs the site list is drawn from
- [`information`](information.md) — deterministic OE analogues of the metrics rolled up here
