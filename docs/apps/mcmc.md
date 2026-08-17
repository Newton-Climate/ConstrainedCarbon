# `ecosys mcmc`

Draw from posterior and prior uncertainty for a site or site set, then use those draws in the repository's cross-site comparison workflows. Where saved chains are available, the command can reuse them; otherwise it uses the available Optimal Estimation result.

```bash
ecosys mcmc configs/multisite/harvard_forest.yaml

ecosys mcmc --site-set configs/site_sets/direct_warming_network_24.yaml --workers 4

# Small check before a larger run
ecosys mcmc harvard_forest --posterior-draws 4 --prior-draws 4 \
  --mc-iterations 50 --null-iterations 20
```

Draw counts, seeds, and iteration counts can be supplied in the command or in the config's `mcmc:` block. See `ecosys mcmc --help` before starting a large run.

## Outputs

The run manifest is written under `outputs/<name>/mcmc/`. The sampler then
writes its Figure-10 CSV, figure, and README bundle below that directory;
consult its generated README for the exact files. The primary tables under
`csv/figure_10/` are `posterior_site_metrics.csv`, draw-level posterior and
prior-null tables, `posterior_regression_summary.csv`, `leave_one_out.csv`,
`predictor_comparison.csv`, and structural-null samples and summary.

Use intervals from `posterior_regression_summary.csv`, not only median slopes
or correlations. A cross-site relationship is more credible when its interval
and sign are stable in `leave_one_out.csv` and it is distinct from the
prior-driven structural-null distribution. It remains conditional on the
selected site set, priors, fitted-model approximation, and shared warming
experiment; it does not establish a causal cross-ecosystem mechanism.
