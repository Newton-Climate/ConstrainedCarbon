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
