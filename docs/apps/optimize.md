# `ecosys optimize`

Fit the configured carbon model to the observations available for one site or a collection of sites. This is normally the first analysis command to run.

```bash
# One included site
ecosys optimize configs/multisite/harvard_forest.yaml

# A named collection of configs
ecosys optimize --site-set configs/site_sets/direct_warming_network_24.yaml \
  --include-er -j 4

# Explore the pool structures declared by a config
ecosys optimize harvard_forest --sweep
```

Useful options include `--observation-path` to choose the observation mix, `--include-er` and `--include-incubation` to add optional constraints, and `--outdir` to choose a result root. `-j` sets the number of worker processes.

Configs are accepted as paths, config stems, or recognized site selectors. Use `ecosys optimize --list` to see the discovered site configs. Run `ecosys optimize --help` for the complete flag reference.
