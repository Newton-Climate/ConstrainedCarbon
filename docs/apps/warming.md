# `ecosys warming`

Run the configured warming experiment for one site or a site set. The command uses the warming settings in each YAML file unless you override them on the command line.

```bash
ecosys warming configs/multisite/harvard_forest.yaml

ecosys warming --site-set configs/site_sets/direct_warming_network_24.yaml \
  --include-er -j 4

ecosys warming configs/multisite/harvard_forest.yaml \
  --horizon-years 200 --warming-delta-c 6
```

Use `--metric transit` for the transit-time variant. The `warming:` block in a site config can set the horizon, temperature change, metric, and optional constraints. See `ecosys warming --help` for all overrides.
