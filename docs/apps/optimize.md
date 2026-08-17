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

## Custom laboratory ¹⁴C data

Set `datasource.radiocarbon_manifest` in a site config to a custom YAML
manifest. The manifest points to a CSV containing bulk, density-fraction, and/or
respired-CO₂ Δ¹⁴C measurements. It takes precedence over the selected
ISRaD `observation_path`, so run the normal optimize command without an
observation-path override:

```bash
ecosys optimize configs/multisite/my_lab_site.yaml --outdir outputs/my_lab_site
```

The example files and full input contract are in
[`examples/custom_14c`](../../examples/custom_14c/README.md). Your config must
still specify `forcing_glob`, because the forward model requires daily climate
and GPP/NPP forcing. `israd_name` is optional for custom-manifest sites.
