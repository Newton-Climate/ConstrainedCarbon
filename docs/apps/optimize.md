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

## Outputs

One site writes to `outputs/<site-id>/optimize/`; a site set additionally
writes `outputs/<site-set>/optimize/network_summary.parquet` (and CSV mirror).
Each site directory contains `manifest.json`, `config.snapshot.yaml`,
`posterior.parquet`, `posterior.npz`, `summary.parquet`, and `diagnostics.json`.

| File / field | Meaning and use |
|---|---|
| `posterior.parquet` | One row per fitted pool. `value_tau_days` and `value_tau_years` are fitted e-folding turnover times—not direct radiocarbon ages or whole-system mean transit times. |
| `posterior.npz` | Raw fitted `log_tau` and transfer logits. Use this for downstream reconstruction, not as a human-readable results table. |
| `summary.parquet` | One site-level row: fitted `tau_<pool>_yr`, modeled SOC and GPP context, constraint counts (`n_*`), final fit status, and objective values. `network_summary` has the corresponding rows for a site set. |
| `diagnostics.json` | `converged`, iterations, objective values, actually used constraint counts, and SOC source. This is the first file to check. |
| `config.snapshot.yaml` and `manifest.json` | Exact resolved configuration and provenance—including Git revision and command inputs. Keep them with every result or figure. |

`J0` and `J_final` are initial and final weighted least-squares objectives; a
smaller final value shows that optimization improved the fit from its starting
point, not that the model is adequate. Interpret turnover estimates only if
`converged` is true and the diagnostic constraint counts and `soc_source` match
the intended evidence. Compare sites only when pool structure, priors,
observation mix, forcing, and options agree. A model-derived SOC value is not
an independent stock constraint.

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
