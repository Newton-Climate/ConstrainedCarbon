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

## Outputs

Each site writes `summary.parquet`/`.csv`, `diagnostics.json`,
`config.snapshot.yaml`, and `manifest.json` under `outputs/<site-id>/warming/`.
A site set also writes `network_warming_summary.parquet`/`.csv` and, when
applicable, `by_biome_group.parquet`/`.csv` under
`outputs/<site-set>/warming/`. The response is a standardized model
counterfactual, not an observed warming response.

| Summary field | Meaning and use |
|---|---|
| `frac_c_loss` | `(baseline final C − warmed final C) / initial C`. Positive means less modeled C remains under warming. |
| `abs_c_loss_gCm2` | Baseline-final minus warmed-final system C. It depends on stock size. |
| `delta_rh_annual_mean_gCm2yr` | Cumulative excess heterotrophic respiration divided by the horizon. |
| `old_fraction_of_excess_rh` | Share of excess Rh attributed to configured slow/passive pools; a model-based attribution, not a measured age fraction. It may be undefined when excess Rh is near zero. |
| `c_initial_gCm2`, `c_base_final_gCm2`, `c_warm_final_gCm2` | Carbon-balance terms for auditing the reported loss. |

Only compare responses with the same warming increment, horizon, metric,
forcing treatment, and constraints. Do not describe the output as a forecast:
it does not by itself include changing vegetation, inputs, moisture,
disturbance, or adaptation. Biome averages summarize the selected sites; they
do not establish a biome effect without uncertainty and coverage assessment.

## Harvard Forest example

```bash
ecosys warming configs/multisite/harvard_forest.yaml \
  --horizon-years 100 --warming-delta-c 4 --outdir outputs/harvard_example
```

For this repeated-forcing, +4 °C experiment, the fitted model retains about
21,930 rather than 26,650 g C m⁻² after 100 years: a 17.9% modeled loss. Read
this as a result of this scenario and fitted model—not as an observed loss or a
forecast that includes changing vegetation, disturbance, or moisture.

![Harvard Forest warming response](artifacts/harvard_forest_warming_response.png)
