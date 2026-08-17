# `ecosys config`

Use these tools when you want to add a study site. They help match a flux tower to an ISRaD location and create a starting configuration file; they do not run the carbon model.

```bash
# Look for an ISRaD match for a tower
ecosys config locate --flux-tower US-Ha1 --out /tmp/harvard_sites.csv

# Make a new site config from the included template
ecosys config build \
  --selector US-Ha1 --tower-id US-Ha1 \
  --lat 42.5378 --lon -72.1715 \
  --biome "temperate deciduous forest" \
  --out configs/multisite/my_site.yaml
```

Open the new YAML file before fitting it. It records the pool structure, priors, observations, and forcing source for the analysis. `config incubation` creates configs from a prepared incubation manifest.

## Outputs and use

`locate` writes the requested candidate-match CSV; it is a geographic/data
discovery aid, not confirmation that a tower and soil profile represent the
same ecosystem. `build` and `incubation` write YAML configuration candidates.
Review pool definitions, depth mapping, priors, observation dates, forcing, and
site metadata before treating the configuration as an analysis protocol.

## Harvard Forest example

```bash
ecosys model validate configs/multisite/harvard_forest.yaml
```

The Harvard config defines one 0–1.3 m soil layer with active, slow, and passive
kinetic pools. That is a modeling choice, not a direct statement that each
depth interval contains only one pool. The later diagnostic below helps check
whether the resulting observation model behaves plausibly.

![Harvard Forest fitted diagnostic](artifacts/harvard_forest_site_diagnostics.png)
