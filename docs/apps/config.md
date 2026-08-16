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
