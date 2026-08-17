# `ecosys model`

Use this app before fitting a new site. It checks that the site YAML resolves,
the forcing can be loaded, and any custom laboratory ¹⁴C CSV and manifest are
valid and map onto the configured model pools.

```bash
ecosys model validate configs/multisite/my_lab_site.yaml
```

Run a forward simulation without fitting parameters:

```bash
ecosys model run configs/multisite/my_lab_site.yaml \
  --outdir outputs/my_lab_site_forward
```

`run` writes the resolved configuration, diagnostics, and `forward_output.npz`
containing time, pool carbon, pool Δ¹⁴C, NEE, GPP, ER, and heterotrophic
respiration. It begins from the configuration's initial state by default. Add
`--spinup-years N` to run a carbon-pool spinup first; this can take
substantially longer.
