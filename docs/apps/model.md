# `ecosys model`

Use this app before fitting a new site. It checks that the site YAML resolves,
the forcing can be loaded, and any custom laboratory ¹⁴C CSV and manifest are
valid and map onto the configured model pools.

```bash
ecosys model validate configs/multisite/my_lab_site.yaml
```

For a custom-data site, a successful validation should report the forcing date
range, configured pool names, and the number of custom ¹⁴C blocks and
respiration dates it found. Fix validation errors before running an inversion.

Run a forward simulation without fitting parameters:

```bash
ecosys model run configs/multisite/my_lab_site.yaml \
  --outdir outputs/my_lab_site_forward
```

For a first forward run, use the default initial state. To inspect a
quasi-steady carbon-pool initialization before simulating, request a bounded
spinup:

```bash
ecosys model run configs/multisite/my_lab_site.yaml \
  --spinup-years 500 \
  --outdir outputs/my_lab_site_spinup
```

`run` writes the resolved configuration, diagnostics, and `forward_output.npz`
containing time, pool carbon, pool Δ¹⁴C, NEE, GPP, ER, and heterotrophic
respiration. It begins from the configuration's initial state by default. Add
`--spinup-years N` to run a carbon-pool spinup first; this can take
substantially longer.

Inspect the saved arrays with NumPy:

```python
import numpy as np

result = np.load("outputs/my_lab_site_forward/forward_output.npz")
print(result["C12"].shape)       # days × pools
print(result["delta14C"].shape)  # days × pools
print(result["Rh"][-1])          # final-day heterotrophic respiration
```
