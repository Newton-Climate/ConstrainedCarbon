# Custom data inputs and fit constraints

This guide describes what can be supplied by a user, how each measurement is
used in a fit, and which constraints still come from the existing tower,
ISRaD, or SoilGrids paths.

## Start with a site config

Copy a working configuration such as
[`configs/multisite/harvard_forest.yaml`](../configs/multisite/harvard_forest.yaml).
Keep its `model`, `external_inputs`, `parameters`, and `inversion` blocks, then
replace the site metadata and datasource settings. A custom laboratory site
needs a daily forcing input and a radiocarbon manifest:

```yaml
site:
  id: my-lab-site
  name: My laboratory site
  lat: 45.0
  lon: -120.0
  biome: temperate conifer forest

datasource:
  forcing_glob: my_tower_data
  forcing_kind: daily
  radiocarbon_manifest: ../../data/custom/my_lab_14c.yaml
```

`forcing_glob` is resolved relative to `data/`. For the `daily` forcing kind,
it identifies a folder containing a FLUXNET-style daily (`*_DD_*.csv`) file
with a usable GPP column. Custom-manifest sites do not require `israd_name`.

Before fitting, check the full input set:

```bash
ecosys model validate configs/multisite/my_lab_site.yaml
```

## Custom laboratory radiocarbon

The custom format uses a human-edited YAML manifest plus a spreadsheet-friendly
CSV. Copy the files in [`examples/custom_14c`](../examples/custom_14c/README.md)
and replace the fake records with your measurements.

```yaml
# data/custom/my_lab_14c.yaml
measurements: my_lab_14c.csv
fraction_rules:
  free_light_fraction: soil_active
  mineral_associated_organic_matter: soil_passive
```

```csv
sample_id,kind,date,depth_top_cm,depth_bottom_cm,delta14c,delta14c_sigma,fraction_property
bulk-1,bulk,2023-07-15,0,10,38,8,
fraction-1,fraction,2023-07-15,0,10,102,12,free_light_fraction
resp-1,respiration,2023-07-15,,,16,15,
```

`kind` must be `bulk`, `fraction`, or `respiration`. Bulk and fraction records
need depth bounds; fraction records also need a `fraction_property` mapped to a
pool in `fraction_rules`. Dates may be ISO dates or decimal years. The importer
uses the supplied Δ¹⁴C uncertainty for bulk and fraction blocks, applies a
15‰ minimum uncertainty, and inverse-variance aggregates replicates collected
on the same date.

## Constraint choices

| Constraint | How to enable it | Data source | Notes |
|---|---|---|---|
| Custom bulk, fraction, or respired Δ¹⁴C | `radiocarbon_manifest` | Your CSV | Used automatically; it replaces the ISRaD ¹⁴C observation path. |
| Tower ER | `ecosys optimize ... --include-er` | Configured tower forcing/ER data | Use only if the forcing product supplies compatible ER observations. |
| Incubation rate | `--include-incubation` | ISRaD incubation table | Not imported from the custom CSV. |
| Incubation CO₂ Δ¹⁴C | `--include-incubation-14c` | ISRaD incubation table | Not imported from the custom CSV. |
| Density-fraction ¹²C | Default for ISRaD `fraction`/`combined`; disable with `--no-fraction-12c` | ISRaD fraction table | Custom manifests currently provide ¹⁴C only. |
| Total SOC stock | Automatic where available for ISRaD `combined` sites | ISRaD or SoilGrids | Custom manifests do not yet import SOC stock observations. |

For a custom ¹⁴C-only site, begin with no optional flags:

```bash
ecosys optimize configs/multisite/my_lab_site.yaml \
  --outdir outputs/my_lab_site
```

Then add a constraint only when you have checked its source and its scientific
compatibility with the field measurements:

```bash
ecosys optimize configs/multisite/my_lab_site.yaml \
  --include-er \
  --outdir outputs/my_lab_site_with_er
```

The resulting `diagnostics.json` reports counts for each constraint type, which
is the quickest way to confirm that the intended observations entered the fit.

## Tune assumptions, not measurements

Use the site YAML to adjust model and inversion assumptions. Common examples:

```yaml
inversion:
  sigma_resp_14C: 20.0       # representativeness uncertainty for respiration Δ¹⁴C
  sigma_soc_fraction: 0.5    # relative uncertainty of a standard SOC constraint

external_inputs:
  CUE: 0.47
  optimize_CUE: false
```

Do not edit the CSV to make observations fit. Preserve the lab value and its
uncertainty, record any filtering decision in the manifest or run notes, and
keep the input files with the resulting output directory.
