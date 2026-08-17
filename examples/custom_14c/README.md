# Custom laboratory ¹⁴C input

Copy these two files into a location appropriate for your project (for example,
`data/custom/`), then point a site configuration at the manifest:

```yaml
datasource:
  radiocarbon_manifest: ../../data/custom/my_site_14c.yaml
```

`israd_name` is not required when a custom manifest is supplied; a valid
`forcing_glob` is still required to provide the meteorological forcing.

The manifest is YAML and contains the CSV filename plus `fraction_rules`
mapping your laboratory fraction labels to model pools. The CSV requires these
columns:

```text
sample_id,kind,date,delta14c,delta14c_sigma
```

Use `bulk`, `fraction`, or `respiration` for `kind`. Bulk and fraction records
also need `depth_top_cm` and `depth_bottom_cm`; fraction records need
`fraction_property`. Dates may be ISO dates (recommended) or decimal years.
All Δ¹⁴C values and uncertainties are in per mil. The importer validates the
table and uses inverse-variance aggregation for replicated measurements from
the same date and observation type.
