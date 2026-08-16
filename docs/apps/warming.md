# `ecosys warming`

Standardized warming-response projection. Takes a fitted site's posterior mean
and repeats its forcing under a uniform soil-temperature perturbation for a
fixed horizon, then reports the resulting change in stocks, respiration, and
(optionally) transit time.

## Synopsis

```bash
ecosys warming [sites ...] [options]
```

## Common invocations

Project the four-degree, hundred-year default for one site:

```bash
ecosys warming configs/multisite/harvard_forest.yaml
```

Project every site in a site-set with tower-ER kept as an OE constraint:

```bash
ecosys warming --site-set configs/site_sets/full_network_41.yaml \
               --include-er -j 8
```

Override the horizon and perturbation from the CLI (they otherwise default
from the config's `warming:` block):

```bash
ecosys warming eight_mile_lake_3pool_config.yaml \
               --horizon-years 200 --warming-delta-c 6
```

Compute the transit-time variant instead of the ΔC / ΔRh vulnerability metric:

```bash
ecosys warming harvard_forest --metric transit
```

## Flags

| Flag | Purpose |
|---|---|
| `--site-set YAML` | run over the site-set's config list |
| `--outdir OUTDIR` | root under which outputs land (default `./outputs/`) |
| `--horizon-years N` | override `warming.horizon_years` from YAML |
| `--warming-delta-c X` | override `warming.warming_delta_c` from YAML |
| `--metric {vulnerability,transit}` | override `warming.metric` from YAML |
| `--include-er` | include tower ER when re-seeding the posterior |
| `--include-incubation` | include ISRaD incubation-rate rows when re-seeding |
| `--include-incubation-14c` | also include the dated ¹⁴C incubation block |
| `-j N`, `--workers N` | parallel-site process count (default 1) |

## Outputs

Runs land under `./outputs/{site_id}/warming/` (or `./outputs/{site_set_name}/warming/`).

```
manifest.json
config.snapshot.yaml
projections.parquet         time, member, delta_C_stock, NEE, ER, tau_effective_by_pool
summary.parquet             metric, horizon_years, warming_delta_c, value, sigma, ci_lo, ci_hi
transit_response.parquet    only when --metric transit
plots/
  vulnerability_timeseries.png
  posterior_vs_prior.png
logs/run.log
```

The `member` column in `projections.parquet` distinguishes posterior draws,
prior draws, and the posterior mean.

## YAML block consulted

```yaml
warming:
  horizon_years: 100          # CLI --horizon-years wins if given
  warming_delta_c: 4.0        # CLI --warming-delta-c wins if given
  metric: vulnerability       # vulnerability | transit
  include_constraints:
    er: true
    incubation: false
    incubation_14c: false
```

The `include_constraints:` sub-block controls which OE rungs seed the
posterior used for the projection; `--include-*` CLI flags override each.

## Related verbs

- [`optimize`](optimize.md) — must be run first to have a posterior to project
- [`mcmc`](mcmc.md) — richer uncertainty envelope than the Gaussian posterior draws written here
