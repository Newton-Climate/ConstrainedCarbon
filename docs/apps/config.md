# `ecosys config`

Config-file utilities. These write or look up YAML — they do not run the
model, so their outputs land under `configs/` (or stdout), not under
`./outputs/`.

## Synopsis

```bash
ecosys config <subverb> [args...]
```

## Subverbs

| Subverb | Purpose |
|---|---|
| `build` | Synthesize a new per-site config from a template + site metadata |
| `incubation` | Generate per-site incubation-experiment configs from an ISRaD extraction |
| `locate` | Look up a site's tower id + coordinates in the co-location table |

## `config build`

Materialize a new per-site YAML by cloning the multi-site template and
substituting site-specific fields.

```bash
ecosys config build \
    --selector US-Ha1 \
    --tower-id US-Ha1 --lat 42.5378 --lon -72.1715 \
    --biome "temperate_deciduous_forest" \
    --observation-path combined \
    --out configs/multisite/harvard_forest.yaml
```

| Flag | Purpose |
|---|---|
| `--selector` | existing site selector, tower id, or site name |
| `--tower-id` | flux tower id |
| `--lat`, `--lon` | tower coordinates |
| `--biome` | biome string override |
| `--observation-path {bulk_resp,fraction,combined}` | which ISRaD observation path the site uses |
| `--template PATH` | YAML template (default `configs/israd_multisite_3pool_config.yaml`) |
| `--out PATH` | output YAML path (default `configs/multisite/<stem>.yaml`) |

Prints the destination path on success.

## `config incubation`

Generate the expansion-site configs used by the ISRaD incubation-rate
constraint from a CSV manifest.

```bash
ecosys config incubation \
    --manifest notebooks/exports/incubation_config_manifest_20260719.csv \
    --template configs/expansion/nahuelbuta.yaml \
    --out-dir configs/expansion
```

| Flag | Purpose |
|---|---|
| `--manifest PATH` | CSV manifest of sites to materialize |
| `--template PATH` | expansion-config template to clone |
| `--out-dir DIR` | destination directory |

## `config locate`

Look up a site in the ISRaD ↔ flux-tower co-location table. Useful before
`config build` when you have a tower id or coordinates but no existing
selector.

```bash
ecosys config locate --flux-tower US-Ha1 --out /tmp/harvard_locate.csv
ecosys config locate --lat 42.54 --lon -72.17 \
                     --max-distance-km 25 \
                     --out /tmp/nearby.csv
```

| Flag | Purpose |
|---|---|
| `--flux-tower` | tower id, tower name, or site name selector |
| `--lat`, `--lon` | query coordinates |
| `--biome` | biome substring filter |
| `--max-distance-km` | radius when querying by coordinates (default 50) |
| `--out PATH` | output CSV path (required) |

Also prints the table to stdout.

## Related verbs

- [`fetch`](fetch.md) — once you have a config, this stages the tower / gridded forcing
- [`optimize`](optimize.md) — the config's ultimate consumer
