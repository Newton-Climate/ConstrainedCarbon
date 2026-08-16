# `ecosys analyze`

Post-hoc analysis over exported inversion artifacts. Every subverb consumes
files that `optimize`, `warming`, or `information` already wrote — nothing
here re-runs the model.

## Synopsis

```bash
ecosys analyze <subverb> [args...]
```

## Subverbs

| Subverb | Purpose |
|---|---|
| `model` | Per-site model export / reload; write a JSON diagnostic bundle |
| `network` | Run canonical inversions across the network and aggregate OE ladder + Shapley diagnostics |
| `transit` | Transit-time diagnostics (`--mode intrinsic \| realized \| gradient`) |
| `transit-vulnerability` | Ridge / LOBO regression testing whether transit metrics predict cross-biome warming vulnerability |
| `cross-ecosystem` | Cross-ecosystem summary markdown + CSV bundle (delegates to the same builder `report cross-ecosystem` uses) |

## `analyze model`

Re-read a site's exported artifacts (or run the inversion and export them),
then emit `analysis.json` with the OE-derived diagnostics.

```bash
ecosys analyze model harvard_forest --from-artifacts \
    --export-dir outputs/israd-multisite-harvard_forest/optimize/
```

| Flag | Purpose |
|---|---|
| `site` | site selector or config path |
| `--export-dir DIR` | artifact directory to write to or read from |
| `--from-artifacts` | only read existing exports; do not re-run the inversion |
| `--observation-path {bulk_resp,fraction,combined}` | override the observation path |

## `analyze network`

Aggregate OE information-content diagnostics across the network.

```bash
ecosys analyze network --site-set configs/site_sets/full_network_41.yaml \
    --include-er-constraint --workers 8
```

| Flag | Purpose |
|---|---|
| `--workers N` | per-site worker processes (default 4) |
| `--include-expansion` | add `configs/expansion/*.yaml` to the default network selection |
| `--site-set YAML` | explicit config list; overrides the default selection |
| `--include-er-constraint` | add annual tower ER as an OE observation block |
| `--include-incubation-constraint` | add ISRaD incubation-rate rows |
| `--include-incubation-14c-constraint` | also add the dated ¹⁴C incubation block |
| `--outdir DIR` | output root |

## `analyze transit`

Transit-time diagnostics; the `--mode` flag picks the underlying driver.

```bash
ecosys analyze transit --mode intrinsic     # network-wide intrinsic MTT
ecosys analyze transit --mode realized      # env-modulated realized MTT
ecosys analyze transit --mode gradient      # four-site gradient with uncertainty
```

`intrinsic` reads the network summary + expansion tables and applies the
configured transfer topology to each site's MAP turnover times.
`realized` scales the intrinsic diagnostic to every optimized site under
its repeating daily forcing. `gradient` re-inverts four gradient sites so
posterior intervals can be quoted.

## `analyze transit-vulnerability`

Ridge / leave-one-biome-out regression asking whether transit-time metrics
add predictive power for cross-biome warming vulnerability beyond turnover
alone. Reads the network + warming exports.

```bash
ecosys analyze transit-vulnerability \
    --network notebooks/exports/network_inversion_fluxcom_er_20260719/site_summary.csv \
    --warming notebooks/exports/warming_vulnerability_fluxcom_er_20260719/site_warming_summary.csv
```

## `analyze cross-ecosystem`

Build the cross-ecosystem summary markdown + CSV bundle. This is the same
implementation as [`report cross-ecosystem`](report.md); either verb can be
used.

## Outputs

Runs land under `./outputs/{site_id or site_set_name}/analyze/{subverb}/`.
Each subverb writes at minimum `manifest.json`, `config.snapshot.yaml`,
`summary.parquet` (subverb-specific columns), and `plots/*.png`.

The `transit` subverb also writes `transit_times.parquet` with mode-dependent
columns (`tau_intrinsic`, `tau_realized`, `tau_gradient`).

## Related verbs

- [`optimize`](optimize.md) — writes the artifacts every subverb here reads
- [`report`](report.md) — cross-site aggregation of these tables
