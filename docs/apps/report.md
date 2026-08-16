# `ecosys report`

Cross-run report generators. `report` operates on top of one or more
`optimize` / `warming` / `analyze` runs — like `analyze`, it never re-runs
the model. Its outputs are the artifacts other consumers (notebooks,
slide-deck builders, papers) read.

## Synopsis

```bash
ecosys report <subverb> [args...]
```

## Subverbs

| Subverb | Purpose |
|---|---|
| `merge` | Append a site-set run to the canonical network / warming / transit CSVs |
| `cross-ecosystem` | Build the cross-ecosystem summary markdown + CSV bundle |

## `report merge`

Fold a site-set's network + warming + (optional) transit outputs into the
canonical multi-run tables so downstream analysis sees the union.

```bash
ecosys report merge \
    --network-addition outputs/expansion_2026/analyze/network/site_summary.csv \
    --warming-addition outputs/expansion_2026/warming/site_warming_summary.csv \
    --outdir outputs/network_inversion_combined_20260804/
```

| Flag | Purpose |
|---|---|
| `--network-addition PATH` | site-set's `analyze network` summary to fold in |
| `--warming-addition PATH` | site-set's `warming` summary to fold in |
| `--outdir DIR` | destination for the merged tables (required) |
| `--base-network PATH` | canonical network summary (default: latest under `notebooks/exports/`) |
| `--base-warming PATH` | canonical warming summary |
| `--base-transit PATH` | canonical transit summary (optional) |
| `--sites SITE ...` | restrict the merge to these site ids |

## `report cross-ecosystem`

Build the cross-ecosystem summary bundle: a markdown report plus per-topic
CSVs, plus regenerated Figure 09 (via
`ecosystem_complexity.visualize.cross_ecosystem`).

```bash
ecosys report cross-ecosystem \
    --network-summary outputs/network_inversion_combined_20260804/site_summary.csv \
    --warming-summary outputs/network_inversion_combined_20260804/site_warming_summary.csv
```

Common flags (defaults resolve to the latest exports under
`notebooks/exports/`):

| Flag | Purpose |
|---|---|
| `--network-summary PATH` | network `site_summary.csv` |
| `--warming-summary PATH` | paired warming `site_warming_summary.csv` |
| `--new-sites PATH ...` | extra expansion-site tables to fold in |
| `--figure-dir DIR` | where Figure 09 lands |
| `--report PATH` | destination markdown path |

This subverb is a re-export of the same implementation
`analyze cross-ecosystem` uses; either verb is fine.

## Outputs

Reports land under `./outputs/{name}/report/` when invoked with `--outdir`.
Otherwise they write to the destinations named by the CLI flags (typically
under `notebooks/exports/` for `merge` and `notebooks/paper_figs/outputs/`
for `cross-ecosystem`).

## Related verbs

- [`optimize`](optimize.md) / [`warming`](warming.md) — produce the site-level tables `merge` folds together
- [`analyze cross-ecosystem`](analyze.md) — same builder, exposed under `analyze`
