# `ecosys analyze`

Use these commands after you have fitted sites and want to answer a broader scientific question.

| Subcommand | Question it helps answer |
|---|---|
| `model` | How well does one fitted model reproduce its observations? |
| `network` | How do results compare across a set of sites? |
| `transit` | How long does carbon remain in the modeled soil system? |
| `transit-vulnerability` | Does transit time help explain warming vulnerability across sites? |
| `cross-ecosystem` | What is the overall pattern across ecosystems? |

```bash
ecosys analyze network \
  --site-set configs/site_sets/direct_warming_network_24.yaml \
  --include-er-constraint --workers 4

ecosys analyze transit --mode intrinsic
```

These analyses use different input tables and assumptions, so check `ecosys analyze <subcommand> --help` before running one.

## Outputs

Every completed command writes to
`outputs/<name>/analyze/<subcommand>/` by default, or under the root passed to
`--outdir`; use `--name` to choose `<name>`. Each run writes `manifest.json`
and `logs/run.log`; the manifest records the supplied arguments and declares
every generated artifact. The app owns all output locations, so use `--outdir`
rather than a subcommand-specific `--out`, `--figure`, or `--export-dir` flag.
On completion it prints a JSON integration record containing `output_dir`, the
manifest path, and absolute paths for every declared artifact.

| Subcommand | Contract payloads |
|---|---|
| `model` | `fit_matrices.npz`, observation/constraint/Shapley/ablation CSVs, `summary.json`, and `site_diagnostics.png` |
| `network` | `site_summary.csv`, `ladder_summary.csv`, `shapley_summary.csv`, `failures.csv`, and `aggregate_summary.json` |
| `transit --mode intrinsic` | `transit_times.csv` and `transit_times.png` (plus a failures CSV when needed) |
| `transit --mode realized` | `realized_transit_times.csv` and `realized_transit_times.png` (plus a failures CSV when needed) |
| `transit --mode gradient` | `gradient_transit_times.csv`, `gradient_transit_draws.csv`, and `gradient_transit_times.png` |
| `transit-vulnerability` | leave-one-biome-out summary, predictions, and site-metrics CSVs |
| `cross-ecosystem` | `report.md` plus the generated figure and CSV bundle |

`model` exports a per-site fit analysis to its contract directory; use it to
inspect the stated fit, not to create a new independent observation. `network`
produces site and observation-information summaries; its DFS and Shapley
outputs have the same local-resolution interpretation described for
`information shapley`. `transit` writes a table and figure at `--out` and
`--figure`: intrinsic transit time reflects turnover and routing under the
configured reference environment, while realized variants also depend on the
forcing used. `transit-vulnerability` writes leave-one-biome-out prediction
tables; judge model additions by held-out error and coverage, not in-sample
association. `cross-ecosystem` writes a figure/table/report bundle whose
cross-site patterns remain conditional on compatible source summaries.

## Harvard Forest example

```bash
ecosys analyze model configs/multisite/harvard_forest.yaml \
  --name harvard_forest_example
```

Use this single-site diagnostic to examine fit quality before moving to a
network analysis. The information panel shows about 2.82 DFS across the
fitted state, so it would be too strong to claim that every fitted parameter is
independently measured.

![Harvard Forest analysis diagnostic](artifacts/harvard_forest_site_diagnostics.png)
