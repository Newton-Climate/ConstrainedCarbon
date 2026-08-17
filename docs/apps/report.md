# `ecosys report`

Use `report` to combine results that already exist. It does not rerun the model or change the original site fits.

```bash
ecosys report merge \
  --network-addition /path/to/site_summary.csv \
  --warming-addition /path/to/site_warming_summary.csv \
  --outdir outputs/combined_results

ecosys report cross-ecosystem \
  --network-summary /path/to/site_summary.csv \
  --warming-summary /path/to/site_warming_summary.csv
```

`merge` combines compatible site summaries. `cross-ecosystem` turns those summaries into a Markdown report, tables, and figure inputs. Use `ecosys report <subcommand> --help` to check the required paths.

## Output status

`report` writes the paths requested by its subcommands and does not yet add a
shared run manifest. Only merge summaries with compatible configurations,
observation constraints, forcing, and warming scenarios. A merged table does
not make heterogeneous runs scientifically comparable.

`merge` writes `site_summary.csv`, `site_warming_summary.csv`, and
`optimized_transit_input.csv` to `--outdir`; they are joined input tables for
later analysis, not a new inversion or uncertainty analysis. `cross-ecosystem`
writes the requested Markdown report plus figure and CSV bundle. Interpret its
biome summaries as descriptive coverage-aware synthesis unless the underlying
site-level uncertainty and comparability have been evaluated.
