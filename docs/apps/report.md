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

## Outputs

Every completed command writes to
`outputs/<name>/report/<subcommand>/` by default, or under the root passed to
`--outdir`; use `--name` to choose `<name>`. Each run writes `manifest.json`
and `logs/run.log`; the manifest records the supplied arguments and declares
every generated artifact. Only merge summaries with compatible configurations,
observation constraints, forcing, and warming scenarios. A merged table does
not make heterogeneous runs scientifically comparable.
On completion it prints a JSON integration record containing `output_dir`, the
manifest path, and absolute paths for every declared artifact.

`merge` always writes `site_summary.csv`, `site_warming_summary.csv`, and
`optimized_transit_input.csv` in its contract directory. `cross-ecosystem`
writes `report.md` plus the figure and CSV bundle in its contract directory.

These merge tables are joined inputs for later analysis, not a new inversion
or uncertainty analysis. Interpret the cross-ecosystem biome summaries as
descriptive coverage-aware synthesis unless the underlying site-level
uncertainty and comparability have been evaluated.
