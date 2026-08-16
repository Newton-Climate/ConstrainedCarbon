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
