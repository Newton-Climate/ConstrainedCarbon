# `ecosys` command guide

`ecosys` is the command-line entry point for the repository's repeatable workflows. Give it a site config, a site-set manifest, or a group of configs where supported.

```bash
ecosys --help
ecosys <command> --help
ecosys <command> <subcommand> --help
```

The built-in help is the authoritative list of flags. A few commands still wrap research-oriented analysis modules, so their exact output location can depend on the selected config and options.

| Command | Purpose |
|---|---|
| [`optimize`](optimize.md) | Fit a site or a group of sites to observations. |
| [`warming`](warming.md) | Run a warming experiment from a fitted model. |
| [`information`](information.md) | Ask which observations constrain the parameters. |
| [`mcmc`](mcmc.md) | Sample uncertainty for cross-site comparisons. |
| [`fetch`](fetch.md) | Download or extract forcing data. |
| [`config`](config.md) | Locate sites and write configs. |
| [`analyze`](analyze.md) | Run network, transit-time, and summary analysis. |
| [`report`](report.md) | Merge results and build cross-ecosystem summaries. |
| [`model`](model.md) | Validate a site input set or run the forward model. |
