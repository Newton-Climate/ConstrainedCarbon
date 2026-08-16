# `ecosys fetch`

Use `fetch` to prepare climate and flux data before fitting a site. You do not need to edit model code to do this.

```bash
# Download AmeriFlux data for a configured site
ecosys fetch flux harvard_forest --accept-policy --accept-license

# Preview the AmeriFlux request without downloading
ecosys fetch flux harvard_forest --dry-run

# Extract forcing for an expansion site from local FLUXCOM grids
ecosys fetch fluxcom configs/expansion/nahuelbuta.yaml
```

For AmeriFlux, supply your account details with `--user-id` and `--email`, or set `AMERIFLUX_USER_ID` and `AMERIFLUX_EMAIL` in `.env`. You must explicitly acknowledge the data-use policy and license. Use `ecosys fetch <subcommand> --help` for data locations and other options.
