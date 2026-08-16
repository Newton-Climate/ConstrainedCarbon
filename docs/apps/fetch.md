# `ecosys fetch`

Download or verify the external data feeding a site inversion. Every subverb
writes a `manifest.json` describing what it fetched (or what was already on
disk).

## Synopsis

```bash
ecosys fetch <subverb> [args...]
```

## Subverbs

| Subverb | Purpose |
|---|---|
| `flux` | Download AmeriFlux (eddy-covariance) forcing + observations for one configured site |
| `fluxcom` | Extract site-level GPP / NEE / ER from the global FLUXCOM-X 2021 grids |
| `clm` | Extract site-level series from a local directory of CLM NetCDFs |
| `israd` | Print where the ISRaD tables live (they are read by the site drivers; no download step) |
| `atm14c` | Verify the atmospheric ¹⁴C record CSVs (Hua, Graven, INTCAL) are staged under `data/` |

## `fetch flux`

Download tower forcing + observations for one site.

```bash
ecosys fetch flux harvard_forest --accept-policy --accept-license
```

| Flag | Purpose |
|---|---|
| `site` | site selector / config stem / tower id |
| `--out-dir DIR` | download root (default `data/`) |
| `--env-file PATH` | dotenv file with `AMERIFLUX_USER_ID` / `AMERIFLUX_EMAIL` (default `.env`) |
| `--user-id ID` | AmeriFlux user id (overrides env-file) |
| `--email ADDR` | AmeriFlux account email (overrides env-file) |
| `--accept-policy` | acknowledge the AmeriFlux data-use policy |
| `--accept-license` | acknowledge the CC-BY license on the data |
| `--keep-archive` | keep the downloaded ZIP after unpacking |
| `--dry-run` | print the download plan and exit |
| `--outdir OUTDIR` | fetch-manifest root (default `./outputs/`) |

Credentials come from CLI or `--env-file`; the YAML config never contains
them.

## `fetch fluxcom`

Extract site-level daily GPP + NEE from the ICOS FLUXCOM-X 2021 grid, with
ER = GPP + NEE. By default runs over every YAML with
`forcing_kind: fluxcom` under `configs/expansion/`.

```bash
ecosys fetch fluxcom                       # every expansion site
ecosys fetch fluxcom configs/expansion/nahuelbuta.yaml
```

| Flag | Purpose |
|---|---|
| `configs` | site config paths; defaults to `configs/expansion/*.yaml` |
| `--overwrite` | replace any existing site CSVs under `data/shared/fluxcom/` |
| `--outdir OUTDIR` | fetch-manifest root (default `./outputs/`) |

## `fetch clm`

Extract site-level series from a local directory of CLM output NetCDFs.

```bash
ecosys fetch clm --source-dir /path/to/CLM_output \
                 configs/multisite/harvard_forest.yaml
```

| Flag | Purpose |
|---|---|
| `--source-dir DIR` | directory of NetCDF files to read from |
| `--variables NAME ...` | override the default variable list |
| `--out-root DIR` | where the extracted CSVs land under `data/` |
| `--overwrite` | replace existing extractions |
| `configs` | site config paths; defaults to `configs/multisite/*.yaml` filtered to `forcing_kind: clm` |

## `fetch israd`

Not a download — ISRaD is read directly from the parser at
`src/ecosystem_complexity/data/parsers_14C.py` by the site drivers. This
subverb prints where the tables live and where they are consumed from, so
you know what to refresh in place.

```bash
ecosys fetch israd
```

## `fetch atm14c`

Verify that the three atmospheric-¹⁴C CSVs (Hua 2021, Graven 2017, INTCAL20)
are staged under `data/shared/atm_14C/`.

```bash
ecosys fetch atm14c
```

Prints `[OK]` / `[MISSING]` per file.

## Notes

`fetch` does not read the model / inversion YAML blocks — it consumes
`site.tower_id` and `site.lat` / `site.lon`. Credentials belong in a
`.env` file (or CLI flags), never in a checked-in config.

## Related verbs

- [`optimize`](optimize.md) — the primary consumer of the data this verb stages
- [`config`](config.md) — turn tower ids / coordinates into a full per-site YAML
