# `ecosys` CLI reference

The `ecosys` command is a single entry point over eight verbs. Each verb has
its own doc page below.

Install the package (`pip install -e .`) and the `ecosys` console script is on
your PATH. Everything can also be invoked as `python -m ecosystem_complexity.cli
<verb> [args...]`.

## Verbs

| Verb | Purpose |
|---|---|
| [`optimize`](optimize.md) | Fit the model to observations at one site, a site set, or a sweep |
| [`warming`](warming.md) | Project a fitted site under a standardized temperature perturbation |
| [`mcmc`](mcmc.md) | Posterior sampling + cross-site rollups + structural-null test |
| [`information`](information.md) | Shapley DFS / AK / gain diagnostics on the OE posterior |
| [`fetch`](fetch.md) | Stage tower, gridded, ISRaD, and atmospheric-¹⁴C inputs |
| [`analyze`](analyze.md) | Post-hoc summaries + transit-time diagnostics over exported artifacts |
| [`config`](config.md) | Build / discover site YAML configs |
| [`report`](report.md) | Cross-run report generators (merged tables + cross-ecosystem summary) |

## Output contract

Every verb writes into `./outputs/{name}/{verb}/`, where `{name}` is the
site id (single-site runs) or the site-set YAML's `name:` field
(multi-site runs). Every run directory contains at least:

```
manifest.json              verb, git_sha, config_hash, config_snapshot, inputs, outputs
config.snapshot.yaml       the fully-resolved config that was executed
logs/run.log               stdout + stderr
```

Per-verb docs enumerate the additional files each writes (parquet tables,
NPZ arrays, PNG plots). `analyze` and `report` only read from `outputs/`,
which makes them cheap to re-run and safely parallelizable across sites.

## Overriding YAML from the CLI

`warming`, `mcmc`, `information`, and the `sweep:` path in `optimize` read
their per-run knobs from optional top-level blocks in the per-site YAML
(`warming:`, `mcmc:`, `information:`, `sweep:`). CLI flags win over YAML on
ties. See each verb's doc for its block and the fields it reads.

## Getting help

```bash
ecosys --help                  # top-level verb list
ecosys <verb> --help           # verb usage + flags
ecosys <verb> <subverb> --help # subverb usage (e.g. `ecosys analyze transit --help`)
```
