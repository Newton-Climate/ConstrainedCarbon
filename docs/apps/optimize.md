# `ecosys optimize`

Run the canonical Optimal-Estimation inversion for one site, a list of sites, a
versioned site-set, or a pool-count / hyperparameter sweep. This is the primary
"fit the model to observations" verb.

## Synopsis

```bash
ecosys optimize [sites ...] [options]
```

`sites` are selectors — each can be a config path (`configs/multisite/harvard_forest.yaml`),
a config stem (`harvard_forest`), or an ISRaD site name.

## Common invocations

Fit one site:

```bash
ecosys optimize configs/multisite/harvard_forest.yaml
```

Fit every config under `configs/multisite/`:

```bash
ecosys optimize --all -j 8
```

Fit a versioned site set (a YAML with an explicit `configs:` list):

```bash
ecosys optimize --site-set configs/site_sets/full_network_41.yaml -j 8
```

Sweep the pool-count / sigma-rule members declared in a config's `sweep:` block:

```bash
ecosys optimize harvard_forest --sweep
```

Sweep every YAML under a directory (equivalent to the old `hf_pool_sweep`
script):

```bash
ecosys optimize --sweep configs/hf_pool_sweep/
```

List discovered sites and exit:

```bash
ecosys optimize --list
```

## Flags

| Flag | Purpose |
|---|---|
| `--all` | run every config under `configs/multisite/` |
| `--site-set YAML` | run the config list in a versioned site-set YAML |
| `--sweep [DIR]` | run each config in `DIR` as a sweep member; if omitted, use the anchor config's `sweep:` block |
| `--list` | list configured sites and exit |
| `--observation-path {bulk_resp,fraction,combined}` | override the observation path each config declares |
| `--outdir DIR` | root under which outputs land (default `./outputs/`) |
| `-j N`, `--workers N` | run `N` sites concurrently in separate processes (default 1) |
| `-q`, `--quiet` | suppress per-site progress logging |
| `--include-incubation` | add ISRaD incubation-rate rows as an OE turnover-rate constraint |
| `--include-incubation-14c` | also add the dated ¹⁴C incubation block (only meaningful with `--include-incubation`) |
| `--include-er` | add annual tower ER as an OE observation block where available |
| `--no-fraction-12c` | drop the fraction-¹²C mass block from the observation vector |
| `--incubation-duration-type CLASS` | restrict incubation rows to one or more ISRaD duration classes |

## Outputs

For each site the run lands under `./outputs/{site_id}/optimize/`. For a
site-set, replace `{site_id}` with the site-set `name:`. Sweep members land
under `./outputs/{site_id}/optimize/sweep/{member_stem}/`.

Every run directory contains:

```
manifest.json               run metadata: verb, git sha, config snapshot, inputs
config.snapshot.yaml        the fully-resolved per-site config that was executed
posterior.parquet           param_name, value, sigma, prior_mean, prior_sigma
state_trajectory.npz        C12(T, n_pools), C14(T, n_pools), time
observation_fit.parquet     time, obs_type, obs, obs_sigma, sim, residual
diagnostics.json            rmse, reduced_chi2, DFS_total, prior_influence
plots/                      fit_fluxes.png, fit_14c.png, loss_history.png
logs/run.log                stdout + stderr
```

The output contract is verb-scoped and stable across sites, so `analyze`,
`report`, and any notebook can concatenate results across runs by reading
these files without introspection.

## YAML blocks consulted

`optimize` reads the canonical `model`, `parameters`, `inversion`,
`external_inputs`, `analysis`, and (when `--sweep` is set) the `sweep:` block.
CLI flags win over YAML on ties.

## Related verbs

- [`warming`](warming.md) — projects a fitted site under a temperature perturbation
- [`information`](information.md) — Fisher / DFS / Shapley diagnostics on the posterior
- [`analyze`](analyze.md) — post-hoc summaries over exported `optimize` artifacts
