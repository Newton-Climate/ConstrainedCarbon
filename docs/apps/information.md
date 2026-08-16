# `ecosys information`

Information-content diagnostics on top of the OE posterior. Only the
`shapley` subverb is fully wired in the current build; `dfs`, `ak`, `gain`,
and `ose` return a stub message pointing back to the matrices `shapley`
already writes to `matrices.npz`.

## Synopsis

```bash
ecosys information <subverb> [args...]
```

## Subverbs

| Subverb | Status | Purpose |
|---|---|---|
| `shapley` | wired | Per-parameter Shapley DFS across the network with an optional biome / pool-count panel |
| `dfs` | stub | Total DFS per site (already in `shapley` matrices.npz) |
| `ak` | stub | Averaging-kernel diagonal (already in `shapley` matrices.npz) |
| `gain` | stub | Gain-matrix diagnostics (already in `shapley` matrices.npz) |
| `ose` | stub | Observation-system experiment scenarios (Session D) |

## `information shapley`

Compute per-parameter Shapley DFS attribution across the network. Members can
be an explicit site set, a directory of YAMLs, or the anchor-YAML `sweep:`
block. Optional plotting groups the result by biome or by soil-pool count.

### Common invocations

Cross-network Shapley with per-biome panels:

```bash
ecosys information shapley \
    --site-set configs/site_sets/full_network_41.yaml \
    --plot-by biome -j 8
```

Pool-count sweep across the sweep-manifest directory:

```bash
ecosys information shapley \
    --configs-dir configs/hf_pool_sweep/ \
    --plot-by pool_count
```

Override the σ_carbon rule (mirrors the old `tight_c` variant):

```bash
ecosys information shapley --site-set configs/site_sets/full_network_41.yaml \
    --sigma-rule 0.20:500
```

### Flags

| Flag | Purpose |
|---|---|
| `sites` | anchor / member config paths |
| `--site-set YAML` | member list from a site-set |
| `--configs-dir DIR` | glob every `*.yaml` under `DIR` as members (replaces old `hf_pool_sweep`) |
| `--outdir OUTDIR` | root under which outputs land (default `./outputs/`) |
| `--sigma-rule REL:ABS` | σ_carbon rule; overrides `information.shapley.sigma_rule` from YAML |
| `--plot-by {biome,pool_count}` | grouping for the summary panel |
| `--include-er` / `--no-include-er` | include annual tower ER when re-running the inversion |
| `--include-incubation` | include ISRaD incubation-rate rows |
| `--include-incubation-14c` | also include the dated ¹⁴C incubation block |
| `-j WORKERS` | parallel-site process count |

### Outputs

```
manifest.json
config.snapshot.yaml
matrices.npz                     averaging_kernel, gain_matrix,
                                 subset_averaging_kernel, subset_gain_matrix
shapley_by_parameter.parquet     per-site, per-constraint, per-parameter attribution
shapley_by_constraint.parquet    per-site, per-constraint totals
plots/
  shapley_by_biome_group.png     (with --plot-by biome)
  shapley_by_pool_count.png      (with --plot-by pool_count)
logs/run.log
```

`matrices.npz` also carries the AK diagonal and gain matrix per site, so the
`dfs` / `ak` / `gain` subverbs can read from `shapley`'s output today.

## YAML block consulted

```yaml
information:
  metrics:
    dfs: true
    averaging_kernel: true
    gain_matrix: true
    posterior_covariance: true
    shapley: true
    ose: false
  shapley:
    sigma_rule: "0.20:500"
    plot_by: biome              # null | biome | pool_count
```

## Related verbs

- [`optimize`](optimize.md) — writes the posterior each metric is computed against
- [`mcmc`](mcmc.md) — sampling analogue of the DFS / posterior-covariance rollups
