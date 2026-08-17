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

## Output status

`analyze` currently delegates to legacy analysis modules. Its output location
is controlled by each subcommand's flags and is not yet covered by the shared
`outputs/<name>/...` contract. Keep the input tables and command invocation
with any generated figure; do not assume a manifest or config snapshot exists.

`model` exports or reloads a per-site fit analysis at `--export-dir`; use it to
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
