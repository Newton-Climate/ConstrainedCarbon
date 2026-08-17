# `ecosys information`

Use this command to examine how the observations constrain fitted parameters. The main workflow is `shapley`, which reruns the relevant fit and attributes degrees of freedom for signal across observation families.

```bash
ecosys information shapley \
  --site-set configs/site_sets/direct_warming_network_24.yaml \
  --plot-by biome -j 4

ecosys information shapley configs/multisite/harvard_forest.yaml
```

You can adjust the carbon-stock uncertainty rule with `--sigma-rule REL:ABS` and add ER or incubation constraints with the matching flags. The command also offers `dfs`, `ak`, `gain`, and `ose` subcommands for focused diagnostics; use their built-in help to check the support and arguments in your checkout.

## Outputs

`shapley` writes per-site `shapley_by_parameter.parquet`, `metrics.parquet`,
and `matrices.npz` under `outputs/<site-id>/information/shapley/`, alongside
the shared manifest and config snapshot. Site-set runs add network and
biome-group tables under the corresponding site-set directory. Shapley values
are DFS attributions, not causal effects or measurement-quality scores.

| File / field | Meaning and use |
|---|---|
| `shapley_by_parameter.parquet` | Long-form attribution of degrees of freedom for signal (DFS) from each observation family to each fitted parameter. Higher `shapley_dfs` means greater marginal local resolution across inclusion orders. |
| `metrics.parquet` | `dfs_total_subset` and per-parameter `dfs_*`. Values near zero indicate a locally prior-dominated parameter; values nearer one indicate stronger local resolution. |
| `matrices.npz` | Technical audit arrays. The averaging kernel maps local true-state changes to retrieved-state changes; its trace is DFS. The gain matrix maps observation perturbations to the retrieved state. |
| `network_site_summary` and biome tables | Site and group rollups for descriptive comparisons, including observation availability. |

DFS and Shapley values are local, linearized optimal-estimation diagnostics
around the fitted solution. They do not show that an observation family causes
an ecological response, nor do they rank measurement quality without regard to
the number and uncertainty of observations. Report `n_obs_family`, uncertainty
rules, and the fitted state alongside any attribution.
