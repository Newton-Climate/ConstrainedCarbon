# `ecosys information`

Use this command to examine how the observations constrain fitted parameters. The main workflow is `shapley`, which reruns the relevant fit and attributes degrees of freedom for signal across observation families.

```bash
ecosys information shapley \
  --site-set configs/site_sets/direct_warming_network_24.yaml \
  --plot-by biome -j 4

ecosys information shapley configs/multisite/harvard_forest.yaml
```

You can adjust the carbon-stock uncertainty rule with `--sigma-rule REL:ABS` and add ER or incubation constraints with the matching flags. The command also offers `dfs`, `ak`, `gain`, and `ose` subcommands for focused diagnostics; use their built-in help to check the support and arguments in your checkout.
