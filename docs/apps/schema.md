# Site configuration schema

A site YAML is the analysis protocol. It says what site is represented, which
data may enter the fit, how carbon moves between pools, and how to interpret
the result.

The complete field reference is [`configs/schema.yaml`](../../configs/schema.yaml).
Runtime validation is the final authority; run this before fitting:

```bash
ecosys model validate configs/multisite/harvard_forest.yaml
```

## Read a config from top to bottom

| Block | What it controls | Question to answer |
|---|---|---|
| `site` | identity, coordinates, tower, biome | Is this the intended ecosystem? |
| `datasource` | forcing, observations, fraction rules | Are dates and products appropriate? |
| `model` | pools, depth ranges, transfer rules, spinup | Do these pools represent the processes to infer? |
| `external_inputs` | carbon input source and partition | Does the model boundary match the study? |
| `parameters` | fixed assumptions | What is held fixed rather than inferred? |
| `inversion` | uncertainties and optimizer settings | Do uncertainties include representativeness error? |
| experiment blocks | warming, MCMC, information settings | Are scenarios stated before outputs are compared? |

## Fields most likely to change a conclusion

`model.soil_layers.som_pools` defines the quantities that can be inferred.
`transfer_rules` defines possible pathways. `datasource.observation_path` and
optional constraints choose the evidence. Finally, `inversion.sigma_*` values
set how strongly each observation family influences the fit.

Change one scientific assumption at a time, give the result a new name, and
compare manifests and diagnostics. Validation can confirm that a config loads;
it cannot decide whether its pool structure or prior is scientifically sound.
