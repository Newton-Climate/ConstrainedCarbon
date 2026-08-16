# How the ecosystem-complexity model works

This is a scientific guide to the model, not a replacement for the YAML file that is run. Start with [`configs/multisite/harvard_forest.yaml`](../configs/multisite/harvard_forest.yaml) for a working example and [`configs/schema.yaml`](../configs/schema.yaml) for the annotated configuration reference.

## The question

Soil carbon is hard to infer from fluxes alone: the same respiration flux can come from a small, fast-turning pool or a large, slow-turning pool. Radiocarbon helps separate those possibilities because its atmospheric history and radioactive decay carry information about the age of carbon moving through the system.

The model asks whether one consistent set of pool sizes, turnover times, and transfers can explain the measurements available at a site. It can use eddy-covariance fluxes, soil-carbon stocks, bulk and fraction radiocarbon, respired radiocarbon, and optional incubation or annual-respiration constraints.

## What the model represents

Each site has one or more soil layers. A layer contains named soil-organic-matter pools, commonly `active`, `slow`, and `passive`. The names are for readability; the turnover-time priors and transfer rules determine their behavior.

At each daily time step, the model receives carbon input, partitions it among pools, adjusts decomposition for temperature, moisture, and thaw where relevant, and then transfers or respires carbon. The same flows are applied to ¹²C and ¹⁴C. The ¹⁴C tracer also decays and receives the site's historical atmospheric radiocarbon signal.

The model reports NEE, GPP, ecosystem respiration, heterotrophic respiration, pool stocks, and pool Δ¹⁴C.

## Fitting a site

The primary fitting workflow uses Optimal Estimation. It finds parameter values that both agree with observations and remain plausible under the prior uncertainty stated in the config. A close fit is not automatically a well-constrained result: posterior uncertainty and the diagnostics show which parameters were genuinely informed by observations.

```bash
ecosys optimize configs/multisite/harvard_forest.yaml
```

Use `--observation-path` to compare a specified subset of measurements. Always record that choice with any result you intend to interpret or publish.

## Information diagnostics

Information analysis answers “which measurement matters for which parameter?” rather than just “did the model fit?”.

- **Degrees of freedom for signal (DFS)** summarize how much observations reduce uncertainty relative to the prior.
- **The averaging kernel** shows which fitted parameters are informed by observations and which remain dominated by the priors.
- **The gain matrix** shows how a change in an observation affects the fitted parameters.
- **Shapley attribution** divides shared information among measurement families, making it possible to compare, for example, fraction radiocarbon with bulk radiocarbon or soil stocks.

These measures depend on assumed observation errors and parameter priors. Use them to compare clearly stated scenarios, not as universal properties of a site.

## Warming and transit time

The warming workflow repeats the configured forcing with a uniform temperature perturbation and evaluates the selected response over the chosen horizon. It is a standardized sensitivity experiment, not a site-specific climate forecast.

Transit-time analyses summarize how long carbon remains in the modeled system. Intrinsic transit time reflects fitted turnover and transfers; realized transit time also reflects the environmental forcing used in the run. Treat both as model-based diagnostics, and report the pool structure and observation set behind them.

## Working with configs

The config is part of the scientific method for a run. Before editing one, make sure you can explain the selected layers and pools, the source and depth of each observation, the turnover-time priors, and the observations included in the fit.

For a new site, use `ecosys config locate` to find likely ISRaD–tower matches and `ecosys config build` to make a starting YAML. Review and adjust the resulting file before downloading data or fitting the site.

## Reproducibility

Keep the exact site YAML, site-set YAML if used, command and options, input data versions, and output tables together. The repository includes exploratory notebooks and historical exports; a config plus an explicit `ecosys` command is the clearest starting point for a reproducible analysis.
