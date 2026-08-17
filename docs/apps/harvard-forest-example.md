# Harvard Forest: reading a complete run

This worked example uses
[`configs/multisite/harvard_forest.yaml`](../../configs/multisite/harvard_forest.yaml).
It shows how to read one fitted site. It is not evidence by itself for a general
ecosystem claim.

## Reproduce the example

```bash
ecosys analyze model configs/multisite/harvard_forest.yaml \
  --name harvard_forest_example
```

The command writes a diagnostic bundle under
`outputs/harvard_forest_example/analyze/model/`. Begin with `manifest.json`,
then read `summary.json`, `observations.csv`, `constraint_ladder.csv`, and the
figure below.

![Harvard Forest model diagnostic](artifacts/harvard_forest_site_diagnostics.png)

*This run uses 44 configured observations: 33 respired-CO₂ Δ¹⁴C values, one
total-carbon stock, five fraction Δ¹⁴C values, one bulk Δ¹⁴C profile, and four
fraction-carbon-share observations.*

## Read the figure in this order

1. **Forcing.** The upper-left panel shows the daily GPP forcing. The result is
   conditional on this time series; it is not independent of the forcing
   product or period.

2. **Total soil carbon.** Dotted gray is the prior simulation; solid blue is
   the fitted (MAP) simulation. Here the MAP is closer to the configured stock
   observation. That is a fit diagnostic, not validation against unused data.

3. **Respired Δ¹⁴C.** Points are observations, dotted gray is the prior, and
   red is the MAP. Look for systematic departures, not just a line passing
   through some points. The model follows the broad decline but not every date.

4. **Observation fit.** Points near the 1:1 line have matching posterior
   predictions and observations. This is a screening panel because the fit
   combines observation types with different units and scales.

5. **Information content.** This run has about 2.82 degrees of freedom for
signal (DFS) across a 12-parameter state. The data locally resolve only a
few independent parameter combinations. These are information diagnostics,
not causal effects.

The reported DFS is the trace of the same averaging kernel used by the fitted
inversion. The analysis command checks that its reconstructed kernel agrees
with the fitted kernel before it writes an artifact.

## What can be said from this run?

The fit has reduced χ² of about 0.89 and weighted RMSE of about 0.99. The
residuals are therefore broadly consistent with the uncertainties specified in
this configuration. This does not prove the model structure is correct, the
uncertainties are perfect, or the result transfers to another site.

The useful next step is a named sensitivity run with one changed assumption:
for example, the observation set, prior, or pool structure. Compare manifests,
parameter estimates, and DFS—not figures alone.
