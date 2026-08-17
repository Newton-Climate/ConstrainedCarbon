# Mapping soil fractions to model pools

Laboratory fractions are operational measurements. Model pools are simplified
kinetic reservoirs. A mapping between them is therefore a scientific
hypothesis—not a label conversion.

The project maps by fractionation *scheme*, not by site name. That keeps the
meaning of a protocol consistent across sites and makes unfamiliar protocols
visible for review.

## What the default mapping says

| Fractionation scheme or property | Model role | Interpretation |
|---|---|---|
| density: free light | active | particulate, relatively fast-cycling material |
| density: occluded light | slow | protected or intermediate-turnover material |
| density: heavy | passive | mineral-associated, slowest modeled material |
| particle size: coarse or sand | active | largely particulate material |
| particle size: silt or silt+sand | slow | intermediate material |
| particle size: fine, clay, or silt+clay | passive | mineral-associated material |
| plant material (non-root) | active | fresh litter enters the configured active pool directly |
| macrofossil | bulk only | contributes to whole-soil, not a separate kinetic pool |
| chemical, aggregate, compound-specific, roots, or unknown schemes | skipped | no defensible default kinetic interpretation |

“Active”, “slow”, and “passive” are model roles. They do not claim that every
particle in a heavy fraction has one age or a unique measured turnover time.

## How to build a defensible mapping

1. Record the separation protocol, not just the fraction name. A density
   `heavy` fraction and a chemical `residual` are not equivalent.
2. Check reservoir membership. Living roots are not soil organic matter merely
   because their Δ¹⁴C is modern.
3. Match the measurement to the pool definition, including depth, inputs, and
   transfer pathways.
4. Preserve uncertainty. Disagreement among properties mapped to one pool must
   widen uncertainty; do not average it away.
5. Make an explicit override only when the site protocol supports it.

## How this project applies the rule

[`fraction_mapping.py`](../../src/ecosystem_complexity/data/fraction_mapping.py)
first checks a site-level `datasource.fraction_rules` override. Otherwise it
applies the scheme policy above and resolves a role such as `active` against
configured pool names. Skipped properties are recorded with a reason. If a
depth-resolved model has more than one pool with the same role, the mapping is
ambiguous and requires an explicit override.

For Harvard Forest, the model has `soil_active`, `soil_slow`, and
`soil_passive` over 0–1.3 m. Its fraction observations help distinguish those
configured pools; they do not directly measure fitted turnover times. See the
[Harvard Forest example](harvard-forest-example.md) for those constraints in a
completed run.

## Override carefully

```yaml
datasource:
  fraction_rules:
    heavy: null
    site_specific_fraction: soil_passive
```

Do not override a mapping because it improves the fit. That changes the
scientific observation model and belongs in a separate, named sensitivity run.

## Layers are not automatically pools

Depth is a location in the soil profile. A kinetic pool is a model reservoir.
They coincide only when the model explicitly defines a depth-resolved pool
structure. Before mapping a layer observation, decide which description the
model uses:

| Model structure | How to interpret a layer measurement |
|---|---|
| Depth-resolved layers | Map the measured interval onto the overlapping model layer. The default is depth-weighted: a measurement contributes in proportion to the thickness it shares with the model layer. |
| Co-located kinetic pools | Do **not** assign a bulk layer to active, slow, or passive simply because it is shallow or deep. Those pools are kinetic fractions that may coexist at the same depth. Use a bulk-mixture operator or a fraction-specific observation instead. |

For the first case, all depths are converted to metres, positive downward. If a
sample crosses a model boundary, its value is split by overlap. The resulting
layer value is the overlap-weighted mean. Report the original interval and the
model boundaries: a 0–20 cm measurement is not a direct observation of either
a 0–10 cm or 10–30 cm pool.

The Harvard Forest configuration has one model soil layer from 0–1.3 m, but
three kinetic pools with nominal ranges of 0–10, 10–30, and 30–130 cm. In the
canonical workflow those ranges help define priors and reporting conventions;
they do **not** turn a bulk 30–130 cm Δ¹⁴C measurement into a direct passive
pool measurement. Density fractions provide the pool-specific evidence instead.

## How bulk observations are used here

A whole-soil bulk Δ¹⁴C measurement contains carbon from every modeled pool.
The model therefore predicts it as a carbon-mass-weighted mixture:

```
bulk Δ¹⁴C = sum(pool carbon × pool Δ¹⁴C) / sum(pool carbon)
```

Depth profiles are first reduced to one whole-profile value per observation
year. The weighting ladder is deliberately simple and uses one consistent basis
within each profile:

1. Measured layer SOC, when it is available for every layer;
2. bulk density × organic carbon × thickness, when those fields are complete;
3. thickness alone, when neither carbon-mass basis is complete.

The chosen basis is recorded in the run log. Thickness weighting is a fallback,
not a claim that carbon density is constant with depth. It can over-weight deep,
carbon-poor material, so use it with appropriate uncertainty.

The profile's spread becomes part of the observation uncertainty. This is
important: a co-located-pool model cannot reproduce a Δ¹⁴C depth gradient, so
the model should not be forced to fit that within-profile variation as if it
were a pool-specific signal. The Harvard example's 1997 bulk profile is one
such whole-profile constraint; its fraction observations supply the separate
active, slow, and passive information.

## What this means for carbon stocks

The canonical workflow uses the total column SOC observation, `sum(pool C)`,
rather than allocating a measured depth-layer stock directly to one kinetic
pool. Density-fraction carbon shares can additionally constrain the modeled
pool partition. This avoids treating a depth boundary as proof that all carbon
inside it belongs to one kinetic reservoir.

If the available profile stops well above the modeled lower boundary, it is not
a complete column-stock observation. Keep that mismatch visible in the input
review or use a deeper, compatible stock product instead of silently assigning
the shallow total to the full model column.
