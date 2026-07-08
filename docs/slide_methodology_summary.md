# Methodology Summary for Slide Results

This note summarizes the methodology behind the results presented in `ecosystem_complexity_presentation.pptx`, with emphasis on slides 6 onward. It reflects the current code path in the repository, especially the canonical site inversions and the CLM/CESM comparison workflow.

## 1. Scientific objective

The project asks three linked questions:

1. How does ecosystem complexity affect the vulnerability of soil carbon stocks?
2. How much do different soil-carbon observations constrain that vulnerability?
3. Do climate-model soil pools imply turnover times that are consistent with radiocarbon-based constraints?

The core strategy is to fit the same compact soil-carbon model to two contrasting ecosystems, quantify which observations constrain which parameters, and then compare those inferred turnover times to CESM2/CLM5.

## 2. Modeling setup

### 2.1 Canonical model structure

The main results are built from a shared 3-pool soil model with:

- `soil_active`
- `soil_slow`
- `soil_passive`

The same model structure is used for both Harvard Forest and Barrow so that cross-site differences reflect the data and forcing, not a change in topology.

Key features:

- Parallel simulation of `12C` and `14C`
- Daily forcing
- JAX-differentiable forward model
- Turnover times represented as `log_tau`
- Inter-pool transfer fractions represented as `log_f_transfer`
- Analytical steady-state replacement for initial `C12` during inversion/Jacobian evaluation to remove spinup drift

Primary code paths:

- `notebooks/sites/canonical.py`
- `src/ecosystem_complexity/optimal_estimation.py`
- `src/ecosystem_complexity/information.py`

### 2.2 Forcing setup

Harvard Forest:

- FluxNet high-resolution forcing
- Analysis window truncated to 1996 onward
- Atmospheric `Δ14C` attached from Hua/Graven/IntCal merged record

Barrow:

- ERA5 + FLUXMET-derived forcing
- Analysis window truncated to 2011 onward
- Atmospheric `Δ14C` attached from the same merged record

The atmospheric radiocarbon record is used so each modeled carbon input carries the appropriate atmospheric `Δ14C` signature through time.

## 3. Inversion method

### 3.1 Optimal estimation

The main site results use `optimize_oe`, a Levenberg-Marquardt optimal-estimation inversion that minimizes:

`J(x) = (y - F(x))^T Se^-1 (y - F(x)) + (x - xa)^T Sa^-1 (x - xa)`

where:

- `x` is the state vector of optimized parameters
- `xa` is the prior parameter vector
- `Se` is the observation error covariance
- `Sa` is the prior error covariance
- `F(x)` is the forward model mapping parameters to observables

For the canonical runs, the optimized field set is:

- `log_tau`
- `log_f_transfer`

The inversion returns:

- MAP parameter estimates
- Posterior covariance `Sx`
- Averaging kernel `A`
- Cost history and convergence diagnostics

### 3.2 Why this formulation matters

This setup makes the project do two jobs at once:

- estimate the best-fitting turnover times
- quantify which observations actually constrain those turnover times

That is why the same framework supports both the fitting slides and the information-content slides.

## 4. Observation mapping to pools

## 4.1 Harvard Forest

Harvard uses three observation families:

1. Respired `Δ14C`
2. Soil C stocks
3. ISRaD density-fraction `Δ14C`

### Respired `Δ14C`

- Source: `hf212-01-14c-no-treat.csv`
- Only the `NWN` site series is used
- Measurements are mapped to the nearest model time index by decimal year

### Soil C stocks

In the canonical Harvard workflow, stock constraints come from ISRaD layer-integrated profiles (`H1` to `H5`) rather than the older local-stock mapping used in some site-specific scripts.

Depth bins used in the canonical 3-pool setup:

- O horizon / above mineral soil -> `soil_active`
- 0 to 15 cm -> `soil_slow`
- 15 to 100 cm -> `soil_passive`

Each layer is assigned by depth midpoint to avoid double counting across boundaries.

### ISRaD density-fraction `Δ14C`

Fraction-to-pool mapping:

- free light -> `soil_active`
- occluded light -> `soil_slow`
- heavy -> `soil_passive`

Entry-year mapping:

- `Gaudinski_2000` -> 1996
- `Savage_unpub` -> 2007
- `McFarlane_2013` -> 2011

Important exclusion:

- the 2011 passive/heavy fraction is excluded because the density cutoff is not consistent with the earlier entries

Interpretation:

- Harvard uses fractionated `Δ14C` to separate younger, intermediate, and mineral-associated carbon in a way that is much sharper than bulk soil alone.

## 4.2 Barrow

Barrow uses:

1. Respired `Δ14C`
2. Soil C stocks
3. ISRaD bulk-layer `Δ14C`

### Respired `Δ14C`

- Source: Vaughn surface-emission measurements
- Same-day measurements are averaged
- Then mapped to the nearest model time index

### Soil C stocks

Individual stock constraints are applied to:

- `soil_active`: 4500 gC m^-2 ± 45%
- `soil_passive`: 12000 gC m^-2 ± 33%

The older Barrow sum constraints exist in helper code:

- `active + slow` organic stock
- `slow + passive` permafrost stock

but the current canonical inversion omits them because they over-determine the slow pool with inconsistent surveys.

### ISRaD bulk-layer `Δ14C`

Bulk layer observations are assigned by depth midpoint:

- `soil_active`: -10 to 14 cm
- `soil_slow`: 14 to 30 cm
- `soil_passive`: 30 to 200 cm

This is a key methodological choice:

- Barrow does **not** use density fractions in the canonical run
- it uses depth-binned bulk soil `Δ14C` aggregated across BEO/Vaughn and Nave profiles

Observation uncertainty is based on spatial variability across plots, not just AMS analytical error, because the site is heterogeneous at small scales.

## 5. Logic of the main experiments

## 5.1 Gap framing and two-site comparison

The paired-site design is intentional:

- Harvard Forest represents a temperate system with relatively modern respired carbon and no permafrost
- Barrow represents a tundra/permafrost system with old deep carbon and weaker coupling between passive stocks and surface respiration

The same 3-pool structure is used at both sites to make the information comparison interpretable.

## 5.2 Harvard validation against literature expectations

The Harvard section first situates the model against the Sierra/Gaudinski literature and then runs the current canonical 3-pool inversion. The logic is:

1. show that the framework can reproduce the classic Harvard bomb-spike behavior
2. then update turnover times using the current OE setup and expanded radiocarbon/stock constraints

This establishes that the framework is not only fitting arbitrary site data; it is grounded in a benchmark system where radiocarbon dynamics are already well known.

## 5.3 Canonical site inversions

For each site:

1. build the site-specific forcing
2. build the site-specific observation vector
3. initialize the model state from literature/observed C and `Δ14C`
4. run prior forward simulation
5. invert for `log_tau` and `log_f_transfer`
6. evaluate posterior fit to respired `Δ14C`

This produces the headline posterior turnover times and the fit-quality metrics.

## 5.4 Information-content experiments

The DFS/averaging-kernel experiments ask not just "what fit best?" but "which observation type contributed what constraint?"

The canonical ablation proceeds by computing the Jacobian at the MAP estimate and then evaluating subsets of the observation rows:

- `C_stocks`
- `pool_delta14C`
- `resp_delta14C`
- combinations of the above

The key quantity is:

- `DFS = trace(A)`

Interpretation:

- high DFS means the data constrain more independent directions in parameter space
- low DFS means the solution remains prior-dominated along many directions

This is how the slides justify statements like "respired `Δ14C` is more informative at Harvard than at Barrow" and "3 pools are about the right resolved complexity for these data."

## 5.5 CLM/CESM comparison

There are two CLM-related workflows in the repo, and the slide logic follows the emulator workflow rather than the simpler proxy comparison.

### A. Proxy comparison

`notebooks/clm/analyze_clm.py` computes implied CLM turnover times via:

- `tau = pool stock / rh`

This is useful as a quick benchmark, but it is not the main inferential comparison.

### B. Emulator comparison used for the main result

`notebooks/clm/fit_clm.py` and `notebooks/clm/clm_emulator_14c.py` do the more important experiment:

1. treat CLM5 recent-decade stocks and total heterotrophic respiration as observations
2. fit the same 3-pool model to those CLM targets with `optimize_oe`
3. obtain a CLM-implied `tau` vector in the same model space as the radiocarbon inversion
4. compare that `tau` vector against the OE posterior
5. hold out observed respired `Δ14C`
6. forward-run the CLM-implied `tau` and test whether it predicts the observed respired radiocarbon

This held-out test is the most important methodological step in the model-comparison section, because it asks whether a stock+flux-consistent CLM solution is also radiocarbon-consistent.

## 6. Interpretation of the result logic

### 6.1 Why Harvard and Barrow behave differently

At Harvard:

- all three pools contribute more directly to respired CO2
- respired `Δ14C` therefore carries strong information about turnover time
- CLM stock+Rh targets and observed radiocarbon are more mutually consistent

At Barrow:

- the passive pool is permafrost-locked and poorly seen by surface respiration alone
- respired `Δ14C` is therefore less informative
- stock constraints and deep/bulk radiocarbon matter more
- CLM can match stocks+Rh while still landing in the wrong radiocarbon-sensitive part of parameter space

### 6.2 Why the information-content results matter

The DFS analysis is not just a diagnostics add-on. It answers the methodological question:

"What observation type is worth collecting in which ecosystem?"

Implications:

- the value of `Δ14C` data is ecosystem-specific
- bulk stocks alone do not constrain the same directions as radiocarbon
- adding complexity beyond what the data can resolve will create prior-dominated parameters

### 6.3 Why the CLM test matters

The CLM comparison is stronger than a simple stock comparison because it asks:

- can a model that matches C stocks and total Rh also match the age signature of respired carbon?

If not, then the model may have the right bulk fluxes for the wrong mechanistic reasons.

That is especially important for warming projections, where the age and depth source of respired carbon determine the long-term carbon-climate feedback.

## 7. Practical implications

## 7.1 For field campaigns

The results imply that observation strategy should depend on ecosystem structure:

- Harvard-like systems benefit strongly from respired `Δ14C`
- Barrow-like systems require stock and depth-resolved/bulk soil `Δ14C` because surface respiration under-sees passive carbon

## 7.2 For model design

The project supports using a compact 3-pool structure for these datasets because:

- it captures the main radiocarbon-resolved dimensions
- added structure is not automatically informative unless new observation types are added

This does **not** mean real soils are only 3 pools; it means the current data resolve roughly three dominant turnover-time dimensions.

## 7.3 For climate-model benchmarking

A climate model should not be considered validated for soil vulnerability solely because it matches:

- total soil carbon
- total respiration

It should also be tested against the age signature of the carbon being respired. In this project, that test is respired `Δ14C`.

## 8. Caveat on slide numbers vs. code state

The deck has been updated to match the current rerun outputs, but the methodology itself is more stable than the exact numbers. Small numerical values can shift when:

- observation selections change
- certain constraints are excluded
- uncertainty assumptions are updated
- the CLM site-cell mapping changes

The methodological structure remains:

- same 3-pool model
- same OE inversion framework
- site-specific observation mapping
- DFS-based information analysis
- CLM emulator + held-out radiocarbon validation

## 9. Files most directly backing the slides

- `notebooks/sites/canonical.py`
- `notebooks/sites/harvard_forest.py`
- `notebooks/sites/barrow.py`
- `src/ecosystem_complexity/optimal_estimation.py`
- `src/ecosystem_complexity/information.py`
- `notebooks/cross_site_information.py`
- `notebooks/headline_figures.py`
- `notebooks/clm/fit_clm.py`
- `notebooks/clm/clm_emulator_14c.py`
- `notebooks/clm/analyze_clm.py`
