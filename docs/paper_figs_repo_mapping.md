# Repository Mapping For Manuscript Figures

## Existing model-output objects

### Canonical OE inversion output

Produced by:

- [notebooks/sites/canonical.py](/Users/newtonnguyen/Documents/ecosystem-complexity/notebooks/sites/canonical.py)
- [notebooks/sites/howland_forest.py](/Users/newtonnguyen/Documents/ecosystem-complexity/notebooks/sites/howland_forest.py)
- [notebooks/sites/eight_mile_lake.py](/Users/newtonnguyen/Documents/ecosystem-complexity/notebooks/sites/eight_mile_lake.py)

Returned variables typically include:

- `model`
- `config`
- `idx`
- `opt_fields`
- `forcing`
- `time_years`
- `obs_full`
- `extra_blocks`
- `state0_obs`
- `state_at_map`
- `params_prior`
- `params_opt`
- `out_prior`
- `out_opt`
- `oe_result`

Key dimensions:

- `forcing.time`: `(T,)`
- `out_opt.C12`: `(T, n_pools)`
- `out_opt.delta14C`: `(T, n_pools)`
- `oe_result.x_opt`: `(n_state,)`
- `oe_result.Sx`: `(n_state, n_state)`
- `oe_result.averaging_kernel`: `(n_state, n_state)`

### Information-content outputs

Available from:

- [src/ecosystem_complexity/information.py](/Users/newtonnguyen/Documents/ecosystem-complexity/src/ecosystem_complexity/information.py)
- [src/ecosystem_complexity/oe_diagnostics.py](/Users/newtonnguyen/Documents/ecosystem-complexity/src/ecosystem_complexity/oe_diagnostics.py)
- [notebooks/cross_site_information.py](/Users/newtonnguyen/Documents/ecosystem-complexity/notebooks/cross_site_information.py)

Existing quantities:

- total DFS
- DFS by observation type
- averaging-kernel matrices
- gain matrices
- posterior covariance and posterior sigma

Existing exported files:

- `notebooks/exports/four_site_constraint_ladder.csv`
- `notebooks/exports/four_site_annotated_averaging_kernel_summary.csv`
- `notebooks/exports/*_averaging_kernel_matrix.csv`

### Warming-vulnerability outputs

Available from:

- [notebooks/uncertainty_projections.py](/Users/newtonnguyen/Documents/ecosystem-complexity/notebooks/uncertainty_projections.py)

Current exported quantities:

- sample-level `frac_loss`
- sample-level `abs_loss`
- sample-level `old_fraction`
- subset-level uncertainty summaries
- MCMC fit diagnostics and retained chains

Existing exported files:

- `notebooks/exports/uncertainty_projections_mcmc_long/uncertainty_projection_samples.csv`
- `notebooks/exports/uncertainty_projections_mcmc_long/uncertainty_projection_summary.csv`
- `notebooks/exports/uncertainty_projections_mcmc_long/uncertainty_projection_panel_c.csv`

### CESM comparison assets

Available from:

- [notebooks/clm/clm_emulator_14c.py](/Users/newtonnguyen/Documents/ecosystem-complexity/notebooks/clm/clm_emulator_14c.py)
- [notebooks/clm/analyze_clm.py](/Users/newtonnguyen/Documents/ecosystem-complexity/notebooks/clm/analyze_clm.py)
- `data/cmip/*.nc`

Current limitation:

- CESM comparison is script-specific and not yet materialized as a unified tidy comparison table.

## Mapping onto the proposed manuscript data model

### Posterior parameter table

Status: partial

- Available: posterior mean state vector, posterior covariance, retained MCMC draws in warming script outputs
- Missing: a standardized tidy table with one row per `ecosystem × observation_subset × draw × mode`

### Observation table

Status: missing as a unified table

- Available: raw site data and ISRaD-derived observation builders in site modules
- Missing: one harmonized table with standardized `observation_type` labels and uncertainties

### Information-metric table

Status: partial

- Available: DFS, averaging kernel diagonal, posterior sigma
- Missing: one complete tidy export spanning all ecosystems and subsets in the manuscript schema

### Warming-output table

Status: partial

- Available: subset-level and draw-level warming summaries
- Missing: annual-by-mode tidy output with cumulative columns in the proposed schema

### CESM comparison table

Status: missing as a unified table

- Available: CLM/CESM scripts and raw CMIP files
- Missing: one tidy comparison table with aligned forcing assumptions and standardized output variables

## Missing quantities by figure

### Figure 1

- No scientific data required
- Can be implemented as a fully schematic figure immediately

### Figure 2

- Needs a unified stored/respired `Δ14C` observation table by ecosystem

### Figure 3

- Needs tidy posterior turnover draws by mode
- DFS-related quantities already exist

### Figure 4

- Mostly supported by current OE diagnostics and exports

### Figure 5

- Needs radiocarbon offset table plus slow-mode observability joined at ecosystem level

### Figure 6

- Mostly supported by current uncertainty-projection outputs

### Figure 7

- Needs a unified CESM comparison table with matched warming assumptions
