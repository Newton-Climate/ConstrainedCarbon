
# Ecosystem Complexity Model — Technical Specification
**Author: Newton H. Nguyen**
**Email: nnewton@stanford.edu**

**Version:** 0.1.0 | **Language:** Python 3.11+ / JAX | **Date:** May 2026
---

## 1. Scientific Objectives

This system implements a flexible, differentiable ecosystem carbon model designed to:

1. **Quantify net carbon exchange responses to climate change** at ecosystem and global scales, using eddy covariance, inventory, and radiocarbon observations jointly.

2. **Benchmark and improve Earth System Models (ESMs)** by providing posterior estimates of carbon turnover times and pool sizes constrained by multi-type observations, identifying where ESMs deviate from data-constrained estimates.

3. **Evaluate radiocarbon as an observational constraint** — systematically measuring the information content of ¹⁴C observations across pool types, ecosystem types, spatial scales, and measurement strategies using Fisher Information, degrees of freedom, and posterior error covariance metrics.

The International Soil Radiocarbon Database (ISRaD) is the primary source of ¹⁴C observational constraints.

---

## 2. Technical Requirements Summary

| Requirement | Implementation |
|---|---|
| Flexible C pool / layer structure | User-defined via YAML |
| ¹²C and ¹⁴C fluxes and stocks | Parallel tracer state |
| Invertible for state/parameter optimization | `optimize()` via `optax` + JAX autodiff |
| Auto-differentiable | Pure JAX throughout; `jax.grad`, `jax.jacfwd` |
| Complexity / information metrics | Fisher Information, DoF, posterior error covariance |
| All settings via YAML | `config.yaml` drives everything |
| Unit and integration tests | `pytest` suite |
| Base cases | Harvard Forest (US-Ha1), Barrow Alaska (US-Brw) |

---

## 3. Configuration System (YAML)

Everything — model structure, site parameters, data paths, inversion settings, and analysis options — is defined in a single YAML file. 

### 3.1 Full YAML Schema

```yaml
# config.yaml — Harvard Forest example

# ─────────────────────────────────────────────
# Site
# ─────────────────────────────────────────────
site:
  id: "US-Ha1"
  name: "Harvard Forest"
  lat: 42.5378
  lon: -72.1715
  elevation_m: 340
  biome: "temperate_deciduous"
  mat_c: 8.5
  map_mm: 1220
  permafrost: false

# ─────────────────────────────────────────────
# Model structure — define pools freely
# ─────────────────────────────────────────────
model:
  dt_days: 1.0
  solver: "euler"           # euler | rk4 | implicit_euler
  spinup_years: 200
  enable_14C: true

  # Above-ground carbon pools
  aboveground_pools:
    - name: leaf
      is_woody: false
      litterfall_to: organic_litter    # target soil pool name
    - name: wood
      is_woody: true
      litterfall_to: mineral_A_fast
    - name: branch
      is_woody: true
      litterfall_to: organic_litter
    - name: root
      is_woody: false
      litterfall_to: mineral_A_fast

  # Soil layers and SOM pools within each layer
  soil_layers:
    - name: organic
      depth_top_m: 0.00
      depth_bot_m: 0.05
      permafrost_eligible: false
      som_pools:
        - name: litter
          tau_prior_days: 180
          tau_prior_std: 60
        - name: fast
          tau_prior_days: 730
          tau_prior_std: 200
    - name: mineral_A
      depth_top_m: 0.05
      depth_bot_m: 0.20
      permafrost_eligible: false
      som_pools:
        - name: fast
          tau_prior_days: 1095
          tau_prior_std: 300
        - name: slow
          tau_prior_days: 10950
          tau_prior_std: 2000
    - name: mineral_B
      depth_top_m: 0.20
      depth_bot_m: 0.60
      permafrost_eligible: false
      som_pools:
        - name: slow
          tau_prior_days: 10950
          tau_prior_std: 2000
        - name: passive
          tau_prior_days: 36500
          tau_prior_std: 10000

  microbial_pool_per_layer: true

  # Carbon transfer rules (source → dest, fraction)
  # remainder of each source outflux is respired
  transfer_rules:
    - [leaf,             organic_litter,  0.95]
    - [branch,           organic_litter,  0.90]
    - [root,             mineral_A_fast,  0.80]
    - [wood,             mineral_A_fast,  0.70]
    - [organic_litter,   organic_fast,    0.30]
    - [organic_litter,   mineral_A_fast,  0.20]
    - [organic_fast,     mineral_A_fast,  0.15]
    - [organic_fast,     mineral_A_slow,  0.05]
    - [mineral_A_fast,   mineral_A_slow,  0.10]
    - [mineral_A_slow,   mineral_B_slow,  0.02]
    - [mineral_A_slow,   mineral_B_passive, 0.01]

# ─────────────────────────────────────────────
# Parameters — priors and initial values
# ─────────────────────────────────────────────
parameters:
  # Environmental response
  Q10:
    value: 2.0
    prior_std: 0.5
    optimize: true
  theta_opt:            # optimal soil moisture (m³/m³), per layer or scalar
    value: 0.30
    prior_std: 0.05
    optimize: true
  gamma_moisture:
    value: 5.0
    prior_std: 2.0
    optimize: true

  # NPP allocation (leaf / wood / branch / root)
  # must sum to 1; stored as log-ratios internally
  alloc:
    leaf: 0.30
    wood: 0.35
    branch: 0.10
    root: 0.25
    optimize: true

  # Tau values per pool are auto-initialized from soil_layers[*].som_pools[*].tau_prior_days
  # Override specific pools here if needed:
  tau_overrides:
    leaf: 365.0
    wood: 18250.0

# ─────────────────────────────────────────────
# Data sources
# ─────────────────────────────────────────────
data:
  flux:
    type: ameriflux_csv
    path: data/harvard_forest/AMF_US-Ha1_FLUXNET_FULLSET_DD_1991-2020.csv
    freq: DD
    gap_fill: true
    qc_threshold: 2

  meteorology:
    type: ameriflux_csv             # same file, different columns
    path: data/harvard_forest/AMF_US-Ha1_FLUXNET_FULLSET_DD_1991-2020.csv

  soil_profile:
    type: harvard_archive_csv
    path: data/harvard_forest/soil_carbon_inventory.csv
    horizon_depth_col: depth_cm
    carbon_col: C_gC_m2

  radiocarbon_14C:
    type: israd_csv                 # ISRaD export format
    path: data/harvard_forest/ISRaD_Harvard_Forest.csv
    pool_name_map:
      free_light:     organic_fast
      occluded_light: mineral_A_fast
      mineral_assoc:  mineral_A_slow
      passive:        mineral_B_passive

  atmospheric_14C:
    type: hua2013
    hemisphere: NH
    path: data/shared/Hua2013_NH.csv  # or auto-download

# ─────────────────────────────────────────────
# Inversion settings
# ─────────────────────────────────────────────
inversion:
  optimizer: adam                   # adam | lbfgs
  lr: 1.0e-3
  n_steps: 2000
  target_vars: [NEE, ER, GPP]
  include_14C: true
  lambda_14C: 1.0                   # weight on 14C loss term
  lambda_reg: 1.0e-3                # weight on prior regularization
  convergence_tol: 1.0e-6

# ─────────────────────────────────────────────
# Analysis / complexity metrics
# ─────────────────────────────────────────────
analysis:
  compute_fisher_information: true
  compute_posterior_covariance: true
  compute_degrees_of_freedom: true
  complexity_lambda_jac: 0.1
  obs_error_sigma:
    NEE: 0.5        # gC m⁻² day⁻¹
    ER: 0.4
    GPP: 0.6
    delta14C: 5.0   # ‰

# ─────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────
output:
  dir: results/harvard_forest/
  save_states: true
  save_fluxes: true
  save_jacobians: true
  save_14C_timeseries: true
```

### 3.2 Barrow, Alaska Config (key differences only)

```yaml
site:
  id: "US-Brw"
  name: "Barrow Alaska"
  lat: 71.3225
  lon: -156.6091
  biome: "arctic_tundra"
  permafrost: true

model:
  spinup_years: 500
  aboveground_pools:
    - name: leaf
      litterfall_to: organic_litter
    - name: root
      litterfall_to: active_fast
    # No wood pool — graminoid/sedge tundra

  soil_layers:
    - name: organic
      depth_top_m: 0.00
      depth_bot_m: 0.10
      permafrost_eligible: false
      som_pools:
        - {name: litter, tau_prior_days: 730,   tau_prior_std: 200}
        - {name: fast,   tau_prior_days: 3650,  tau_prior_std: 1000}
    - name: active
      depth_top_m: 0.10
      depth_bot_m: 0.35
      permafrost_eligible: true
      som_pools:
        - {name: fast,   tau_prior_days: 3650,  tau_prior_std: 1000}
        - {name: slow,   tau_prior_days: 36500, tau_prior_std: 10000}
    - name: permafrost
      depth_top_m: 0.35
      depth_bot_m: 2.00
      permafrost_eligible: true
      som_pools:
        - {name: slow,    tau_prior_days: 73000,  tau_prior_std: 20000}
        - {name: passive, tau_prior_days: 365000, tau_prior_std: 100000}

data:
  radiocarbon_14C:
    type: israd_csv
    path: data/barrow/ISRaD_Barrow.csv
    # Permafrost pools: initialize C14 from obs, not spinup
    permafrost_init_from_obs: true
    permafrost_pools: [permafrost_slow, permafrost_passive]
```

### 3.3 Config Loading

```python
def load_config(path: str) -> ModelConfig:
    """
    Parse YAML → validated ModelConfig dataclass tree.
    Raises ConfigValidationError with descriptive message on:
      - Unknown pool names in transfer_rules
      - Transfer row sums > 1
      - Overlapping layer depths
      - Missing required fields
      - Allocation fractions not summing to 1
    """
```

---

## 4. Model Architecture

### 4.1 State Variables

The full model state is a flat JAX pytree. Pool ordering follows the YAML definition order, which is fixed at `build_model()` time. A `PoolIndex` helper (generated from config) maps names to integer indices throughout.

```python
class EcosystemState(NamedTuple):
    # ¹²C pool sizes — shape (n_total_pools,), gC m⁻²
    # Order: [ag_pools... | layer0_som... layer0_mic | layer1_som... | ...]
    C12: jnp.ndarray

    # ¹⁴C tracer pools — shape (n_total_pools,), gC m⁻²
    # Tracks ¹⁴C mass (not ratio). Zero if enable_14C=False.
    C14: jnp.ndarray

    # Per-layer environmental state — shape (n_layers,)
    soil_temp:     jnp.ndarray   # °C
    soil_moisture: jnp.ndarray   # m³ m⁻³

    # Differentiable freeze/thaw fraction — shape (n_layers,)
    # 1.0 = fully thawed, 0.0 = frozen
    # frozen_frac = sigmoid(k * soil_temp), k=10 → ~0.1°C transition width
    # Always 1.0 for non-permafrost layers
    frozen_frac: jnp.ndarray

    # Active layer depth — scalar, meters
    # jnp.inf for non-permafrost sites
    active_layer_depth: jnp.ndarray

    # Time — days since 1970-01-01
    time: jnp.ndarray
```

### 4.2 Parameters

```python
class ModelParams(NamedTuple):
    # Turnover times — shape (n_total_pools,), stored in log-space
    # Exponentiated before use: tau = exp(log_tau), units: days
    log_tau: jnp.ndarray

    # Transfer matrix — shape (n_total_pools, n_total_pools + 1)
    # Stored as raw logits; converted via softmax (last col = respiration)
    # f_transfer[i, j] = fraction of pool i outflux going to pool j
    log_f_transfer: jnp.ndarray

    # NPP allocation — shape (n_ag_pools,), stored as log-ratios
    # Converted via softmax to ensure sum = 1
    log_alloc: jnp.ndarray

    # Environmental response — per layer or scalar (broadcast)
    log_Q10:        jnp.ndarray   # Q10 temperature sensitivity
    log_theta_opt:  jnp.ndarray   # optimal soil moisture
    log_gamma_moist: jnp.ndarray  # moisture curve shape

    # Microbial priming — shape (n_mic_pools,)
    alpha_priming: jnp.ndarray

    # ¹⁴C decay constant — FIXED, not optimized
    # lambda_14C = ln(2) / (5730 * 365.25) day⁻¹ = 3.317e-7
    lambda_14C: jnp.ndarray
```

### 4.3 Flux Equations

All flux functions are pure JAX — no Python control flow, no side effects.

```
# Decomposition flux from pool i (gC m⁻² day⁻¹)
F_decomp_i = C12_i / tau_i * f_temp(T_layer) * f_moisture(θ_layer) * frozen_frac_layer

# Temperature scalar (Lloyd-Taylor Q10 form)
f_temp(T) = Q10 ^ ((T - T_ref) / 10)         T_ref = 15°C

# Moisture scalar (Gaussian)
f_moist(θ) = exp(-gamma * (θ - θ_opt)²)

# NPP partitioning
alloc = softmax(log_alloc)                    # shape (n_ag_pools,)
F_npp_i = GPP * CUE * alloc_i

# Heterotrophic respiration
F_rh = Σ_i (1 - Σ_j f_ij) * F_decomp_i      # unallocated fraction respired

# Net ecosystem exchange
NEE = F_rh + Ra - GPP                         # Ra = autotrophic respiration
```

### 4.4 ¹²C / ¹⁴C Parallel Fluxes

The ¹⁴C pools are driven by the same transfer structure as ¹²C, with two additions: radioactive decay and isotopically-weighted input from the atmosphere.

```
# ¹⁴C ratio of pool i
R_i = C14_i / max(C12_i, ε)

# ¹⁴C outflux from pool i (same tau, same transfer rules)
F14_out_i = C14_i / tau_i * f_temp * f_moist * frozen_frac

# ¹⁴C transfer to downstream pools j
F14_in_j = Σ_i f_ij * F14_out_i

# ¹⁴C input from atmosphere (AG pools receiving NPP only)
R_atm(t) = R_std * (1 + Δ¹⁴C_atm(t) / 1000)
F14_npp_i = F_npp_i * R_atm(t)              # only for aboveground pools

# Radioactive decay
decay_i = lambda_14C * C14_i

# Full ¹⁴C tendency
dC14_i/dt = F14_npp_i + F14_in_i - F14_out_i - decay_i
```

---

## 5. Data Ingestion

### 5.1 Forcing and Observation Structures

```python
class ForcingData(NamedTuple):
    time:           jnp.ndarray   # (T,) days since 1970-01-01
    air_temp:       jnp.ndarray   # (T,) °C
    sw_radiation:   jnp.ndarray   # (T,) W m⁻²
    precip:         jnp.ndarray   # (T,) mm day⁻¹
    vpd:            jnp.ndarray   # (T,) hPa
    soil_temp:      jnp.ndarray   # (T, n_layers) °C
    soil_moisture:  jnp.ndarray   # (T, n_layers) m³ m⁻³
    snow_depth:     jnp.ndarray   # (T,) m
    active_layer:   jnp.ndarray   # (T,) m — inf for non-permafrost
    delta14C_atm:   jnp.ndarray   # (T,) Δ¹⁴C ‰, atmospheric record

class ObservationData(NamedTuple):
    time:           jnp.ndarray   # (T,)
    NEE:            jnp.ndarray   # (T,) gC m⁻² day⁻¹, NaN=missing
    GPP:            jnp.ndarray   # (T,)
    ER:             jnp.ndarray   # (T,)

    # Radiocarbon observations from ISRaD — sparse
    # {pool_name: (delta14C_permil, uncertainty_permil, decimal_year)}
    delta14C_obs:   dict[str, tuple[float, float, float]]

    # Bulk C pool observations — sparse, optional
    # {pool_name: (gC_m2, uncertainty)}
    C_pools_obs:    dict[str, tuple[float, float]]
```

### 5.2 ISRaD Parser

The International Soil Radiocarbon Database is the primary ¹⁴C data source. ISRaD exports have a specific multi-table structure:

```python
def load_israd(
    filepath: str,
    config: ModelConfig,
    pool_name_map: dict[str, str],     # ISRaD fraction → model pool name
    site_filter: str | None = None,    # filter by ISRaD site name
    min_year: float | None = None,     # earliest observation to include
) -> dict[str, tuple[float, float, float]]:
    """
    Parse ISRaD flat export CSV into ObservationData.delta14C_obs format.

    ISRaD tables used:
      - lyr (layer data): bulk soil Δ¹⁴C by depth horizon
      - frc (fraction data): density/size fraction Δ¹⁴C — preferred when available
      - pro (profile metadata): links to site and collection year

    Depth horizons are mapped to model layers via align_to_layers().
    Fraction names are mapped to model pool names via pool_name_map.

    ISRaD Δ¹⁴C column: 'lyr_14c' or 'frc_14c' (‰)
    ISRaD uncertainty:  'lyr_14c_sd' or 'frc_14c_sd'
    ISRaD year:         derived from 'pro_treatment_date' or 'pro_date_collected'
    """
```

### 5.3 Other Parsers

```python
def load_ameriflux_csv(filepath, config, freq="DD", gap_fill=True, qc_threshold=2)
    -> tuple[ForcingData, ObservationData]
    # Unit conversions: μmol CO₂ m⁻² s⁻¹ → gC m⁻² day⁻¹
    # QC filtering on NEE_VUT_REF_QC, RECO_NT_VUT_REF_QC
    # soil_temp/moisture returned as (T, n_layers) aligned to config layers

def load_era5_netcdf(filepath, lat, lon, config) -> ForcingData
    # t2m: K → °C; ssrd: J m⁻² → W m⁻² (÷86400); tp: m → mm day⁻¹
    # Bilinear interpolation to (lat, lon)

def load_atmospheric_14C(source="hua2013", hemisphere="NH", path=None) -> jnp.ndarray
    # Loads Hua et al. 2013 record + Levin extension to present
    # Interpolates to daily resolution via cubic spline
    # Bomb spike peak: ~1963, ~+900‰ NH

def align_to_layers(
    depths_top: list[float],
    depths_bot: list[float],
    values: list[float],
    uncertainties: list[float],
    config: ModelConfig,
    method: str = "depth_weighted",    # depth_weighted | nearest | top_match
) -> tuple[jnp.ndarray, jnp.ndarray]
    # Maps measured horizon data onto model layer grid
    # Returns (values_on_layers, uncertainties_on_layers)
    # NaN for layers with no overlapping measurement

def validate_forcing(forcing: ForcingData, config: ModelConfig) -> list[str]
    # Returns list of warning strings
    # Checks: NaN fraction, physical ranges, temporal gaps > 7 days
```

---

## 6. High-Level API

### 6.1 `build_model`

```python
def build_model(config_path: str) -> EcosystemModel:
    """
    Load YAML config and instantiate a fully JIT-compiled EcosystemModel.

    Performs at build time:
      1. Parses and validates YAML → ModelConfig
      2. Resolves PoolIndex from pool/layer definitions
      3. Builds initial ModelParams from priors in YAML
      4. Constructs and validates transfer matrix from transfer_rules
      5. JIT-compiles forward step, ¹⁴C step, and diagnostic functions

    Example:
        model = build_model("configs/harvard_forest.yaml")
        model = build_model("configs/barrow_alaska.yaml")
        model = build_model("configs/my_custom_5layer_model.yaml")
    """
```

### 6.2 `run_model`

```python
def run_model(
    model: EcosystemModel,
    forcing: ForcingData,
    state0: EcosystemState | None = None,
) -> ModelOutput:
    """
    Run forward model over full forcing timeseries via jax.lax.scan.

    ModelOutput fields:
        .C12          (T, n_pools)   bulk carbon timeseries
        .C14          (T, n_pools)   ¹⁴C tracer timeseries
        .delta14C     (T, n_pools)   Δ¹⁴C ‰ per pool
        .fluxes       dict of (T,) arrays: NEE, GPP, ER, Rh, Ra, NPP
        .complexity   (T,)           complexity index
        .jacobian     (T, n_p, n_p)  flux Jacobian (written to disk if large)
    """
```

### 6.3 `spinup`

```python
def spinup(
    model: EcosystemModel,
    forcing: ForcingData,
    n_years: int | None = None,          # from YAML if None
    convergence_tol: float = 1e-4,
    spinup_14C_from_year: int = 1500,    # start ¹⁴C spinup from this year
    permafrost_14C_init: dict | None = None,  # override for ancient pools
) -> EcosystemState:
    """
    Two-phase spinup:
      Phase 1: Run bulk ¹²C to steady state (cyclostationary equilibrium).
      Phase 2: Run ¹⁴C tracer forward from ~1500 CE through the
               historical atmospheric record with fixed ¹²C pools.
               This correctly captures the bomb spike signature in fast pools
               and pre-industrial depletion in slow pools.

    For permafrost sites (Barrow): after Phase 2, override C14 in
    permafrost_eligible layers with permafrost_14C_init values
    (measured ages from cores) rather than the spun-up values,
    since Pleistocene-age carbon cannot be initialized from the
    atmospheric record.
    """
```

### 6.4 `optimize`

```python
def optimize(
    model: EcosystemModel,
    forcing: ForcingData,
    observations: ObservationData,
    state0: EcosystemState | None = None,
) -> OptimizationResult:
    """
    Jointly invert model parameters against flux and ¹⁴C observations.
    All settings (optimizer, lambda_14C, n_steps, etc.) read from YAML.

    Combined loss (all terms in normalized units):

        L = Σ_v  Σ_t  ((sim_v(t) - obs_v(t)) / σ_v)²      # flux terms
          + Σ_p       ((sim_Δ14C_p - obs_Δ14C_p) / σ_p)²  # ¹⁴C terms
          + Σ_i       ((log_tau_i - μ_i) / σ_τi)²          # prior reg.

    Gradient computed via jax.grad (reverse-mode AD).
    Parameters constrained via log/softmax reparameterization (no bounds needed).

    Returns OptimizationResult:
        .params_opt         optimized ModelParams
        .state_opt          optimized initial EcosystemState
        .loss_history       (n_steps,) total loss
        .loss_components    dict of per-term loss histories
        .tau_history        (n_steps, n_pools) tau evolution
        .converged          bool
        .n_iter             int
    """
```

### 6.5 `compute_complexity`

```python
def compute_complexity(
    model: EcosystemModel,
    state: EcosystemState,
) -> ComplexityResult:
    """
    Compute ecosystem complexity index at a given state.
    Settings (method, lambda_jac) read from YAML analysis section.

    C_total = H_shannon + lambda_jac * ||J||_1

    H_shannon = -Σ_i p_i * log(p_i)    # entropy over pool proportions
    J_ij = ∂F_i/∂C12_j                  # flux Jacobian via jax.jacfwd

    Returns ComplexityResult:
        .C_total        scalar index
        .H_shannon      entropy component
        .J_norm         Jacobian L1 norm component
        .jacobian       (n_pools, n_pools) full Jacobian
        .pool_fractions (n_pools,) normalized pool sizes
    """
```

---

## 7. Information Content Analysis

This is the core analytical layer for Objective 3 — quantifying how much ¹⁴C observations constrain carbon turnover times.

### 7.1 Fisher Information Matrix

```python
def compute_fisher_information(
    model: EcosystemModel,
    forcing: ForcingData,
    params: ModelParams,
    state0: EcosystemState,
    obs_error_sigma: dict[str, float],   # from YAML analysis section
    obs_types: list[str] = ["NEE", "ER", "GPP", "delta14C"],
) -> FisherResult:
    """
    Compute the Fisher Information Matrix (FIM) for model parameters
    with respect to a set of observation types.

    FIM_ij = Σ_t (1/σ²_obs) * (∂y_t/∂θ_i) * (∂y_t/∂θ_j)

    where y_t are simulated observables, θ are the parameters (log_tau etc.),
    and σ²_obs is the observation error variance from YAML.

    Sensitivities ∂y/∂θ computed via jax.jacrev over the full forward run.
    FIM computed separately for each obs_type and jointly, enabling
    direct comparison of information content per observation type.

    Returns FisherResult:
        .FIM_total      (n_params, n_params) joint FIM
        .FIM_per_type   dict {obs_type: (n_params, n_params) FIM}
        .eigenvalues    FIM eigenvalues (measure of well-constrained directions)
        .eigenvectors   FIM eigenvectors (constrained parameter combinations)
    """
```

### 7.2 Degrees of Information (Degrees of Freedom for Signal)

```python
def compute_degrees_of_freedom(
    FIM: jnp.ndarray,
    prior_covariance: jnp.ndarray,
) -> DofResult:
    """
    Compute degrees of freedom for signal (DFS) — the number of
    independent pieces of information the observations provide about
    the parameters, beyond the prior.

    DFS = trace(I - (C_prior⁻¹ + FIM)⁻¹ * C_prior⁻¹)
        = trace(A)   where A = posterior gain matrix

    Interpretation:
      DFS = n_params → observations fully constrain all parameters
      DFS = 0        → observations add no information beyond prior
      DFS per obs type quantifies relative value of each data stream

    Also computes per-parameter DFS (diagonal of A) — which individual
    parameters are most constrained by which observation types.

    Returns DofResult:
        .dfs_total          scalar — total degrees of freedom for signal
        .dfs_per_obs_type   dict {obs_type: scalar}
        .dfs_per_param      (n_params,) per-parameter DFS
        .averaging_kernel   (n_params, n_params) matrix A
    """
```

### 7.3 Posterior Error Covariance

```python
def compute_posterior_covariance(
    FIM: jnp.ndarray,
    prior_covariance: jnp.ndarray,
) -> PosteriorResult:
    """
    Compute the Bayesian posterior parameter error covariance matrix.

    C_post = (C_prior⁻¹ + FIM)⁻¹

    Diagonal elements: posterior variance on each parameter (τ_i, alloc, etc.)
    sqrt(C_post_ii): posterior 1-sigma uncertainty on parameter i

    Uncertainty reduction:
        UR_i = 1 - sqrt(C_post_ii) / sqrt(C_prior_ii)

    Computes uncertainty reduction per parameter per observation type —
    the key diagnostic for Objective 3: which ¹⁴C measurements most
    reduce uncertainty on which turnover times.

    Returns PosteriorResult:
        .C_post                 (n_params, n_params)
        .posterior_sigma        (n_params,) 1-sigma uncertainties
        .uncertainty_reduction  dict {obs_type: (n_params,) UR}
        .correlation_matrix     (n_params, n_params) posterior correlations
    """
```

### 7.4 Observation System Experiment (OSE) Interface

```python
def run_observation_system_experiment(
    model: EcosystemModel,
    forcing: ForcingData,
    params_true: ModelParams,
    ose_config: dict,
) -> OseResult:
    """
    Systematically evaluate the information content of different
    ¹⁴C measurement strategies — directly addressing Objective 3.

    ose_config (from YAML or dict) specifies:
      - which pools to sample (e.g., all layers vs surface only)
      - which years to sample (pre-bomb, post-bomb, recent)
      - measurement uncertainty scenarios

    For each scenario:
      1. Simulate 'true' observations from params_true
      2. Compute FIM, DFS, posterior covariance
      3. Report uncertainty reduction on all tau parameters

    Returns OseResult with ranked list of measurement strategies by
    total DFS and by uncertainty reduction on key slow/passive tau values.

    Example use: identify whether sampling permafrost layers at Barrow
    in 2010 provides more ¹⁴C constraint than sampling surface organic
    layers in 1975 (post-bomb peak).
    """
```

---

## 8. Data Sources by Site

### 8.1 Harvard Forest (US-Ha1)

| Variable | Source | Format | Notes |
|---|---|---|---|
| NEE / GPP / ER | AmeriFlux US-Ha1 | FLUXNET2015 CSV | 30-min → daily |
| Meteorology | AmeriFlux US-Ha1 | FLUXNET2015 CSV | Same file |
| Soil T & moisture | HF EMS tower | CSV | Multi-depth |
| Soil ¹⁴C | ISRaD | CSV | Density fractions preferred |
| Atmospheric ¹⁴C | Hua et al. 2013 | CSV | NH zone 2 |
| Soil C inventory | Harvard Forest Data Archive | CSV | By horizon |
| LAI | MODIS MCD15A3H | HDF/NetCDF | 4-day, 500m |

### 8.2 Barrow, Alaska (US-Brw)

| Variable | Source | Format | Notes |
|---|---|---|---|
| NEE / GPP / ER | AmeriFlux US-Brw | FLUXNET CSV | Growing season focus |
| Meteorology | ERA5 reanalysis | NetCDF | t2m, ssrd, tp |
| Soil T profile | NGEE-Arctic | CSV/JSON | 5–40 cm depths |
| Active layer depth | NGEE-Arctic | CSV | Daily, Stefan equation |
| Soil ¹⁴C | ISRaD + NGEE-Arctic | CSV | Includes permafrost cores |
| Atmospheric ¹⁴C | Hua et al. 2013 | CSV | NH zone 1 |
| Permafrost C init | Direct measurement | JSON | Ancient pools only |

---

## 9. Implementation

### 9.1 Key JAX Patterns

```python
# --- Forward scan (no Python loop in compiled code) ---
@partial(jax.jit, static_argnames=["model"])
def _forward(params, state0, forcing, model):
    def step(state, t):
        forcing_t = jax.tree.map(lambda x: x[t], forcing)
        state_new  = model.step_12C(state, params, forcing_t)
        state_new  = model.step_14C(state_new, params, forcing_t)
        obs_sim    = model.diagnose(state_new, params, forcing_t)
        return state_new, obs_sim
    T = forcing.time.shape[0]
    return jax.lax.scan(step, state0, jnp.arange(T))

# --- Gradient of loss (reverse-mode AD) ---
grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))

# --- Fisher Information sensitivities (forward-mode AD) ---
# Forward-mode efficient when n_params >> n_obs per timestep
sens_fn = jax.jit(jax.jacfwd(simulate_obs, argnums=0))

# --- Complexity Jacobian (forward-mode: n_pools small) ---
jac_fn = jax.jit(jax.jacfwd(flux_fn, argnums=0))

# --- Positivity / sum constraints ---
tau   = jnp.exp(params.log_tau)                # tau > 0
alloc = jax.nn.softmax(params.log_alloc)        # Σ = 1
F_mat = jax.nn.softmax(params.log_f_transfer, axis=-1)[:, :-1]  # rows ≤ 1
```

### 9.2 Permafrost Differentiable Mask

```python
def frozen_frac(soil_temp: jnp.ndarray, steepness: float = 10.0) -> jnp.ndarray:
    """Smooth [0,1] thaw fraction. Differentiable at freeze/thaw front."""
    return jax.nn.sigmoid(steepness * soil_temp)

def effective_decomp(C12, tau, f_temp, f_moist, frozen_frac):
    """Decomposition gated by thaw state — differentiable."""
    return C12 / tau * f_temp * f_moist * frozen_frac
```

---

## 10. Tests

### 10.1 Unit Tests

```
tests/
├── test_config.py
│     load_config(valid_yaml)              → no exception
│     load_config(bad_transfer_sum)        → ConfigValidationError
│     load_config(unknown_pool_name)       → ConfigValidationError
│     pool_index round-trip               → names ↔ indices bijective
│
├── test_parsers.py
│     load_ameriflux_csv (Harvard)         → NEE in [-20,20] gC/m²/d; no NaN forcing
│     load_era5_netcdf (Barrow)            → T in [-50,10]°C; precip ≥ 0
│     load_israd (Harvard)                 → all pool names in config; Δ¹⁴C finite
│     load_atmospheric_14C                 → 1963 peak > 500‰; post-2000 < 100‰
│     align_to_layers (depth mismatch)    → NaN where no overlap; no NaN where overlap
│
├── test_model.py
│     ¹²C mass balance (Harvard 10yr)      → |ΔC_total - net_input| < 1e-4
│     ¹²C mass balance (Barrow 10yr)       → same
│     ¹⁴C mass balance                    → ΣC14 + cumul_decay = ΣC14_init + ΣF14_npp
│     frozen_frac at T=0°C                → ≈ 0.5; at T=-5°C → < 0.1
│     no NaN (Harvard full run)            → jnp.isfinite(output.C12).all()
│     no NaN (Barrow full run)             → same
│
├── test_14C.py
│     bomb spike propagation              → fast pool Δ¹⁴C peaks ~1970; passive flat
│     permafrost init override            → permafrost pools hold init Δ¹⁴C after spinup
│     delta14C diagnostic                 → formula: (R/R_std - 1)*1000; matches analytic
│     grad of 14C loss w.r.t. log_tau    → finite, non-zero for slow pools
│
├── test_inversion.py
│     synthetic tau recovery (flux only)  → recover within 5% after 1000 steps
│     synthetic tau recovery (+14C)       → slow pool within 10%; passive within 20%
│     loss decreases monotonically        → loss_history[-1] < loss_history[0]
│     gradient finite everywhere          → no NaN in jax.grad output
│
├── test_information.py
│     FIM positive semi-definite          → all eigenvalues ≥ 0
│     DFS ≤ n_params                      → always
│     DFS(flux + 14C) > DFS(flux only)   → 14C adds information
│     posterior variance < prior          → UR > 0 for all constrained params
│     OSE ranking consistent              → deep layer 14C > surface for passive tau
│
└── test_complexity.py
      Shannon entropy ≥ 0                 → always
      Jacobian shape                      → (n_pools, n_pools) per config
      grad of complexity w.r.t. C12      → finite, non-zero
```

### 10.2 Integration Tests

| Test | Site | Pass Criterion |
|---|---|---|
| End-to-end Harvard | US-Ha1 2000–2015 | NEE R² > 0.65 daily; RMSE < 1.5 gC m⁻² d⁻¹ |
| End-to-end Barrow | US-Brw 2000–2015 | Annual NEE budget sign correct; RMSE < 0.8 |
| Passive SOM ¹⁴C | Harvard | Δ¹⁴C within 50‰ of ISRaD obs post-inversion |
| Permafrost ¹⁴C | Barrow | Δ¹⁴C within 100‰ of NGEE-Arctic obs |
| Custom 2-layer config | Synthetic | Runs without error; mass balance holds |
| Custom 6-layer config | Synthetic | Runs without error; FIM shape correct |

---

## 11. Directory Structure

```
ecosystem_complexity/
├── configs/
│   ├── harvard_forest.yaml
│   ├── barrow_alaska.yaml
│   └── schema.yaml              # JSON Schema for config validation
├── src/ecosystem_complexity/
│   ├── config.py                # load_config, ModelConfig, validation
│   ├── state.py                 # EcosystemState, ModelParams, PoolIndex
│   ├── model.py                 # EcosystemModel, step_12C, step_14C
│   ├── api.py                   # build_model, run_model, spinup, optimize
│   │                            #   compute_complexity
│   ├── fluxes.py                # All differentiable flux equations
│   ├── tracer_14C.py            # ¹⁴C step, spinup_14C, delta14C diagnostic
│   │                            #   initialize_permafrost_14C
│   ├── transfer.py              # build_transfer_matrix, validation
│   ├── information.py           # compute_fisher_information
│   │                            #   compute_degrees_of_freedom
│   │                            #   compute_posterior_covariance
│   │                            #   run_observation_system_experiment
│   ├── complexity.py            # Shannon entropy + Jacobian metric
│   └── data/
│       ├── parsers.py           # load_ameriflux_csv, load_era5_netcdf
│       ├── parsers_14C.py       # load_israd, load_atmospheric_14C
│       │                        #   load_permafrost_14C_json
│       ├── alignment.py         # align_to_layers
│       └── schemas.py           # ForcingData, ObservationData
├── tests/
│   ├── test_config.py
│   ├── test_parsers.py
│   ├── test_model.py
│   ├── test_14C.py
│   ├── test_inversion.py
│   ├── test_information.py
│   └── test_complexity.py
├── notebooks/
│   ├── harvard_forest_demo.ipynb
│   ├── barrow_alaska_demo.ipynb
│   └── ose_14C_strategy_demo.ipynb
└── pyproject.toml
```

---

## 12. Dependencies

```toml
[project]
name = "ecosystem-complexity"
requires-python = ">=3.11"
dependencies = [
    "jax>=0.4.25",
    "jaxlib>=0.4.25",
    "optax>=0.2.2",
    "numpy>=1.26",
    "xarray>=2024.1",
    "netCDF4>=1.6",
    "pandas>=2.2",
    "scipy>=1.12",
    "pyyaml>=6.0",
    "jsonschema>=4.21",     # YAML config validation
    "h5py>=3.10",
]

[project.optional-dependencies]
gpu  = ["jax[cuda12]>=0.4.25"]
test = ["pytest>=8.0", "pytest-cov>=5.0"]
```

---

A few design decisions to flag:

**Fisher Information vs MCMC.** The FIM/DFS approach is fast and fully differentiable (sensitivities from `jax.jacrev`), which makes it ideal for systematic OSE experiments across many sites and pool configurations. It assumes Gaussian errors and a linearized model, which is a reasonable approximation near the optimum. If you need full non-Gaussian posteriors, a Hamiltonian Monte Carlo sampler via `blackjax` can be added later using the same JAX model.

**ISRaD pool mapping.** ISRaD fraction names (free light, occluded light, mineral-associated) don't map 1:1 to model pools — the `pool_name_map` in the YAML is where you encode this science decision site by site. The spec leaves this explicit rather than trying to automate it.

**DFS interpretation for Objective 3.** The `run_observation_system_experiment` function is the direct computational answer to "which ¹⁴C measurements provide the strongest constraints." Running it across Harvard Forest and Barrow with different pool/layer sampling strategies should produce the ranked comparison your objectives require.