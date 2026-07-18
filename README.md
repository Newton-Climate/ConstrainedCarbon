# ecosystem-complexity

**An inference-first, differentiable ecosystem carbon-cycle model.**

`ecosystem-complexity` is a flexible, fully auto-differentiable model of the terrestrial
carbon cycle that treats ¹²C and ¹⁴C (radiocarbon) as parallel tracers. It is built to
answer a specific scientific question: **how much information do radiocarbon observations
actually add** when constraining soil-carbon turnover times and pool sizes?

The model is written in [JAX](https://jax.readthedocs.io/), so the same forward model can be:

- **run** forward over a site's meteorological forcing,
- **inverted** to fit eddy-covariance fluxes, radiocarbon, and stock observations
  (gradient descent *or* Optimal Estimation), and
- **analysed** with information-theoretic diagnostics (Fisher Information, degrees of
  freedom for signal, posterior error covariance).

Everything — pool structure, layers, site parameters, data paths, and inversion
settings — is defined in a single YAML file.

---

## Scientific objectives

1. **Quantify net carbon-exchange responses to climate** using eddy covariance,
   inventory, and radiocarbon observations *jointly*.
2. **Benchmark Earth System Models** with data-constrained posterior estimates of
   turnover times and pool sizes.
3. **Evaluate radiocarbon as a constraint** — measure the information content of ¹⁴C
   across pool types, biomes, and measurement strategies.

The [International Soil Radiocarbon Database (ISRaD)](https://soilradiocarbon.org) is the
primary source of ¹⁴C observations. Reference sites include Harvard Forest (US-Ha1),
Howland Forest (US-Ho1), Barrow Alaska (US-A10/US-Brw), and Eight Mile Lake (US-EML).

See [`docs/TECHSPEC.md`](docs/TECHSPEC.md) for the full technical specification.

---

## Architecture

```
                          ┌─────────────────────┐
                          │   config.yaml        │   site, pools, layers,
                          │  (one file drives     │   inversion + analysis
                          │   everything)         │   settings
                          └──────────┬───────────┘
                                     │ load_config / PoolIndex
                                     ▼
        data/ (AmeriFlux, ISRaD,  ┌─────────────────────┐
        CMIP, atm ¹⁴C record)     │  config.py           │
              │                   │  state.py            │  frozen dataclasses:
              │  parsers / loaders│  ModelParams,        │  validated config +
              ▼                   │  EcosystemState      │  tracer state
     ┌──────────────────┐         └──────────┬───────────┘
     │ data/            │                    │
     │  parsers.py      │  ForcingData        │
     │  parsers_14C.py  │  ObservationData    ▼
     │  loaders.py      │─────────►┌─────────────────────────────────────┐
     │  israd_*.py      │          │        FORWARD MODEL (pure JAX)      │
     │  alignment.py    │          │                                     │
     │  schemas.py      │          │  climate.py   f_temp / f_moisture /  │
     └──────────────────┘          │               thawed_frac           │
                                   │  soil.py      SOM decomposition      │
                                   │  above_ground.py  GPP / autotrophic  │
                                   │  transfer.py  inter-pool transfers   │
                                   │  tracer_14C.py  ¹⁴C step + Δ¹⁴C      │
                                   │  model.py     step_12C / step_14C /  │
                                   │               diagnose               │
                                   └──────────────────┬──────────────────┘
                                                      │ api.py
                                          build_model / run_model / spinup
                                                      │
                     ┌────────────────────────────────┼────────────────────────────┐
                     ▼                                 ▼                            ▼
          ┌────────────────────┐        ┌──────────────────────────┐   ┌────────────────────┐
          │  INVERSION         │        │  OPTIMAL ESTIMATION       │   │  INFORMATION        │
          │  inversion.py      │        │  optimal_estimation.py    │   │  information.py     │
          │                    │        │  oe_utils.py              │   │  complexity.py      │
          │  optimize()        │        │  optimize_oe()            │   │  sensitivity.py     │
          │  Adam / L-BFGS     │        │  Levenberg–Marquardt      │   │                     │
          │  (optax)           │        │  + posterior covariance   │   │  Fisher Information  │
          │                    │        │  oe_diagnostics.py        │   │  DFS, posterior Sₓ  │
          └────────────────────┘        └──────────────────────────┘   └────────────────────┘
```

All traced computation lives in pure, module-level JAX functions so the whole pipeline is
safe under `jax.jit`, `jax.lax.scan`, and `jax.grad`. Structural arguments (pool counts,
pool→layer mapping, timestep) are held static.

### Key modules (`src/ecosystem_complexity/`)

| Module | Responsibility |
|---|---|
| `config.py` | Parse & validate YAML → frozen `ModelConfig`; `PoolIndex` maps pool names → state-vector positions |
| `state.py` | `EcosystemState` (¹²C/¹⁴C pools), `ModelParams`, default constructors |
| `model.py` | `EcosystemModel`: `step_12C`, `step_14C`, `diagnose` (NEE/GPP/ER/Rh/Ra) |
| `climate.py` | Abiotic response functions (Lloyd–Taylor Q10, moisture, freeze/thaw). `fluxes.py` is a back-compat shim |
| `soil.py` / `above_ground.py` / `transfer.py` | Soil decomposition, GPP & autotrophic respiration, inter-pool carbon transfers |
| `tracer_14C.py` | ¹⁴C tracer step, Δ¹⁴C computation, historical-atmosphere spin-up |
| `api.py` | Public runtime: `build_model`, `run_model`, `spinup` |
| `inversion.py` | Gradient-based inversion (`optimize`) via `optax` + autodiff |
| `optimal_estimation.py` / `oe_utils.py` / `oe_diagnostics.py` | Optimal Estimation (`optimize_oe`, Levenberg–Marquardt) with posterior covariance |
| `information.py` / `complexity.py` / `sensitivity.py` | Fisher Information, degrees of freedom for signal, posterior error covariance |
| `data/` | Parsers, loaders, ISRaD observations, alignment, and `ForcingData`/`ObservationData` schemas |

---

## Installation

Requires **Python 3.11+**.

### Option A — conda (recommended)

```bash
make env                              # create the conda env from environment.yaml
conda activate ecosystem-complexity
make install                          # pip install -e .
```

### Option B — pip / venv

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"               # editable install + dev tools
```

### Required: JAX and optax

The forward model and inversions depend on **`jax`** and **`optax`**, which are not yet
pinned in `environment.yaml` / `pyproject.toml`. Install them explicitly:

```bash
pip install jax optax                 # CPU build; see JAX docs for GPU/TPU wheels
```

> For GPU/TPU support, follow the platform-specific instructions in the
> [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

Verify the install:

```bash
python -c "import ecosystem_complexity, jax, optax; print('OK')"
make test
```

---

## Quick start

### 1. Run a forward simulation

```python
from ecosystem_complexity.api import build_model, run_model, spinup
from ecosystem_complexity.data.loaders import load_forcing  # site-specific loader

model = build_model("configs/harvard_3pool_config.yaml")
forcing = load_forcing("configs/harvard_3pool_config.yaml")

# Spin up to quasi-steady state (12C convergence + 14C historical spin-up)
state0 = spinup(model, forcing)

out = run_model(model, forcing, state0=state0)
print(out.NEE.shape, out.delta14C.shape)   # (T,), (T, n_pools)
```

`ModelOutput` carries `C12`, `C14`, `delta14C`, `NEE`, `GPP`, `ER`, `Rh`, `Ra`, and the
`final_state`.

### 2. Invert against observations

```python
# Gradient-based (Adam / L-BFGS)
from ecosystem_complexity.api import optimize
result = optimize(model, forcing, observations, ...)

# Optimal Estimation (Levenberg–Marquardt + posterior covariance)
from ecosystem_complexity.api import optimize_oe
oe = optimize_oe(model, forcing, observations, ...)
```

### 3. Analyse information content

```python
from ecosystem_complexity.information import analyze_information_content
info = analyze_information_content(model, forcing, ...)   # Fisher, DFS, posterior Sₓ
```

Full, runnable end-to-end examples live in [`notebooks/`](notebooks/).

---

## Configuration

A single YAML file defines the site, model structure (pools and layers), spin-up, ¹⁴C
options, data paths, inversion settings, and analysis options. Ready-made configs live in
[`configs/`](configs/):

| Config | Site |
|---|---|
| `harvard_3pool_config.yaml`, `harvard_4pool_config.yaml` | Harvard Forest (US-Ha1) |
| `howland_forest_3pool_config.yaml` | Howland Forest (US-Ho1) |
| `barrow_3pool_config.yaml`, `barrow_alaska.yaml` | Barrow, Alaska (permafrost) |
| `eight_mile_lake_3pool_config.yaml` | Eight Mile Lake (US-EML, permafrost) |
| `schema.yaml` | Annotated reference schema |

Pools are named `{layer}_{som_pool}` (e.g. `organic_litter`); microbial pools are
`{layer}_mic`. See TECHSPEC §3 for the full schema.

---

## Repository layout

```
ecosystem-complexity/
├── src/ecosystem_complexity/   # the package (see module table above)
│   └── data/                   # parsers, loaders, schemas, ISRaD observations
│   └── sites/                  # reusable per-site inversion modules
├── configs/                    # per-site YAML configurations
├── data/                       # AmeriFlux, ISRaD, CMIP forcing & observations
├── notebooks/                  # remaining analysis scripts & figure studies
├── apps/                       # reusable CLI apps for lookup, download, config, fit, analysis
├── docs/                       # TECHSPEC.md + methodology notes
├── tests/                      # pytest suite (unit + integration)
├── environment.yaml            # conda environment
├── pyproject.toml              # package metadata & tooling config
└── Makefile                    # env / install / lint / format / test
```

The application entry points under `apps/` now cover the generalized site
workflows (`locate_site.py`, `download_flux.py`, `build_site_config.py`,
`optim_site_main.py`, `analyze_model.py`). The `notebooks/` directory remains
for figure-generation and one-off analyses such as `cross_site_information.py`.

---

## Development

```bash
make lint      # ruff check + mypy (strict)
make format    # black + ruff --fix
make test      # pytest -v with coverage
make clean     # remove caches and build artifacts
```

Scientific naming conventions (`C12`, `C14`, `log_Q10`, `GPP`, `Ra`, …) are intentionally
preserved; the corresponding lint rules are disabled in `pyproject.toml`.

---

## License

See [`LICENSE`](LICENSE).

## Citation / contact

Newton H. Nguyen — nnewton@stanford.edu
