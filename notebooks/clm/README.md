# CLM / CESM2 CMIP6 comparison

The active workflow now reads the global CESM2 historical NetCDFs already
stored in `data/cmip` and selects the nearest model grid cell for the four
canonical analysis sites:

- Harvard Forest (`US-Ha1`)
- Barrow, Alaska (`US-A10`)
- Howland Forest (`US-Ho1`)
- Eight-mile Lake (`US-EML`)

The older `download_clm.py` path that wrote one NetCDF per site/grid-cell is
deprecated and kept only for reference.

## Why CESM2?

CESM2's land surface is **CLM5**, the same Community Land Model used by NCAR
in CMIP6. Its soil-C pool decomposition follows a Century-style cascade
(active → slow → passive) — directly comparable to our 3-pool inversion.

## Variables used from `data/cmip`

| CMIP6 short name | Long name | Units | Use |
|---|---|---|---|
| `cSoilFast`   | Fast-pool soil C | kg C m⁻² | τ_active comparison |
| `cSoilMedium` | Medium-pool soil C | kg C m⁻² | τ_slow comparison |
| `cSoilSlow`   | Slow/passive-pool soil C | kg C m⁻² | τ_passive comparison |
| `cSoil`       | Total soil C | kg C m⁻² | bulk-soil check |
| `rhSoil`      | Soil heterotrophic respiration | kg C m⁻² s⁻¹ | implied τ = pool / flux |
| `npp`         | Net primary production | kg C m⁻² s⁻¹ | flux context |

The local global archive does not include the older per-site `cLitter` files,
so the CLM-emulator active-pool target is based on `cSoilFast` alone.

## Experiments

| Experiment | Years | Use |
|---|---|---|
| `historical` | 1850–2014 | Match our HF (1996–2021) + Barrow (2011+) windows |
| `ssp585`     | 2015–2100 | High-emission warming forecast |
| `ssp245`     | 2015–2100 | Mid-emission scenario |
| `piControl`  | 800 yr | Equilibrium reference |

## Usage

```bash
# Compare implied CESM2 turnover against the OE posterior
python notebooks/clm/analyze_clm.py

# Fit our 3-pool model to CESM2 C + Rh at the four sites
python notebooks/clm/fit_clm.py

# Diagnose how the CESM2-emulated τ vectors miss observed respired Δ14C
python notebooks/clm/clm_emulator_14c.py
```
