# Per-parameter Shapley DFS — 24 sites

## Network-mean Shapley DFS (family × parameter)

Family | tau_active | tau_slow | tau_passive | f_a_to_s | f_s_to_p
---|---|---|---|---|---
C_stocks | 0.062 | 0.078 | 0.080 | 0.140 | 0.055
bulk_14C | 0.051 | 0.098 | 0.181 | 0.020 | 0.117
fraction_14C | 0.007 | 0.002 | 0.000 | 0.000 | 0.000
resp_14C | 0.085 | 0.006 | 0.012 | 0.088 | 0.055
ER_annual | 0.065 | 0.002 | 0.001 | 0.027 | 0.006
fraction_12C | 0.099 | 0.117 | 0.083 | 0.049 | 0.061

## Best observation family per parameter (network mean)

- **$\tau_{\rm active}$** — best: `fraction_12C` (Shapley DFS 0.10, 27% of parameter's total). Runner-up: `resp_14C` (0.09).
- **$\tau_{\rm slow}$** — best: `fraction_12C` (Shapley DFS 0.12, 39% of parameter's total). Runner-up: `bulk_14C` (0.10).
- **$\tau_{\rm passive}$** — best: `bulk_14C` (Shapley DFS 0.18, 51% of parameter's total). Runner-up: `fraction_12C` (0.08).
- **$f_{a\to s}$** — best: `C_stocks` (Shapley DFS 0.14, 43% of parameter's total). Runner-up: `resp_14C` (0.09).
- **$f_{s\to p}$** — best: `bulk_14C` (Shapley DFS 0.12, 40% of parameter's total). Runner-up: `fraction_12C` (0.06).

## Biome-specific dominant families per parameter

Biome | tau_active | tau_slow | tau_passive | f_a_to_s | f_s_to_p
---|---|---|---|---|---
Arctic / permafrost | resp_14C (0.11) | bulk_14C (0.17) | bulk_14C (0.29) | resp_14C (0.43) | resp_14C (0.41)
Boreal | C_stocks (0.07) | C_stocks (0.10) | bulk_14C (0.15) | C_stocks (0.17) | bulk_14C (0.12)
Peatland | resp_14C (0.34) | C_stocks (0.05) | bulk_14C (0.08) | C_stocks (0.08) | bulk_14C (0.06)
Temperate forest | ER_annual (0.13) | fraction_12C (0.13) | bulk_14C (0.20) | C_stocks (0.13) | bulk_14C (0.14)
Tropical | fraction_12C (0.11) | fraction_12C (0.12) | bulk_14C (0.18) | C_stocks (0.17) | bulk_14C (0.12)
Grassland / Med. | fraction_12C (0.24) | fraction_12C (0.28) | fraction_12C (0.19) | C_stocks (0.18) | fraction_12C (0.14)
