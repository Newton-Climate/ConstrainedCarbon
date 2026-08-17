# Per-parameter Shapley DFS — 24 sites, combined observation path

## Network-mean Shapley DFS (family × parameter)

Family | tau_active | tau_slow | tau_passive | f_a_to_s | f_s_to_p
---|---|---|---|---|---
C_stocks | 0.052 | 0.094 | 0.081 | 0.146 | 0.056
bulk_14C | 0.041 | 0.093 | 0.158 | 0.014 | 0.105
fraction_14C | 0.025 | 0.006 | 0.003 | 0.005 | 0.001
resp_14C | 0.062 | 0.005 | 0.009 | 0.079 | 0.051
ER_annual | 0.054 | 0.003 | 0.002 | 0.021 | 0.002
fraction_12C | 0.133 | 0.141 | 0.118 | 0.064 | 0.088

## Best family per parameter (network mean)

- **$\tau_{\rm active}$** best: `fraction_12C` (0.13, 36% of param). Runner-up: `resp_14C` (0.06).
- **$\tau_{\rm slow}$** best: `fraction_12C` (0.14, 41% of param). Runner-up: `C_stocks` (0.09).
- **$\tau_{\rm passive}$** best: `bulk_14C` (0.16, 43% of param). Runner-up: `fraction_12C` (0.12).
- **$f_{a\to s}$** best: `C_stocks` (0.15, 44% of param). Runner-up: `resp_14C` (0.08).
- **$f_{s\to p}$** best: `bulk_14C` (0.11, 35% of param). Runner-up: `fraction_12C` (0.09).

## Dominant family per (biome, parameter)

Biome | tau_active | tau_slow | tau_passive | f_a_to_s | f_s_to_p
---|---|---|---|---|---
Arctic / permafrost | resp_14C (0.11) | bulk_14C (0.17) | bulk_14C (0.29) | resp_14C (0.43) | resp_14C (0.41)
Boreal | C_stocks (0.07) | C_stocks (0.10) | bulk_14C (0.15) | C_stocks (0.17) | bulk_14C (0.12)
Peatland | resp_14C (0.34) | C_stocks (0.05) | bulk_14C (0.08) | C_stocks (0.08) | bulk_14C (0.06)
Temperate forest | fraction_12C (0.18) | fraction_12C (0.18) | fraction_12C (0.18) | C_stocks (0.14) | fraction_12C (0.13)
Tropical | fraction_12C (0.11) | fraction_12C (0.12) | bulk_14C (0.18) | C_stocks (0.17) | bulk_14C (0.12)
Grassland / Med. | fraction_12C (0.24) | fraction_12C (0.28) | fraction_12C (0.19) | C_stocks (0.18) | fraction_12C (0.14)
