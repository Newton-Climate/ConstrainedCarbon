# Cross-Ecosystem 14C Inversion Summary

Figure: ![Cross-ecosystem summary](../paper_figs/outputs/cross_ecosystem_summary/figures/figure_09.png)

## Scope

- Total unique inverted site-ecosystems: **34**
- Sites with direct warming-vulnerability outputs: **24**
- Additional turnover-only expansion sites: **10**
- Direct-warming DFS range: **0.45 to 3.60** with median **1.23**
- Source tables: `notebooks/exports/network_inversion_fluxcom_er_20260719/site_summary.csv`, `notebooks/exports/warming_vulnerability_fluxcom_er_20260719/site_warming_summary.csv`, `notebooks/exports/new_sites_incubation_20260719.csv`, `notebooks/exports/incubation_new_sites_runnable_20260719.csv`

## Main Results

- The full union sits in a common three-pool turnover regime: active turnover remains near 2 years, while slow and passive pools span the ecological gradient.
- Warming vulnerability is strongest in **boreal systems fractionally** and in **arctic/permafrost systems for old-carbon release and absolute loss**.
- Constrainability is driven primarily by observation family rather than biome identity. Bulk 14C, respired 14C, and annual ER carry most of the leverage where they exist.
- The added incubation-expansion sites mostly extend the **old-C tail** rather than creating a new turnover cluster.
- Mean old-carbon share of excess RH by biome group: Arctic / permafrost: 0.88, Grassland / Mediterranean: 0.84, Tropical: 0.81, Temperate forest: 0.77, Boreal: 0.66, Peatland: 0.46.

## Biome-Level Vulnerability

| biome_label               |   n_sites |   dfs_mean |   frac_loss_mean |   abs_loss_mean |   old_share_mean |
|:--------------------------|----------:|-----------:|-----------------:|----------------:|-----------------:|
| Boreal                    |         2 |       1    |            0.312 |           2,241 |             0.66 |
| Arctic / permafrost       |         2 |       1.98 |            0.259 |           4,596 |             0.88 |
| Tropical                  |         6 |       0.95 |            0.211 |           1,562 |             0.81 |
| Peatland                  |         1 |       1.14 |            0.209 |           1,396 |             0.46 |
| Temperate forest          |        10 |       1.45 |            0.181 |           2,214 |             0.77 |
| Grassland / Mediterranean |         3 |       0.78 |            0.165 |             569 |             0.84 |

## Most Constrained Direct-Warming Sites

| site               | biome_group         |   dfs_total | dominant_family   |
|:-------------------|:--------------------|------------:|:------------------|
| Howland Forest     | Temperate forest    |        3.6  | ER_annual         |
| EML                | Arctic / permafrost |        2.46 | resp_14C          |
| FLONA              | Tropical            |        1.59 | bulk_14C          |
| Adventdalen Valley | Arctic / permafrost |        1.49 | resp_14C          |
| Harvard Forest     | Temperate forest    |        1.49 | resp_14C          |
| Appi forest        | Temperate forest    |        1.43 | bulk_14C          |

## Most Vulnerable Direct-Warming Sites

| site                | biome_group         |   frac_c_loss |   abs_c_loss_gCm2 |   old_fraction_of_excess_rh |
|:--------------------|:--------------------|--------------:|------------------:|----------------------------:|
| CZ_1964burn_NSA     | Boreal              |         0.404 |             2,621 |                        0.75 |
| EML                 | Arctic / permafrost |         0.391 |             6,258 |                        0.83 |
| Harvard Forest      | Temperate forest    |         0.239 |             1,975 |                        0.33 |
| ZF2                 | Tropical            |         0.232 |             1,824 |                        0.8  |
| BCI                 | Tropical            |         0.225 |             1,216 |                        0.82 |
| CZ_Old_Black_Spruce | Boreal              |         0.22  |             1,861 |                        0.56 |

## Long-Tail Expansion Sites

These sites do not yet have direct warming projections in this summary, but they most strongly extend the old-carbon turnover tail.

| site              | biome_group               |   tau_slow_yr |   tau_passive_yr |   n_incubation |
|:------------------|:--------------------------|--------------:|-----------------:|---------------:|
| Dinesen           | Grassland / Mediterranean |          22.1 |            817   |              1 |
| Treynor           | Grassland / Mediterranean |          56.9 |            422.9 |              1 |
| Trumbore Ahwahnee | Grassland / Mediterranean |          70.8 |            294.4 |              1 |
| CZ_1981burn       | Arctic / permafrost       |          20.1 |            253.6 |              1 |
| La Campana        | Grassland / Mediterranean |          27.5 |            205.7 |              1 |
| Biadaski          | Temperate forest          |          20.5 |            181.3 |              1 |

## Constraint Structure by Biome

| biome_group               | dominant_constraint   |   stock_share |   bulk+respired_share |
|:--------------------------|:----------------------|--------------:|----------------------:|
| Arctic / permafrost       | Respired 14C          |          0.15 |                  0.84 |
| Boreal                    | C stocks              |          0.46 |                  0.52 |
| Grassland / Mediterranean | C stocks              |          0.79 |                  0.21 |
| Peatland                  | Respired 14C          |          0.22 |                  0.65 |
| Temperate forest          | Bulk 14C              |          0.37 |                  0.49 |
| Tropical                  | C stocks              |          0.56 |                  0.44 |

## Interpretation

- This ensemble is large enough to show that old-carbon vulnerability is not restricted to permafrost. Boreal, temperate, and tropical systems all mobilize meaningfully old carbon once warming increases RH.
- Several weakly constrained sites remain stock-dominated, so the main limit on inference is still missing radiocarbon geometry, not ecosystem diversity.
- The expansion sites make the passive-pool tail broader and older, which increases confidence that the long-lived tail is a general ecosystem property rather than a four-site artifact.
- The strongest next step is to run standardized warming projections for the turnover-only expansion sites with the largest passive tails, especially `Dinesen`, `Treynor`, `Trumbore Ahwahnee`, `Trumbore Musick`, and `Nahuelbuta`.
