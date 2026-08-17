# Deprecated notebook workflows

The files below are retained as reproducible research records, but their
standard operational workflow is now provided by the `ecosys` app commands.
For new runs, use the corresponding command rather than adapting a notebook.

| Notebook(s) | Replacement |
| --- | --- |
| `barrow_model.py`, `harvard_optimal_model.py`, `invert_hf_sierra.py`, `sites/barrow.py`, `sites/canonical.py`, `sites/eight_mile_lake.py`, `sites/harvard_forest.py`, `sites/howland_forest.py` | `ecosys optimize` for site inversions; `ecosys model` for forward-model checks |
| `uncertainty_projections.py` | `ecosys mcmc` for posterior sampling and `ecosys warming` for projections |
| `averaging_kernel_figure.py`, `bulk_respired_on_par_with_fraction_14c.py`, `compare_14c_pathways.py`, `cross_site_information.py`, `export_four_site_information_tables.py`, `export_fraction_12c_ladder.py`, `export_fraction_12c_shapley.py`, `export_multisite_constraint_ladder.py`, `export_oe_constraint_ladder.py`, `four_site_ak_gain_figures.py`, `gain_matrix_observation_key.py`, `gain_obs_metadata.py`, `hf_bulk_vs_fraction_14c_information.py`, `multisite_information_metrics.py` | `ecosys information` for standard constraint and information-content analyses |
| `download_soilgrids_soc.py`, `clm/download_clm.py` | `ecosys fetch` for supported data staging |
| `paper_figs/build_cross_ecosystem_figures.py` | `ecosys report cross-ecosystem` for the standard summary bundle |

The remaining notebooks are exploratory methods, specialized data preparation,
or publication-figure scripts and are not deprecated by this registry.
