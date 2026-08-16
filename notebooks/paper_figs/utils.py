"""Backwards-compat shim; see ``ecosystem_complexity.visualize.utils``."""
from ecosystem_complexity.visualize.utils import *  # noqa: F401,F403
from ecosystem_complexity.visualize.utils import (  # noqa: F401
    DataLike,
    add_one_to_one,
    bootstrap_mean_ci,
    close_or_show,
    coerce_table,
    finalize_figure,
    maybe_add_zero_line,
    order_categories,
    panelize,
    prepare_output_paths,
    select_observation_subset,
    setup_figure_config,
    standard_figure_parser,
    summarize_quantiles,
)
