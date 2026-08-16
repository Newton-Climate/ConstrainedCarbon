"""Backwards-compat shim; see ``ecosystem_complexity.visualize.io``."""
from ecosystem_complexity.visualize.io import *  # noqa: F401,F403
from ecosystem_complexity.visualize.io import (  # noqa: F401
    TableSource,
    expand_table_sources,
    load_table,
    load_tables,
)
