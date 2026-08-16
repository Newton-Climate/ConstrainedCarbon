"""Backwards-compat shim; see ``ecosystem_complexity.visualize.cross_ecosystem``."""
from ecosystem_complexity.biome import (  # noqa: F401
    BIOME_GROUP_COLORS,
    BIOME_GROUP_LABELS,
    BIOME_GROUP_ORDER,
    biome_group as _biome_group,
)
from ecosystem_complexity.visualize.cross_ecosystem import *  # noqa: F401,F403
from ecosystem_complexity.visualize.cross_ecosystem import (  # noqa: F401
    FAMILY_COLORS,
    FAMILY_LABELS,
    FAMILY_ORDER,
    _load_new_site_tables,
    build_cross_ecosystem_tables,
    make_figure_09,
)
