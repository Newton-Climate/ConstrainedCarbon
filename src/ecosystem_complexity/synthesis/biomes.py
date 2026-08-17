"""Biome-group taxonomy shared across analysis and plotting code.

Single source of truth for the six-plus-``other`` grouping used by the
network inversion, MCMC posterior analysis, cross-ecosystem summary,
and paper Figure 09 / 10 renderers. The keyword sets below were
consolidated from three previously divergent copies (in fig_09,
mcmc.chain, and network.inversions) — this module is the union so no
site can silently fall into ``other`` because one call site missed a
keyword.
"""
from __future__ import annotations

BIOME_GROUP_ORDER: list[str] = [
    "arctic_permafrost",
    "boreal",
    "peatland",
    "temperate_forest",
    "grassland_mediterranean",
    "tropical",
]

BIOME_GROUP_LABELS: dict[str, str] = {
    "arctic_permafrost": "Arctic / permafrost",
    "boreal": "Boreal",
    "peatland": "Peatland",
    "temperate_forest": "Temperate forest",
    "grassland_mediterranean": "Grassland / Mediterranean",
    "tropical": "Tropical",
    "other": "Other",
}

BIOME_GROUP_COLORS: dict[str, str] = {
    "arctic_permafrost": "#355C7D",
    "boreal": "#6C8E4E",
    "peatland": "#7A4E7A",
    "temperate_forest": "#C06C2B",
    "grassland_mediterranean": "#C89B2B",
    "tropical": "#2C8C7B",
    "other": "#808080",
}


def biome_group(biome: str) -> str:
    """Bucket a free-text biome string into a canonical group key.

    Keyword coverage is the union of every prior duplicate — including
    ``cropland`` and ``shrubland`` for the grassland/mediterranean
    bucket and ``conifer`` for the temperate-forest bucket. Falls back
    to ``"other"`` when nothing matches.
    """
    b = str(biome).lower()
    if any(k in b for k in ("arctic", "tundra", "permafrost")):
        return "arctic_permafrost"
    if "boreal" in b:
        return "boreal"
    if any(k in b for k in ("peatland", "moss")):
        return "peatland"
    if "tropical" in b:
        return "tropical"
    if any(k in b for k in ("grassland", "mollisol", "mediterranean", "cropland", "shrubland")):
        return "grassland_mediterranean"
    if "temperate" in b or "conifer" in b:
        return "temperate_forest"
    return "other"
