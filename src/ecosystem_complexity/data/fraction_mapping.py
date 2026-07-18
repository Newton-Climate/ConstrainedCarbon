"""ISRaD fraction → kinetic pool mapping, keyed on the fractionation scheme.

The mapping from an ISRaD fraction to a model pool is a statement about the
**fractionation protocol**, not about a site: "free light" means the same thing
at Blodgett as at Solling. Encoding it per site — as the old
``if israd_name == "Appi forest"`` branch did — makes every new site a code
change and invites the category error that branch contained (it mapped
``residual``, a *chemical-extraction* product, onto a pool otherwise defined by
*density* separation).

ISRaD already labels the protocol in ``frc_scheme``, so the policy is keyed on
that rather than on a hand-maintained list of property strings. A new ISRaD
release that adds a property inherits its scheme's policy automatically instead
of silently falling through. The observed vocabulary is:

    density            free light, heavy, occluded light
    particle size      clay, silt, silt+clay, silt+sand, sand, coarse, fine
    chemical           carbonate, extracted, fulvic/humic acid, humin, residual
    physical (other)   macrofossil, plant material (non-root), roots, pyrogenic C
    aggregate          macroaggregate, microaggregate
    compound specific  lipids

Each scheme gets one of three policies:

``MAP``   the separation isolates carbon whose turnover the model's pools
         represent, so its properties map to pool roles. Density separation is
         the canonical case; particle-size separation is included because the
         size continuum tracks mineral protection — coarse separates hold
         particulate OM that cycles fast, fine separates hold the
         mineral-associated carbon the heavy density fraction isolates.
``BULK``  recognisable plant fragments — not a kinetic fraction of the soil
         matrix, so folded into the whole-sample bulk observation instead.
         Mapping EML's macrofossil to the active pool once asserted +283‰ for a
         τ=2 yr pool while the bulk layers put that pool at −176‰ in the same
         year: a >10σ contradiction.
``SKIP``  no defensible kinetic interpretation. Resistance to a chemical
         reagent, or membership of an aggregate size class, is not the same
         property as slow turnover.

Per-site overrides live in the config's ``datasource.fraction_rules`` and are
merged over this default, so a genuinely unusual protocol stays a config change:

    datasource:
      fraction_rules:
        residual: passive      # map a normally-skipped property
        heavy: null            # or skip a normally-mapped one
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MAP, BULK, SKIP = "map", "bulk", "skip"

# Policy per ISRaD ``frc_scheme``. Unknown schemes are skipped (and logged),
# so new protocols surface rather than being silently reinterpreted.
SCHEME_POLICY: dict[str, str] = {
    "density": MAP,
    "particle size": MAP,
    "physical (other)": SKIP,
    "chemical": SKIP,
    "aggregate": SKIP,
    "compound specific": SKIP,
}

# Property-level refinements, applied over SCHEME_POLICY.
#
# ``physical (other)`` is a grab-bag rather than one protocol, so it cannot take
# a single policy. Its members differ in whether they are soil organic matter at
# all:
#
# * ``macrofossil`` — dead plant fragments genuinely within the soil matrix, so
#   part of what a whole-sample bulk measurement captures. Its Δ¹⁴C spread
#   (−599…+675‰) is real and is carried into the bulk block's σ.
# * ``roots`` — living/recent root biomass, which is not SOM; the standard
#   protocol removes roots before fractionation. Δ¹⁴C averages +155‰ (max +285)
#   because it tracks recent photosynthate, and against free light in the same
#   layer it scatters −73…+157‰ — no stable kinetic relationship. Folding it
#   into bulk biases bulk enriched, so it is excluded.
# * ``plant material (non-root)`` — litter, and therefore *inside* the active
#   pool by construction. The configs set ``aboveground_pools: []`` and
#   ``external_inputs.partition.soil_active: 1.0``, so there is no separate
#   litter reservoir: 100% of fresh plant carbon enters soil_active directly.
#   The pool is defined as litter + fast-cycling SOM, so litter is a valid
#   observation of it. This is borne out at Harvard 2007, the only site with
#   the property: the MAP active pool sits at +78‰, against +101‰ for plant
#   material and +16‰ for free light — the fast pool looks far more like litter
#   than like the free light fraction it was previously constrained by alone.
#   Both map to active and are aggregated per (pool, year), so their spread is
#   carried into the block's σ (84‰ combined vs 88‰ for free light alone —
#   free light is already the more heterogeneous of the two).
# * ``roots`` — excluded, on reservoir membership rather than isotopic grounds.
#   Live root biomass is not part of the soil carbon stock the model tracks, and
#   the standard protocol removes roots before fractionation, so using it as an
#   observation of a *soil* pool equates that pool with living biomass. Note the
#   empirical case is weak in isolation: roots (+75‰) match the MAP active pool
#   (+78‰) closely, because a pool fed entirely by fresh input and a live root
#   both track recent atmosphere. The exclusion is a definitional call.
# * ``pyrogenic c`` — charcoal; distinct kinetics, 1 row.
PROPERTY_POLICY: dict[str, str] = {
    "macrofossil": BULK,
    "plant material (non-root)": MAP,
}

# Kinetic role per property, for the schemes whose policy is MAP. Roles resolve
# against the model's pool_index (see resolve_pool), so a config whose layer is
# not called "soil", or which defines a 4th pool, still works.
#
# Particle-size roles follow the size continuum: coarse/sand separates are
# largely particulate OM (fast), silt-dominated separates are intermediate, and
# clay-bearing separates carry the mineral-associated, slowest carbon.
PROPERTY_ROLES: dict[str, str] = {
    # density
    "free light": "active",
    "occluded light": "slow",
    "heavy": "passive",
    # particle size
    "coarse": "active",
    "sand": "active",
    "silt": "slow",
    "silt+sand": "slow",
    "fine": "passive",
    "clay": "passive",
    "silt+clay": "passive",
    # litter — the active pool receives 100% of fresh plant input, so litter is
    # the leading edge of that pool rather than a reservoir of its own
    "plant material (non-root)": "active",
}


def normalize(value: str) -> str:
    """Canonical form of an ISRaD scheme/property string (trimmed, lowercased)."""
    return str(value).strip().lower()


def policy_for(scheme: str, prop: str) -> str | None:
    """Policy for one (scheme, property) pair; ``None`` if the scheme is unknown.

    Property-level entries win over the scheme default — that is how the
    ``physical (other)`` grab-bag is resolved member by member.
    """
    prop = normalize(prop)
    if prop in PROPERTY_POLICY:
        return PROPERTY_POLICY[prop]
    return SCHEME_POLICY.get(normalize(scheme))


# Properties folded into the bulk observation. Derived from the same policy the
# fraction mapping applies, so the bulk builder and this module cannot drift
# apart into double-counting (or dropping) a row.
BULK_PROPERTIES: frozenset[str] = frozenset(
    p for p, policy in PROPERTY_POLICY.items() if policy == BULK
)


@dataclass(frozen=True)
class FractionMapping:
    """Resolved property → pool-name mapping for one site/model pairing."""

    pool_by_property: dict[str, str]
    skipped: dict[str, str]          # property → why it was skipped

    def as_rules(self) -> list[tuple[str, str]]:
        """(pool_name, property) pairs, ordered for stable block naming."""
        return sorted((pool, prop) for prop, pool in self.pool_by_property.items())


def resolve_pool(role: str, pool_names: list[str]) -> str | None:
    """Find the pool implementing ``role`` (``active`` → ``soil_active``).

    Matches on the role suffix rather than an exact name so the mapping does not
    hard-code the layer prefix from the canonical configs.
    """
    exact = [n for n in pool_names if n == role]
    if exact:
        return exact[0]
    suffixed = [n for n in pool_names if n.rsplit("_", 1)[-1] == role]
    if len(suffixed) == 1:
        return suffixed[0]
    if len(suffixed) > 1:
        # Several layers expose the same role; a depth-resolved config needs an
        # explicit rule to disambiguate.
        logger.warning(
            "fraction mapping: role %r is ambiguous across pools %s — "
            "add an explicit datasource.fraction_rules entry",
            role, suffixed,
        )
    return None


def build_fraction_mapping(
    observed: list[tuple[str, str]],
    pool_names: list[str],
    overrides: dict[str, str | None] | None = None,
) -> FractionMapping:
    """Resolve the (scheme, property) pairs a site reports against its pools.

    ``overrides`` (from ``datasource.fraction_rules``) maps a property to a pool
    role, an explicit pool name, or ``None`` to skip a normally-mapped property.
    Overrides bypass the scheme policy entirely — that is their purpose.
    """
    overrides = {normalize(k): v for k, v in (overrides or {}).items()}

    pool_by_property: dict[str, str] = {}
    skipped: dict[str, str] = {}

    for raw_scheme, raw_prop in observed:
        scheme, prop = normalize(raw_scheme), normalize(raw_prop)

        if prop in overrides:
            target = overrides[prop]
            if target is None:
                skipped[prop] = "skipped by config override"
                continue
            pool = target if target in pool_names else resolve_pool(target, pool_names)
            if pool is None:
                raise ValueError(
                    f"datasource.fraction_rules maps {prop!r} to {target!r}, which is "
                    f"neither a pool name nor a resolvable role in {pool_names}"
                )
            pool_by_property[prop] = pool
            continue

        policy = policy_for(scheme, prop)
        if policy is None:
            skipped[prop] = f"unknown frc_scheme {scheme!r}"
            continue
        if policy == BULK:
            skipped[prop] = f"{scheme} — folded into the bulk observation"
            continue
        if policy == SKIP:
            skipped[prop] = f"{scheme} — no kinetic interpretation"
            continue

        role = PROPERTY_ROLES.get(prop)
        if role is None:
            skipped[prop] = f"{scheme} property has no role assignment"
            continue
        pool = resolve_pool(role, pool_names)
        if pool is None:
            skipped[prop] = f"no pool implements role {role!r}"
            continue
        pool_by_property[prop] = pool

    return FractionMapping(pool_by_property=pool_by_property, skipped=skipped)
