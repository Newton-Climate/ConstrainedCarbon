from ecosystem_complexity.oe_diagnostics import (
    ALL_FAMILIES,
    LADDER_STEPS,
    classify_block,
    ladder_family,
)


def test_er_annual_block_is_classified_as_its_own_family() -> None:
    assert classify_block("er_annual") == "ER_annual"
    assert ladder_family("er_annual") == "ER_annual"


def test_er_annual_is_included_in_network_diagnostic_families() -> None:
    assert "ER_annual" in ALL_FAMILIES
    # ER_annual is no longer the terminal rung (inc_rate was added after it),
    # but it must still be carried by the fully-loaded rung.
    assert "ER_annual" in LADDER_STEPS[-1][1]


def test_incubation_rate_blocks_get_their_own_family() -> None:
    """``israd_inc_rate_*`` must not fall through to the generic ``israd`` prefix.

    The catch-all maps ``israd*`` to bulk_14C, so a mis-ordered prefix table
    would silently book a *rate* constraint as radiocarbon — the DFS
    attribution would still sum correctly and nothing would look wrong.
    """
    assert ladder_family("israd_inc_rate_15C") == "inc_rate"
    assert ladder_family("israd_inc_rate_4C") == "inc_rate"
    # neighbours in the same table stay put
    assert ladder_family("israd_bulk_1996") == "bulk_14C"
    assert ladder_family("israd_fraction_heavy") == "fraction_14C"


def test_inc_rate_is_a_family_and_the_terminal_rung() -> None:
    assert "inc_rate" in ALL_FAMILIES
    assert LADDER_STEPS[-1][1][-1] == "inc_rate"
    # every earlier family survives into the final rung
    assert set(LADDER_STEPS[-2][1]).issubset(set(LADDER_STEPS[-1][1]))
