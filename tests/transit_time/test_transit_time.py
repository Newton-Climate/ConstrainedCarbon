import numpy as np

from ecosystem_complexity.transit_time import (
    intrinsic_mean_transit_time,
    realized_mean_transit_time,
)


def _two_pool_logits() -> np.ndarray:
    # pool 0: 25% transfers to pool 1; pool 1 respires all outflux.
    return np.log(np.array([[1e-12, 0.25, 0.75], [1e-12, 1e-12, 1.0]]))


def test_intrinsic_transit_time_includes_transfer_pathway():
    mean, by_pool = intrinsic_mean_transit_time(
        np.log(np.array([2.0, 10.0])), _two_pool_logits(), np.array([1.0, 0.0])
    )
    assert np.allclose(by_pool, [4.5, 10.0])
    assert np.isclose(mean, 4.5)


def test_realized_transit_time_matches_constant_modifier_scaling():
    intrinsic, _ = intrinsic_mean_transit_time(
        np.log(np.array([365.25, 3652.5])), _two_pool_logits(), np.array([1.0, 0.0])
    )
    realized, by_phase = realized_mean_transit_time(
        np.log(np.array([365.25, 3652.5])),
        _two_pool_logits(),
        np.full((12, 2), 0.5),
        np.ones(12),
        np.array([1.0, 0.0]),
    )
    assert np.isclose(realized, intrinsic / 0.5, rtol=1e-8)
    assert by_phase.shape == (12, 2)


def test_realized_transit_time_respects_input_timing():
    # With a one-pool system, the same annual modifiers give a shorter mean
    # when carbon enters during the warm (fast-decay) phase.
    logits = np.log(np.array([[1e-12, 1.0]]))
    modifier = np.array([[0.25], [2.0]])
    warm_input, _ = realized_mean_transit_time(
        np.log(np.array([100.0])), logits, modifier, np.array([0.0, 1.0]), np.array([1.0])
    )
    cold_input, _ = realized_mean_transit_time(
        np.log(np.array([100.0])), logits, modifier, np.array([1.0, 0.0]), np.array([1.0])
    )
    assert warm_input < cold_input
