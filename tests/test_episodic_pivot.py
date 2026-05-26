"""Tests for episodic_pivot strategy (Chris Flanders EP)."""

import trendspec.strategy.examples  # noqa: F401 — triggers @register_strategy decorators

from trendspec.strategy.base import create_strategy, get_strategy


def test_strategy_registered() -> None:
    """Strategy registers under name `episodic_pivot` and instance has expected defaults."""
    cls = get_strategy("episodic_pivot")
    assert cls is not None
    assert cls.name == "episodic_pivot"

    instance = create_strategy("episodic_pivot")
    assert instance.get_param("gap_pct") == 0.05
    assert instance.get_param("volume_multiplier") == 3.0
    assert instance.get_param("max_positions") == 10
