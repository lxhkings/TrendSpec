"""Tests for EMACluster Pullback strategy."""
import pytest


def test_strategy_registered():
    """Strategy registers under the name 'ema_cluster_pullback'."""
    from trendspec.strategy.base import get_strategy
    import trendspec.strategy.examples.ema_cluster_pullback  # noqa: F401

    cls = get_strategy("ema_cluster_pullback")
    assert cls is not None
    assert cls.name == "ema_cluster_pullback"


def test_strategy_default_params():
    """Strategy ships with spec's default param values."""
    from trendspec.strategy.examples.ema_cluster_pullback import EMAClusterPullback
    s = EMAClusterPullback()
    assert s.get_param("ema_short") == 20
    assert s.get_param("ema_mid") == 60
    assert s.get_param("ema_long") == 120
    assert s.get_param("daily_cluster_threshold") == 0.04
    assert s.get_param("weekly_proximity_threshold") == 0.025
    assert s.get_param("stop_loss_pct") == 0.08
    assert s.get_param("confirmation_days") == 2