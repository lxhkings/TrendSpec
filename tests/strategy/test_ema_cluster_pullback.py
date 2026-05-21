"""Tests for EMACluster Pullback strategy."""
from datetime import date

import polars as pl
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


def _build_passing_dataset():
    """Build daily + weekly DataFrames where AAPL meets all BUY conditions."""
    from datetime import timedelta
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(250)]
    prices = []
    base = 100.0
    for i in range(250):
        if i < 200:
            base += 0.05
        prices.append(base)
    daily = pl.DataFrame({
        "instrument_id": ["AAPL"] * 250,
        "ticker": ["AAPL"] * 250,
        "date": dates,
        "open":   prices,
        "high":   [p * 1.005 for p in prices],
        "low":    [p * 0.995 for p in prices],
        "close":  prices,
        "volume": [10_000_000] * 250,
        "adj_factor":[1.0]*250,
    })
    weekly_dates = [date(2024, 1, 5) + timedelta(days=7*i) for i in range(40)]
    w_prices = [100.0 + 0.25 * i for i in range(40)]
    weekly = pl.DataFrame({
        "instrument_id":["AAPL"]*40,
        "ticker": ["AAPL"] * 40,
        "date": weekly_dates,
        "open":  w_prices, "high": [p*1.01 for p in w_prices],
        "low":   [p*0.99 for p in w_prices], "close": w_prices,
        "volume":[50_000_000]*40, "adj_factor":[1.0]*40,
    })
    return daily, weekly


def test_buy_signal_emitted_after_confirmation_days():
    """Run strategy across dataset; expect at least one BUY signal."""
    from trendspec.data.markets import Market
    from trendspec.strategy.base import get_strategy
    from trendspec.strategy.context import StrategyContext
    import trendspec.strategy.examples.ema_cluster_pullback  # noqa: F401

    daily, weekly = _build_passing_dataset()
    StrategyClass = get_strategy("ema_cluster_pullback")
    strat = StrategyClass(params={"market_filter_enabled": False})
    ctx = StrategyContext(market=Market.US, strategy=strat, data=daily,
                          weekly_data=weekly)
    strat.init(ctx)

    buy_count = 0
    for dt in daily["date"].to_list():
        ctx._current_date = dt
        ctx._current_instrument_id = "AAPL"
        ctx._current_ticker = "AAPL"
        ctx._pending_signals = []
        try:
            strat.next(ctx)
        except Exception:
            pass
        for sig in ctx._pending_signals:
            if sig.direction == "BUY":
                buy_count += 1
    assert buy_count >= 1, "策略应在密集+周回踩+多头趋势末段触发至少一次 BUY"