"""Tests for episodic_pivot strategy (Chris Flanders EP)."""

import polars as pl
from datetime import date, timedelta

import trendspec.strategy.examples  # noqa: F401 — triggers @register_strategy decorators

from trendspec.data.markets import Market
from trendspec.strategy.base import create_strategy, get_strategy
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.examples.episodic_pivot import EpisodicPivot


def test_strategy_registered() -> None:
    """Strategy registers under name `episodic_pivot` and instance has expected defaults."""
    cls = get_strategy("episodic_pivot")
    assert cls is not None
    assert cls.name == "episodic_pivot"

    instance = create_strategy("episodic_pivot")
    assert instance.get_param("gap_pct") == 0.05
    assert instance.get_param("volume_multiplier") == 3.0
    assert instance.get_param("max_positions") == 10


def _make_bars(iid: str, n: int = 250, start_close: float = 100.0) -> pl.DataFrame:
    """Synthetic OHLCV: gentle uptrend, predictable values for cache assertions."""
    rows = []
    close = start_close
    # Extract ticker from instrument_id (e.g., "AAPL_US" -> "AAPL")
    ticker = iid.split("_")[0]
    for i in range(n):
        bd = date(2024, 1, 1) + timedelta(days=i)
        rows.append({
            "instrument_id": iid,
            "date": bd,
            "ticker": ticker,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
            "adj_factor": 1.0,
        })
        close *= 1.001
    return pl.DataFrame(rows)


def test_init_precomputes_indicators_and_caches() -> None:
    """init() populates indicator cache and per-iid OHLCV/date dicts."""
    df = _make_bars("AAPL_US", n=250)
    strat = EpisodicPivot()
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)

    # Indicator cache populated
    assert strat._iid_dates.get("AAPL_US") is not None
    assert len(strat._iid_dates["AAPL_US"]) == 250

    # OHLCV cache: a date in middle of series
    mid_date = df["date"][125]
    bar = strat._iid_ohlcv["AAPL_US"][mid_date]
    assert "close" in bar and "high" in bar and "low" in bar and "open" in bar and "volume" in bar

    # ADV20 available via context (proves precompute ran)
    adv = ctx.indicator_value("ADV", "AAPL_US", df["date"][50], period=20)
    assert adv is not None and adv > 0


def test_prev_bar_returns_t_minus_1_ohlcv() -> None:
    """_prev_bar(iid, T) returns OHLCV dict for the previous trading day."""
    df = _make_bars("AAPL_US", n=10)
    ctx = StrategyContext(market=Market.US, strategy=EpisodicPivot(), data=df)
    strat = EpisodicPivot()
    strat.init(ctx)

    t_date = df["date"][5]
    prev_date = df["date"][4]
    prev_bar = strat._prev_bar("AAPL_US", t_date)
    assert prev_bar is not None
    assert prev_bar["close"] == strat._iid_ohlcv["AAPL_US"][prev_date]["close"]


def test_prev_bar_returns_none_at_first_bar() -> None:
    """First bar in series has no T-1."""
    df = _make_bars("AAPL_US", n=10)
    ctx = StrategyContext(market=Market.US, strategy=EpisodicPivot(), data=df)
    strat = EpisodicPivot()
    strat.init(ctx)

    first_date = df["date"][0]
    assert strat._prev_bar("AAPL_US", first_date) is None


def test_prev_bar_returns_none_for_unknown_iid() -> None:
    df = _make_bars("AAPL_US", n=10)
    ctx = StrategyContext(market=Market.US, strategy=EpisodicPivot(), data=df)
    strat = EpisodicPivot()
    strat.init(ctx)

    assert strat._prev_bar("UNKNOWN", df["date"][5]) is None
