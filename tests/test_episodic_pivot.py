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


def _make_ep_setup_bars(iid: str = "AAPL_US") -> pl.DataFrame:
    """
    Build OHLCV that satisfies all 6 BUY conditions on the last bar (T).

    Layout:
      - Bars 0..200: flat consolidation around price 100 with low volatility (base compression)
      - Bars 201..220: very tight base (ATR10 small)
      - Bar 221 (T): gap-up open 105, close 108, volume 5M, high 108.5, low 105
    """
    rows = []
    base_price = 100.0
    start = date(2024, 1, 1)
    ticker = iid.split("_")[0]

    # Long flat consolidation: 220 bars, drift up slowly so EMA50 > EMA200
    for i in range(221):
        d = start + timedelta(days=i)
        # Gentle uptrend so EMA50 > EMA200 by bar 200+
        drift = 1.0 + (i * 0.0003)
        close_i = base_price * drift
        # Very tight intraday range for last 30 bars (base compression)
        range_factor = 0.005 if i >= 190 else 0.015
        rows.append({
            "instrument_id": iid,
            "date": d,
            "ticker": ticker,
            "open": close_i,
            "high": close_i * (1 + range_factor),
            "low": close_i * (1 - range_factor),
            "close": close_i,
            "volume": 1_000_000,
            "adj_factor": 1.0,
        })

    # T bar (gap day): index 221
    t_date = start + timedelta(days=221)
    prev_close = rows[-1]["close"]
    gap_open = prev_close * 1.06       # 6% gap up
    t_high = gap_open * 1.03
    t_low = gap_open
    # close_in_range = (t_close - t_low) / (t_high - t_low) >= 0.80
    # Set t_close near high: close_in_range = 0.97
    t_range = t_high - t_low
    t_close = t_low + 0.97 * t_range
    rows.append({
        "instrument_id": iid,
        "date": t_date,
        "ticker": ticker,
        "open": gap_open,
        "high": t_high,
        "low": t_low,
        "close": t_close,
        "volume": 6_000_000,           # 6x the 1M baseline
        "adj_factor": 1.0,
    })

    return pl.DataFrame(rows)


def test_buy_all_conditions_pass() -> None:
    df = _make_ep_setup_bars()
    iid = "AAPL_US"
    t_date = df["date"][-1]

    strat = EpisodicPivot()
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)

    assert strat._check_buy(ctx, iid, t_date) is True


def test_buy_rejected_no_gap() -> None:
    df = _make_ep_setup_bars()
    # Override T bar to no gap
    last_idx = df.height - 1
    prev_close = df["close"][last_idx - 1]
    df = df.with_columns(
        pl.when(pl.col("date") == df["date"][-1])
        .then(pl.lit(prev_close * 1.01))  # 1% gap (below 5% threshold)
        .otherwise(pl.col("open"))
        .alias("open")
    )
    strat = EpisodicPivot()
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)
    assert strat._check_buy(ctx, "AAPL_US", df["date"][-1]) is False


def test_buy_rejected_no_volume() -> None:
    df = _make_ep_setup_bars()
    df = df.with_columns(
        pl.when(pl.col("date") == df["date"][-1])
        .then(pl.lit(1_500_000))  # only 1.5x baseline (below 3x ADV20)
        .otherwise(pl.col("volume"))
        .alias("volume")
    )
    strat = EpisodicPivot()
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)
    assert strat._check_buy(ctx, "AAPL_US", df["date"][-1]) is False


def test_buy_rejected_close_low_in_range() -> None:
    df = _make_ep_setup_bars()
    # Force close near low (close_in_range approx 0.1)
    t_date = df["date"][-1]
    t_high = df["high"][-1]
    t_low = df["low"][-1]
    bad_close = t_low + (t_high - t_low) * 0.1
    df = df.with_columns(
        pl.when(pl.col("date") == t_date)
        .then(pl.lit(bad_close))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    strat = EpisodicPivot()
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)
    assert strat._check_buy(ctx, "AAPL_US", t_date) is False


def test_buy_rejected_no_trend() -> None:
    """Force EMA50 < EMA200 by making history a downtrend."""
    rows = []
    base_price = 200.0
    start = date(2024, 1, 1)
    ticker = "AAPL"
    for i in range(221):
        d = start + timedelta(days=i)
        # Downtrend: price drops over time -> EMA50 < EMA200
        close_i = base_price * (1 - i * 0.001)
        rows.append({
            "instrument_id": "AAPL_US",
            "date": d,
            "ticker": ticker,
            "open": close_i,
            "high": close_i * 1.005,
            "low": close_i * 0.995,
            "close": close_i,
            "volume": 1_000_000,
            "adj_factor": 1.0,
        })
    # Same EP-style T bar
    t_date = start + timedelta(days=221)
    prev_close = rows[-1]["close"]
    rows.append({
        "instrument_id": "AAPL_US",
        "date": t_date,
        "ticker": ticker,
        "open": prev_close * 1.06,
        "high": prev_close * 1.09,
        "low": prev_close * 1.06,
        "close": prev_close * 1.085,
        "volume": 6_000_000,
        "adj_factor": 1.0,
    })
    df = pl.DataFrame(rows)

    strat = EpisodicPivot()
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)
    assert strat._check_buy(ctx, "AAPL_US", t_date) is False


def test_buy_rejected_no_base_compression() -> None:
    """Force ATR10 approx ATR30 (no compression)."""
    df = _make_ep_setup_bars()
    # Replace last 30 bars' range to be wide (no compression)
    n = df.height
    new_rows = df.to_dicts()
    for i in range(n - 31, n - 1):
        close_i = new_rows[i]["close"]
        new_rows[i]["high"] = close_i * 1.04   # wide range
        new_rows[i]["low"] = close_i * 0.96
    df2 = pl.DataFrame(new_rows)
    strat = EpisodicPivot()
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df2)
    strat.init(ctx)
    assert strat._check_buy(ctx, "AAPL_US", df2["date"][-1]) is False


def test_buy_rejected_low_liquidity() -> None:
    df = _make_ep_setup_bars()
    # Override entire volume to be too small for $20M ADV20
    # At price approx 100, need volume x 100 = 20M -> volume = 200K. Use 50K to fail.
    df = df.with_columns(pl.lit(50_000).alias("volume"))
    # But still need a 3x volume spike on T bar (relative to the baseline 50K -> T = 150K)
    df = df.with_columns(
        pl.when(pl.col("date") == df["date"][-1])
        .then(pl.lit(200_000))
        .otherwise(pl.col("volume"))
        .alias("volume")
    )
    strat = EpisodicPivot()
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)
    assert strat._check_buy(ctx, "AAPL_US", df["date"][-1]) is False
