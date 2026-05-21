"""End-to-end smoke test for ema_cluster_pullback."""

from datetime import date, timedelta

import polars as pl


def test_screen_run_does_not_crash(tmp_path):
    """ScreeningEngine + ema_cluster_pullback completes without error."""
    import trendspec.strategy.examples  # noqa: F401
    from trendspec.data.markets import Market
    from trendspec.engine.base_engine import EngineConfig
    from trendspec.engine.screening_engine import ScreeningEngine
    from trendspec.ingest.writer import write_parquet
    from trendspec.strategy.base import get_strategy

    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(150)]
    closes = [100.0 + 0.01 * i for i in range(150)]
    daily = pl.DataFrame(
        {
            "instrument_id": ["AAPL"] * 150,
            "date": dates,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [10_000_000] * 150,
            "adj_factor": [1.0] * 150,
            "ticker": ["AAPL"] * 150,
        }
    )
    weekly_dates = [date(2024, 1, 5) + timedelta(days=7 * i) for i in range(25)]
    w_closes = [100.0 + 0.1 * i for i in range(25)]
    weekly = pl.DataFrame(
        {
            "instrument_id": ["AAPL"] * 25,
            "date": weekly_dates,
            "open": w_closes,
            "high": [c * 1.01 for c in w_closes],
            "low": [c * 0.99 for c in w_closes],
            "close": w_closes,
            "volume": [50_000_000] * 25,
            "adj_factor": [1.0] * 25,
            "ticker": ["AAPL"] * 25,
        }
    )
    write_parquet(daily, Market.US, "daily", str(tmp_path), overwrite=True)
    write_parquet(weekly, Market.US, "weekly", str(tmp_path), overwrite=True)

    StrategyClass = get_strategy("ema_cluster_pullback")

    config = EngineConfig(
        market=Market.US,
        start_date=dates[-1],
        end_date=dates[-1],
        root=str(tmp_path),
    )
    engine = ScreeningEngine(config)
    result = engine.run(
        StrategyClass,
        params={"market_filter_enabled": False, "adv_threshold_us": 0},
    )
    assert result is not None
