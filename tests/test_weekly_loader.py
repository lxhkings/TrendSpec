"""Tests for bars() frequency parameter."""
import tempfile
from datetime import date

import polars as pl
import pytest


@pytest.fixture
def lake_with_weekly(weekly_prices_db):
    """Build a data_lake with weekly Parquet via the weekly ingestor."""
    from trendspec.data.markets import Market
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.stocks_db_ingestor import ingest_us_weekly

    with tempfile.TemporaryDirectory() as d:
        manifest = Manifest(Market.US, d)
        ingest_us_weekly(weekly_prices_db, manifest, d, full_sync=True)
        yield d


def test_bars_loads_weekly_frequency(lake_with_weekly):
    """bars(frequency='weekly') returns weekly Parquet data (AAPL + MSFT, both SP500)."""
    from trendspec.data.markets import Market
    from trendspec.data.parquet_loader import bars

    df = bars(market=Market.US, frequency="weekly", root=lake_with_weekly)
    assert len(df) == 3   # 2 AAPL + 1 MSFT
    assert sorted(df["instrument_id"].unique().to_list()) == ["AAPL", "MSFT"]
    assert sorted(df.filter(pl.col("instrument_id") == "AAPL")["date"].to_list()) == [
        date(2024, 1, 5), date(2024, 1, 12)
    ]


def test_bars_defaults_to_daily(lake_with_weekly):
    """bars() without frequency arg still loads daily (backward compat)."""
    from trendspec.data.markets import Market
    from trendspec.data.parquet_loader import bars

    # No daily data → empty DataFrame
    df = bars(market=Market.US, root=lake_with_weekly)
    assert df.is_empty()
