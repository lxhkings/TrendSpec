"""Tests for bars() frequency parameter."""
import tempfile
from datetime import date

import polars as pl
import pytest
from sqlalchemy import create_engine, text


@pytest.fixture
def lake_with_weekly():
    """Build a data_lake with weekly Parquet via the weekly ingestor."""
    from trendspec.data.markets import Market
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.stocks_db_ingestor import ingest_us_weekly

    with tempfile.TemporaryDirectory() as d:
        engine = create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE weekly_prices (ticker TEXT, date DATE, open REAL, high REAL, low REAL, close REAL, volume INTEGER)"))
            conn.execute(text("CREATE TABLE index_constituents (index_id TEXT, snapshot_date DATE, ticker TEXT)"))
            conn.execute(text("INSERT INTO index_constituents VALUES ('SP500', '2024-01-01', 'AAPL')"))
            conn.execute(text("""
                INSERT INTO weekly_prices VALUES
                ('AAPL', '2024-01-05', 180.0, 188.0, 179.0, 187.0, 250000000),
                ('AAPL', '2024-01-12', 187.0, 192.0, 185.0, 190.0, 260000000)
            """))
            conn.commit()
        manifest = Manifest(Market.US, d)
        ingest_us_weekly(engine, manifest, d, full_sync=True)
        yield d


def test_bars_loads_weekly_frequency(lake_with_weekly):
    """bars(frequency='weekly') returns weekly Parquet data."""
    from trendspec.data.markets import Market
    from trendspec.data.parquet_loader import bars

    df = bars(market=Market.US, frequency="weekly", root=lake_with_weekly)
    assert len(df) == 2
    assert df["instrument_id"].unique().to_list() == ["AAPL"]
    assert sorted(df["date"].to_list()) == [date(2024, 1, 5), date(2024, 1, 12)]


def test_bars_defaults_to_daily(lake_with_weekly):
    """bars() without frequency arg still loads daily (backward compat)."""
    from trendspec.data.markets import Market
    from trendspec.data.parquet_loader import bars

    # No daily data → empty DataFrame
    df = bars(market=Market.US, root=lake_with_weekly)
    assert df.is_empty()