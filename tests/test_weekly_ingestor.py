"""Tests for weekly ingestor."""
import tempfile
import polars as pl
import pytest
from sqlalchemy import create_engine, text


@pytest.fixture
def stocks_db_with_weekly():
    """SQLite mock of Synology DB with weekly_prices table."""
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE prices (ticker TEXT, date DATE, open REAL, high REAL,
                                  low REAL, close REAL, volume INTEGER)
        """))
        conn.execute(text("""
            CREATE TABLE weekly_prices (ticker TEXT, date DATE, open REAL, high REAL,
                                         low REAL, close REAL, volume INTEGER)
        """))
        conn.execute(text("""
            CREATE TABLE stocks (ticker TEXT PRIMARY KEY, exchange TEXT,
                                  gics_sector TEXT, gics_industry TEXT, is_active INTEGER)
        """))
        conn.execute(text("""
            CREATE TABLE index_constituents (index_id TEXT, snapshot_date DATE, ticker TEXT)
        """))
        conn.execute(text("""
            INSERT INTO index_constituents VALUES
            ('SP500', '2024-01-01', 'AAPL'),
            ('SP500', '2024-01-01', 'MSFT')
        """))
        conn.execute(text("""
            INSERT INTO stocks VALUES
            ('AAPL', 'NYSE', 'Tech', 'Hardware', 1),
            ('MSFT', 'Nasdaq', 'Tech', 'Software', 1),
            ('600000', 'SSE', 'Financials', 'Banks', 1),
            ('000001', 'SZSE', 'Financials', 'Banks', 1)
        """))
        conn.execute(text("""
            INSERT INTO weekly_prices VALUES
            ('AAPL',   '2024-01-05', 180.0, 188.0, 179.0, 187.0, 250000000),
            ('AAPL',   '2024-01-12', 187.0, 192.0, 185.0, 190.0, 260000000),
            ('MSFT',   '2024-01-05', 365.0, 375.0, 364.0, 373.0, 100000000),
            ('600000', '2024-01-05', 7.0,   7.3,   6.9,   7.2,   50000000),
            ('000001', '2024-01-05', 10.0,  10.5,  9.9,   10.3,  80000000)
        """))
        conn.commit()
    yield engine


@pytest.fixture
def temp_root():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_ingest_us_weekly_writes_parquet(stocks_db_with_weekly, temp_root):
    """US weekly Parquet has correct schema and data."""
    from trendspec.data.markets import Market
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.stocks_db_ingestor import ingest_us_weekly

    manifest = Manifest(Market.US, temp_root)
    result = ingest_us_weekly(stocks_db_with_weekly, manifest, temp_root, full_sync=True)

    assert result["row_count"] == 3   # 2 AAPL + 1 MSFT
    assert result["instrument_count"] == 2

    from trendspec.data.parquet_loader import scan_parquet
    lf = scan_parquet(temp_root, Market.US, "weekly")
    df = lf.collect()
    assert set(df.columns) >= {"instrument_id", "date", "open", "high", "low",
                                "close", "volume", "adj_factor"}
    assert df["adj_factor"].unique().to_list() == [1.0]
    assert sorted(df["instrument_id"].unique().to_list()) == ["AAPL", "MSFT"]