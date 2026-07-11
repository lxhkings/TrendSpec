"""Tests for weekly ingestor."""
import tempfile

import pytest


@pytest.fixture
def temp_root():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_ingest_us_weekly_writes_parquet(weekly_prices_db, temp_root):
    """US weekly Parquet has correct schema and data."""
    from trendspec.data.markets import Market
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.stocks_db_ingestor import ingest_us_weekly

    manifest = Manifest(Market.US, temp_root)
    result = ingest_us_weekly(weekly_prices_db, manifest, temp_root, full_sync=True)

    assert result["row_count"] == 3   # 2 AAPL + 1 MSFT
    assert result["instrument_count"] == 2

    from trendspec.data.parquet_loader import scan_parquet
    lf = scan_parquet(temp_root, Market.US, "weekly")
    df = lf.collect()
    assert set(df.columns) >= {"instrument_id", "date", "open", "high", "low",
                                "close", "volume", "adj_factor"}
    assert df["adj_factor"].unique().to_list() == [1.0]
    assert sorted(df["instrument_id"].unique().to_list()) == ["AAPL", "MSFT"]


def test_ingest_cn_weekly_derives_instrument_id(weekly_prices_db, temp_root):
    """CN weekly produces SH/SZ-prefixed instrument_id."""
    from trendspec.data.markets import Market
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_weekly

    manifest = Manifest(Market.CN, temp_root)
    result = ingest_cn_weekly(weekly_prices_db, manifest, temp_root, full_sync=True)

    assert result["row_count"] == 2
    assert result["instrument_count"] == 2

    from trendspec.data.parquet_loader import scan_parquet
    df = scan_parquet(temp_root, Market.CN, "weekly").collect()
    assert sorted(df["instrument_id"].unique().to_list()) == ["SH600000", "SZ000001"]
