"""Tests for Synology stocks DB custom ingestor."""

import tempfile

import polars as pl
import pytest
from sqlalchemy import create_engine, text

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def stocks_db():
    """SQLite in-memory mock of the Synology stocks DB."""
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE prices (
                ticker TEXT,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
            )
        """))
        conn.execute(text("""
            CREATE TABLE stocks (
                ticker TEXT PRIMARY KEY,
                exchange TEXT,
                gics_sector TEXT,
                gics_industry TEXT,
                is_active INTEGER
            )
        """))
        conn.execute(text("""
            CREATE TABLE constituent_changes (
                index_id TEXT,
                ticker TEXT,
                change_type TEXT,
                change_date DATE
            )
        """))
        conn.execute(text("""
            CREATE TABLE index_constituents (
                index_id TEXT,
                snapshot_date DATE,
                ticker TEXT
            )
        """))
        conn.execute(text("""
            INSERT INTO index_constituents VALUES
            ('SP500', '2024-01-01', 'AAPL'),
            ('SP500', '2024-01-01', 'MSFT'),
            ('RUSSELL1000', '2024-01-01', 'AAPL'),
            ('RUSSELL1000', '2024-01-01', 'MSFT'),
            ('RUSSELL1000', '2024-01-01', 'JPM')
        """))
        # US stocks metadata
        conn.execute(text("""
            INSERT INTO stocks VALUES
            ('AAPL', 'NYSE', 'Information Technology', 'Technology Hardware', 1),
            ('MSFT', 'Nasdaq', 'Information Technology', 'Systems Software', 1),
            ('JPM', 'NYSE', 'Financials', 'Diversified Banks', 1)
        """))
        # US price data
        conn.execute(text("""
            INSERT INTO prices VALUES
            ('AAPL', '2024-01-02', 185.0, 186.0, 183.0, 185.5, 50000000),
            ('AAPL', '2024-01-03', 185.5, 187.0, 184.0, 186.0, 55000000),
            ('MSFT', '2024-01-02', 370.0, 372.0, 368.0, 371.0, 20000000),
            ('MSFT', '2024-01-03', 371.0, 373.0, 369.0, 372.0, 22000000),
            ('JPM',  '2024-01-02', 150.0, 152.0, 149.0, 151.0, 10000000),
            ('JPM',  '2024-01-03', 151.0, 153.0, 150.0, 152.0, 11000000)
        """))
        conn.commit()
    yield engine


@pytest.fixture
def temp_root():
    with tempfile.TemporaryDirectory() as d:
        yield d


# =============================================================================
# US daily tests
# =============================================================================

def test_ingest_us_daily_schema(stocks_db, temp_root):
    """US daily Parquet has correct columns and types."""
    from trendspec.data.markets import Market
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.stocks_db_ingestor import ingest_us_daily

    manifest = Manifest(Market.US, temp_root)
    result = ingest_us_daily(stocks_db, manifest, temp_root)

    assert result["row_count"] == 6
    assert result["instrument_count"] == 3

    df = pl.read_parquet(f"{temp_root}/us/daily/")
    assert set(df.columns) >= {"instrument_id", "date", "ticker", "open", "high", "low", "close", "volume", "adj_factor"}


def test_ingest_us_daily_instrument_id_equals_ticker(stocks_db, temp_root):
    """For US stocks, instrument_id == ticker."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_daily
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    ingest_us_daily(stocks_db, manifest, temp_root)

    df = pl.read_parquet(f"{temp_root}/us/daily/")
    mismatched = df.filter(pl.col("instrument_id") != pl.col("ticker"))
    assert len(mismatched) == 0, f"instrument_id != ticker: {mismatched}"


def test_ingest_us_daily_adj_factor_is_one(stocks_db, temp_root):
    """adj_factor must be 1.0 (prices already adjusted via Yahoo API)."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_daily
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    ingest_us_daily(stocks_db, manifest, temp_root)

    df = pl.read_parquet(f"{temp_root}/us/daily/")
    assert df["adj_factor"].unique().to_list() == [1.0]


def test_ingest_us_daily_incremental(stocks_db, temp_root):
    """Second run with same data is a no-op (already synced)."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_daily
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    r1 = ingest_us_daily(stocks_db, manifest, temp_root)
    assert r1["row_count"] == 6

    manifest2 = Manifest(Market.US, temp_root)  # reload manifest from disk
    r2 = ingest_us_daily(stocks_db, manifest2, temp_root)
    assert r2["row_count"] == 0  # no new rows


def test_ingest_us_daily_since_inclusive(stocks_db, temp_root):
    """`since` filters from that date inclusive, overriding manifest."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_daily
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    # since=2024-01-03 → only the 2024-01-03 rows (3 tickers), not 2024-01-02
    result = ingest_us_daily(stocks_db, manifest, temp_root, since="2024-01-03")

    assert result["row_count"] == 3
    df = pl.read_parquet(f"{temp_root}/us/daily/")
    assert df["date"].cast(pl.Utf8).unique().to_list() == ["2024-01-03"]


# =============================================================================
# US components tests
# =============================================================================

@pytest.fixture
def stocks_db_with_changes(stocks_db):
    """Add SP500 constituent changes to the fixture DB."""
    with stocks_db.connect() as conn:
        conn.execute(text("""
            INSERT INTO constituent_changes VALUES
            ('SP500', 'AAPL', 'ADDED', '2020-01-15'),
            ('SP500', 'MSFT', 'ADDED', '2019-06-01'),
            ('SP500', 'JPM', 'REMOVED', '2023-03-10')
        """))
        conn.commit()
    return stocks_db


def test_ingest_us_components_schema(stocks_db_with_changes, temp_root):
    """US components Parquet has correct columns."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_components
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    result = ingest_us_components(stocks_db_with_changes, manifest, temp_root)

    assert result["row_count"] > 0

    df = pl.read_parquet(f"{temp_root}/us/components/")
    assert set(df.columns) >= {"instrument_id", "date", "event", "event_details"}


def test_ingest_us_components_event_mapping(stocks_db_with_changes, temp_root):
    """ADDED maps to IPO, REMOVED maps to DELIST."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_components
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    ingest_us_components(stocks_db_with_changes, manifest, temp_root)

    df = pl.read_parquet(f"{temp_root}/us/components/")
    events = df["event"].unique().to_list()
    assert "IPO" in events
    assert "DELIST" in events
    assert "ADDED" not in events
    assert "REMOVED" not in events


def test_ingest_us_components_all_tickers_have_ipo(stocks_db_with_changes, temp_root):
    """Every US ticker in prices should have at least one IPO event."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_components
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    ingest_us_components(stocks_db_with_changes, manifest, temp_root)

    df = pl.read_parquet(f"{temp_root}/us/components/")
    ipos = df.filter(pl.col("event") == "IPO")["instrument_id"].unique().to_list()
    assert "AAPL" in ipos
    assert "MSFT" in ipos
    assert "JPM" in ipos


# =============================================================================
# US sectors tests
# =============================================================================

def test_ingest_us_sectors_schema(stocks_db, temp_root):
    """US sectors Parquet has correct columns."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_sectors
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    result = ingest_us_sectors(stocks_db, manifest, temp_root)

    assert result["instrument_count"] == 3

    df = pl.read_parquet(f"{temp_root}/us/sectors/")
    assert set(df.columns) >= {"instrument_id", "date", "sector", "sector_name"}


def test_ingest_us_sectors_static_date(stocks_db, temp_root):
    """All sector rows have assign_date = 2000-01-01."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_sectors
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    ingest_us_sectors(stocks_db, manifest, temp_root)

    df = pl.read_parquet(f"{temp_root}/us/sectors/")
    dates = df["date"].unique().to_list()
    assert len(dates) == 1
    assert str(dates[0]) == "2000-01-01"


# =============================================================================
# CN fixtures + tests
# =============================================================================

@pytest.fixture
def stocks_db_cn(stocks_db):
    """Add CN stocks and prices to the fixture DB."""
    with stocks_db.connect() as conn:
        conn.execute(text("""
            INSERT INTO stocks VALUES
            ('600000', 'SSE', 'Financials', 'Banks', 1),
            ('000001', 'SZSE', 'Financials', 'Banks', 1),
            ('600036', 'SH', 'Financials', 'Banks', 1)
        """))
        conn.execute(text("""
            INSERT INTO index_constituents VALUES
            ('CSI800', '2024-01-01', '600000'),
            ('CSI800', '2024-01-01', '000001'),
            ('CSI800', '2024-01-01', '600036')
        """))
        conn.execute(text("""
            INSERT INTO prices VALUES
            ('600000', '2024-01-02', 10.0, 10.5, 9.8, 10.2, 1000000),
            ('600000', '2024-01-03', 10.2, 10.8, 10.0, 10.5, 1100000),
            ('000001', '2024-01-02', 20.0, 20.5, 19.8, 20.2, 500000),
            ('000001', '2024-01-03', 20.2, 20.8, 20.0, 20.5, 550000),
            ('600036', '2024-01-02', 15.0, 15.5, 14.8, 15.2, 800000),
            ('600036', '2024-01-03', 15.2, 15.8, 15.0, 15.5, 850000)
        """))
        conn.commit()
    return stocks_db


def test_ingest_cn_daily_instrument_id_prefix(stocks_db_cn, temp_root):
    """CN instrument_id has SH/SZ prefix based on exchange."""
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_daily
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.CN, temp_root)
    result = ingest_cn_daily(stocks_db_cn, manifest, temp_root)

    assert result["row_count"] == 6
    assert result["instrument_count"] == 3

    df = pl.read_parquet(f"{temp_root}/cn/daily/")
    ids = sorted(df["instrument_id"].unique().to_list())
    assert "SH600000" in ids
    assert "SZ000001" in ids
    assert "SH600036" in ids  # SH exchange → SH prefix


def test_ingest_cn_daily_adj_factor_is_one(stocks_db_cn, temp_root):
    """CN adj_factor = 1.0 (Tushare prices already backward-adjusted)."""
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_daily
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.CN, temp_root)
    ingest_cn_daily(stocks_db_cn, manifest, temp_root)

    df = pl.read_parquet(f"{temp_root}/cn/daily/")
    assert df["adj_factor"].unique().to_list() == [1.0]


def test_ingest_cn_components_has_ipo_events(stocks_db_cn, temp_root):
    """Every CN ticker gets an IPO event from MIN(date) in prices."""
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_components
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.CN, temp_root)
    result = ingest_cn_components(stocks_db_cn, manifest, temp_root)

    assert result["row_count"] == 3  # one IPO per ticker

    df = pl.read_parquet(f"{temp_root}/cn/components/")
    assert set(df["event"].unique().to_list()) == {"IPO"}
    ids = df["instrument_id"].unique().to_list()
    assert "SH600000" in ids
    assert "SZ000001" in ids


def test_ingest_cn_sectors_maps_to_correct_market(stocks_db_cn, temp_root):
    """CN sectors use CN instrument_id format (SH/SZ prefix)."""
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_sectors
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.CN, temp_root)
    result = ingest_cn_sectors(stocks_db_cn, manifest, temp_root)

    assert result["instrument_count"] == 3

    df = pl.read_parquet(f"{temp_root}/cn/sectors/")
    ids = df["instrument_id"].unique().to_list()
    assert all(id_.startswith(("SH", "SZ")) for id_ in ids)


def test_ingest_cn_sectors_not_limited_to_csi800(stocks_db, temp_root):
    """去掉 CSI800 JOIN 后，非 CSI800 成分股只要 gics_sector 非空也要被摄入。"""
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_sectors
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    with stocks_db.connect() as conn:
        conn.execute(text("""
            INSERT INTO stocks VALUES
            ('600000', 'SSE', 'Financials', 'Banks', 1),
            ('688001', 'SSE', 'Technology', 'Semiconductors', 1)
        """))
        # 只有 600000 在 CSI800，688001 不在任何指数成分表里
        conn.execute(text("""
            INSERT INTO index_constituents VALUES
            ('CSI800', '2024-01-01', '600000')
        """))
        conn.commit()

    manifest = Manifest(Market.CN, temp_root)
    result = ingest_cn_sectors(stocks_db, manifest, temp_root)

    assert result["instrument_count"] == 2  # 600000 + 688001，非 CSI800 的也在

    df = pl.read_parquet(f"{temp_root}/cn/sectors/")
    ids = df["instrument_id"].unique().to_list()
    assert "SH688001" in ids  # 非 CSI800 成分股同样被摄入
