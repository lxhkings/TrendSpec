"""
Shared test fixtures for TrendSpec PIT tests.

Provides:
- Mini Parquet samples (daily, components, sectors)
- SQLite in-memory MariaDB mock
- Sample data with known IPO/delist/halt events
- Sample data with sector reclassifications

Key test cases from plan:
1. 600631 商业城 - Delisted 2016, should be in 2015-06-01 universe
2. Sector reclassification - Before date = sector A, after date = sector B
3. Forward adjustment - Price * (adj_factor / latest_adj_factor)
"""

import os
import tempfile
from datetime import date
from unittest.mock import patch

import polars as pl
import pytest
from sqlalchemy import create_engine, text

from trendspec.config.settings import Settings
from trendspec.data.markets import Market
from trendspec.data.sectors import clear_sector_cache
from trendspec.data.universe.cn import IPO_EVENT, DELIST_EVENT, HALT_EVENT, RESUME_EVENT
from trendspec.ingest.writer import write_parquet


# =============================================================================
# Basic Fixtures
# =============================================================================


@pytest.fixture
def temp_root() -> str:
    """Create temporary directory for data_lake."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_settings(temp_root: str) -> Settings:
    """Create mock settings with temp data_lake root."""
    settings = Settings()
    # Override data_lake_root for tests
    with patch(
        "trendspec.config.settings.get_settings",
        return_value=settings
    ):
        # Manually set data_lake_root
        settings.data_lake.data_lake_root = temp_root
        yield settings


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before each test to ensure isolation."""
    clear_sector_cache()
    yield
    clear_sector_cache()


# =============================================================================
# SQLite In-Memory MariaDB Mock
# =============================================================================


@pytest.fixture
def sqlite_engine():
    """Create SQLite in-memory engine with mock schema."""
    engine = create_engine("sqlite:///:memory:")

    # Create CN_A tables (matching naming convention: {market.path}_{dataset})
    with engine.connect() as conn:
        # Daily OHLCV table
        conn.execute(text("""
            CREATE TABLE cn_daily (
                instrument_id TEXT,
                trade_date DATE,
                ticker TEXT,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                adj_factor REAL
            )
        """))

        # Component events table (IPO, delist, halt)
        conn.execute(text("""
            CREATE TABLE cn_components (
                instrument_id TEXT,
                event_date DATE,
                event_type TEXT,
                event_details TEXT
            )
        """))

        # Sector assignments table
        conn.execute(text("""
            CREATE TABLE cn_sectors (
                instrument_id TEXT,
                assign_date DATE,
                sector_code TEXT,
                sector_name TEXT
            )
        """))

        # US tables
        conn.execute(text("""
            CREATE TABLE us_daily (
                instrument_id TEXT,
                trade_date DATE,
                ticker TEXT,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                adj_factor REAL
            )
        """))

        conn.execute(text("""
            CREATE TABLE us_components (
                instrument_id TEXT,
                event_date DATE,
                event_type TEXT,
                event_details TEXT
            )
        """))

        conn.execute(text("""
            CREATE TABLE us_sectors (
                instrument_id TEXT,
                assign_date DATE,
                sector_code TEXT,
                sector_name TEXT
            )
        """))

        conn.commit()

    yield engine


def insert_sqlite_data(engine, table_name: str, data: list[tuple]) -> None:
    """
    Insert data into SQLite table using proper SQLAlchemy 2.x syntax.
    Helper function for tests.
    """
    with engine.connect() as conn:
        # Column names for each table
        if table_name == "cn_daily":
            columns = ["instrument_id", "trade_date", "ticker", "open_price",
                       "high_price", "low_price", "close_price", "volume", "adj_factor"]
        elif table_name == "cn_components":
            columns = ["instrument_id", "event_date", "event_type", "event_details"]
        elif table_name == "cn_sectors":
            columns = ["instrument_id", "assign_date", "sector_code", "sector_name"]
        elif table_name == "us_daily":
            columns = ["instrument_id", "trade_date", "ticker", "open_price",
                       "high_price", "low_price", "close_price", "volume", "adj_factor"]
        elif table_name == "us_components":
            columns = ["instrument_id", "event_date", "event_type", "event_details"]
        elif table_name == "us_sectors":
            columns = ["instrument_id", "assign_date", "sector_code", "sector_name"]
        else:
            raise ValueError(f"Unknown table: {table_name}")

        # Build INSERT statement
        col_list = ", ".join(columns)
        param_list = ", ".join([f":{col}" for col in columns])
        sql = text(f"INSERT INTO {table_name} ({col_list}) VALUES ({param_list})")

        # Convert data to dicts
        params = [dict(zip(columns, row)) for row in data]
        conn.execute(sql, params)
        conn.commit()


# =============================================================================
# Sample Daily OHLCV Data
# =============================================================================


@pytest.fixture
def cn_daily_basic() -> pl.DataFrame:
    """Basic CN_A daily OHLCV data for 3 instruments."""
    return pl.DataFrame({
        "instrument_id": [
            "SH600000", "SH600000", "SH600000",
            "SZ000001", "SZ000001", "SZ000001",
            "SH600631", "SH600631", "SH600631",
        ],
        "date": [
            date(2015, 1, 5), date(2015, 6, 1), date(2015, 12, 31),
            date(2015, 1, 5), date(2015, 6, 1), date(2015, 12, 31),
            date(2015, 1, 5), date(2015, 6, 1), date(2015, 12, 31),
        ],
        "ticker": [
            "600000", "600000", "600000",
            "000001", "000001", "000001",
            "600631", "600631", "600631",
        ],
        "open": [10.0, 10.5, 11.0, 20.0, 20.5, 21.0, 8.0, 8.5, 9.0],
        "high": [10.5, 11.0, 11.5, 20.5, 21.0, 21.5, 8.5, 9.0, 9.5],
        "low": [9.8, 10.2, 10.8, 19.8, 20.2, 20.8, 7.8, 8.2, 8.8],
        "close": [10.2, 10.8, 11.2, 20.2, 20.8, 21.2, 8.2, 8.8, 9.2],
        "volume": [1000000, 1100000, 1200000, 500000, 550000, 600000, 800000, 850000, 900000],
        "adj_factor": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })


@pytest.fixture
def cn_daily_with_adjustment() -> pl.DataFrame:
    """
    CN_A daily data with corporate action (dividend/split) requiring adjustment.

    Simulates a dividend on Jan 15, 2024:
    - adj_factor changes from 1.0 to 0.95 (5% dividend)
    - Prices drop proportionally

    Forward adjustment formula: adjusted = raw * (adj_factor / latest_adj_factor)
    """
    return pl.DataFrame({
        "instrument_id": ["SH600000"] * 8,
        "date": [
            date(2024, 1, 10), date(2024, 1, 11), date(2024, 1, 12),
            date(2024, 1, 13), date(2024, 1, 14),  # Before dividend
            date(2024, 1, 15),  # Dividend date
            date(2024, 1, 16), date(2024, 1, 17),  # After dividend
        ],
        "ticker": ["600000"] * 8,
        "open": [10.0, 10.1, 10.2, 10.3, 10.4, 9.5, 9.55, 9.6],
        "high": [10.5, 10.6, 10.7, 10.8, 10.9, 9.8, 9.85, 9.9],
        "low": [9.8, 9.9, 10.0, 10.1, 10.2, 9.2, 9.25, 9.3],
        "close": [10.2, 10.3, 10.4, 10.5, 10.6, 9.5, 9.55, 9.6],
        "volume": [1000000, 1100000, 1200000, 1300000, 1400000, 2000000, 1500000, 1600000],
        "adj_factor": [1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.95, 0.95],
    })


@pytest.fixture
def cn_daily_with_split() -> pl.DataFrame:
    """
    CN_A daily data with stock split requiring adjustment.

    Simulates a 2:1 split on March 1, 2024:
    - adj_factor changes from 1.0 to 0.5
    - Prices halved, volume doubled
    """
    return pl.DataFrame({
        "instrument_id": ["SH600000"] * 6,
        "date": [
            date(2024, 2, 26), date(2024, 2, 27), date(2024, 2, 28),  # Before split
            date(2024, 3, 1),  # Split date
            date(2024, 3, 2), date(2024, 3, 3),  # After split
        ],
        "ticker": ["600000"] * 6,
        "open": [100.0, 101.0, 102.0, 50.0, 50.5, 51.0],
        "high": [105.0, 106.0, 107.0, 52.5, 53.0, 53.5],
        "low": [98.0, 99.0, 100.0, 49.0, 49.5, 50.0],
        "close": [102.0, 103.0, 104.0, 51.0, 51.5, 52.0],
        "volume": [500000, 550000, 600000, 1200000, 1100000, 1150000],  # Volume doubled
        "adj_factor": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
    })


# =============================================================================
# Sample Component Events Data (IPO, Delist, Halt)
# =============================================================================


@pytest.fixture
def cn_components_basic() -> pl.DataFrame:
    """Basic CN_A component events for 3 instruments."""
    return pl.DataFrame({
        "instrument_id": ["SH600000", "SZ000001", "SH600631"],
        "date": [date(1999, 11, 10), date(1991, 4, 3), date(1996, 12, 2)],
        "event": [IPO_EVENT, IPO_EVENT, IPO_EVENT],
        "event_details": ["浦发银行 IPO", "平安银行 IPO", "商业城 IPO"],
    })


@pytest.fixture
def cn_components_with_delist() -> pl.DataFrame:
    """
    CN_A component events with delisting - critical for survivorship bias tests.

    Key test case: 600631 商业城 delisted in 2016
    - Should be in 2015-06-01 universe (before delisting)
    - Should NOT be in 2017-01-01 universe (after delisting)
    """
    return pl.DataFrame({
        "instrument_id": [
            "SH600000",  # Active
            "SZ000001",  # Active
            "SH600631",  # Delisted in 2016
            "SH600631",  # Delisting event
            "SZ000002",  # IPO after 2015
        ],
        "date": [
            date(1999, 11, 10),  # SH600000 IPO
            date(1991, 4, 3),     # SZ000001 IPO
            date(1996, 12, 2),    # SH600631 IPO (商业城)
            date(2016, 3, 29),    # SH600631 delist date
            date(2016, 4, 1),     # SZ000002 IPO (after 2015)
        ],
        "event": [
            IPO_EVENT,
            IPO_EVENT,
            IPO_EVENT,
            DELIST_EVENT,  # Delisting event
            IPO_EVENT,
        ],
        "event_details": [
            "浦发银行 IPO",
            "平安银行 IPO",
            "商业城 IPO",
            "商业城 delisted",
            "New stock IPO",
        ],
    })


@pytest.fixture
def cn_components_with_halt() -> pl.DataFrame:
    """
    CN_A component events with trading halt.

    Test case: SH600000 halted from 2024-03-01 to 2024-03-15
    - Should NOT be in active universe during halt period
    - Should be back in universe after RESUME event
    """
    return pl.DataFrame({
        "instrument_id": [
            "SH600000", "SH600000", "SH600000",  # IPO, Halt, Resume
            "SZ000001",  # IPO
        ],
        "date": [
            date(1999, 11, 10),  # IPO
            date(2024, 3, 1),    # Halt start
            date(2024, 3, 15),   # Resume
            date(1991, 4, 3),    # SZ000001 IPO
        ],
        "event": [
            IPO_EVENT,
            HALT_EVENT,
            RESUME_EVENT,
            IPO_EVENT,
        ],
        "event_details": [
            "IPO",
            "Trading suspended",
            "Trading resumed",
            "IPO",
        ],
    })


@pytest.fixture
def cn_components_with_ipos() -> pl.DataFrame:
    """
    CN_A component events with IPO timeline for survivorship bias tests.

    Tests that stocks IPO-ing after a date are NOT in universe at that date.
    """
    return pl.DataFrame({
        "instrument_id": [
            "SH600000",  # IPO 1999
            "SZ000001",  # IPO 1991
            "SH600036",  # IPO 2003
            "SZ002475",  # IPO 2010
            "SH688981",  # IPO 2022 (STAR Market)
        ],
        "date": [
            date(1999, 11, 10),
            date(1991, 4, 3),
            date(2003, 8, 22),
            date(2010, 5, 6),
            date(2022, 1, 28),
        ],
        "event": [IPO_EVENT] * 5,
        "event_details": ["IPO"] * 5,
    })


@pytest.fixture
def us_components_basic() -> pl.DataFrame:
    """Basic US component events."""
    return pl.DataFrame({
        "instrument_id": ["AAPL", "MSFT", "JPM"],
        "date": [date(1980, 12, 12), date(1986, 3, 13), date(1799, 1, 1)],
        "event": [IPO_EVENT, IPO_EVENT, IPO_EVENT],
        "event_details": ["Apple IPO", "Microsoft IPO", "JPMorgan IPO"],
    })


# =============================================================================
# Sample Sector Assignment Data (with Reclassification)
# =============================================================================


@pytest.fixture
def cn_sectors_basic() -> pl.DataFrame:
    """Basic CN_A sector assignments (Shenwan Level 1)."""
    return pl.DataFrame({
        "instrument_id": ["SH600000", "SZ000001", "SH600036"],
        "date": [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 1)],
        "sector": ["15", "16", "15"],  # Banking, Non-bank Finance, Banking
        "sector_name": ["银行", "非银金融", "银行"],
    })


@pytest.fixture
def cn_sectors_with_reclassification() -> pl.DataFrame:
    """
    CN_A sector assignments with reclassification - critical for PIT sector tests.

    Test case: SH600000 changed sector on 2024-01-15
    - Before 2024-01-15: sector "10" (农林牧渔 - Agriculture)
    - After 2024-01-15: sector "15" (银行 - Banking)

    This tests that sector attribution uses the correct sector at each date.
    """
    return pl.DataFrame({
        "instrument_id": [
            "SH600000", "SH600000",  # Sector change
            "SZ000001",  # No change
            "SH600036",  # No change
        ],
        "date": [
            date(2020, 1, 1),   # Initial assignment
            date(2024, 1, 15),  # Reclassification date
            date(2020, 1, 1),
            date(2020, 1, 1),
        ],
        "sector": [
            "10",   # Agriculture (before)
            "15",   # Banking (after)
            "16",   # Non-bank Finance
            "15",   # Banking
        ],
        "sector_name": [
            "农林牧渔",  # Agriculture
            "银行",      # Banking
            "非银金融",  # Non-bank Finance
            "银行",      # Banking
        ],
    })


@pytest.fixture
def cn_sectors_multiple_changes() -> pl.DataFrame:
    """
    CN_A sector assignments with multiple reclassifications.

    Test case: Instrument changes sector multiple times
    - 2018-01-01: sector "03" (化工 - Chemicals)
    - 2020-06-01: sector "06" (电子 - Electronics)
    - 2023-01-01: sector "23" (计算机 - Computers)

    Tests binary search on sorted dates for correct sector at each point.
    """
    return pl.DataFrame({
        "instrument_id": ["SH600001"] * 3,
        "date": [
            date(2018, 1, 1),
            date(2020, 6, 1),
            date(2023, 1, 1),
        ],
        "sector": ["03", "06", "23"],
        "sector_name": ["化工", "电子", "计算机"],
    })


@pytest.fixture
def us_sectors_basic() -> pl.DataFrame:
    """Basic US sector assignments (GICS)."""
    return pl.DataFrame({
        "instrument_id": ["AAPL", "MSFT", "JPM", "XOM"],
        "date": [date(2020, 1, 1)] * 4,
        "sector": ["45", "45", "40", "10"],  # Tech, Tech, Financials, Energy
        "sector_name": ["Information Technology", "Information Technology",
                        "Financials", "Energy"],
    })


@pytest.fixture
def us_sectors_with_reclassification() -> pl.DataFrame:
    """
    US sector assignments with GICS reclassification.

    Historical GICS change examples:
    - AOL changed from Technology to Telecommunications in 2000s
    - REITs moved from Financials to Real Estate in 2016

    Test case: AAPL hypothetical sector change
    - Before 2022-07-01: GICS "45" (Information Technology)
    - After 2022-07-01: GICS "50" (Communication Services)
    """
    return pl.DataFrame({
        "instrument_id": ["AAPL", "AAPL", "MSFT"],
        "date": [
            date(2020, 1, 1),
            date(2022, 7, 1),  # Reclassification
            date(2020, 1, 1),
        ],
        "sector": ["45", "50", "45"],
        "sector_name": ["Information Technology", "Communication Services",
                        "Information Technology"],
    })


# =============================================================================
# Instrument Identity Test Data (Ticker Rename, Code Reuse)
# =============================================================================


@pytest.fixture
def cn_a_ticker_rename() -> pl.DataFrame:
    """
    CN_A daily data with ticker rename (same instrument_id).

    Test case: Company changed ticker symbol but instrument_id stays same.
    - SH600000 ticker was "600000", changed to "PFYH" (hypothetical)
    - Same instrument_id throughout

    This tests that instrument_id is immutable and used for historical continuity.
    """
    return pl.DataFrame({
        "instrument_id": ["SH600000"] * 6,
        "date": [
            date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3),  # Before rename
            date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3),  # After rename
        ],
        "ticker": [
            "600000", "600000", "600000",  # Old ticker
            "PFYH", "PFYH", "PFYH",  # New ticker (rename)
        ],
        "open": [10.0, 10.1, 10.2, 11.0, 11.1, 11.2],
        "high": [10.5, 10.6, 10.7, 11.5, 11.6, 11.7],
        "low": [9.8, 9.9, 10.0, 10.8, 10.9, 11.0],
        "close": [10.2, 10.3, 10.4, 11.2, 11.3, 11.4],
        "volume": [1000000, 1100000, 1200000, 1300000, 1400000, 1500000],
        "adj_factor": [1.0] * 6,
    })


@pytest.fixture
def cn_a_code_reuse() -> pl.DataFrame:
    """
    CN_A daily data with ticker code reuse (different instrument_ids).

    Test case: After delisting, same ticker code assigned to new company.
    - Old instrument_id: SH_OLD001 (ticker "600001")
    - New instrument_id: SH_NEW001 (ticker "600001" reused)
    - Different instrument_ids, same ticker

    This tests that instrument_id distinguishes between old and new companies.
    """
    return pl.DataFrame({
        "instrument_id": [
            "SH_OLD001", "SH_OLD001", "SH_OLD001",  # Old company
            "SH_NEW001", "SH_NEW001", "SH_NEW001",  # New company
        ],
        "date": [
            date(2014, 1, 1), date(2014, 1, 2), date(2014, 1, 3),  # Old company dates
            date(2016, 5, 1), date(2016, 5, 2), date(2016, 5, 3),  # New company dates
        ],
        "ticker": [
            "600001", "600001", "600001",  # Same ticker
            "600001", "600001", "600001",  # Reused ticker
        ],
        "open": [5.0, 5.1, 5.2, 10.0, 10.1, 10.2],
        "high": [5.5, 5.6, 5.7, 10.5, 10.6, 10.7],
        "low": [4.8, 4.9, 5.0, 9.8, 9.9, 10.0],
        "close": [5.2, 5.3, 5.4, 10.2, 10.3, 10.4],
        "volume": [500000, 550000, 600000, 1000000, 1100000, 1200000],
        "adj_factor": [1.0] * 6,
    })


@pytest.fixture
def cn_components_code_reuse() -> pl.DataFrame:
    """
    CN_A component events for ticker code reuse scenario.

    - Old company delists on 2015-03-01
    - New company IPOs on 2016-05-01 with same ticker code
    """
    return pl.DataFrame({
        "instrument_id": ["SH_OLD001", "SH_OLD001", "SH_NEW001"],
        "date": [date(2010, 1, 1), date(2015, 3, 1), date(2016, 5, 1)],
        "event": [IPO_EVENT, DELIST_EVENT, IPO_EVENT],
        "event_details": ["Old company IPO", "Old company delisted", "New company IPO"],
    })


# =============================================================================
# Populated Data Lake Fixtures
# =============================================================================


@pytest.fixture
def populated_cn_a_basic(
    temp_root: str,
    cn_daily_basic: pl.DataFrame,
    cn_components_basic: pl.DataFrame,
    cn_sectors_basic: pl.DataFrame,
) -> str:
    """Populate data_lake with basic CN_A data."""
    write_parquet(cn_daily_basic, Market.CN, "daily", temp_root)
    write_parquet(cn_components_basic, Market.CN, "components", temp_root)
    write_parquet(cn_sectors_basic, Market.CN, "sectors", temp_root)
    return temp_root


@pytest.fixture
def populated_cn_a_with_delist(
    temp_root: str,
    cn_daily_with_delist: pl.DataFrame,
    cn_components_with_delist: pl.DataFrame,
) -> str:
    """Populate data_lake with delisting scenario."""
    write_parquet(cn_daily_with_delist, Market.CN, "daily", temp_root)
    write_parquet(cn_components_with_delist, Market.CN, "components", temp_root)
    return temp_root


@pytest.fixture
def cn_daily_with_delist() -> pl.DataFrame:
    """CN_A daily data extending through delisting date."""
    return pl.DataFrame({
        "instrument_id": [
            "SH600000", "SH600000",
            "SZ000001", "SZ000001",
            "SH600631", "SH600631", "SH600631",  # 商业城 before delist
            "SZ000002",  # IPO after 2015
        ],
        "date": [
            date(2015, 1, 5), date(2015, 12, 31),
            date(2015, 1, 5), date(2015, 12, 31),
            date(2015, 1, 5), date(2015, 6, 1), date(2016, 3, 28),  # Last day before delist
            date(2016, 4, 5),  # After IPO
        ],
        "ticker": [
            "600000", "600000",
            "000001", "000001",
            "600631", "600631", "600631",
            "000002",
        ],
        "open": [10.0, 11.0, 20.0, 21.0, 8.0, 8.5, 7.0, 15.0],
        "high": [10.5, 11.5, 20.5, 21.5, 8.5, 9.0, 7.5, 15.5],
        "low": [9.8, 10.8, 19.8, 20.8, 7.8, 8.2, 6.8, 14.8],
        "close": [10.2, 11.2, 20.2, 21.2, 8.2, 8.8, 7.2, 15.2],
        "volume": [1000000, 1200000, 500000, 600000, 800000, 850000, 200000, 300000],
        "adj_factor": [1.0] * 9,
    })


@pytest.fixture
def populated_cn_a_with_halt(
    temp_root: str,
    cn_daily_with_halt: pl.DataFrame,
    cn_components_with_halt: pl.DataFrame,
) -> str:
    """Populate data_lake with halt scenario."""
    write_parquet(cn_daily_with_halt, Market.CN, "daily", temp_root)
    write_parquet(cn_components_with_halt, Market.CN, "components", temp_root)
    return temp_root


@pytest.fixture
def cn_daily_with_halt() -> pl.DataFrame:
    """CN_A daily data with halt period."""
    return pl.DataFrame({
        "instrument_id": ["SH600000"] * 5 + ["SZ000001"] * 2,
        "date": [
            date(2024, 2, 28),  # Before halt
            date(2024, 3, 1),   # Halt starts (no trading)
            date(2024, 3, 5),   # During halt (no trading)
            date(2024, 3, 15),  # Resume
            date(2024, 3, 16),  # After halt
            date(2024, 3, 5), date(2024, 3, 16),
        ],
        "ticker": ["600000"] * 5 + ["000001"] * 2,
        "open": [10.0, None, None, 9.8, 9.9, 20.0, 20.1],
        "high": [10.5, None, None, 10.2, 10.3, 20.5, 20.6],
        "low": [9.8, None, None, 9.6, 9.7, 19.8, 19.9],
        "close": [10.2, None, None, 10.0, 10.1, 20.2, 20.3],
        "volume": [1000000, 0, 0, 500000, 600000, 500000, 550000],
        "adj_factor": [1.0] * 7,
    }).filter(pl.col("close").is_not_null())  # Filter out halted days with no data


@pytest.fixture
def populated_cn_a_with_reclassification(
    temp_root: str,
    cn_daily_basic: pl.DataFrame,
    cn_components_basic: pl.DataFrame,
    cn_sectors_with_reclassification: pl.DataFrame,
) -> str:
    """Populate data_lake with sector reclassification scenario."""
    write_parquet(cn_daily_basic, Market.CN, "daily", temp_root)
    write_parquet(cn_components_basic, Market.CN, "components", temp_root)
    write_parquet(cn_sectors_with_reclassification, Market.CN, "sectors", temp_root)
    return temp_root


@pytest.fixture
def populated_us_basic(
    temp_root: str,
    us_daily_basic: pl.DataFrame,
    us_components_basic: pl.DataFrame,
    us_sectors_basic: pl.DataFrame,
) -> str:
    """Populate data_lake with basic US data."""
    write_parquet(us_daily_basic, Market.US, "daily", temp_root)
    write_parquet(us_components_basic, Market.US, "components", temp_root)
    write_parquet(us_sectors_basic, Market.US, "sectors", temp_root)
    return temp_root


@pytest.fixture
def us_daily_basic() -> pl.DataFrame:
    """Basic US daily OHLCV data."""
    return pl.DataFrame({
        "instrument_id": ["AAPL", "AAPL", "MSFT", "MSFT", "JPM", "JPM"],
        "date": [
            date(2024, 1, 2), date(2024, 1, 3),
            date(2024, 1, 2), date(2024, 1, 3),
            date(2024, 1, 2), date(2024, 1, 3),
        ],
        "ticker": ["AAPL", "AAPL", "MSFT", "MSFT", "JPM", "JPM"],
        "open": [180.0, 181.0, 370.0, 371.0, 150.0, 151.0],
        "high": [185.0, 186.0, 375.0, 376.0, 155.0, 156.0],
        "low": [178.0, 179.0, 368.0, 369.0, 148.0, 149.0],
        "close": [182.0, 183.0, 372.0, 373.0, 152.0, 153.0],
        "volume": [50000000, 55000000, 20000000, 22000000, 10000000, 11000000],
        "adj_factor": [1.0] * 6,
    })


@pytest.fixture
def populated_cn_a_with_adjustment(
    temp_root: str,
    cn_daily_with_adjustment: pl.DataFrame,
    cn_components_basic: pl.DataFrame,
) -> str:
    """Populate data_lake with adjustment factor scenario."""
    write_parquet(cn_daily_with_adjustment, Market.CN, "daily", temp_root)
    write_parquet(cn_components_basic, Market.CN, "components", temp_root)
    return temp_root


@pytest.fixture
def populated_cn_a_with_split(
    temp_root: str,
    cn_daily_with_split: pl.DataFrame,
    cn_components_basic: pl.DataFrame,
) -> str:
    """Populate data_lake with stock split scenario."""
    write_parquet(cn_daily_with_split, Market.CN, "daily", temp_root)
    write_parquet(cn_components_basic, Market.CN, "components", temp_root)
    return temp_root


@pytest.fixture
def populated_cn_a_with_identity(
    temp_root: str,
    cn_a_ticker_rename: pl.DataFrame,
    cn_a_code_reuse: pl.DataFrame,
    cn_components_code_reuse: pl.DataFrame,
) -> str:
    """Populate data_lake with instrument identity scenarios."""
    # Combine rename and reuse daily data
    combined_daily = pl.concat([cn_a_ticker_rename, cn_a_code_reuse])
    write_parquet(combined_daily, Market.CN, "daily", temp_root)
    write_parquet(cn_components_code_reuse, Market.CN, "components", temp_root)
    return temp_root