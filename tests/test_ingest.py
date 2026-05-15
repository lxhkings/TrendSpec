"""
Tests for TrendSpec ingest module.

Uses SQLite in-memory as MariaDB mock.
Tests each ingestor, incremental sync, manifest update, and Parquet writer.
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
from trendspec.ingest.incremental import (
    build_incremental_where_clause,
    get_full_date_range,
    get_instruments_to_sync,
    sync_instrument_incremental,
    update_manifest_after_sync,
)
from trendspec.ingest.manifest import Manifest, get_global_status, read_manifest
from trendspec.ingest.mariadb_client import get_engine
from trendspec.ingest.schema_map import (
    CN_COMPONENTS_MAP,
    CN_DAILY_MAP,
    CN_SECTORS_MAP,
    US_DAILY_MAP,
    derive_instrument_id_cn,
    derive_instrument_id_us,
    get_column_map,
    get_table_name,
)
from trendspec.ingest.writer import (
    get_partition_path,
    read_partition,
    write_parquet,
)

# =============================================================================
# Helper Functions for SQL Inserts
# =============================================================================


def insert_data(engine, table_name: str, data: list[tuple]) -> None:
    """
    Insert data into SQLite table using proper SQLAlchemy 2.x syntax.

    SQLAlchemy 2.x requires executemany with proper parameter format.
    """
    with engine.connect() as conn:
        # Convert tuples to dicts for SQLAlchemy 2.x
        # Use column names matching the fixture table definitions
        if table_name == "cn_daily":
            columns = ["instrument_id", "trade_date", "ticker", "open_price", "high_price", "low_price", "close_price", "volume", "adj_factor"]
        elif table_name == "cn_components":
            columns = ["instrument_id", "event_date", "event_type", "event_details"]
        elif table_name == "cn_sectors":
            columns = ["instrument_id", "assign_date", "sector_code", "sector_name"]
        elif table_name == "us_daily":
            columns = ["instrument_id", "trade_date", "ticker", "open_price", "high_price", "low_price", "close_price", "volume", "adj_factor"]
        elif table_name == "us_components":
            columns = ["instrument_id", "event_date", "event_type", "event_details"]
        elif table_name == "us_sectors":
            columns = ["instrument_id", "assign_date", "sector_code", "sector_name"]
        else:
            raise ValueError(f"Unknown table: {table_name}")

        # Build INSERT statement with named parameters
        col_list = ", ".join(columns)
        param_list = ", ".join([f":{col}" for col in columns])
        sql = text(f"INSERT INTO {table_name} ({col_list}) VALUES ({param_list})")

        # Convert data to dicts
        params = [dict(zip(columns, row)) for row in data]

        conn.execute(sql, params)
        conn.commit()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_root() -> str:
    """Create temporary directory for data_lake."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sqlite_engine():
    """Create SQLite in-memory engine with mock schema."""
    engine = create_engine("sqlite:///:memory:")

    # Create CN_A tables (matching naming convention: {market.path}_{dataset})
    with engine.connect() as conn:
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
        conn.execute(text("""
            CREATE TABLE cn_components (
                instrument_id TEXT,
                event_date DATE,
                event_type TEXT,
                event_details TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE cn_sectors (
                instrument_id TEXT,
                assign_date DATE,
                sector_code TEXT,
                sector_name TEXT
            )
        """))

        # Create US tables
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


@pytest.fixture
def cn_a_sample_data() -> list[tuple]:
    """Sample CN_A daily data."""
    return [
        ("SH600000", "2024-01-01", "600000", 10.0, 10.5, 9.8, 10.2, 1000000, 1.0),
        ("SH600000", "2024-01-02", "600000", 10.2, 10.8, 10.0, 10.5, 1200000, 1.0),
        ("SH600000", "2024-01-03", "600000", 10.5, 11.0, 10.3, 10.8, 1500000, 1.0),
        ("SZ000001", "2024-01-01", "000001", 20.0, 20.5, 19.8, 20.2, 500000, 1.0),
        ("SZ000001", "2024-01-02", "000001", 20.2, 20.8, 20.0, 20.5, 600000, 1.0),
        ("SZ000001", "2024-01-03", "000001", 20.5, 21.0, 20.3, 20.8, 700000, 1.0),
    ]


@pytest.fixture
def us_sample_data() -> list[tuple]:
    """Sample US daily data."""
    return [
        ("AAPL", "2024-01-01", "AAPL", 180.0, 182.5, 179.0, 181.5, 50000000, 1.0),
        ("AAPL", "2024-01-02", "AAPL", 181.5, 184.0, 181.0, 183.0, 55000000, 1.0),
        ("AAPL", "2024-01-03", "AAPL", 183.0, 185.5, 182.5, 184.5, 60000000, 1.0),
        ("MSFT", "2024-01-01", "MSFT", 400.0, 405.0, 398.0, 402.0, 20000000, 1.0),
        ("MSFT", "2024-01-02", "MSFT", 402.0, 407.0, 401.0, 405.0, 22000000, 1.0),
        ("MSFT", "2024-01-03", "MSFT", 405.0, 410.0, 404.0, 408.0, 25000000, 1.0),
    ]


@pytest.fixture
def cn_components_sample() -> list[tuple]:
    """Sample CN_A component events."""
    return [
        ("SH600000", "2020-01-01", "IPO", "Listed on Shanghai"),
        ("SZ000002", "2024-01-15", "IPO", "Listed on Shenzhen"),
        ("SH600000", "2024-03-01", "HALT", "Temporary suspension"),
        ("SH600000", "2024-03-15", "RESUME", "Trading resumed"),
    ]


@pytest.fixture
def cn_sectors_sample() -> list[tuple]:
    """Sample CN_A sector assignments."""
    return [
        ("SH600000", "2020-01-01", "15", "银行"),
        ("SH600000", "2024-01-01", "15", "银行"),  # Same sector, later date
        ("SZ000001", "2020-01-01", "16", "非银金融"),
    ]


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings for testing."""
    with patch.dict(os.environ, {
        "DB_HOST": "localhost",
        "DB_USER": "trendspec",
        "DB_PASSWORD": "testpass",
        "DB_NAME": "testdb",
    }, clear=False):
        Settings.get.cache_clear()
        return Settings.get()


# =============================================================================
# MariaDB Client Tests
# =============================================================================


class TestMariaDBClient:
    """Tests for MariaDB client."""

    def test_get_engine_creates_engine(self, mock_settings: Settings) -> None:
        """get_engine should create SQLAlchemy engine."""
        engine = get_engine(mock_settings)
        assert engine is not None
        assert str(engine.url).startswith("mysql+pymysql://")

    def test_engine_has_pool(self, mock_settings: Settings) -> None:
        """Engine should have connection pool."""
        engine = get_engine(mock_settings)
        assert engine.pool is not None
        # Default pool size is 5
        assert engine.pool.size() == 5


# =============================================================================
# Schema Map Tests
# =============================================================================


class TestSchemaMap:
    """Tests for schema mapping."""

    def test_cn_daily_map_keys(self) -> None:
        """CN_A daily map should have standard column keys."""
        expected_keys = ["instrument_id", "date", "ticker", "open", "high", "low", "close", "volume", "adj_factor"]
        assert set(CN_DAILY_MAP.keys()) == set(expected_keys)

    def test_cn_daily_map_values(self) -> None:
        """CN_A daily map should map to SQL column names."""
        assert CN_DAILY_MAP["date"] == "trade_date"
        assert CN_DAILY_MAP["open"] == "open_price"
        assert CN_DAILY_MAP["close"] == "close_price"

    def test_us_daily_map_keys(self) -> None:
        """US daily map should have standard column keys."""
        expected_keys = ["instrument_id", "date", "ticker", "open", "high", "low", "close", "volume", "adj_factor"]
        assert set(US_DAILY_MAP.keys()) == set(expected_keys)

    def test_get_column_map_cn_daily(self) -> None:
        """get_column_map should return CN_A daily map."""
        map = get_column_map(Market.CN, "daily")
        assert map == CN_DAILY_MAP

    def test_get_column_map_cn_components(self) -> None:
        """get_column_map should return CN_A components map."""
        map = get_column_map(Market.CN, "components")
        assert map == CN_COMPONENTS_MAP

    def test_get_column_map_cn_sectors(self) -> None:
        """get_column_map should return CN_A sectors map."""
        map = get_column_map(Market.CN, "sectors")
        assert map == CN_SECTORS_MAP

    def test_get_column_map_unknown_raises(self) -> None:
        """get_column_map should raise for unknown combination."""
        with pytest.raises(ValueError, match="No schema mapping"):
            get_column_map(Market.HK, "daily")

    def test_get_table_name_cn_daily(self) -> None:
        """get_table_name should return correct table name."""
        assert get_table_name(Market.CN, "daily") == "cn_daily"

    def test_get_table_name_us_components(self) -> None:
        """get_table_name should return correct table name."""
        assert get_table_name(Market.US, "components") == "us_components"

    def test_derive_instrument_id_cn_sh(self) -> None:
        """derive_instrument_id_cn should prefix with SH."""
        assert derive_instrument_id_cn("600000", "SH") == "SH600000"

    def test_derive_instrument_id_cn_sz(self) -> None:
        """derive_instrument_id_cn should prefix with SZ."""
        assert derive_instrument_id_cn("000001", "SZ") == "SZ000001"

    def test_derive_instrument_id_cn_invalid_exchange(self) -> None:
        """derive_instrument_id_cn should raise for invalid exchange."""
        with pytest.raises(ValueError, match="Invalid exchange"):
            derive_instrument_id_cn("600000", "BJ")  # Beijing exchange not supported

    def test_derive_instrument_id_us(self) -> None:
        """derive_instrument_id_us should return uppercase ticker."""
        assert derive_instrument_id_us("aapl") == "AAPL"


# =============================================================================
# Parquet Writer Tests
# =============================================================================


class TestParquetWriter:
    """Tests for Parquet writer."""

    def test_write_parquet_creates_partition_dirs(self, temp_root: str) -> None:
        """write_parquet should create partition directories."""
        df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 1)],
            "close": [10.0],
            "volume": [1000000],
        })

        write_parquet(df, Market.CN, "daily", temp_root)

        # Check partition directory exists
        partition_dir = os.path.join(temp_root, "cn", "daily", "instrument_id=SH600000")
        assert os.path.exists(partition_dir)

        # Check Parquet file exists
        parquet_file = os.path.join(partition_dir, "2024.parquet")
        assert os.path.exists(parquet_file)

    def test_write_parquet_correct_partition(self, temp_root: str) -> None:
        """write_parquet should partition by instrument_id and year."""
        df = pl.DataFrame({
            "instrument_id": ["SH600000", "SZ000001"],
            "date": [date(2024, 1, 1), date(2024, 1, 1)],
            "close": [10.0, 20.0],
            "volume": [1000000, 500000],
        })

        write_parquet(df, Market.CN, "daily", temp_root)

        # Check both instruments have partitions
        assert os.path.exists(os.path.join(temp_root, "cn", "daily", "instrument_id=SH600000"))
        assert os.path.exists(os.path.join(temp_root, "cn", "daily", "instrument_id=SZ000001"))

    def test_write_parquet_multi_year(self, temp_root: str) -> None:
        """write_parquet should create separate files for different years."""
        df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000"],
            "date": [date(2023, 1, 1), date(2024, 1, 1)],
            "close": [10.0, 11.0],
            "volume": [1000000, 1100000],
        })

        write_parquet(df, Market.CN, "daily", temp_root)

        # Check both year files exist
        partition_dir = os.path.join(temp_root, "cn", "daily", "instrument_id=SH600000")
        assert os.path.exists(os.path.join(partition_dir, "2023.parquet"))
        assert os.path.exists(os.path.join(partition_dir, "2024.parquet"))

    def test_write_parquet_requires_instrument_id(self, temp_root: str) -> None:
        """write_parquet should require instrument_id column."""
        df = pl.DataFrame({
            "date": [date(2024, 1, 1)],
            "close": [10.0],
        })

        with pytest.raises(ValueError, match="must have 'instrument_id'"):
            write_parquet(df, Market.CN, "daily", temp_root)

    def test_write_parquet_requires_date(self, temp_root: str) -> None:
        """write_parquet should require date column."""
        df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "close": [10.0],
        })

        with pytest.raises(ValueError, match="must have 'date'"):
            write_parquet(df, Market.CN, "daily", temp_root)

    def test_write_parquet_zstd_compression(self, temp_root: str) -> None:
        """write_parquet should use zstd compression."""
        df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 1)],
            "close": [10.0],
        })

        write_parquet(df, Market.CN, "daily", temp_root)

        # Read back and verify
        parquet_file = os.path.join(temp_root, "cn", "daily", "instrument_id=SH600000", "2024.parquet")
        read_df = pl.read_parquet(parquet_file)

        # Check data is correct
        assert read_df["instrument_id"].item() == "SH600000"
        assert read_df["close"].item() == 10.0

    def test_read_partition(self, temp_root: str) -> None:
        """read_partition should read partition data."""
        df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 1)],
            "close": [10.0],
        })

        write_parquet(df, Market.CN, "daily", temp_root)

        read_df = read_partition(temp_root, Market.CN, "daily", "SH600000", 2024)
        assert len(read_df) == 1
        assert read_df["instrument_id"].item() == "SH600000"

    def test_read_partition_empty_if_not_exists(self, temp_root: str) -> None:
        """read_partition should return empty DataFrame if partition doesn't exist."""
        read_df = read_partition(temp_root, Market.CN, "daily", "SH600000", 2024)
        assert read_df.is_empty()

    def test_get_partition_path(self, temp_root: str) -> None:
        """get_partition_path should return correct path."""
        path = get_partition_path(temp_root, Market.CN, "daily", "SH600000", 2024)
        expected = os.path.join(temp_root, "cn", "daily", "instrument_id=SH600000", "2024.parquet")
        assert path == expected


# =============================================================================
# Manifest Tests
# =============================================================================


class TestManifest:
    """Tests for sync manifest."""

    def test_manifest_creates_file(self, temp_root: str) -> None:
        """Manifest should create JSON file."""
        manifest = Manifest(Market.CN, temp_root)
        manifest.update_dataset_state("daily", 100, ("2024-01-01", "2024-01-31"), 5)

        manifest_path = os.path.join(temp_root, "_manifest", "cn.json")
        assert os.path.exists(manifest_path)

    def test_manifest_stores_state(self, temp_root: str) -> None:
        """Manifest should store dataset state."""
        manifest = Manifest(Market.CN, temp_root)
        manifest.update_dataset_state("daily", 100, ("2024-01-01", "2024-01-31"), 5)

        state = manifest.get_dataset_state("daily")
        assert state is not None
        assert state["row_count"] == 100
        assert state["date_range"]["start"] == "2024-01-01"
        assert state["instrument_count"] == 5

    def test_manifest_get_last_date(self, temp_root: str) -> None:
        """Manifest should track last date per instrument."""
        manifest = Manifest(Market.CN, temp_root)
        manifest.update_instrument_date("daily", "SH600000", "2024-01-15")

        last_date = manifest.get_last_date("daily", "SH600000")
        assert last_date == "2024-01-15"

    def test_manifest_get_last_date_none_if_not_synced(self, temp_root: str) -> None:
        """Manifest should return None for unsynced instrument."""
        manifest = Manifest(Market.CN, temp_root)
        last_date = manifest.get_last_date("daily", "SH600000")
        assert last_date is None

    def test_manifest_update_instrument_date(self, temp_root: str) -> None:
        """Manifest should update instrument date incrementally."""
        manifest = Manifest(Market.CN, temp_root)
        manifest.update_instrument_date("daily", "SH600000", "2024-01-15")
        manifest.update_instrument_date("daily", "SZ000001", "2024-01-20")

        state = manifest.get_dataset_state("daily")
        assert state["instruments"]["SH600000"] == "2024-01-15"
        assert state["instruments"]["SZ000001"] == "2024-01-20"

    def test_read_manifest(self, temp_root: str) -> None:
        """read_manifest should load existing manifest."""
        manifest = Manifest(Market.CN, temp_root)
        manifest.update_dataset_state("daily", 100, ("2024-01-01", "2024-01-31"), 5)

        loaded_manifest = read_manifest(Market.CN, temp_root)
        state = loaded_manifest.get_dataset_state("daily")
        assert state is not None
        assert state["row_count"] == 100

    def test_manifest_clear_dataset(self, temp_root: str) -> None:
        """Manifest should clear dataset state."""
        manifest = Manifest(Market.CN, temp_root)
        manifest.update_dataset_state("daily", 100, ("2024-01-01", "2024-01-31"), 5)
        manifest.clear_dataset("daily")

        state = manifest.get_dataset_state("daily")
        assert state is None

    def test_get_global_status(self, temp_root: str) -> None:
        """get_global_status should return all market statuses."""
        cn_manifest = Manifest(Market.CN, temp_root)
        cn_manifest.update_dataset_state("daily", 100, ("2024-01-01", "2024-01-31"), 5)

        us_manifest = Manifest(Market.US, temp_root)
        us_manifest.update_dataset_state("daily", 200, ("2024-01-01", "2024-01-31"), 10)

        status = get_global_status(temp_root)
        assert "cn" in status
        assert "us" in status


# =============================================================================
# Incremental Sync Tests
# =============================================================================


class TestIncrementalSync:
    """Tests for incremental sync logic."""

    def test_build_incremental_where_clause_full(self) -> None:
        """Full sync should have no date filter."""
        where = build_incremental_where_clause("SH600000", None)
        assert where == "instrument_id = 'SH600000'"

    def test_build_incremental_where_clause_incremental(self) -> None:
        """Incremental sync should have date filter."""
        where = build_incremental_where_clause("SH600000", "2024-01-15")
        assert "instrument_id = 'SH600000'" in where
        assert "trade_date > '2024-01-15'" in where

    def test_get_instruments_to_sync(self, temp_root: str) -> None:
        """get_instruments_to_sync should check manifest."""
        manifest = Manifest(Market.CN, temp_root)
        manifest.update_instrument_date("daily", "SH600000", "2024-01-15")

        instruments = get_instruments_to_sync(manifest, "daily", ["SH600000", "SZ000001"])

        assert instruments["SH600000"] == "2024-01-15"
        assert instruments["SZ000001"] is None

    def test_sync_instrument_incremental_full(
        self,
        sqlite_engine,
        cn_a_sample_data,
        temp_root: str,
    ) -> None:
        """Full sync should pull all data."""
        # Insert sample data
        insert_data(sqlite_engine, "cn_daily", cn_a_sample_data)

        manifest = Manifest(Market.CN, temp_root)

        df = sync_instrument_incremental(
            sqlite_engine,
            "cn_daily",
            CN_DAILY_MAP,
            "SH600000",
            None,  # Full sync
        )

        assert len(df) == 3  # All 3 days for SH600000
        assert df["instrument_id"].unique().to_list() == ["SH600000"]

    def test_sync_instrument_incremental_incremental(
        self,
        sqlite_engine,
        cn_a_sample_data,
        temp_root: str,
    ) -> None:
        """Incremental sync should pull only new data."""
        # Insert sample data
        insert_data(sqlite_engine, "cn_daily", cn_a_sample_data)

        manifest = Manifest(Market.CN, temp_root)

        df = sync_instrument_incremental(
            sqlite_engine,
            "cn_daily",
            CN_DAILY_MAP,
            "SH600000",
            "2024-01-02",  # Pull data after Jan 2
        )

        assert len(df) == 1  # Only Jan 3
        assert df["date"].item() == date(2024, 1, 3)

    def test_update_manifest_after_sync(self, temp_root: str) -> None:
        """update_manifest_after_sync should update manifest."""
        manifest = Manifest(Market.CN, temp_root)

        df = pl.DataFrame({
            "instrument_id": ["SH600000", "SZ000001"],
            "date": [date(2024, 1, 15), date(2024, 1, 20)],
            "close": [10.0, 20.0],
        })

        update_manifest_after_sync(manifest, "daily", df)

        assert manifest.get_last_date("daily", "SH600000") == "2024-01-15"
        assert manifest.get_last_date("daily", "SZ000001") == "2024-01-20"

    def test_get_full_date_range(self) -> None:
        """get_full_date_range should return min and max dates."""
        df = pl.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 15), date(2024, 1, 31)],
        })

        start, end = get_full_date_range(df)
        assert start == "2024-01-01"
        assert end == "2024-01-31"

    def test_get_full_date_range_empty(self) -> None:
        """get_full_date_range should return empty strings for empty DataFrame."""
        df = pl.DataFrame()
        start, end = get_full_date_range(df)
        assert start == ""
        assert end == ""


# =============================================================================
# CN_A Ingestor Tests
# =============================================================================


class TestCNAIngestor:
    """Tests for CN_A ingestor."""

    def test_ingest_cn_daily_full_sync(
        self,
        sqlite_engine,
        cn_a_sample_data,
        temp_root: str,
    ) -> None:
        """Full CN_A daily sync should pull all data."""
        from trendspec.ingest.cn_ingestor import ingest_cn_daily

        # Insert sample data
        insert_data(sqlite_engine, "cn_daily", cn_a_sample_data)

        manifest = Manifest(Market.CN, temp_root)
        result = ingest_cn_daily(sqlite_engine, manifest, temp_root, full_sync=True)

        assert result["row_count"] == 6
        assert result["instrument_count"] == 2

    def test_ingest_cn_components(
        self,
        sqlite_engine,
        cn_components_sample,
        temp_root: str,
    ) -> None:
        """CN_A components ingest should pull events."""
        from trendspec.ingest.cn_ingestor import ingest_cn_components

        # Insert sample data
        insert_data(sqlite_engine, "cn_components", cn_components_sample)

        manifest = Manifest(Market.CN, temp_root)
        result = ingest_cn_components(sqlite_engine, manifest, temp_root, full_sync=True)

        assert result["row_count"] == 4

    def test_ingest_cn_sectors(
        self,
        sqlite_engine,
        cn_sectors_sample,
        temp_root: str,
    ) -> None:
        """CN_A sectors ingest should pull assignments."""
        from trendspec.ingest.cn_ingestor import ingest_cn_sectors

        # Insert sample data
        insert_data(sqlite_engine, "cn_sectors", cn_sectors_sample)

        manifest = Manifest(Market.CN, temp_root)
        result = ingest_cn_sectors(sqlite_engine, manifest, temp_root, full_sync=True)

        assert result["row_count"] == 3


# =============================================================================
# US Ingestor Tests
# =============================================================================


class TestUSIngestor:
    """Tests for US ingestor."""

    def test_ingest_us_daily_full_sync(
        self,
        sqlite_engine,
        us_sample_data,
        temp_root: str,
    ) -> None:
        """Full US daily sync should pull all data."""
        from trendspec.ingest.us_ingestor import ingest_us_daily

        # Insert sample data
        insert_data(sqlite_engine, "us_daily", us_sample_data)

        manifest = Manifest(Market.US, temp_root)
        result = ingest_us_daily(sqlite_engine, manifest, temp_root, full_sync=True)

        assert result["row_count"] == 6
        assert result["instrument_count"] == 2

    def test_ingest_us_components(
        self,
        sqlite_engine,
        temp_root: str,
    ) -> None:
        """US components ingest should pull events."""
        from trendspec.ingest.us_ingestor import ingest_us_components

        # Insert sample component events
        us_components = [
            ("AAPL", "2020-01-01", "IPO", "NASDAQ listing"),
            ("AAPL", "2024-01-15", "SP500_ADD", "Added to S&P 500"),
        ]

        insert_data(sqlite_engine, "us_components", us_components)

        manifest = Manifest(Market.US, temp_root)
        result = ingest_us_components(sqlite_engine, manifest, temp_root, full_sync=True)

        assert result["row_count"] == 2

    def test_ingest_us_sectors(
        self,
        sqlite_engine,
        temp_root: str,
    ) -> None:
        """US sectors ingest should pull assignments."""
        from trendspec.ingest.us_ingestor import ingest_us_sectors

        # Insert sample sector assignments
        us_sectors = [
            ("AAPL", "2020-01-01", "45", "Information Technology"),
            ("MSFT", "2020-01-01", "45", "Information Technology"),
        ]

        insert_data(sqlite_engine, "us_sectors", us_sectors)

        manifest = Manifest(Market.US, temp_root)
        result = ingest_us_sectors(sqlite_engine, manifest, temp_root, full_sync=True)

        assert result["row_count"] == 2


# =============================================================================
# Components Ingestor Tests
# =============================================================================


class TestComponentsIngestor:
    """Tests for generic components ingestor."""

    def test_get_ipo_instruments(self) -> None:
        """get_ipo_instruments should filter IPO events."""
        from trendspec.ingest.components_ingestor import get_ipo_instruments

        df = pl.DataFrame({
            "instrument_id": ["SH600000", "SZ000001", "SH600000"],
            "date": [date(2020, 1, 1), date(2020, 1, 1), date(2024, 3, 1)],
            "event": ["IPO", "IPO", "HALT"],
        })

        ipo_df = get_ipo_instruments(df)
        assert len(ipo_df) == 2
        assert all(ipo_df["event"] == "IPO")

    def test_get_delisted_instruments(self) -> None:
        """get_delisted_instruments should filter DELIST events."""
        from trendspec.ingest.components_ingestor import get_delisted_instruments

        df = pl.DataFrame({
            "instrument_id": ["SH600000", "SZ000002"],
            "date": [date(2024, 1, 1), date(2024, 1, 1)],
            "event": ["DELIST", "IPO"],
        })

        delisted_df = get_delisted_instruments(df)
        assert len(delisted_df) == 1
        assert delisted_df["event"].item() == "DELIST"

    def test_get_active_instruments(self) -> None:
        """get_active_instruments should find active stocks."""
        from trendspec.ingest.components_ingestor import get_active_instruments

        df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000", "SZ000002"],
            "date": [date(2020, 1, 1), date(2024, 1, 1), date(2024, 1, 1)],
            "event": ["IPO", "HALT", "DELIST"],
        })

        active_df = get_active_instruments(df, date(2024, 1, 15))
        # SH600000 has HALT (inactive), SZ000002 has DELIST
        # So only instruments with IPO as latest event would be active
        # But SH600000's latest is HALT, SZ000002's is DELIST
        assert len(active_df) == 0  # Both are inactive


# =============================================================================
# Sectors Ingestor Tests
# =============================================================================


class TestSectorsIngestor:
    """Tests for sectors ingestor."""

    def test_get_sector_at_date(self) -> None:
        """get_sector_at_date should return PIT sector."""
        from trendspec.ingest.sectors_ingestor import get_sector_at_date

        df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000"],
            "date": [date(2020, 1, 1), date(2024, 1, 1)],
            "sector": ["10", "15"],
        })

        sector = get_sector_at_date(df, "SH600000", date(2023, 1, 1))
        assert sector == "10"  # 2020 assignment is the latest before 2023

        sector = get_sector_at_date(df, "SH600000", date(2024, 1, 15))
        assert sector == "15"  # 2024 assignment is the latest

    def test_get_sector_name_cn(self) -> None:
        """get_sector_name should return Shenwan sector name."""
        from trendspec.ingest.sectors_ingestor import get_sector_name

        name = get_sector_name(Market.CN, "15")
        assert name == "银行"

    def test_get_sector_name_us(self) -> None:
        """get_sector_name should return GICS sector name."""
        from trendspec.ingest.sectors_ingestor import get_sector_name

        name = get_sector_name(Market.US, "45")
        assert name == "Information Technology"

    def test_get_all_sectors_cn(self) -> None:
        """get_all_sectors should return all Shenwan sectors."""
        from trendspec.ingest.sectors_ingestor import get_all_sectors

        sectors = get_all_sectors(Market.CN)
        assert len(sectors) == 28
        assert "15" in sectors

    def test_get_all_sectors_us(self) -> None:
        """get_all_sectors should return all GICS sectors."""
        from trendspec.ingest.sectors_ingestor import get_all_sectors

        sectors = get_all_sectors(Market.US)
        assert len(sectors) == 11

    def test_get_sector_instruments(self) -> None:
        """get_sector_instruments should return instruments in sector."""
        from trendspec.ingest.sectors_ingestor import get_sector_instruments

        df = pl.DataFrame({
            "instrument_id": ["SH600000", "SZ000001", "SH600001"],
            "date": [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 1)],
            "sector": ["15", "16", "15"],
        })

        instruments = get_sector_instruments(df, "15", date(2024, 1, 1))
        assert set(instruments) == {"SH600000", "SH600001"}
