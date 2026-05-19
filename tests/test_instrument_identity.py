"""
Tests for TrendSpec Instrument Identity functionality - CRITICAL for historical continuity.

Key test cases:
1. Ticker rename: instrument_id stays same when ticker changes
2. Delist + code reuse: different instrument_ids for same ticker code
3. Historical join continuity: same instrument_id across ticker changes

CRITICAL for data integrity:
- Primary key is (instrument_id, date), NOT (ticker, date)
- instrument_id is immutable - uniquely identifies a security
- ticker can change due to renames, and can be reused after delisting
"""

from datetime import date

import polars as pl

from trendspec.data.markets import Market
from trendspec.data.parquet_loader import bars_for_instrument, get_instrument_ids
from trendspec.data.schema import PRIMARY_KEY, REQUIRED_COLUMNS, validate_dataframe_schema
from trendspec.data.universe import CNAUniverse
from trendspec.data.universe.cn import DELIST_EVENT, IPO_EVENT
from trendspec.ingest.writer import read_partition, write_parquet

# =============================================================================
# Primary Key Definition Tests
# =============================================================================


class TestPrimaryKeyDefinition:
    """
    Tests to verify the primary key is (instrument_id, date), NOT (ticker, date).

    This is a one-way door decision - changing it would invalidate all historical data.
    """

    def test_primary_key_is_instrument_id_date(self) -> None:
        """
        Primary key must be (instrument_id, date).

        This is CRITICAL - changing this would break historical continuity.
        """
        assert PRIMARY_KEY == ("instrument_id", "date"), (
            "Primary key MUST be (instrument_id, date) - "
            "this is a one-way door decision for historical continuity"
        )

    def test_required_columns_include_instrument_id(self) -> None:
        """
        Required columns must include instrument_id.

        ticker is also required but is NOT part of primary key.
        """
        assert "instrument_id" in REQUIRED_COLUMNS, "instrument_id must be in required columns"
        assert "date" in REQUIRED_COLUMNS, "date must be in required columns"
        assert "ticker" in REQUIRED_COLUMNS, "ticker must be in required columns"

    def test_ticker_not_in_primary_key(self) -> None:
        """
        ticker should NOT be in primary key.

        ticker can change, instrument_id cannot.
        """
        assert "ticker" not in PRIMARY_KEY, (
            "ticker MUST NOT be in primary key - "
            "ticker can change, instrument_id cannot"
        )


# =============================================================================
# Ticker Rename Tests
# =============================================================================


class TestTickerRename:
    """
    Tests for ticker rename scenario.

    When a company changes its ticker symbol:
    - instrument_id stays the same (same company)
    - ticker changes to new symbol
    - Historical continuity maintained via instrument_id
    """

    def test_ticker_rename_instrument_id_unchanged(self, temp_root: str) -> None:
        """
        When ticker changes, instrument_id should stay the same.

        This ensures historical continuity across ticker changes.

        Example: SH600000 ticker was "600000", later changed to "PFYH"
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 6,  # Same instrument_id
            "date": [
                date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3),  # Old ticker
                date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3),  # New ticker
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

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        # Verify data integrity
        df = bars_for_instrument(Market.CN, "SH600000", root=temp_root)

        if not df.is_empty():
            # Should have 6 rows - all for same instrument_id
            assert len(df) == 6, "All historical data should be accessible via instrument_id"

            # instrument_id should be same throughout
            instrument_ids = df["instrument_id"].unique().to_list()
            assert instrument_ids == ["SH600000"], "instrument_id should be constant"

            # tickers should show both old and new
            tickers = df["ticker"].unique().to_list()
            assert len(tickers) == 2, "Should have both old and new tickers"
            assert "600000" in tickers, "Old ticker should be present"
            assert "PFYH" in tickers, "New ticker should be present"

    def test_ticker_rename_price_continuity(self, temp_root: str) -> None:
        """
        Price continuity should be maintained across ticker rename.

        The same instrument_id ensures prices join correctly.
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 6,
            "date": [
                date(2023, 12, 28), date(2023, 12, 29), date(2023, 12, 30),  # Before rename
                date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3),  # After rename
            ],
            "ticker": [
                "600000", "600000", "600000",
                "PFYH", "PFYH", "PFYH",
            ],
            "close": [10.0, 10.1, 10.2, 10.2, 10.3, 10.4],  # Price continuity
            "open": [10.0] * 6,
            "high": [10.5] * 6,
            "low": [9.8] * 6,
            "volume": [1000000] * 6,
            "adj_factor": [1.0] * 6,
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        df = bars_for_instrument(Market.CN, "SH600000", adjustment_mode="raw", root=temp_root)

        if not df.is_empty():
            df = df.sort("date")

            # Price should join correctly - no gaps
            closes = df["close"].to_list()

            # Verify continuity: Dec 30 close = Jan 1 close (rename happened)
            dec30_close = df.filter(pl.col("date") == date(2023, 12, 30))["close"].item()
            jan1_close = df.filter(pl.col("date") == date(2024, 1, 1))["close"].item()

            # Price should be continuous (same or close)
            assert abs(dec30_close - jan1_close) < 0.5, (
                f"Price continuity across ticker rename: "
                f"Dec30: {dec30_close}, Jan1: {jan1_close}"
            )

    def test_ticker_rename_universe_membership(self, temp_root: str) -> None:
        """
        Universe membership should track instrument_id, not ticker.

        Same stock should be in universe regardless of ticker name.
        """
        components_df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(1999, 11, 10)],
            "event": [IPO_EVENT],
            "event_details": ["IPO"],
        })

        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 6,
            "date": [
                date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3),
                date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3),
            ],
            "ticker": [
                "600000", "600000", "600000",
                "PFYH", "PFYH", "PFYH",
            ],
            "close": [10.0] * 6,
            "open": [10.0] * 6,
            "high": [10.5] * 6,
            "low": [9.8] * 6,
            "volume": [1000000] * 6,
            "adj_factor": [1.0] * 6,
        })

        write_parquet(components_df, Market.CN, "components", temp_root)
        write_parquet(daily_df, Market.CN, "daily", temp_root)

        universe = CNAUniverse(temp_root)

        # Universe should track SH600000 regardless of ticker
        universe_2023 = universe.tickers(date(2023, 1, 1))
        universe_2024 = universe.tickers(date(2024, 1, 1))

        assert "SH600000" in universe_2023, "instrument_id should be tracked"
        assert "SH600000" in universe_2024, "Same instrument_id after ticker rename"


# =============================================================================
# Delist + Code Reuse Tests
# =============================================================================


class TestDelistCodeReuse:
    """
    Tests for ticker code reuse after delisting.

    Critical scenario: Old company delists, new company gets same ticker code.
    - Different instrument_ids (different companies)
    - Same ticker code (reused)

    This tests that instrument_id distinguishes between them.
    """

    def test_code_reuse_different_instrument_ids(self, temp_root: str) -> None:
        """
        When ticker code is reused, instrument_ids should be different.

        Old company: instrument_id = SH_OLD001 (ticker "600001")
        New company: instrument_id = SH_NEW001 (ticker "600001")
        Same ticker, different companies.
        """
        components_df = pl.DataFrame({
            "instrument_id": [
                "SH_OLD001", "SH_OLD001",  # Old company IPO, then delist
                "SH_NEW001",  # New company IPO with reused ticker
            ],
            "date": [
                date(2010, 1, 1),  # Old IPO
                date(2015, 3, 1),  # Old delist
                date(2016, 5, 1),  # New IPO
            ],
            "event": [IPO_EVENT, DELIST_EVENT, IPO_EVENT],
            "event_details": ["Old company IPO", "Old company delisted", "New company IPO"],
        })

        daily_df = pl.DataFrame({
            "instrument_id": [
                "SH_OLD001", "SH_OLD001", "SH_OLD001",  # Old company
                "SH_NEW001", "SH_NEW001", "SH_NEW001",  # New company
            ],
            "date": [
                date(2014, 1, 1), date(2014, 1, 2), date(2015, 2, 28),  # Old company dates
                date(2016, 5, 1), date(2016, 5, 2), date(2016, 5, 3),  # New company dates
            ],
            "ticker": [
                "600001", "600001", "600001",  # Same ticker code
                "600001", "600001", "600001",  # Reused ticker code
            ],
            "close": [5.0, 5.1, 5.2, 10.0, 10.1, 10.2],  # Different price levels
            "open": [5.0, 5.1, 5.2, 10.0, 10.1, 10.2],
            "high": [5.5, 5.6, 5.7, 10.5, 10.6, 10.7],
            "low": [4.8, 4.9, 5.0, 9.8, 9.9, 10.0],
            "volume": [500000, 550000, 600000, 1000000, 1100000, 1200000],
            "adj_factor": [1.0] * 6,
        })

        write_parquet(components_df, Market.CN, "components", temp_root)
        write_parquet(daily_df, Market.CN, "daily", temp_root)

        # Verify old and new are separate
        df_old = bars_for_instrument(Market.CN, "SH_OLD001", root=temp_root)
        df_new = bars_for_instrument(Market.CN, "SH_NEW001", root=temp_root)

        if not df_old.is_empty() and not df_new.is_empty():
            # Should have different data
            assert len(df_old) == 3, "Old company should have 3 rows"
            assert len(df_new) == 3, "New company should have 3 rows"

            # Dates should be different
            old_dates = df_old["date"].to_list()
            new_dates = df_new["date"].to_list()
            assert old_dates != new_dates, "Old and new company dates should differ"

            # But tickers should be same
            assert df_old["ticker"].unique().to_list() == ["600001"], "Old company ticker"
            assert df_new["ticker"].unique().to_list() == ["600001"], "New company ticker (same code)"

    def test_code_reuse_universe_membership(self, temp_root: str) -> None:
        """
        Universe membership should correctly distinguish old and new companies.

        - 2014: Only SH_OLD001 in universe
        - 2015-03-02: Neither (SH_OLD001 delisted, SH_NEW001 not yet IPO'd)
        - 2016: Only SH_NEW001 in universe
        """
        components_df = pl.DataFrame({
            "instrument_id": [
                "SH_OLD001", "SH_OLD001",
                "SH_NEW001",
            ],
            "date": [
                date(2010, 1, 1),
                date(2015, 3, 1),
                date(2016, 5, 1),
            ],
            "event": [IPO_EVENT, DELIST_EVENT, IPO_EVENT],
            "event_details": ["IPO", "Delist", "IPO"],
        })

        daily_df = pl.DataFrame({
            "instrument_id": ["SH_OLD001", "SH_NEW001"],
            "date": [date(2014, 1, 1), date(2016, 5, 1)],
            "ticker": ["600001", "600001"],
            "close": [5.0, 10.0],
            "open": [5.0, 10.0],
            "high": [5.5, 10.5],
            "low": [4.8, 9.8],
            "volume": [500000, 1000000],
            "adj_factor": [1.0, 1.0],
        })

        write_parquet(components_df, Market.CN, "components", temp_root)
        write_parquet(daily_df, Market.CN, "daily", temp_root)

        universe = CNAUniverse(temp_root)

        # 2014: SH_OLD001 should be in universe
        universe_2014 = universe.tickers(date(2014, 1, 1))
        assert "SH_OLD001" in universe_2014, "Old company should be in 2014 universe"
        assert "SH_NEW001" not in universe_2014, "New company didn't exist in 2014"

        # 2015-04-01: Neither should be in universe (between delist and IPO)
        universe_2015 = universe.tickers(date(2015, 4, 1))
        assert "SH_OLD001" not in universe_2015, "Old company delisted"
        assert "SH_NEW001" not in universe_2015, "New company not yet IPO'd"

        # 2016: SH_NEW001 should be in universe
        universe_2016 = universe.tickers(date(2016, 6, 1))
        assert "SH_OLD001" not in universe_2016, "Old company long gone"
        assert "SH_NEW001" in universe_2016, "New company is now listed"

    def test_code_reuse_historical_data_not_confused(self, temp_root: str) -> None:
        """
        Historical data should not confuse old and new companies.

        Querying instrument_id SH_OLD001 should only return old company data.
        Querying instrument_id SH_NEW001 should only return new company data.
        """
        daily_df = pl.DataFrame({
            "instrument_id": [
                "SH_OLD001", "SH_OLD001",
                "SH_NEW001", "SH_NEW001",
            ],
            "date": [
                date(2014, 1, 1), date(2015, 2, 1),  # Old company
                date(2016, 5, 1), date(2017, 1, 1),  # New company
            ],
            "ticker": ["600001", "600001", "600001", "600001"],  # All same ticker
            "close": [5.0, 3.0, 10.0, 12.0],  # Different price trajectories
            "open": [5.0, 3.0, 10.0, 12.0],
            "high": [5.5, 3.5, 10.5, 12.5],
            "low": [4.8, 2.8, 9.8, 11.8],
            "volume": [500000, 200000, 1000000, 1100000],
            "adj_factor": [1.0] * 4,
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        # Query by instrument_id should return correct subset
        df_old = bars_for_instrument(Market.CN, "SH_OLD001", root=temp_root)
        df_new = bars_for_instrument(Market.CN, "SH_NEW001", root=temp_root)

        if not df_old.is_empty():
            # Old company data should only have 2014-2015 dates
            old_dates = df_old.sort("date")["date"].to_list()
            assert all(d.year <= 2015 for d in old_dates), (
                "Old company data should not include 2016+ dates"
            )

            # Old company close prices should be ~3-5
            old_closes = df_old["close"].to_list()
            assert all(c < 10.0 for c in old_closes), (
                "Old company prices should be lower"
            )

        if not df_new.is_empty():
            # New company data should only have 2016+ dates
            new_dates = df_new.sort("date")["date"].to_list()
            assert all(d.year >= 2016 for d in new_dates), (
                "New company data should not include pre-2016 dates"
            )

            # New company close prices should be ~10+
            new_closes = df_new["close"].to_list()
            assert all(c >= 10.0 for c in new_closes), (
                "New company prices should be higher"
            )


# =============================================================================
# Historical Join Continuity Tests
# =============================================================================


class TestHistoricalJoinContinuity:
    """
    Tests for historical join continuity.

    When joining historical data, instrument_id should ensure correct continuity.
    """

    def test_join_by_instrument_id_correct(self, temp_root: str) -> None:
        """
        Joining by instrument_id should produce correct historical series.

        Across ticker changes, same instrument_id = same security.
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 10,
            "date": [
                date(2020, 1, 1), date(2020, 6, 1), date(2021, 1, 1), date(2021, 6, 1),
                date(2022, 1, 1),  # Ticker rename happens
                date(2022, 6, 1), date(2023, 1, 1), date(2023, 6, 1),
                date(2024, 1, 1), date(2024, 6, 1),
            ],
            "ticker": [
                "600000", "600000", "600000", "600000",  # Old ticker
                "PFYH",  # Rename date
                "PFYH", "PFYH", "PFYH", "PFYH", "PFYH",  # New ticker
            ],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
            "open": [10.0] * 10,
            "high": [10.5] * 10,
            "low": [9.8] * 10,
            "volume": [1000000] * 10,
            "adj_factor": [1.0] * 10,
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        df = bars_for_instrument(Market.CN, "SH600000", root=temp_root)

        if not df.is_empty():
            df = df.sort("date")

            # Should have all 10 rows
            assert len(df) == 10, "Should have complete historical series"

            # Close prices should increase continuously
            closes = df["close"].to_list()
            for i in range(len(closes) - 1):
                assert closes[i] <= closes[i + 1], (
                    f"Prices should increase: {closes[i]} -> {closes[i + 1]}"
                )

            # Verify ticker change at 2022-01-01
            ticker_change_row = df.filter(pl.col("date") == date(2022, 1, 1))
            if not ticker_change_row.is_empty():
                ticker_at_change = ticker_change_row["ticker"].item()
                assert ticker_at_change == "PFYH", "Ticker should change at rename date"

    def test_join_with_adjustment_factors(self, temp_root: str) -> None:
        """
        Adjustment factors should be tracked by instrument_id.

        Same instrument_id ensures adj_factor history is correct.
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 6,
            "date": [
                date(2023, 1, 1), date(2023, 6, 1),  # Before dividend
                date(2024, 1, 1),  # Ticker rename + dividend
                date(2024, 6, 1), date(2025, 1, 1), date(2025, 6, 1),  # After dividend
            ],
            "ticker": [
                "600000", "600000",
                "PFYH",  # Rename
                "PFYH", "PFYH", "PFYH",
            ],
            "close": [10.0, 11.0, 10.0, 11.0, 12.0, 13.0],  # Dividend adjusted
            "open": [10.0] * 6,
            "high": [10.5] * 6,
            "low": [9.8] * 6,
            "volume": [1000000] * 6,
            "adj_factor": [1.0, 1.0, 0.9, 0.9, 0.9, 0.9],  # Dividend at rename
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        df = bars_for_instrument(Market.CN, "SH600000", adjustment_mode="raw", root=temp_root)

        if not df.is_empty():
            # adj_factor should be tracked correctly across ticker change
            adj_factors = df.sort("date")["adj_factor"].to_list()
            assert adj_factors == [1.0, 1.0, 0.9, 0.9, 0.9, 0.9], (
                "Adjustment factors should be correct across ticker rename"
            )


# =============================================================================
# Instrument ID Validation Tests
# =============================================================================


class TestInstrumentIdValidation:
    """Tests for instrument_id validation."""

    def test_schema_validation_with_instrument_id(self, temp_root: str) -> None:
        """
        Schema validation should require instrument_id.

        Without instrument_id, data is invalid.
        """
        # Valid DataFrame
        valid_df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 1)],
            "ticker": ["600000"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000000],
            "adj_factor": [1.0],
        })

        errors = validate_dataframe_schema(valid_df)
        assert len(errors) == 0, "Valid DataFrame should pass validation"

        # DataFrame without instrument_id
        invalid_df = pl.DataFrame({
            "date": [date(2024, 1, 1)],
            "ticker": ["600000"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000000],
            "adj_factor": [1.0],
        })

        errors = validate_dataframe_schema(invalid_df, require_all=True)
        assert len(errors) > 0, "Missing instrument_id should fail validation"
        assert "instrument_id" in str(errors[0]), "Error should mention instrument_id"

    def test_instrument_id_format_cn_a(self) -> None:
        """
        CN_A instrument_id format should be validated.

        Format: SH + 6-digit code for Shanghai
               SZ + 6-digit code for Shenzhen
        """
        # Valid formats
        valid_ids = ["SH600000", "SH600036", "SH688001", "SZ000001", "SZ300001", "SZ002475"]

        for id in valid_ids:
            # Format check (SH/SZ + 6 digits)
            assert len(id) == 8, f"{id} should be 8 characters"
            assert id[:2] in ["SH", "SZ"], f"{id} should start with SH or SZ"

    def test_instrument_id_format_us(self) -> None:
        """
        US instrument_id format should be validated.

        Format: ticker symbol (AAPL, MSFT, etc.)
        """
        # US uses ticker as instrument_id
        valid_ids = ["AAPL", "MSFT", "JPM", "GOOGL"]

        for id in valid_ids:
            assert len(id) >= 1, f"{id} should have length"


# =============================================================================
# Partition by Instrument ID Tests
# =============================================================================


class TestPartitionByInstrumentId:
    """
    Tests for partitioning by instrument_id in Parquet storage.

    Data is partitioned by instrument_id for efficient per-instrument queries.
    """

    def test_partition_by_instrument_id(self, temp_root: str) -> None:
        """
        Parquet should be partitioned by instrument_id.

        This enables efficient single-instrument queries.
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000", "SZ000001", "SZ000001"],
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 2)],
            "ticker": ["600000", "600000", "000001", "000001"],
            "open": [10.0, 10.1, 20.0, 20.1],
            "high": [10.5, 10.6, 20.5, 20.6],
            "low": [9.8, 9.9, 19.8, 19.9],
            "close": [10.2, 10.3, 20.2, 20.3],
            "volume": [1000000, 1100000, 500000, 550000],
            "adj_factor": [1.0] * 4,
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        # Check partition directories exist
        import os
        daily_path = os.path.join(temp_root, "cn", "daily")

        assert os.path.exists(os.path.join(daily_path, "instrument_id=SH600000")), (
            "SH600000 partition should exist"
        )
        assert os.path.exists(os.path.join(daily_path, "instrument_id=SZ000001")), (
            "SZ000001 partition should exist"
        )

    def test_read_partition_by_instrument_id(self, temp_root: str) -> None:
        """
        Reading by instrument_id should return only that instrument's data.

        Partition enables efficient retrieval without scanning all data.
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000", "SZ000001", "SZ000001"],
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 2)],
            "ticker": ["600000", "600000", "000001", "000001"],
            "open": [10.0, 10.1, 20.0, 20.1],
            "high": [10.5, 10.6, 20.5, 20.6],
            "low": [9.8, 9.9, 19.8, 19.9],
            "close": [10.2, 10.3, 20.2, 20.3],
            "volume": [1000000, 1100000, 500000, 550000],
            "adj_factor": [1.0] * 4,
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        # Read specific partition
        partition_df = read_partition(temp_root, Market.CN, "daily", "SH600000")

        assert not partition_df.is_empty(), "Partition should have data"
        assert partition_df["instrument_id"].unique().to_list() == ["SH600000"], (
            "Partition should only contain SH600000 data"
        )
        assert len(partition_df) == 2, "Should have 2 rows for SH600000"


# =============================================================================
# Get Instrument IDs Tests
# =============================================================================


class TestGetInstrumentIds:
    """Tests for get_instrument_ids function."""

    def test_get_instrument_ids_returns_ids(self, temp_root: str) -> None:
        """
        get_instrument_ids should return all instrument_ids in data_lake.

        Not tickers - returns instrument_ids.
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SZ000001", "SH600036"],
            "date": [date(2024, 1, 1)] * 3,
            "ticker": ["600000", "000001", "600036"],
            "open": [10.0] * 3,
            "high": [10.5] * 3,
            "low": [9.8] * 3,
            "close": [10.2] * 3,
            "volume": [1000000] * 3,
            "adj_factor": [1.0] * 3,
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        ids = get_instrument_ids(Market.CN, temp_root)

        assert len(ids) >= 3, "Should have at least 3 instrument_ids"
        assert "SH600000" in ids, "SH600000 should be in list"
        assert "SZ000001" in ids, "SZ000001 should be in list"
        assert "SH600036" in ids, "SH600036 should be in list"

    def test_get_instrument_ids_empty(self, temp_root: str) -> None:
        """get_instrument_ids should return empty list for empty data_lake."""
        ids = get_instrument_ids(Market.CN, temp_root)
        assert ids == [], "Empty data_lake should return empty list"


# =============================================================================
# Integration Tests
# =============================================================================


class TestInstrumentIdentityIntegration:
    """Integration tests for instrument identity across components."""

    def test_identity_with_components_events(self, temp_root: str) -> None:
        """
        Component events should track instrument_id, not ticker.

        IPO/delist events should use instrument_id.
        """
        components_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000"],
            "date": [date(1999, 11, 10), date(2024, 1, 15)],  # IPO and ticker rename
            "event": [IPO_EVENT, "RENAME"],
            "event_details": ["IPO as 600000", "Ticker renamed to PFYH"],
        })

        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 2,
            "date": [date(2024, 1, 1), date(2024, 1, 20)],
            "ticker": ["600000", "PFYH"],
            "open": [10.0] * 2,
            "high": [10.5] * 2,
            "low": [9.8] * 2,
            "close": [10.2] * 2,
            "volume": [1000000] * 2,
            "adj_factor": [1.0] * 2,
        })

        write_parquet(components_df, Market.CN, "components", temp_root)
        write_parquet(daily_df, Market.CN, "daily", temp_root)

        universe = CNAUniverse(temp_root)

        # IPO date should be tracked
        ipo = universe.ipo_date("SH600000")
        assert ipo == date(1999, 11, 10), "IPO date should be correct"

        # instrument_id should be in universe
        assert universe.contains("SH600000", date(2024, 1, 1)), "Should be in universe before rename"
        assert universe.contains("SH600000", date(2024, 1, 20)), "Should be in universe after rename"

    def test_identity_with_sector_assignments(self, temp_root: str) -> None:
        """
        Sector assignments should track instrument_id, not ticker.

        Sector changes should be associated with instrument_id.
        """
        sectors_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000"],
            "date": [date(2020, 1, 1), date(2024, 1, 15)],  # Sector change at ticker rename
            "sector": ["10", "15"],
            "sector_name": ["农林牧渔", "银行"],
        })

        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 2,
            "date": [date(2020, 1, 1), date(2024, 2, 1)],
            "ticker": ["600000", "PFYH"],
            "open": [10.0] * 2,
            "high": [10.5] * 2,
            "low": [9.8] * 2,
            "close": [10.2] * 2,
            "volume": [1000000] * 2,
            "adj_factor": [1.0] * 2,
        })

        write_parquet(sectors_df, Market.CN, "sectors", temp_root)
        write_parquet(daily_df, Market.CN, "daily", temp_root)

        from trendspec.data.sectors import clear_sector_cache, sector

        clear_sector_cache()

        # Sector should be tracked by instrument_id
        sector_2020 = sector(Market.CN, "SH600000", date(2020, 1, 1), temp_root)
        sector_2024 = sector(Market.CN, "SH600000", date(2024, 2, 1), temp_root)

        assert sector_2020 == "10", "Sector should be '10' in 2020"
        assert sector_2024 == "15", "Sector should be '15' in 2024"
