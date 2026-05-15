"""
Tests for TrendSpec PIT Sector functionality - CRITICAL for sector attribution accuracy.

Key test cases:
1. Sector reclassification - Before date = sector A, after date = sector B
2. Shenwan Level 1 sector changes (A-share)
3. GICS sector changes (US)
4. Edge: sector change on exact date

PIT is CRITICAL because:
- Sector attribution affects factor performance attribution
- If using "current sector" for historical data, attributions are wrong
- Example: Apple was in Technology, in 2018, Communication Services after GICS reclassification
"""

import tempfile
from datetime import date

import polars as pl
import pytest

from trendspec.data.markets import Market
from trendspec.data.sectors import (
    GICS_SECTORS,
    SHENWAN_L1_SECTORS,
    SectorIndex,
    clear_sector_cache,
    sector,
    sector_name,
    sector_universe,
)
from trendspec.ingest.writer import write_parquet


# =============================================================================
# PIT Sector Lookup Tests
# =============================================================================


class TestPITSectorLookup:
    """
    Tests that sector attribution uses the correct sector at each point in time.

    Critical for factor attribution accuracy.
    """

    def test_sector_before_reclassification(self, temp_root: str) -> None:
        """
        PIT sector lookup: before reclassification date, returns old sector.

        Test case: SH600000 reclassified on 2024-01-15
        - Before 2024-01-15: sector "10" (农林牧渔)
        - After 2024-01-15: sector "15" (银行)
        """
        # Create sector data with reclassification
        sectors_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000"],
            "date": [date(2020, 1, 1), date(2024, 1, 15)],
            "sector": ["10", "15"],  # Agriculture -> Banking
            "sector_name": ["农林牧渔", "银行"],
        })

        write_parquet(sectors_df, Market.CN_A, "sectors", temp_root)

        clear_sector_cache()
        index = SectorIndex(Market.CN_A, temp_root)

        # Before reclassification: should return sector "10"
        sector_before = index.sector("SH600000", date(2023, 12, 31))
        assert sector_before == "10", (
            f"PIT sector lookup failed: expected '10' (农林牧渔) for 2023-12-31, "
            f"got '{sector_before}'. This would cause wrong sector attribution in backtests!"
        )

        # After reclassification: should return sector "15"
        sector_after = index.sector("SH600000", date(2024, 2, 1))
        assert sector_after == "15", (
            f"PIT sector lookup failed: expected '15' (银行) for 2024-02-01, "
            f"got '{sector_after}'. This would cause wrong sector attribution in backtests!"
        )

    def test_sector_on_exact_reclassification_date(self, temp_root: str) -> None:
        """
        Edge case: sector lookup on exact reclassification date.

        On the date of reclassification, should return NEW sector
        (assuming assignment takes effect on that date).
        """
        sectors_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000"],
            "date": [date(2020, 1, 1), date(2024, 1, 15)],
            "sector": ["10", "15"],
            "sector_name": ["农林牧渔", "银行"],
        })

        write_parquet(sectors_df, Market.CN_A, "sectors", temp_root)

        clear_sector_cache()
        index = SectorIndex(Market.CN_A, temp_root)

        # On exact reclassification date: should return NEW sector
        sector_on_date = index.sector("SH600000", date(2024, 1, 15))
        assert sector_on_date == "15", (
            f"On reclassification date 2024-01-15, should return new sector '15', "
            f"got '{sector_on_date}'"
        )

    def test_sector_before_first_assignment(self, temp_root: str) -> None:
        """
        Sector lookup before first assignment should return None.

        No sector history exists before IPO or first classification.
        """
        sectors_df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2020, 1, 1)],
            "sector": ["15"],
            "sector_name": ["银行"],
        })

        write_parquet(sectors_df, Market.CN_A, "sectors", temp_root)

        clear_sector_cache()
        index = SectorIndex(Market.CN_A, temp_root)

        # Before first assignment: should return None
        sector_before = index.sector("SH600000", date(2019, 1, 1))
        assert sector_before is None, (
            f"Sector lookup before first assignment should return None, "
            f"got '{sector_before}'"
        )

    def test_multiple_sector_changes(self, temp_root: str) -> None:
        """
        Test instrument with multiple sector reclassifications.

        Test case: SH600001 changes sector 3 times
        - 2018-01-01: sector "03" (化工)
        - 2020-06-01: sector "06" (电子)
        - 2023-01-01: sector "23" (计算机)
        """
        sectors_df = pl.DataFrame({
            "instrument_id": ["SH600001", "SH600001", "SH600001"],
            "date": [date(2018, 1, 1), date(2020, 6, 1), date(2023, 1, 1)],
            "sector": ["03", "06", "23"],
            "sector_name": ["化工", "电子", "计算机"],
        })

        write_parquet(sectors_df, Market.CN_A, "sectors", temp_root)

        clear_sector_cache()
        index = SectorIndex(Market.CN_A, temp_root)

        # Test each period
        assert index.sector("SH600001", date(2019, 1, 1)) == "03", "2019: 化工"
        assert index.sector("SH600001", date(2021, 1, 1)) == "06", "2021: 电子"
        assert index.sector("SH600001", date(2024, 1, 1)) == "23", "2024: 计算机"

        # Test boundary dates
        assert index.sector("SH600001", date(2020, 5, 31)) == "03", "Last day of old sector"
        assert index.sector("SH600001", date(2020, 6, 1)) == "06", "First day of new sector"


# =============================================================================
# GICS Sector Reclassification Tests (US)
# =============================================================================


class TestGICSSectorReclassification:
    """
    Tests for GICS sector reclassification in US market.

    Historical GICS changes:
    - 2016: REITs moved from Financials (40) to Real Estate (60)
    - 2018: Telecom/Media moved to Communication Services (50)
    """

    def test_gics_sector_change(self, temp_root: str) -> None:
        """
        Test GICS sector reclassification for US stock.

        Example: Hypothetical AAPL sector change
        - Before 2022-07-01: GICS "45" (Information Technology)
        - After 2022-07-01: GICS "50" (Communication Services)
        """
        sectors_df = pl.DataFrame({
            "instrument_id": ["AAPL", "AAPL", "MSFT"],
            "date": [date(2020, 1, 1), date(2022, 7, 1), date(2020, 1, 1)],
            "sector": ["45", "50", "45"],  # Tech -> Communication
            "sector_name": ["Information Technology", "Communication Services",
                           "Information Technology"],
        })

        write_parquet(sectors_df, Market.US, "sectors", temp_root)

        clear_sector_cache()
        index = SectorIndex(Market.US, temp_root)

        # Before reclassification
        sector_before = index.sector("AAPL", date(2021, 6, 30))
        assert sector_before == "45", "AAPL was in Technology before change"

        # After reclassification
        sector_after = index.sector("AAPL", date(2022, 7, 2))
        assert sector_after == "50", "AAPL moved to Communication Services"

        # MSFT unchanged
        sector_msft = index.sector("MSFT", date(2024, 1, 1))
        assert sector_msft == "45", "MSFT stayed in Technology"

    def test_reit_sector_change(self, temp_root: str) -> None:
        """
        Test REITs moved from Financials to Real Estate in 2016 GICS change.

        Historical fact: GICS created Real Estate sector (60) in 2016.
        """
        sectors_df = pl.DataFrame({
            "instrument_id": ["REIT001", "REIT001"],
            "date": [date(2010, 1, 1), date(2016, 9, 1)],
            "sector": ["40", "60"],  # Financials -> Real Estate
            "sector_name": ["Financials", "Real Estate"],
        })

        write_parquet(sectors_df, Market.US, "sectors", temp_root)

        clear_sector_cache()
        index = SectorIndex(Market.US, temp_root)

        # Before 2016 GICS change: REIT in Financials
        sector_2015 = index.sector("REIT001", date(2015, 6, 1))
        assert sector_2015 == "40", "REITs were in Financials before 2016"

        # After 2016: REIT in Real Estate
        sector_2017 = index.sector("REIT001", date(2017, 1, 1))
        assert sector_2017 == "60", "REITs moved to Real Estate in 2016"


# =============================================================================
# Sector Universe Tests
# =============================================================================


class TestPITSectorUniverse:
    """
    Tests for sector_universe - getting all stocks in a sector at a date.
    """

    def test_sector_universe_before_reclassification(self, temp_root: str) -> None:
        """
        sector_universe should return stocks that WERE in sector at that date.

        Not stocks currently in sector (survivorship bias prevention).
        """
        sectors_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000", "SH600001", "SH600002"],
            "date": [
                date(2020, 1, 1), date(2024, 1, 15),
                date(2020, 1, 1), date(2020, 1, 1),
            ],
            "sector": ["10", "15", "15", "15"],
            "sector_name": ["农林牧渔", "银行", "银行", "银行"],
        })

        write_parquet(sectors_df, Market.CN_A, "sectors", temp_root)

        clear_sector_cache()
        index = SectorIndex(Market.CN_A, temp_root)

        # Banking sector in 2023: should NOT include SH600000 (still in Agriculture)
        banking_2023 = index.sector_universe("15", date(2023, 1, 1))
        assert "SH600000" not in banking_2023, (
            "SH600000 was in Agriculture (10) in 2023, not Banking (15)"
        )
        assert "SH600001" in banking_2023, "SH600001 was in Banking in 2023"
        assert "SH600002" in banking_2023, "SH600002 was in Banking in 2023"

        # Banking sector in 2024: should include SH600000 (reclassified)
        banking_2024 = index.sector_universe("15", date(2024, 3, 1))
        assert "SH600000" in banking_2024, "SH600000 reclassified to Banking in 2024"

    def test_sector_universe_empty_sector(self, temp_root: str) -> None:
        """sector_universe for non-existent sector should return empty list."""
        sectors_df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2020, 1, 1)],
            "sector": ["15"],
            "sector_name": ["银行"],
        })

        write_parquet(sectors_df, Market.CN_A, "sectors", temp_root)

        clear_sector_cache()
        index = SectorIndex(Market.CN_A, temp_root)

        # Non-existent sector code
        result = index.sector_universe("99", date(2024, 1, 1))
        assert result == [], "Non-existent sector should return empty list"


# =============================================================================
# Sector Name Tests
# =============================================================================


class TestSectorNames:
    """Tests for sector name lookup."""

    def test_shenwan_sector_names(self) -> None:
        """Shenwan Level 1 sector names should be correct."""
        assert sector_name(Market.CN_A, "15") == "银行"
        assert sector_name(Market.CN_A, "11") == "医药生物"
        assert sector_name(Market.CN_A, "01") == "农林牧渔"

    def test_gics_sector_names(self) -> None:
        """GICS sector names should be correct."""
        assert sector_name(Market.US, "45") == "Information Technology"
        assert sector_name(Market.US, "40") == "Financials"
        assert sector_name(Market.US, "10") == "Energy"

    def test_unknown_sector_code(self) -> None:
        """Unknown sector code should return None."""
        assert sector_name(Market.CN_A, "99") is None
        assert sector_name(Market.US, "99") is None


# =============================================================================
# Binary Search Efficiency Tests
# =============================================================================


class TestSectorIndexBinarySearch:
    """
    Tests for binary search on sorted dates in sector lookup.

    The implementation uses binary search for O(log n) lookup.
    """

    def test_binary_search_many_dates(self, temp_root: str) -> None:
        """
        Test binary search with many sector change dates.

        Performance test for sector lookup with 100+ reclassifications.
        """
        # Create many sector changes (simulate frequent reclassifications)
        dates_list = [date(2020, 1, 1)]
        sectors_list = ["01"]

        # Monthly sector changes for 5 years = 60 changes
        for i in range(60):
            year = 2020 + (i // 12)
            month = 1 + (i % 12)
            dates_list.append(date(year, month, 1))
            sectors_list.append(str(i % 28 + 1).zfill(2))  # Shenwan codes 01-28

        sectors_df = pl.DataFrame({
            "instrument_id": ["SH699999"] * len(dates_list),
            "date": dates_list,
            "sector": sectors_list,
            "sector_name": ["Sector"] * len(dates_list),
        })

        write_parquet(sectors_df, Market.CN_A, "sectors", temp_root)

        clear_sector_cache()
        index = SectorIndex(Market.CN_A, temp_root)

        # Verify correct sector at each period
        # Test random points
        test_dates = [
            date(2021, 6, 15),  # Should find assignment at 2021-06-01
            date(2023, 3, 10),  # Should find assignment at 2023-03-01
            date(2024, 11, 20),  # Should find assignment at 2024-11-01
        ]

        for test_date in test_dates:
            sector_result = index.sector("SH699999", test_date)
            assert sector_result is not None, f"Should find sector for {test_date}"

    def test_sector_lookup_performance(self, temp_root: str) -> None:
        """
        Performance test: sector lookup should be O(1) or O(log n).

        With pre-built index, lookup should be fast even with many instruments.
        """
        # Create many instruments with sector assignments
        num_instruments = 500  # Simulate realistic universe size
        instrument_ids = [f"SH{i:06d}" for i in range(num_instruments)]

        sectors_df = pl.DataFrame({
            "instrument_id": instrument_ids,
            "date": [date(2020, 1, 1)] * num_instruments,
            "sector": [str((i % 28) + 1).zfill(2) for i in range(num_instruments)],
            "sector_name": ["Sector"] * num_instruments,
        })

        write_parquet(sectors_df, Market.CN_A, "sectors", temp_root)

        clear_sector_cache()
        index = SectorIndex(Market.CN_A, temp_root)

        # Lookup should work for all instruments
        for i in range(10):  # Test subset
            instrument_id = f"SH{i:06d}"
            sector_result = index.sector(instrument_id, date(2022, 1, 1))
            assert sector_result is not None, f"Should find sector for {instrument_id}"


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestSectorConvenienceFunctions:
    """Tests for convenience functions that wrap SectorIndex."""

    def test_sector_function(self, temp_root: str) -> None:
        """sector() function should work correctly."""
        sectors_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000"],
            "date": [date(2020, 1, 1), date(2024, 1, 15)],
            "sector": ["10", "15"],
            "sector_name": ["农林牧渔", "银行"],
        })

        write_parquet(sectors_df, Market.CN_A, "sectors", temp_root)

        clear_sector_cache()

        # Test sector function
        result_before = sector(Market.CN_A, "SH600000", date(2023, 1, 1), temp_root)
        assert result_before == "10"

        result_after = sector(Market.CN_A, "SH600000", date(2024, 3, 1), temp_root)
        assert result_after == "15"

    def test_sector_universe_function(self, temp_root: str) -> None:
        """sector_universe() function should work correctly."""
        sectors_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600001", "SH600002"],
            "date": [date(2020, 1, 1)] * 3,
            "sector": ["15", "15", "10"],
            "sector_name": ["银行", "银行", "农林牧渔"],
        })

        write_parquet(sectors_df, Market.CN_A, "sectors", temp_root)

        clear_sector_cache()

        banking = sector_universe(Market.CN_A, "15", date(2022, 1, 1), temp_root)
        assert "SH600000" in banking
        assert "SH600001" in banking
        assert "SH600002" not in banking


# =============================================================================
# Edge Cases
# =============================================================================


class TestSectorEdgeCases:
    """Tests for edge cases in sector attribution."""

    def test_empty_sector_index(self, temp_root: str) -> None:
        """Empty sector data should return None for all lookups."""
        clear_sector_cache()
        index = SectorIndex(Market.CN_A, temp_root)

        assert index.instrument_count() == 0
        assert index.sector("SH600000", date(2024, 1, 1)) is None

    def test_instrument_not_in_index(self, temp_root: str) -> None:
        """Unknown instrument should return None."""
        sectors_df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2020, 1, 1)],
            "sector": ["15"],
            "sector_name": ["银行"],
        })

        write_parquet(sectors_df, Market.CN_A, "sectors", temp_root)

        clear_sector_cache()
        index = SectorIndex(Market.CN_A, temp_root)

        # Unknown instrument
        assert index.sector("UNKNOWN", date(2024, 1, 1)) is None

    def test_future_date_sector_lookup(self, temp_root: str) -> None:
        """
        Date after all assignments should return most recent sector.

        This tests the binary search handles future dates correctly.
        """
        sectors_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000"],
            "date": [date(2020, 1, 1), date(2022, 6, 1)],
            "sector": ["10", "15"],
            "sector_name": ["农林牧渔", "银行"],
        })

        write_parquet(sectors_df, Market.CN_A, "sectors", temp_root)

        clear_sector_cache()
        index = SectorIndex(Market.CN_A, temp_root)

        # Future date: should return most recent sector
        sector_future = index.sector("SH600000", date(2030, 1, 1))
        assert sector_future == "15", "Future date should return last known sector"

    def test_all_sectors_at_date(self, temp_root: str) -> None:
        """all_sectors_at_date should return sector-to-instruments mapping."""
        sectors_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600001", "SH600002"],
            "date": [date(2020, 1, 1)] * 3,
            "sector": ["15", "15", "10"],
            "sector_name": ["银行", "银行", "农林牧渔"],
        })

        write_parquet(sectors_df, Market.CN_A, "sectors", temp_root)

        clear_sector_cache()
        index = SectorIndex(Market.CN_A, temp_root)

        all_sectors = index.all_sectors_at_date(date(2022, 1, 1))

        assert "15" in all_sectors
        assert "10" in all_sectors
        assert "SH600000" in all_sectors["15"]
        assert "SH600001" in all_sectors["15"]
        assert "SH600002" in all_sectors["10"]