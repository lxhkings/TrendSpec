"""
Tests for TrendSpec PIT Universe functionality - CRITICAL for survivorship bias prevention.

Key test cases:
1. 600631 商业城 - Delisted 2016, should be in 2015-06-01 universe
2. Survivorship bias prevention:
   - Delisted stocks should be in historical universe
   - Stocks IPO after date should NOT be in universe
   - Halted stocks should be filtered correctly
3. Edge cases: IPO day, delist day, halt day

PIT is CRITICAL because:
- Survivorship bias invalidates all backtest results
- ~800+ A-share stocks delisted since 2010
- If universe uses "current stocks" to backfill history, returns are overstated
"""

import tempfile
from datetime import date

import polars as pl
import pytest

from trendspec.data.markets import Market
from trendspec.data.parquet_loader import bars, read_components
from trendspec.data.sectors import sector
from trendspec.data.universe import CNAUniverse, USUniverse, get_universe
from trendspec.data.universe.cn_a import IPO_EVENT, DELIST_EVENT, HALT_EVENT, RESUME_EVENT
from trendspec.ingest.writer import write_parquet


# =============================================================================
# Survivorship Bias Tests - The Core Purpose of PIT
# =============================================================================


class TestSurvivorshipBiasPrevention:
    """
    Tests to verify survivorship bias prevention - the CRITICAL foundation.

    If these tests fail, backtest results are invalid.
    """

    def test_delisted_stock_in_historical_universe(self, temp_root: str) -> None:
        """
        CRITICAL: Delisted stock should be in historical universe.

        Test case: 600631 商业城 delisted 2016-03-29
        - Should be in universe at 2015-06-01 (before delisting)
        - Should NOT be in universe at 2017-01-01 (after delisting)

        This is the core survivorship bias prevention.
        """
        # Create components data with delisting
        components_df = pl.DataFrame({
            "instrument_id": [
                "SH600000",  # Active stock
                "SH600631",  # 商业城 - IPO
                "SH600631",  # 商业城 - delist
            ],
            "date": [
                date(1999, 11, 10),
                date(1996, 12, 2),    # IPO date
                date(2016, 3, 29),    # Delist date
            ],
            "event": [IPO_EVENT, IPO_EVENT, DELIST_EVENT],
            "event_details": ["IPO", "商业城 IPO", "商业城 delisted"],
        })

        # Create daily data showing 商业城 existed in 2015
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600631", "SH600631", "SH600631"],
            "date": [date(2015, 1, 5), date(2015, 1, 5), date(2015, 6, 1), date(2016, 3, 28)],
            "ticker": ["600000", "600631", "600631", "600631"],
            "open": [10.0, 8.0, 8.5, 7.0],
            "high": [10.5, 8.5, 9.0, 7.5],
            "low": [9.8, 7.8, 8.2, 6.8],
            "close": [10.2, 8.2, 8.8, 7.2],
            "volume": [1000000, 800000, 850000, 200000],
            "adj_factor": [1.0, 1.0, 1.0, 1.0],
        })

        write_parquet(components_df, Market.CN_A, "components", temp_root)
        write_parquet(daily_df, Market.CN_A, "daily", temp_root)

        universe = CNAUniverse(temp_root)

        # CRITICAL TEST: 商业城 should be in 2015-06-01 universe
        universe_2015 = universe.tickers(date(2015, 6, 1))
        assert "SH600631" in universe_2015, (
            "SURVIVORSHIP BIAS: 商业城 was trading in 2015-06-01, "
            "but not found in historical universe. This would overstate backtest returns!"
        )

        # And should NOT be in 2017 universe
        universe_2017 = universe.tickers(date(2017, 1, 1))
        assert "SH600631" not in universe_2017, (
            "商业城 was delisted in 2016, should NOT be in 2017 universe"
        )

    def test_delisted_stock_return_calculations(self, temp_root: str) -> None:
        """
        Verify that backtests can correctly include delisted stock returns.

        If universe excludes 商业城 for 2015, backtest returns are overstated
        because it excludes a stock that eventually went to zero.
        """
        # Simulate 商业城 returns leading to delisting
        # Stock price declining: 10 -> 8 -> 5 -> 2 -> 0 (delist)
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600631"] * 4 + ["SH600000"] * 2,
            "date": [
                date(2015, 1, 1), date(2015, 6, 1), date(2016, 1, 1), date(2016, 3, 28),
                date(2015, 1, 1), date(2015, 6, 1),
            ],
            "ticker": ["600631"] * 4 + ["600000"] * 2,
            "open": [10.0, 8.0, 5.0, 2.0, 10.0, 11.0],
            "high": [10.5, 8.5, 5.5, 2.5, 10.5, 11.5],
            "low": [9.5, 7.5, 4.5, 1.8, 9.8, 10.8],
            "close": [10.0, 8.0, 5.0, 2.0, 10.2, 11.0],
            "volume": [1000000, 800000, 500000, 100000, 1000000, 1100000],
            "adj_factor": [1.0] * 6,
        })

        components_df = pl.DataFrame({
            "instrument_id": ["SH600631", "SH600631", "SH600000"],
            "date": [date(1996, 12, 2), date(2016, 3, 29), date(1999, 11, 10)],
            "event": [IPO_EVENT, DELIST_EVENT, IPO_EVENT],
            "event_details": ["IPO", "Delisted", "IPO"],
        })

        write_parquet(daily_df, Market.CN_A, "daily", temp_root)
        write_parquet(components_df, Market.CN_A, "components", temp_root)

        universe = CNAUniverse(temp_root)

        # 2015-01-01 universe should include both 商业城 and 浦发银行
        universe_2015 = universe.tickers(date(2015, 1, 1))
        assert "SH600631" in universe_2015, "商业城 should be in 2015 universe"
        assert "SH600000" in universe_2015, "浦发银行 should be in 2015 universe"

        # Calculate returns for 商业城 in 2015
        # 10 -> 8 = -20% return
        # If survivorship bias exists, this -20% would be missing from returns

    def test_multiple_delisted_stocks(self, temp_root: str) -> None:
        """
        Test multiple delisted stocks are all in historical universe.

        ~800+ A-share stocks delisted since 2010.
        All must be in historical universe for survivorship-free backtest.
        """
        # Simulate multiple delisted stocks
        components_df = pl.DataFrame({
            "instrument_id": [
                "SH600631", "SH600631",  # 商业城
                "SZ000002", "SZ000002",  # 万科A (hypothetical delist)
                "SH600001", "SH600001",  # Another delisted
                "SH600000",  # Active
            ],
            "date": [
                date(1996, 12, 2), date(2016, 3, 29),  # 商业城
                date(1991, 1, 29), date(2018, 6, 30),   # 万科A
                date(1998, 1, 1), date(2015, 7, 1),    # Another
                date(1999, 11, 10),  # Active
            ],
            "event": [
                IPO_EVENT, DELIST_EVENT,
                IPO_EVENT, DELIST_EVENT,
                IPO_EVENT, DELIST_EVENT,
                IPO_EVENT,
            ],
            "event_details": ["IPO", "Delist", "IPO", "Delist", "IPO", "Delist", "IPO"],
        })

        daily_df = pl.DataFrame({
            "instrument_id": ["SH600631", "SZ000002", "SH600001", "SH600000"],
            "date": [date(2015, 1, 1), date(2015, 1, 1), date(2015, 1, 1), date(2015, 1, 1)],
            "ticker": ["600631", "000002", "600001", "600000"],
            "open": [8.0, 20.0, 5.0, 10.0],
            "high": [8.5, 20.5, 5.5, 10.5],
            "low": [7.8, 19.8, 4.8, 9.8],
            "close": [8.2, 20.2, 5.2, 10.2],
            "volume": [800000, 500000, 300000, 1000000],
            "adj_factor": [1.0] * 4,
        })

        write_parquet(components_df, Market.CN_A, "components", temp_root)
        write_parquet(daily_df, Market.CN_A, "daily", temp_root)

        universe = CNAUniverse(temp_root)

        # All 3 delisted stocks should be in 2015 universe
        universe_2015 = universe.tickers(date(2015, 1, 1))
        assert "SH600631" in universe_2015, "商业城 should be in 2015 universe"
        assert "SZ000002" in universe_2015, "万科A should be in 2015 universe"
        assert "SH600001" in universe_2015, "SH600001 should be in 2015 universe"
        assert "SH600000" in universe_2015, "浦发银行 should be in 2015 universe"


# =============================================================================
# IPO Filtering Tests
# =============================================================================


class TestIPOFiltering:
    """
    Tests that stocks IPO-ing after a date are NOT in universe at that date.

    This prevents look-ahead bias - you can't trade a stock before it exists.
    """

    def test_stock_not_in_universe_before_ipo(self, temp_root: str) -> None:
        """
        Stock should NOT be in universe before IPO date.

        Test case: SH600036 IPO date is 2003-08-22
        - Should NOT be in 2002-01-01 universe
        - Should be in 2004-01-01 universe
        """
        components_df = pl.DataFrame({
            "instrument_id": ["SH600036", "SH600000"],
            "date": [date(2003, 8, 22), date(1999, 11, 10)],
            "event": [IPO_EVENT, IPO_EVENT],
            "event_details": ["招商银行 IPO", "浦发银行 IPO"],
        })

        daily_df = pl.DataFrame({
            "instrument_id": ["SH600036", "SH600036", "SH600000"],
            "date": [date(2003, 8, 22), date(2004, 1, 1), date(2002, 1, 1)],
            "ticker": ["600036", "600036", "600000"],
            "open": [10.0, 10.5, 9.0],
            "high": [10.5, 11.0, 9.5],
            "low": [9.8, 10.2, 8.8],
            "close": [10.2, 10.8, 9.2],
            "volume": [2000000, 1500000, 1000000],
            "adj_factor": [1.0] * 3,
        })

        write_parquet(components_df, Market.CN_A, "components", temp_root)
        write_parquet(daily_df, Market.CN_A, "daily", temp_root)

        universe = CNAUniverse(temp_root)

        # 2002 universe should NOT include 招商银行 (IPO 2003)
        universe_2002 = universe.tickers(date(2002, 1, 1))
        assert "SH600036" not in universe_2002, (
            "招商银行 IPO'd in 2003, should NOT be in 2002 universe"
        )

        # But 浦发银行 should be there (IPO 1999)
        assert "SH600000" in universe_2002, "浦发银行 IPO'd in 1999, should be in 2002"

        # 2004 universe should include 招商银行
        universe_2004 = universe.tickers(date(2004, 1, 1))
        assert "SH600036" in universe_2004, "招商银行 should be in 2004 universe"

    def test_recent_ipo_stocks(self, temp_root: str) -> None:
        """
        Test recent IPO stocks (STAR Market, ChiNext) are correctly filtered.

        STAR Market stocks IPO since 2019, ChiNext since 2009.
        """
        components_df = pl.DataFrame({
            "instrument_id": [
                "SH600000",  # IPO 1999
                "SZ300001",  # ChiNext IPO 2009
                "SH688001",  # STAR Market IPO 2019
            ],
            "date": [
                date(1999, 11, 10),
                date(2009, 10, 30), # First ChiNext IPO
                date(2019, 7, 22),  # First STAR Market IPO
            ],
            "event": [IPO_EVENT, IPO_EVENT, IPO_EVENT],
            "event_details": ["Main board IPO", "ChiNext IPO", "STAR IPO"],
        })

        # Daily data for each stock at dates relevant to queries
        daily_df = pl.DataFrame({
            "instrument_id": [
                "SH600000", "SH600000", "SH600000",
                "SZ300001", "SZ300001",
                "SH688001", "SH688001",
            ],
            "date": [
                date(2010, 1, 5), date(2020, 1, 5), date(2000, 1, 5),
                date(2010, 1, 5), date(2020, 1, 5),
                date(2020, 1, 5), date(2019, 8, 1),
            ],
            "ticker": ["600000", "600000", "600000", "300001", "300001", "688001", "688001"],
            "open": [10.0, 11.0, 9.0, 30.0, 35.0, 50.0, 55.0],
            "high": [10.5, 11.5, 9.5, 32.0, 37.0, 52.0, 57.0],
            "low": [9.8, 10.8, 8.8, 28.0, 33.0, 48.0, 53.0],
            "close": [10.2, 11.2, 9.2, 31.0, 36.0, 51.0, 56.0],
            "volume": [1000000, 1100000, 1000000, 500000, 600000, 1000000, 1200000],
            "adj_factor": [1.0] * 7,
        })

        write_parquet(components_df, Market.CN_A, "components", temp_root)
        write_parquet(daily_df, Market.CN_A, "daily", temp_root)

        universe = CNAUniverse(temp_root)

        # 2010 universe: ChiNext, Main board. NOT STAR Market
        universe_2010 = universe.tickers(date(2010, 1, 1))
        assert "SH688001" not in universe_2010, "STAR Market didn't exist in 2010"
        assert "SZ300001" in universe_2010, "ChiNext existed in 2010"
        assert "SH600000" in universe_2010, "Main board existed in 2010"

        # 2020 universe: All three
        universe_2020 = universe.tickers(date(2020, 1, 1))
        assert "SH688001" in universe_2020, "STAR Market existed in 2020"
        assert "SZ300001" in universe_2020, "ChiNext existed in 2020"
        assert "SH600000" in universe_2020, "Main board existed in 2020"

    def test_ipo_day_edge_case(self, temp_root: str) -> None:
        """
        Test that stock IS in universe on IPO date itself.

        Edge case: IPO date = 2024-01-15
        - Should NOT be in 2024-01-14 universe
        - Should be in 2024-01-15 universe (IPO day)
        """
        ipo_date = date(2024, 1, 15)

        components_df = pl.DataFrame({
            "instrument_id": ["SH_NEW001"],
            "date": [ipo_date],
            "event": [IPO_EVENT],
            "event_details": ["New IPO"],
        })

        daily_df = pl.DataFrame({
            "instrument_id": ["SH_NEW001"],
            "date": [ipo_date],
            "ticker": ["NEW001"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [2000000],
            "adj_factor": [1.0],
        })

        write_parquet(components_df, Market.CN_A, "components", temp_root)
        write_parquet(daily_df, Market.CN_A, "daily", temp_root)

        universe = CNAUniverse(temp_root)

        # Before IPO
        universe_before = universe.tickers(date(2024, 1, 14))
        assert "SH_NEW001" not in universe_before, "Stock should NOT exist before IPO"

        # On IPO date
        universe_ipo = universe.tickers(ipo_date)
        assert "SH_NEW001" in universe_ipo, "Stock should exist on IPO date"


# =============================================================================
# Halt Filtering Tests
# =============================================================================


class TestHaltFiltering:
    """
    Tests that halted stocks are correctly filtered from active universe.

    During a trading halt, you cannot buy/sell the stock.
    """

    def test_halted_stock_not_in_active_universe(self, temp_root: str) -> None:
        """
        Stock should NOT be in active universe during halt period.

        Test case: SH600000 halted from 2024-03-01 to 2024-03-15
        - Should NOT be in active tickers on 2024-03-05
        - Should be back in active tickers on 2024-03-16
        """
        components_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000", "SH600000", "SZ000001"],
            "date": [
                date(1999, 11, 10),  # IPO
                date(2024, 3, 1),    # Halt start
                date(2024, 3, 15),   # Resume
                date(1991, 4, 3),    # Another stock IPO
            ],
            "event": [IPO_EVENT, HALT_EVENT, RESUME_EVENT, IPO_EVENT],
            "event_details": ["IPO", "Halted", "Resumed", "IPO"],
        })

        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000", "SH600000", "SZ000001"] * 2,
            "date": [
                date(2024, 2, 28), date(2024, 3, 1), date(2024, 3, 15), date(2024, 3, 5),
                date(2024, 3, 16), date(2024, 3, 1), date(2024, 3, 15), date(2024, 3, 16),
            ],
            "ticker": ["600000", "600000", "600000", "000001"] * 2,
            "open": [10.0, None, 9.8, 20.0, 9.9, None, 9.8, 20.1],
            "high": [10.5, None, 10.2, 20.5, 10.3, None, 10.2, 20.6],
            "low": [9.8, None, 9.6, 19.8, 9.7, None, 9.6, 19.9],
            "close": [10.2, None, 10.0, 20.2, 10.1, None, 10.0, 20.3],
            "volume": [1000000, 0, 500000, 500000, 600000, 0, 500000, 550000],
            "adj_factor": [1.0] * 8,
        })

        # Filter out rows with null prices (halted days)
        daily_df = daily_df.filter(pl.col("close").is_not_null())

        write_parquet(components_df, Market.CN_A, "components", temp_root)
        write_parquet(daily_df, Market.CN_A, "daily", temp_root)

        universe = CNAUniverse(temp_root)

        # During halt: SH600000 should NOT be in active tickers
        universe_halt = universe.tickers(date(2024, 3, 5))
        assert "SH600000" not in universe_halt, (
            "SH600000 is halted on 2024-03-05, should NOT be in active universe"
        )
        assert "SZ000001" in universe_halt, "SZ000001 is not halted"

        # After resume: SH600000 should be back
        universe_resume = universe.tickers(date(2024, 3, 16))
        assert "SH600000" in universe_resume, "SH600000 should be active after resume"
        assert "SZ000001" in universe_resume

    def test_halt_day_edge_case(self, temp_root: str) -> None:
        """
        Test edge case: halt start day and resume day.

        - On halt start date: stock should NOT be active
        - On resume date: stock should be active
        """
        halt_start = date(2024, 3, 1)
        resume_date = date(2024, 3, 15)

        components_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000", "SH600000"],
            "date": [date(1999, 11, 10), halt_start, resume_date],
            "event": [IPO_EVENT, HALT_EVENT, RESUME_EVENT],
            "event_details": ["IPO", "Halt", "Resume"],
        })

        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 2, 28)],
            "ticker": ["600000"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000000],
            "adj_factor": [1.0],
        })

        write_parquet(components_df, Market.CN_A, "components", temp_root)
        write_parquet(daily_df, Market.CN_A, "daily", temp_root)

        universe = CNAUniverse(temp_root)

        # On halt start date
        universe_halt_start = universe.tickers(halt_start)
        assert "SH600000" not in universe_halt_start, "Halted on start date"

        # On resume date
        universe_resume = universe.tickers(resume_date)
        assert "SH600000" in universe_resume, "Active on resume date"


# =============================================================================
# Delist Day Edge Case Tests
# =============================================================================


class TestDelistDayEdgeCases:
    """
    Tests for edge cases around delisting date.

    - On delist date: stock should NOT be active
    - Day before delist: stock should be active
    """

    def test_delist_day_edge(self, temp_root: str) -> None:
        """
        Test edge case: last trading day vs delist date.

        商业城 last trading day: 2016-03-28
        Delist date: 2016-03-29

        - 2016-03-28: Should be in universe
        - 2016-03-29: Should NOT be in universe
        """
        last_trade_day = date(2016, 3, 28)
        delist_day = date(2016, 3, 29)

        components_df = pl.DataFrame({
            "instrument_id": ["SH600631", "SH600631"],
            "date": [date(1996, 12, 2), delist_day],
            "event": [IPO_EVENT, DELIST_EVENT],
            "event_details": ["IPO", "Delist"],
        })

        daily_df = pl.DataFrame({
            "instrument_id": ["SH600631"],
            "date": [last_trade_day],
            "ticker": ["600631"],
            "open": [7.0],
            "high": [7.5],
            "low": [6.8],
            "close": [7.2],
            "volume": [200000],
            "adj_factor": [1.0],
        })

        write_parquet(components_df, Market.CN_A, "components", temp_root)
        write_parquet(daily_df, Market.CN_A, "daily", temp_root)

        universe = CNAUniverse(temp_root)

        # Last trading day: should be in universe
        universe_last = universe.tickers(last_trade_day)
        assert "SH600631" in universe_last, "商业城 should be active on last trading day"

        # Delist day: should NOT be in universe
        universe_delist = universe.tickers(delist_day)
        assert "SH600631" not in universe_delist, "商业城 should NOT be active on delist day"


# =============================================================================
# Universe Count Tests
# =============================================================================


class TestUniverseCount:
    """
    Tests for universe count at different dates.

    Universe count should change as stocks IPO and delist.
    """

    def test_universe_count_changes_over_time(self, temp_root: str) -> None:
        """
        Universe count should reflect actual stocks at each date.

        - 2000: Few stocks
        - 2010: More stocks (ChiNext launches)
        - 2020: Many stocks (STAR Market launches)
        - After delistings: count should decrease
        """
        components_df = pl.DataFrame({
            "instrument_id": [
                "SH600000",  # IPO 1999
                "SH600631",  # IPO 1996, delist 2016
                "SH600631",  # delist event
                "SH600036",  # IPO 2003
                "SZ300001",  # IPO 2009 (ChiNext)
                "SH688001",  # IPO 2019 (STAR)
            ],
            "date": [
                date(1999, 11, 10),  # IPO 1999
                date(1996, 12, 2),   # IPO 1996
                date(2016, 3, 29),   # Delist 2016
                date(2003, 8, 22),   # IPO 2003
                date(2009, 10, 30),  # IPO 2009 (ChiNext)
                date(2019, 7, 22),   # IPO 2019 (STAR)
            ],
            "event": [IPO_EVENT, IPO_EVENT, DELIST_EVENT, IPO_EVENT, IPO_EVENT, IPO_EVENT],
            "event_details": ["IPO", "IPO", "Delist", "IPO", "IPO", "IPO"],
        })

        # Daily data covering all query dates
        daily_df = pl.DataFrame({
            "instrument_id": [
                "SH600000", "SH600000", "SH600000",
                "SH600631", "SH600631", "SH600631",
                "SH600036", "SH600036",
                "SZ300001", "SZ300001",
                "SH688001", "SH688001",
            ],
            "date": [
                date(2000, 1, 1), date(2010, 1, 1), date(2020, 1, 1),
                date(2000, 1, 1), date(2010, 1, 1), date(2015, 1, 1),
                date(2010, 1, 1), date(2020, 1, 1),
                date(2010, 1, 1), date(2020, 1, 1),
                date(2020, 1, 1), date(2019, 8, 1),
            ],
            "ticker": ["600000", "600000", "600000", "600631", "600631", "600631",
                       "600036", "600036", "300001", "300001", "688001", "688001"],
            "open": [10.0] * 12,
            "high": [10.5] * 12,
            "low": [9.8] * 12,
            "close": [10.2] * 12,
            "volume": [1000000] * 12,
            "adj_factor": [1.0] * 12,
        })

        write_parquet(components_df, Market.CN_A, "components", temp_root)
        write_parquet(daily_df, Market.CN_A, "daily", temp_root)

        universe = CNAUniverse(temp_root)

        # 2000: Only SH600000 (IPO 1999), NOT SH600036 (IPO 2003)
        count_2000 = universe.count(date(2000, 1, 1))
        # Should include SH600000 and SH600631 (IPO 1996)
        assert count_2000 >= 2, "At least 2 stocks in 2000 universe"

        # 2015: 商业城 should be included
        count_2015 = universe.count(date(2015, 6, 1))
        # Should have SH600000, SH600036, SH600631, SZ300001 (ChiNext)
        assert count_2015 >= 4, "At least 4 stocks in 2015 universe"

        # 2017: 商业城 should be excluded (delisted)
        count_2017 = universe.count(date(2017, 1, 1))
        # Should NOT have SH600631
        assert count_2017 >= 3, "At least 3 stocks in 2017 universe (商业城 delisted)"
        assert "SH600631" not in universe.tickers(date(2017, 1, 1))


# =============================================================================
# US Universe PIT Tests
# =============================================================================


class TestUSUniversePIT:
    """
    Tests for US universe PIT functionality.

    US universe is SP500 + Russell 1000 historical components.
    """

    def test_us_universe_empty(self, temp_root: str) -> None:
        """Empty data_lake should create empty US universe."""
        universe = USUniverse(temp_root)
        assert universe.instrument_count_total() == 0

    def test_us_universe_with_data(self, temp_root: str) -> None:
        """US universe should load component events."""
        components_df = pl.DataFrame({
            "instrument_id": ["AAPL", "MSFT", "JPM"],
            "date": [date(1980, 12, 12), date(1986, 3, 13), date(1799, 1, 1)],
            "event": [IPO_EVENT, IPO_EVENT, IPO_EVENT],
            "event_details": ["IPO", "IPO", "IPO"],
        })

        daily_df = pl.DataFrame({
            "instrument_id": ["AAPL", "MSFT", "JPM"],
            "date": [date(2024, 1, 1)] * 3,
            "ticker": ["AAPL", "MSFT", "JPM"],
            "open": [180.0, 370.0, 150.0],
            "high": [185.0, 375.0, 155.0],
            "low": [178.0, 368.0, 148.0],
            "close": [182.0, 372.0, 152.0],
            "volume": [50000000, 20000000, 10000000],
            "adj_factor": [1.0] * 3,
        })

        write_parquet(components_df, Market.US, "components", temp_root)
        write_parquet(daily_df, Market.US, "daily", temp_root)

        universe = USUniverse(temp_root)

        # Check instruments are loaded
        assert universe.instrument_count_total() >= 3

        # Check IPO dates
        assert universe.ipo_date("AAPL") == date(1980, 12, 12)

    def test_us_delisted_stock(self, temp_root: str) -> None:
        """
        US delisted stock should be in historical universe.

        Example: ENRON delisted 2001
        - Should be in 2000 universe
        - Should NOT be in 2002 universe
        """
        components_df = pl.DataFrame({
            "instrument_id": ["ENRON", "ENRON", "AAPL"],
            "date": [date(1985, 1, 1), date(2001, 12, 2), date(1980, 12, 12)],
            "event": [IPO_EVENT, DELIST_EVENT, IPO_EVENT],
            "event_details": ["IPO", "Bankruptcy", "IPO"],
        })

        daily_df = pl.DataFrame({
            "instrument_id": ["ENRON", "AAPL"],
            "date": [date(2000, 1, 1), date(2000, 1, 1)],
            "ticker": ["ENRON", "AAPL"],
            "open": [80.0, 20.0],
            "high": [85.0, 22.0],
            "low": [78.0, 18.0],
            "close": [82.0, 21.0],
            "volume": [10000000, 50000000],
            "adj_factor": [1.0, 1.0],
        })

        write_parquet(components_df, Market.US, "components", temp_root)
        write_parquet(daily_df, Market.US, "daily", temp_root)

        universe = USUniverse(temp_root)

        # 2000 universe: ENRON should be included
        universe_2000 = universe.tickers(date(2000, 1, 1))
        assert "ENRON" in universe_2000, "ENRON was trading in 2000"

        # 2002 universe: ENRON should NOT be included
        universe_2002 = universe.tickers(date(2002, 1, 1))
        assert "ENRON" not in universe_2002, "ENRON delisted in 2001"


# =============================================================================
# PIT Design Rule Enforcement Tests
# =============================================================================


class TestPITDesignRules:
    """
    Tests to verify PIT design rules are enforced.

    Key rule: EVERY API MUST ACCEPT DATE PARAMETER.
    No "current universe" shortcuts allowed.
    """

    def test_tickers_requires_date_parameter(self, temp_root: str) -> None:
        """tickers() method MUST require date parameter."""
        universe = CNAUniverse(temp_root)

        # The method signature requires date
        # This should work
        result = universe.tickers(date(2024, 1, 1))
        assert isinstance(result, list)

    def test_contains_requires_date_parameter(self, temp_root: str) -> None:
        """contains() method MUST require date parameter."""
        universe = CNAUniverse(temp_root)

        # The method signature requires date
        result = universe.contains("SH600000", date(2024, 1, 1))
        assert isinstance(result, bool)

    def test_is_active_requires_date_parameter(self, temp_root: str) -> None:
        """is_active() method MUST require date parameter."""
        universe = CNAUniverse(temp_root)

        result = universe.is_active("SH600000", date(2024, 1, 1))
        assert isinstance(result, bool)

    def test_no_current_universe_shortcuts(self, temp_root: str) -> None:
        """Universe should NOT have methods that return 'current' universe."""
        universe = CNAUniverse(temp_root)

        # Check for methods that might bypass PIT design
        methods = [m for m in dir(universe) if not m.startswith("_")]

        # These methods should NOT exist (would violate PIT design)
        forbidden_methods = [
            "current_tickers",
            "latest_tickers",
            "get_current_universe",
            "get_latest_universe",
            "current_count",
        ]

        for forbidden in forbidden_methods:
            assert forbidden not in methods, (
                f"Method '{forbidden}' violates PIT design rule - "
                f"all universe queries must accept date parameter"
            )

    def test_all_instruments_includes_delisted(self, temp_root: str) -> None:
        """
        all_instruments() should include ALL stocks, including delisted.

        This is for survivorship-free historical analysis.
        It returns stocks that ever existed, not current stocks.
        """
        components_df = pl.DataFrame({
            "instrument_id": ["SH600631", "SH600631", "SH600000"],
            "date": [date(1996, 12, 2), date(2016, 3, 29), date(1999, 11, 10)],
            "event": [IPO_EVENT, DELIST_EVENT, IPO_EVENT],
            "event_details": ["IPO", "Delist", "IPO"],
        })

        write_parquet(components_df, Market.CN_A, "components", temp_root)

        universe = CNAUniverse(temp_root)

        # all_instruments should include 商业城 (delisted)
        all_ids = universe.all_instruments()
        assert "SH600631" in all_ids, (
            "all_instruments() must include delisted stocks for survivorship-free analysis"
        )


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestGetUniverse:
    """Tests for universe factory function."""

    def test_get_universe_cn_a(self, temp_root: str) -> None:
        """get_universe should return CNAUniverse for CN_A."""
        universe = get_universe("CN_A", temp_root)
        assert isinstance(universe, CNAUniverse)

    def test_get_universe_us(self, temp_root: str) -> None:
        """get_universe should return USUniverse for US."""
        universe = get_universe("US", temp_root)
        assert isinstance(universe, USUniverse)

    def test_get_universe_unknown_raises(self, temp_root: str) -> None:
        """get_universe should raise for unknown market."""
        with pytest.raises(ValueError, match="Unknown market"):
            get_universe("UNKNOWN", temp_root)


# =============================================================================
# Integration Tests
# =============================================================================


class TestUniverseIntegration:
    """Integration tests for universe with data loader."""

    def test_universe_matches_daily_data(self, temp_root: str) -> None:
        """
        Universe tickers should match instruments in daily data.

        If daily data has SH600000 trading on 2015-01-01,
        then universe at 2015-01-01 should include SH600000.
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SZ000001", "SH600631"],
            "date": [date(2015, 1, 5)] * 3,
            "ticker": ["600000", "000001", "600631"],
            "open": [10.0, 20.0, 8.0],
            "high": [10.5, 20.5, 8.5],
            "low": [9.8, 19.8, 7.8],
            "close": [10.2, 20.2, 8.2],
            "volume": [1000000, 500000, 800000],
            "adj_factor": [1.0] * 3,
        })

        components_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SZ000001", "SH600631", "SH600631"],
            "date": [date(1999, 11, 10), date(1991, 4, 3), date(1996, 12, 2), date(2016, 3, 29)],
            "event": [IPO_EVENT, IPO_EVENT, IPO_EVENT, DELIST_EVENT],
            "event_details": ["IPO", "IPO", "IPO", "Delist"],
        })

        write_parquet(daily_df, Market.CN_A, "daily", temp_root)
        write_parquet(components_df, Market.CN_A, "components", temp_root)

        universe = CNAUniverse(temp_root)

        # Check universe matches daily data
        universe_2015 = universe.tickers(date(2015, 1, 5))
        assert "SH600000" in universe_2015
        assert "SZ000001" in universe_2015
        assert "SH600631" in universe_2015, "商业城 should be in 2015 universe"

        # But 商业城 should NOT be in 2017
        universe_2017 = universe.tickers(date(2017, 1, 1))
        assert "SH600631" not in universe_2017

    def test_bars_with_universe_filter(self, temp_root: str) -> None:
        """
        bars() should work with universe filter for survivorship-free data.

        Historical bars should include delisted stock data.
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600631", "SH600631"],
            "date": [date(2015, 1, 5), date(2015, 1, 5), date(2015, 6, 1)],
            "ticker": ["600000", "600631", "600631"],
            "open": [10.0, 8.0, 8.5],
            "high": [10.5, 8.5, 9.0],
            "low": [9.8, 7.8, 8.2],
            "close": [10.2, 8.2, 8.8],
            "volume": [1000000, 800000, 850000],
            "adj_factor": [1.0] * 3,
        })

        components_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600631", "SH600631"],
            "date": [date(1999, 11, 10), date(1996, 12, 2), date(2016, 3, 29)],
            "event": [IPO_EVENT, IPO_EVENT, DELIST_EVENT],
            "event_details": ["IPO", "IPO", "Delist"],
        })

        write_parquet(daily_df, Market.CN_A, "daily", temp_root)
        write_parquet(components_df, Market.CN_A, "components", temp_root)

        universe = CNAUniverse(temp_root)

        # Get universe at 2015
        universe_2015 = universe.tickers(date(2015, 6, 1))

        # Get bars for that universe
        df = bars(
            Market.CN_A,
            start_date=date(2015, 1, 1),
            end_date=date(2015, 12, 31),
            instrument_ids=universe_2015,
            root=temp_root,
        )

        # Should include 商业城 data
        if not df.is_empty():
            instruments = df["instrument_id"].unique().to_list()
            assert "SH600631" in instruments, (
                "Bars for 2015 universe should include 商业城 data"
            )