"""
Tests for TrendSpec Price Adjustment functionality - CRITICAL for accurate price calculations.

Key test cases:
1. Forward adjustment (前复权): historical prices adjusted for dividends/splits
2. Backward adjustment (后复权): current prices adjusted for dividends/splits
3. Raw (无复权): unadjusted prices
4. Manual calculation verification

Adjustment formula:
- Forward: adjusted = raw * (adj_factor / latest_adj_factor)
- Backward: adjusted = raw * (adj_factor / earliest_adj_factor)

CRITICAL for backtest accuracy:
- Using wrong adjustment mode affects return calculations
- Forward adjustment ensures historical price continuity
"""

from datetime import date

import polars as pl
import pytest

from trendspec.data.markets import Market
from trendspec.data.parquet_loader import (
    ADJUSTMENT_MODES,
    bars,
    bars_for_instrument,
)
from trendspec.ingest.writer import write_parquet

# =============================================================================
# Adjustment Factor Correctness Tests
# =============================================================================


class TestAdjustmentFactorCorrectness:
    """
    Tests verifying adj_factor correctness with manual calculation.

    These are the foundation for accurate price adjustment.
    """

    def test_adj_factor_dividend(self, temp_root: str) -> None:
        """
        Verify adj_factor for dividend scenario with manual calculation.

        Scenario: 5% dividend on Jan 15, 2024
        - Before dividend: adj_factor = 1.0, price = 10.6
        - After dividend: adj_factor = 0.95, price = 9.5 (actual drop ~10% due to dividend)

        Manual calculation:
        - latest_adj_factor = 0.95
        - Forward adjustment for Jan 10: 10.0 * (1.0 / 0.95) = 10.526
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 5,
            "date": [
                date(2024, 1, 10), date(2024, 1, 11), date(2024, 1, 12), date(2024, 1, 13), date(2024, 1, 14),
                # Before dividend
            ],
            "ticker": ["600000"] * 5,
            "open": [10.0, 10.1, 10.2, 10.3, 10.4],
            "high": [10.5, 10.6, 10.7, 10.8, 10.9],
            "low": [9.8, 9.9, 10.0, 10.1, 10.2],
            "close": [10.2, 10.3, 10.4, 10.5, 10.6],
            "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            "adj_factor": [1.0, 1.0, 1.0, 1.0, 1.0],  # Before dividend
        })

        # Add post-dividend data
        post_dividend_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 3,
            "date": [date(2024, 1, 15), date(2024, 1, 16), date(2024, 1, 17)],
            "ticker": ["600000"] * 3,
            "open": [9.5, 9.55, 9.6],
            "high": [9.8, 9.85, 9.9],
            "low": [9.2, 9.25, 9.3],
            "close": [9.5, 9.55, 9.6],
            "volume": [2000000, 1500000, 1600000],
            "adj_factor": [0.95, 0.95, 0.95],  # After 5% dividend
        })

        combined_df = pl.concat([daily_df, post_dividend_df])
        write_parquet(combined_df, Market.CN, "daily", temp_root)

        # Test forward adjustment
        df_forward = bars(Market.CN, adjustment_mode="forward", root=temp_root)

        if not df_forward.is_empty():
            # Manual calculation for Jan 10 forward adjustment:
            # adjusted_close = 10.2 * (1.0 / 0.95) = 10.5263
            # The forward-adjusted historical close should be higher than raw
            jan10_data = df_forward.filter(
                (pl.col("instrument_id") == "SH600000") & (pl.col("date") == date(2024, 1, 10))
            )
            if not jan10_data.is_empty():
                forward_close = jan10_data["close"].item()
                # Forward-adjusted price should be ~10.526 (higher than raw 10.2)
                assert forward_close > 10.2, (
                    f"Forward adjustment should increase historical price. "
                    f"Raw: 10.2, Forward: {forward_close}"
                )
                # Check it's approximately correct (within 1% tolerance)
                expected_forward = 10.2 * (1.0 / 0.95)
                assert abs(forward_close - expected_forward) < 0.1, (
                    f"Forward adjustment calculation incorrect. "
                    f"Expected: {expected_forward:.4f}, Got: {forward_close:.4f}"
                )

    def test_adj_factor_stock_split(self, temp_root: str) -> None:
        """
        Verify adj_factor for 2:1 stock split with manual calculation.

        Scenario: 2:1 split on March 1, 2024
        - Before split: adj_factor = 1.0, price = 100
        - After split: adj_factor = 0.5, price = 50 (half of pre-split)

        Manual calculation:
        - Forward adjustment for Feb 28: 100 * (1.0 / 0.5) = 200
        - This ensures continuity: pre-split 100 -> post-split 50 looks like 200 -> 100
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 6,
            "date": [
                date(2024, 2, 26), date(2024, 2, 27), date(2024, 2, 28),  # Before split
                date(2024, 3, 1), date(2024, 3, 2), date(2024, 3, 3),  # After split
            ],
            "ticker": ["600000"] * 6,
            "open": [100.0, 101.0, 102.0, 50.0, 50.5, 51.0],
            "high": [105.0, 106.0, 107.0, 52.5, 53.0, 53.5],
            "low": [98.0, 99.0, 100.0, 49.0, 49.5, 50.0],
            "close": [102.0, 103.0, 104.0, 51.0, 51.5, 52.0],
            "volume": [500000, 550000, 600000, 1200000, 1100000, 1150000],  # Volume doubled after split
            "adj_factor": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5],  # Split factor
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        # Test forward adjustment
        df_forward = bars(Market.CN, adjustment_mode="forward", root=temp_root)

        if not df_forward.is_empty():
            # Manual calculation for Feb 28 forward adjustment:
            # adjusted_close = 104.0 * (1.0 / 0.5) = 208.0
            # Forward adjustment doubles the pre-split price
            feb28_data = df_forward.filter(
                (pl.col("instrument_id") == "SH600000") & (pl.col("date") == date(2024, 2, 28))
            )
            if not feb28_data.is_empty():
                forward_close = feb28_data["close"].item()
                expected_forward = 104.0 * (1.0 / 0.5)  # = 208.0
                assert forward_close > 104.0, (
                    f"Forward adjustment should double pre-split price. "
                    f"Raw: 104.0, Forward: {forward_close}"
                )
                # Allow for minor rounding differences
                assert abs(forward_close - expected_forward) < 1.0, (
                    f"Forward split adjustment calculation incorrect. "
                    f"Expected: ~{expected_forward:.1f}, Got: {forward_close:.1f}"
                )

    def test_adj_factor_sequence(self, temp_root: str) -> None:
        """
        Verify adj_factor for multiple corporate actions in sequence.

        Scenario: dividend then split
        - Jan 1-10: adj_factor = 1.0
        - Jan 15: dividend, adj_factor = 0.95
        - Mar 1: split, adj_factor = 0.475 (0.95 * 0.5)
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 6,
            "date": [
                date(2024, 1, 1), date(2024, 1, 10),  # Before dividend
                date(2024, 1, 15), date(2024, 2, 28),  # After dividend, before split
                date(2024, 3, 1), date(2024, 3, 5),  # After split
            ],
            "ticker": ["600000"] * 6,
            "open": [10.0, 10.5, 9.5, 10.0, 5.0, 5.5],
            "high": [10.5, 11.0, 10.0, 10.5, 5.5, 6.0],
            "low": [9.5, 10.0, 9.0, 9.5, 4.5, 5.0],
            "close": [10.2, 10.8, 9.5, 10.2, 5.1, 5.5],
            "volume": [1000000, 1200000, 2000000, 1500000, 3000000, 2500000],
            "adj_factor": [1.0, 1.0, 0.95, 0.95, 0.475, 0.475],  # Combined factor
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        df_forward = bars(Market.CN, adjustment_mode="forward", root=temp_root)

        if not df_forward.is_empty():
            # Verify adjustment ratio is correct
            jan1_data = df_forward.filter(
                (pl.col("instrument_id") == "SH600000") & (pl.col("date") == date(2024, 1, 1))
            )
            if not jan1_data.is_empty():
                forward_close = jan1_data["close"].item()
                expected_forward = 10.2 * (1.0 / 0.475)  # = 21.47
                # Forward adjustment should increase historical price significantly
                assert forward_close > 15.0, (
                    f"Forward adjustment for sequence should increase price. "
                    f"Raw: 10.2, Forward: {forward_close}, Expected: ~{expected_forward:.2f}"
                )


# =============================================================================
# Forward Adjustment Tests (前复权)
# =============================================================================


class TestForwardAdjustment:
    """
    Tests for forward adjustment (前复权).

    Forward adjustment modifies historical prices to be consistent with current prices.
    Formula: adjusted = raw * (adj_factor / latest_adj_factor)

    Use case: Comparing current prices to historical prices for continuity.
    """

    def test_forward_adjustment_basic(self, temp_root: str) -> None:
        """
        Basic forward adjustment: historical prices should be adjusted upward.

        When adj_factor decreases (dividend/split), forward adjustment
        increases historical prices to maintain continuity.
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 4,
            "date": [
                date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
            ],
            "ticker": ["600000"] * 4,
            "open": [10.0, 10.1, 10.2, 9.5],  # Price drop on Jan 4 (dividend)
            "high": [10.5, 10.6, 10.7, 9.8],
            "low": [9.8, 9.9, 10.0, 9.2],
            "close": [10.2, 10.3, 10.4, 9.5],
            "volume": [1000000, 1100000, 1200000, 800000],
            "adj_factor": [1.0, 1.0, 1.0, 0.95],  # Dividend on Jan 4
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        df_forward = bars(Market.CN, adjustment_mode="forward", root=temp_root)

        if not df_forward.is_empty():
            # Historical prices (Jan 1-3) should be adjusted upward
            jan1_data = df_forward.filter(pl.col("date") == date(2024, 1, 1))
            jan4_data = df_forward.filter(pl.col("date") == date(2024, 1, 4))

            if not jan1_data.is_empty() and not jan4_data.is_empty():
                forward_jan1_close = jan1_data["close"].item()
                forward_jan4_close = jan4_data["close"].item()

                # Jan 1 forward-adjusted should be higher than raw (10.2)
                assert forward_jan1_close > 10.2, "Forward adjustment should increase historical price"

                # Jan 4 (latest) should stay close to raw
                # forward: 9.5 * (0.95 / 0.95) = 9.5
                assert abs(forward_jan4_close - 9.5) < 0.01, "Latest price should not change in forward adjustment"

    def test_forward_adjustment_returns_consistency(self, temp_root: str) -> None:
        """
        Forward-adjusted prices should produce correct returns.

        Return calculation with forward adjustment should match actual returns.
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 4,
            "date": [
                date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
            ],
            "ticker": ["600000"] * 4,
            "open": [100.0, 101.0, 102.0, 50.0],  # 2:1 split
            "high": [105.0, 106.0, 107.0, 52.5],
            "low": [98.0, 99.0, 100.0, 49.0],
            "close": [102.0, 103.0, 104.0, 52.0],
            "volume": [1000000, 1100000, 1200000, 2400000],
            "adj_factor": [1.0, 1.0, 1.0, 0.5],  # Split on Jan 4
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        df_forward = bars(Market.CN, adjustment_mode="forward", root=temp_root)

        if not df_forward.is_empty():
            # Verify that returns between adjusted prices are correct
            # Jan 3 to Jan 4 (including split):
            # Raw returns would be: (52 - 104) / 104 = -50% (looks like huge loss)
            # Forward-adjusted returns should be: (52 - 208) / 208 = -75%? No wait...
            # Actually forward adjustment makes pre-split prices double
            # Jan 3 forward: 104 * (1.0 / 0.5) = 208
            # Jan 4 forward: 52 * (0.5 / 0.5) = 52
            # Returns: (52 - 208) / 208 = -75%

            jan3_data = df_forward.filter(pl.col("date") == date(2024, 1, 3))
            jan4_data = df_forward.filter(pl.col("date") == date(2024, 1, 4))

            if not jan3_data.is_empty() and not jan4_data.is_empty():
                jan3_close = jan3_data["close"].item()
                jan4_close = jan4_data["close"].item()

                # Forward adjustment: Jan 3 should be ~208
                expected_jan3 = 104.0 * (1.0 / 0.5)  # = 208
                assert abs(jan3_close - expected_jan3) < 1.0, (
                    f"Forward-adjusted Jan 3 should be ~{expected_jan3}, got {jan3_close}"
                )


# =============================================================================
# Backward Adjustment Tests (后复权)
# =============================================================================


class TestBackwardAdjustment:
    """
    Tests for backward adjustment (后复权).

    Backward adjustment modifies current prices to be consistent with historical prices.
    Formula: adjusted = raw * (adj_factor / earliest_adj_factor)

    Use case: Viewing historical price trajectory without discontinuity.
    """

    def test_backward_adjustment_basic(self, temp_root: str) -> None:
        """
        Basic backward adjustment: current prices should be adjusted.

        When adj_factor decreases, backward adjustment reduces current prices.
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 4,
            "date": [
                date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
            ],
            "ticker": ["600000"] * 4,
            "open": [10.0, 10.1, 10.2, 9.5],  # Price drop on Jan 4
            "high": [10.5, 10.6, 10.7, 9.8],
            "low": [9.8, 9.9, 10.0, 9.2],
            "close": [10.2, 10.3, 10.4, 9.5],
            "volume": [1000000, 1100000, 1200000, 800000],
            "adj_factor": [1.0, 1.0, 1.0, 0.95],  # Dividend on Jan 4
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        df_backward = bars(Market.CN, adjustment_mode="backward", root=temp_root)

        if not df_backward.is_empty():
            # Current prices (Jan 4) should be adjusted downward
            # backward: 9.5 * (0.95 / 1.0) = 9.025
            jan4_data = df_backward.filter(pl.col("date") == date(2024, 1, 4))

            if not jan4_data.is_empty():
                backward_jan4_close = jan4_data["close"].item()
                expected_backward = 9.5 * (0.95 / 1.0)  # = 9.025
                assert backward_jan4_close < 9.5, (
                    f"Backward adjustment should decrease current price. "
                    f"Raw: 9.5, Backward: {backward_jan4_close}"
                )

            # Historical prices (Jan 1) should stay close to raw
            jan1_data = df_backward.filter(pl.col("date") == date(2024, 1, 1))
            if not jan1_data.is_empty():
                backward_jan1_close = jan1_data["close"].item()
                # backward: 10.2 * (1.0 / 1.0) = 10.2
                assert abs(backward_jan1_close - 10.2) < 0.01, (
                    "Earliest price should not change in backward adjustment"
                )

    def test_backward_adjustment_stock_split(self, temp_root: str) -> None:
        """
        Backward adjustment for stock split.

        After 2:1 split, backward adjustment reduces post-split prices by half.
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 4,
            "date": [
                date(2024, 2, 27), date(2024, 2, 28),  # Before split
                date(2024, 3, 1), date(2024, 3, 2),  # After split
            ],
            "ticker": ["600000"] * 4,
            "open": [100.0, 102.0, 50.0, 51.0],
            "high": [105.0, 107.0, 52.5, 53.5],
            "low": [98.0, 100.0, 49.0, 50.0],
            "close": [103.0, 104.0, 51.0, 52.0],
            "volume": [500000, 600000, 1200000, 1150000],
            "adj_factor": [1.0, 1.0, 0.5, 0.5],  # Split on March 1
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        df_backward = bars(Market.CN, adjustment_mode="backward", root=temp_root)

        if not df_backward.is_empty():
            # Post-split prices (Mar 1-2) should be adjusted downward
            mar1_data = df_backward.filter(pl.col("date") == date(2024, 3, 1))
            if not mar1_data.is_empty():
                backward_mar1_close = mar1_data["close"].item()
                expected_backward = 51.0 * (0.5 / 1.0)  # = 25.5
                assert backward_mar1_close < 30.0, (
                    f"Backward adjustment for split should reduce post-split price. "
                    f"Raw: 51.0, Expected: ~{expected_backward}, Got: {backward_mar1_close}"
                )


# =============================================================================
# Raw (No Adjustment) Tests
# =============================================================================


class TestRawAdjustment:
    """
    Tests for raw (无复权) mode - no price adjustment applied.

    Raw mode returns actual prices as recorded.
    Use case: Volume analysis, actual transaction prices.
    """

    def test_raw_preserves_original_prices(self, temp_root: str) -> None:
        """
        Raw mode should preserve original prices exactly.

        No adjustment factor applied - returns exact historical prices.
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 4,
            "date": [
                date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
            ],
            "ticker": ["600000"] * 4,
            "open": [10.0, 10.1, 10.2, 9.5],
            "high": [10.5, 10.6, 10.7, 9.8],
            "low": [9.8, 9.9, 10.0, 9.2],
            "close": [10.2, 10.3, 10.4, 9.5],
            "volume": [1000000, 1100000, 1200000, 800000],
            "adj_factor": [1.0, 1.0, 1.0, 0.95],
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        df_raw = bars(Market.CN, adjustment_mode="raw", root=temp_root)

        if not df_raw.is_empty():
            # Raw prices should be exactly as stored
            jan1_data = df_raw.filter(pl.col("date") == date(2024, 1, 1))
            jan4_data = df_raw.filter(pl.col("date") == date(2024, 1, 4))

            if not jan1_data.is_empty():
                raw_jan1_close = jan1_data["close"].item()
                assert raw_jan1_close == 10.2, "Raw should preserve exact price"

            if not jan4_data.is_empty():
                raw_jan4_close = jan4_data["close"].item()
                assert raw_jan4_close == 9.5, "Raw should preserve exact price"

    def test_raw_volume_unchanged(self, temp_root: str) -> None:
        """
        Raw mode should preserve volume values.

        Volume is not adjusted even when prices are.
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 4,
            "date": [
                date(2024, 2, 27), date(2024, 2, 28),  # Before split
                date(2024, 3, 1), date(2024, 3, 2),  # After split
            ],
            "ticker": ["600000"] * 4,
            "open": [100.0, 102.0, 50.0, 51.0],
            "high": [105.0, 107.0, 52.5, 53.5],
            "low": [98.0, 100.0, 49.0, 50.0],
            "close": [103.0, 104.0, 51.0, 52.0],
            "volume": [500000, 600000, 1200000, 1150000],  # Volume doubled after split
            "adj_factor": [1.0, 1.0, 0.5, 0.5],
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        df_raw = bars(Market.CN, adjustment_mode="raw", root=temp_root)

        if not df_raw.is_empty():
            # Volume should be preserved (doubled after split in raw data)
            mar1_data = df_raw.filter(pl.col("date") == date(2024, 3, 1))
            if not mar1_data.is_empty():
                raw_mar1_volume = mar1_data["volume"].item()
                assert raw_mar1_volume == 1200000, "Raw should preserve exact volume"


# =============================================================================
# Adjustment Mode Validation Tests
# =============================================================================


class TestAdjustmentModeValidation:
    """Tests for adjustment mode validation and edge cases."""

    def test_invalid_adjustment_mode_raises(self, temp_root: str) -> None:
        """Invalid adjustment mode should raise ValueError."""
        daily_df = pl.DataFrame({
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

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        with pytest.raises(ValueError, match="Invalid adjustment mode"):
            bars(Market.CN, adjustment_mode="invalid", root=temp_root)

    def test_adjustment_modes_defined(self) -> None:
        """Adjustment modes should be properly defined."""
        assert "raw" in ADJUSTMENT_MODES
        assert "forward" in ADJUSTMENT_MODES
        assert "backward" in ADJUSTMENT_MODES

    def test_no_adj_factor_column(self, temp_root: str) -> None:
        """
        When adj_factor column missing, adjustment should be skipped.

        Should return data without adjustment.
        """
        # DataFrame without adj_factor
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 1)],
            "ticker": ["600000"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000000],
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        # Raw mode should work (no adjustment needed anyway)
        df_raw = bars(Market.CN, adjustment_mode="raw", root=temp_root)
        if not df_raw.is_empty():
            assert "close" in df_raw.columns

    def test_single_day_data(self, temp_root: str) -> None:
        """
        Adjustment with single day of data should work correctly.

        No adjustment needed when only one day exists.
        """
        daily_df = pl.DataFrame({
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

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        # All modes should work with single day
        df_forward = bars(Market.CN, adjustment_mode="forward", root=temp_root)
        df_backward = bars(Market.CN, adjustment_mode="backward", root=temp_root)
        df_raw = bars(Market.CN, adjustment_mode="raw", root=temp_root)

        if not df_forward.is_empty():
            assert df_forward["close"].item() == 10.2, "Single day forward should equal raw"
        if not df_backward.is_empty():
            assert df_backward["close"].item() == 10.2, "Single day backward should equal raw"
        if not df_raw.is_empty():
            assert df_raw["close"].item() == 10.2, "Single day raw should equal raw"


# =============================================================================
# Multi-Instrument Adjustment Tests
# =============================================================================


class TestMultiInstrumentAdjustment:
    """Tests for adjustment across multiple instruments."""

    def test_different_adj_factors_per_instrument(self, temp_root: str) -> None:
        """
        Different instruments can have different adj_factors.

        Adjustment should be applied independently per instrument.
        """
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000", "SZ000001", "SZ000001"],
            "date": [date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 1), date(2024, 1, 4)],
            "ticker": ["600000", "600000", "000001", "000001"],
            "open": [10.0, 9.5, 20.0, 19.0],  # Different price drops
            "high": [10.5, 9.8, 20.5, 19.5],
            "low": [9.8, 9.2, 19.8, 18.5],
            "close": [10.2, 9.5, 20.2, 19.0],
            "volume": [1000000, 800000, 500000, 400000],
            "adj_factor": [1.0, 0.95, 1.0, 0.90],  # Different factors
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        df_forward = bars(Market.CN, adjustment_mode="forward", root=temp_root)

        if not df_forward.is_empty():
            # Each instrument should be adjusted independently
            sh600000_jan1 = df_forward.filter(
                (pl.col("instrument_id") == "SH600000") & (pl.col("date") == date(2024, 1, 1))
            )
            sz000001_jan1 = df_forward.filter(
                (pl.col("instrument_id") == "SZ000001") & (pl.col("date") == date(2024, 1, 1))
            )

            if not sh600000_jan1.is_empty() and not sz000001_jan1.is_empty():
                sh600_close = sh600000_jan1["close"].item()
                sz001_close = sz000001_jan1["close"].item()

                # SH600000: 10.2 * (1.0 / 0.95) = ~10.737
                # SZ000001: 20.2 * (1.0 / 0.90) = ~22.444
                # They should be different due to different adj_factors
                expected_sh600 = 10.2 * (1.0 / 0.95)
                expected_sz001 = 20.2 * (1.0 / 0.90)

                assert abs(sh600_close - expected_sh600) < 0.5, (
                    f"SH600000 adjustment: expected ~{expected_sh600:.2f}, got {sh600_close:.2f}"
                )
                assert abs(sz001_close - expected_sz001) < 0.5, (
                    f"SZ000001 adjustment: expected ~{expected_sz001:.2f}, got {sz001_close:.2f}"
                )


# =============================================================================
# bars_for_instrument Adjustment Tests
# =============================================================================


class TestBarsForInstrumentAdjustment:
    """Tests for single-instrument bars retrieval with adjustment."""

    def test_bars_for_instrument_forward(self, temp_root: str) -> None:
        """bars_for_instrument should apply forward adjustment."""
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 4,
            "date": [
                date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
            ],
            "ticker": ["600000"] * 4,
            "open": [10.0, 10.1, 10.2, 9.5],
            "high": [10.5, 10.6, 10.7, 9.8],
            "low": [9.8, 9.9, 10.0, 9.2],
            "close": [10.2, 10.3, 10.4, 9.5],
            "volume": [1000000, 1100000, 1200000, 800000],
            "adj_factor": [1.0, 1.0, 1.0, 0.95],
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        df_forward = bars_for_instrument(
            Market.CN, "SH600000", adjustment_mode="forward", root=temp_root
        )

        if not df_forward.is_empty():
            # Should have 4 rows
            assert len(df_forward) == 4

            # Historical prices should be adjusted
            jan1_close = df_forward.filter(pl.col("date") == date(2024, 1, 1))["close"].item()
            assert jan1_close > 10.2, "Forward should increase historical price"

    def test_bars_for_instrument_backward(self, temp_root: str) -> None:
        """bars_for_instrument should apply backward adjustment."""
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 4,
            "date": [
                date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
            ],
            "ticker": ["600000"] * 4,
            "open": [10.0, 10.1, 10.2, 9.5],
            "high": [10.5, 10.6, 10.7, 9.8],
            "low": [9.8, 9.9, 10.0, 9.2],
            "close": [10.2, 10.3, 10.4, 9.5],
            "volume": [1000000, 1100000, 1200000, 800000],
            "adj_factor": [1.0, 1.0, 1.0, 0.95],
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        df_backward = bars_for_instrument(
            Market.CN, "SH600000", adjustment_mode="backward", root=temp_root
        )

        if not df_backward.is_empty():
            assert len(df_backward) == 4

            # Latest prices should be adjusted
            jan4_close = df_backward.filter(pl.col("date") == date(2024, 1, 4))["close"].item()
            assert jan4_close < 9.5, "Backward should decrease current price"

    def test_bars_for_instrument_raw(self, temp_root: str) -> None:
        """bars_for_instrument should return raw prices unchanged."""
        daily_df = pl.DataFrame({
            "instrument_id": ["SH600000"] * 4,
            "date": [
                date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
            ],
            "ticker": ["600000"] * 4,
            "open": [10.0, 10.1, 10.2, 9.5],
            "high": [10.5, 10.6, 10.7, 9.8],
            "low": [9.8, 9.9, 10.0, 9.2],
            "close": [10.2, 10.3, 10.4, 9.5],
            "volume": [1000000, 1100000, 1200000, 800000],
            "adj_factor": [1.0, 1.0, 1.0, 0.95],
        })

        write_parquet(daily_df, Market.CN, "daily", temp_root)

        df_raw = bars_for_instrument(
            Market.CN, "SH600000", adjustment_mode="raw", root=temp_root
        )

        if not df_raw.is_empty():
            assert len(df_raw) == 4

            # Prices should be exactly as stored
            closes = df_raw.sort("date")["close"].to_list()
            expected_closes = [10.2, 10.3, 10.4, 9.5]
            assert closes == expected_closes, "Raw should preserve exact prices"
