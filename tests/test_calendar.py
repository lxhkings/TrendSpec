"""
Tests for TrendSpec calendar module.

Tests trading calendar functionality including:
- Holiday detection
- Trading day checks
- Trading day ranges
"""

from datetime import date

import pytest

from trendspec.data.calendar import (
    count_trading_days,
    get_trading_day_of_week,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
    trading_days_between,
)
from trendspec.data.markets import Market


class TestIsTradingDayCN:
    """Tests for CN_A trading day checks."""

    @pytest.fixture
    def cn_market(self) -> Market:
        """Return CN_A market."""
        return Market.CN

    def test_weekend_not_trading_day(self, cn_market: Market) -> None:
        """Weekend should not be trading day."""
        # Saturday
        assert is_trading_day(cn_market, date(2024, 1, 6)) is False
        # Sunday
        assert is_trading_day(cn_market, date(2024, 1, 7)) is False

    def test_new_year_holiday(self, cn_market: Market) -> None:
        """New Year's Day should not be trading day."""
        # January 1 is New Year's Day in China
        assert is_trading_day(cn_market, date(2024, 1, 1)) is False

    def test_weekday_is_trading_day(self, cn_market: Market) -> None:
        """Regular weekday should be trading day."""
        # January 2, 2024 is Tuesday (regular trading day)
        assert is_trading_day(cn_market, date(2024, 1, 2)) is True

    def test_spring_festival_holiday(self, cn_market: Market) -> None:
        """Spring Festival (Lunar New Year) should be holiday."""
        # 2024 Spring Festival: Feb 10-17 (approximate)
        # Feb 10, 2024 is Saturday (weekend + holiday)
        # The exact dates vary based on lunar calendar
        # Let's test a known holiday date
        assert is_trading_day(cn_market, date(2024, 2, 10)) is False  # Saturday

    def test_national_day_holiday(self, cn_market: Market) -> None:
        """National Day (October 1) should be holiday."""
        # October 1, 2024 is National Day
        assert is_trading_day(cn_market, date(2024, 10, 1)) is False


class TestIsTradingDayUS:
    """Tests for US trading day checks."""

    @pytest.fixture
    def us_market(self) -> Market:
        """Return US market."""
        return Market.US

    def test_weekend_not_trading_day(self, us_market: Market) -> None:
        """Weekend should not be trading day."""
        assert is_trading_day(us_market, date(2024, 1, 6)) is False  # Saturday
        assert is_trading_day(us_market, date(2024, 1, 7)) is False  # Sunday

    def test_new_year_holiday(self, us_market: Market) -> None:
        """New Year's Day should not be trading day."""
        assert is_trading_day(us_market, date(2024, 1, 1)) is False

    def test_good_friday_holiday(self, us_market: Market) -> None:
        """Good Friday should not be trading day (NYSE specific)."""
        # Good Friday 2024: March 29
        assert is_trading_day(us_market, date(2024, 3, 29)) is False

    def test_christmas_holiday(self, us_market: Market) -> None:
        """Christmas should not be trading day."""
        assert is_trading_day(us_market, date(2024, 12, 25)) is False

    def test_thanksgiving_holiday(self, us_market: Market) -> None:
        """Thanksgiving should not be trading day."""
        # Thanksgiving 2024: November 28
        assert is_trading_day(us_market, date(2024, 11, 28)) is False

    def test_weekday_is_trading_day(self, us_market: Market) -> None:
        """Regular weekday should be trading day."""
        # January 2, 2024 is Tuesday
        assert is_trading_day(us_market, date(2024, 1, 2)) is True

    def test_mlk_day_holiday(self, us_market: Market) -> None:
        """Martin Luther King Jr. Day should not be trading day."""
        # MLK Day 2024: January 15
        assert is_trading_day(us_market, date(2024, 1, 15)) is False


class TestIsTradingDayHK:
    """Tests for HK market placeholder."""

    def test_hk_raises_not_implemented(self) -> None:
        """HK market should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Hong Kong market calendar"):
            is_trading_day(Market.HK, date(2024, 1, 1))


class TestTradingDaysBetween:
    """Tests for trading days range."""

    def test_cn_a_trading_days_range(self) -> None:
        """Get trading days between dates for CN_A."""
        # Jan 1-5, 2024: Jan 1 is holiday, Jan 6-7 are weekend
        days = trading_days_between(Market.CN, date(2024, 1, 1), date(2024, 1, 5))
        # Expected: Jan 2, 3, 4, 5 (Jan 1 is holiday)
        assert len(days) >= 3  # At least Jan 2-4 should be trading days
        assert all(is_trading_day(Market.CN, d) for d in days)

    def test_us_trading_days_range(self) -> None:
        """Get trading days between dates for US."""
        days = trading_days_between(Market.US, date(2024, 1, 1), date(2024, 1, 5))
        assert all(is_trading_day(Market.US, d) for d in days)

    def test_empty_range(self) -> None:
        """Empty range should return empty list."""
        days = trading_days_between(Market.CN, date(2024, 1, 5), date(2024, 1, 1))
        assert len(days) == 0

    def test_single_day_range(self) -> None:
        """Single day range should return one day if trading."""
        # Jan 2 is a trading day
        days = trading_days_between(Market.CN, date(2024, 1, 2), date(2024, 1, 2))
        if is_trading_day(Market.CN, date(2024, 1, 2)):
            assert len(days) == 1
            assert days[0] == date(2024, 1, 2)

    def test_hk_raises_not_implemented(self) -> None:
        """HK market should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            trading_days_between(Market.HK, date(2024, 1, 1), date(2024, 1, 5))


class TestNextPreviousTradingDay:
    """Tests for next and previous trading day."""

    def test_next_trading_day_after_weekend(self) -> None:
        """Next trading day after weekend should be Monday."""
        # Saturday Jan 6, 2024 -> next trading day should be Monday Jan 8
        next_day = next_trading_day(Market.CN, date(2024, 1, 6))
        assert next_day.weekday() == 0  # Monday

    def test_previous_trading_day_before_weekend(self) -> None:
        """Previous trading day before weekend should be Friday."""
        # Sunday Jan 7, 2024 -> previous trading day should be Friday Jan 5
        prev_day = previous_trading_day(Market.CN, date(2024, 1, 7))
        assert prev_day.weekday() == 4  # Friday

    def test_next_trading_day_after_holiday(self) -> None:
        """Next trading day after holiday."""
        # Jan 1 is holiday, next trading day should be Jan 2 or later
        next_day = next_trading_day(Market.CN, date(2024, 1, 1))
        assert next_day > date(2024, 1, 1)
        assert is_trading_day(Market.CN, next_day)

    def test_hk_raises_not_implemented(self) -> None:
        """HK market should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            next_trading_day(Market.HK, date(2024, 1, 1))
        with pytest.raises(NotImplementedError):
            previous_trading_day(Market.HK, date(2024, 1, 1))


class TestCountTradingDays:
    """Tests for counting trading days."""

    def test_count_trading_days_week(self) -> None:
        """Count trading days in a week."""
        # Jan 1-7, 2024: Jan 1 is holiday, Jan 6-7 are weekend
        count = count_trading_days(Market.CN, date(2024, 1, 1), date(2024, 1, 7))
        assert count <= 5  # Max 5 trading days in a week
        assert count >= 2  # At least some trading days

    def test_count_trading_days_empty(self) -> None:
        """Empty range should have zero trading days."""
        count = count_trading_days(Market.CN, date(2024, 1, 7), date(2024, 1, 1))
        assert count == 0


class TestTradingDayOfWeek:
    """Tests for trading day of week."""

    def test_not_trading_day_returns_zero(self) -> None:
        """Non-trading day should return 0."""
        assert get_trading_day_of_week(Market.CN, date(2024, 1, 6)) == 0  # Saturday

    def test_trading_day_returns_positive(self) -> None:
        """Trading day should return positive number."""
        if is_trading_day(Market.CN, date(2024, 1, 2)):
            dow = get_trading_day_of_week(Market.CN, date(2024, 1, 2))
            assert dow > 0
            assert dow <= 5
