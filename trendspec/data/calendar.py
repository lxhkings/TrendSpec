"""
Trading calendar for TrendSpec.

Provides trading calendar functionality with exchange-specific holidays.
Key design: date parameter is required for all APIs (PIT design rule).

Supported markets:
- CN_A: SSE/SZSE (Shanghai/Shenzhen) - Chinese holidays
- US: NYSE/NASDAQ - US holidays
- HK: Placeholder (not implemented)
"""

from datetime import date, timedelta
from functools import lru_cache
from typing import Final

import holidays as holidays_lib

from trendspec.data.markets import Market

# =============================================================================
# Exchange Holiday Configuration
# =============================================================================

# SSE/SZSE trading calendar years (Chinese stock exchange)
# Note: Chinese holidays have specific rules:
# - Spring Festival (Lunar New Year): 7-day break
# - National Day: 7-day break
# - Other holidays: 1-3 days
CN_A_HOLIDAY_YEARS: Final[range] = range(1990, 2030)  # SSE founded 1990

# NYSE/NASDAQ trading calendar years
US_HOLIDAY_YEARS: Final[range] = range(1900, 2030)


def _get_cn_a_holidays() -> holidays_lib.HolidayBase:
    """
    Get China A-share exchange holidays.

    SSE and SZSE share the same holiday calendar.
    Uses the 'holidays' library with CN (China) country code.

    Returns:
        Holiday calendar for China
    """
    # China holidays from the holidays library
    # This includes: New Year, Spring Festival, Qingming, Labor Day,
    # Dragon Boat, Mid-Autumn, National Day
    return holidays_lib.CN(years=CN_A_HOLIDAY_YEARS)


def _get_us_holidays() -> holidays_lib.HolidayBase:
    """
    Get US exchange holidays (NYSE/NASDAQ).

    NYSE holidays are:
    - New Year's Day
    - Martin Luther King Jr. Day
    - Presidents' Day
    - Good Friday (NYSE specific, not federal)
    - Memorial Day
    - Independence Day
    - Labor Day
    - Thanksgiving
    - Christmas

    Returns:
        Holiday calendar for US exchanges
    """
    # NYSE/NASDAQ uses US Federal holidays plus Good Friday
    us_holidays = holidays_lib.US(years=US_HOLIDAY_YEARS)

    # Add Good Friday (NYSE-specific, not in federal holidays)
    # Good Friday is 2 days before Easter Sunday
    # We need to add this manually for each year
    for year in US_HOLIDAY_YEARS:
        # Calculate Easter Sunday using Anonymous Gregorian algorithm
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1

        easter_sunday = date(year, month, day)
        good_friday = easter_sunday - timedelta(days=2)
        us_holidays[good_friday] = "Good Friday"

    return us_holidays


# =============================================================================
# Cached Holiday Calendars
# =============================================================================

# Pre-built holiday calendars for O(1) lookup
_CN_A_HOLIDAYS: holidays_lib.HolidayBase = _get_cn_a_holidays()
_US_HOLIDAYS: holidays_lib.HolidayBase = _get_us_holidays()


# =============================================================================
# Trading Calendar API
# =============================================================================


def is_trading_day(market: Market, as_of_date: date) -> bool:
    """
    Check if a date is a trading day for a specific market.

    PIT design: as_of_date parameter is required.

    Args:
        market: Market enum (CN_A, US, HK)
        as_of_date: Date to check

    Returns:
        True if trading day, False if holiday/weekend

    Raises:
        NotImplementedError: For HK market (not implemented)

    Example:
        >>> is_trading_day(Market.CN_A, date(2024, 1, 1))
        False  # New Year's Day
        >>> is_trading_day(Market.US, date(2024, 1, 1))
        False  # New Year's Day
        >>> is_trading_day(Market.CN_A, date(2024, 1, 2))
        True   # Regular trading day
    """
    if market == Market.HK:
        raise NotImplementedError(
            "Hong Kong market calendar not yet implemented. "
            "HK market support planned for future release."
        )

    # Check weekend first (Saturday=5, Sunday=6)
    weekday = as_of_date.weekday()
    if weekday >= 5:  # Saturday or Sunday
        return False

    # Check holidays based on market
    if market == Market.CN_A:
        return as_of_date not in _CN_A_HOLIDAYS
    elif market == Market.US:
        return as_of_date not in _US_HOLIDAYS

    return True  # Default to True for unknown markets


def trading_days_between(
    market: Market,
    start_date: date,
    end_date: date,
) -> list[date]:
    """
    Get list of trading days between start and end dates.

    PIT design: start_date and end_date parameters are required.

    Args:
        market: Market enum (CN_A, US, HK)
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        List of trading days in the range

    Raises:
        NotImplementedError: For HK market (not implemented)

    Example:
        >>> trading_days_between(Market.CN_A, date(2024, 1, 1), date(2024, 1, 5))
        [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
        # Jan 1 is holiday (New Year), Jan 6-7 are weekend
    """
    if market == Market.HK:
        raise NotImplementedError(
            "Hong Kong market calendar not yet implemented. "
            "HK market support planned for future release."
        )

    trading_days: list[date] = []
    current_date = start_date

    while current_date <= end_date:
        if is_trading_day(market, current_date):
            trading_days.append(current_date)
        current_date += timedelta(days=1)

    return trading_days


def next_trading_day(market: Market, as_of_date: date) -> date:
    """
    Get the next trading day after a given date.

    PIT design: as_of_date parameter is required.

    Args:
        market: Market enum (CN_A, US, HK)
        as_of_date: Starting date

    Returns:
        Next trading day

    Raises:
        NotImplementedError: For HK market (not implemented)
    """
    if market == Market.HK:
        raise NotImplementedError(
            "Hong Kong market calendar not yet implemented."
        )

    next_date = as_of_date + timedelta(days=1)

    # Search forward for trading day
    # Limit search to 10 days to avoid infinite loop
    for _ in range(10):
        if is_trading_day(market, next_date):
            return next_date
        next_date += timedelta(days=1)

    # If no trading day found within 10 days, raise error
    raise ValueError(
        f"No trading day found within 10 days after {as_of_date} for {market}"
    )


def previous_trading_day(market: Market, as_of_date: date) -> date:
    """
    Get the previous trading day before a given date.

    PIT design: as_of_date parameter is required.

    Args:
        market: Market enum (CN_A, US, HK)
        as_of_date: Starting date

    Returns:
        Previous trading day

    Raises:
        NotImplementedError: For HK market (not implemented)
    """
    if market == Market.HK:
        raise NotImplementedError(
            "Hong Kong market calendar not yet implemented."
        )

    prev_date = as_of_date - timedelta(days=1)

    # Search backward for trading day
    for _ in range(10):
        if is_trading_day(market, prev_date):
            return prev_date
        prev_date -= timedelta(days=1)

    raise ValueError(
        f"No trading day found within 10 days before {as_of_date} for {market}"
    )


def count_trading_days(
    market: Market,
    start_date: date,
    end_date: date,
) -> int:
    """
    Count number of trading days between start and end dates.

    PIT design: start_date and end_date parameters are required.

    Args:
        market: Market enum (CN_A, US, HK)
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        Number of trading days

    Raises:
        NotImplementedError: For HK market (not implemented)
    """
    return len(trading_days_between(market, start_date, end_date))


@lru_cache(maxsize=128)
def get_trading_day_of_week(market: Market, as_of_date: date) -> int:
    """
    Get the trading day sequence number within a week.

    Useful for weekly rebalancing strategies.

    Args:
        market: Market enum (CN_A, US, HK)
        as_of_date: Date to check

    Returns:
        Trading day number (1-5) or 0 if not a trading day
    """
    if not is_trading_day(market, as_of_date):
        return 0

    # Count trading days from the start of the week (Monday)
    week_start = as_of_date - timedelta(days=as_of_date.weekday())
    trading_days = trading_days_between(market, week_start, as_of_date)
    return len(trading_days)