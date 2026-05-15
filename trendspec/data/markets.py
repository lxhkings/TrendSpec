"""
Market abstraction layer for TrendSpec.

Defines market-specific configuration including trading calendars,
price precision, sector classifications, and trading rules.

Supported markets:
- CN_A: China A-shares (Shanghai/Shenzhen)
- US: US stocks (NYSE/NASDAQ)
- HK: Hong Kong stocks (placeholder for future)
"""

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Literal


@dataclass(frozen=True, slots=True)
class TradingHours:
    """Trading hours for a market session."""

    pre_market_open: str | None  # HH:MM format, None if no pre-market
    market_open: str  # HH:MM format
    market_close: str  # HH:MM format
    after_market_close: str | None  # HH:MM format, None if no after-hours
    timezone: str  # IANA timezone identifier


@dataclass(frozen=True, slots=True)
class PriceLimitRule:
    """Price limit rules for a market."""

    has_limit: bool
    regular_limit_pct: float | None = None  # e.g., 0.10 for +/-10%
    special_limit_pct: float | None = None  # e.g., 0.20 for STAR/ChiNext
    circuit_breaker_pct: float | None = None  # e.g., 0.07 for US circuit breaker
    description: str = ""


@dataclass(frozen=True, slots=True)
class CommissionRule:
    """Commission and fee structure for a market."""

    commission_rate: float  # Base commission rate
    commission_min: float  # Minimum commission per trade
    stamp_duty: float  # Stamp duty (0 if none)
    stamp_duty_side: Literal["buy", "sell", "both", "none"]  # Which side has stamp duty
    transfer_fee: float  # Transfer fee rate (0 if none)
    description: str = ""


class Market(StrEnum):
    """
    Supported markets with their configuration metadata.

    Values are market codes used throughout the system.
    """

    CN = "CN"  # China A-shares
    US = "US"  # US stocks (NYSE/NASDAQ)
    HK = "HK"  # Hong Kong stocks (placeholder)

    @property
    def path(self) -> str:
        """Data lake subdirectory for this market."""
        return _MARKET_METADATA[self].path

    @property
    def price_precision(self) -> int:
        """Decimal places for price display."""
        return _MARKET_METADATA[self].price_precision

    @property
    def trading_calendar(self) -> str:
        """Exchange calendar identifier."""
        return _MARKET_METADATA[self].trading_calendar

    @property
    def sector_classification(self) -> str:
        """Sector taxonomy name."""
        return _MARKET_METADATA[self].sector_classification

    @property
    def sector_count(self) -> int:
        """Number of sectors in classification."""
        return _MARKET_METADATA[self].sector_count

    @property
    def currency(self) -> str:
        """Trading currency code (ISO 4217)."""
        return _MARKET_METADATA[self].currency

    @property
    def price_limit_rules(self) -> PriceLimitRule:
        """Daily price movement limits."""
        return _MARKET_METADATA[self].price_limit_rules

    @property
    def commission_rules(self) -> CommissionRule:
        """Commission and stamp duty structure."""
        return _MARKET_METADATA[self].commission_rules

    @property
    def trading_hours(self) -> TradingHours:
        """Market open/close times."""
        return _MARKET_METADATA[self].trading_hours

    def data_path(self, root: str) -> str:
        """
        Construct full path to market's data_lake directory.

        Args:
            root: Root directory for data_lake (e.g., "./data_lake")

        Returns:
            Full path to market's data directory
        """
        import os

        return os.path.join(root, self.path)

    def is_trading_day(self, date: date | datetime) -> bool:
        """
        Check if the given date is a trading day for this market.

        Note: This is a placeholder implementation. Actual calendar
        integration will be added in Phase 4 with exchange calendar data.

        Args:
            date: Date to check

        Returns:
            True if trading day (placeholder: excludes weekends)

        Raises:
            NotImplementedError: For HK market (not yet implemented)
        """
        if self == Market.HK:
            raise NotImplementedError(
                "Hong Kong market calendar not yet implemented. "
                "HK market support planned for future release."
            )

        # Placeholder: exclude weekends
        # TODO Phase 4: Integrate with exchange_calendar library
        weekday = date.weekday()
        return weekday < 5  # Monday=0, Friday=4


@dataclass(frozen=True, slots=True)
class MarketMetadata:
    """Complete metadata for a market."""

    path: str
    price_precision: int
    trading_calendar: str
    sector_classification: str
    sector_count: int
    currency: str
    price_limit_rules: PriceLimitRule
    commission_rules: CommissionRule
    trading_hours: TradingHours


# Market metadata configuration
_MARKET_METADATA: dict[Market, MarketMetadata] = {
    Market.CN: MarketMetadata(
        path="cn",
        price_precision=2,
        trading_calendar="SSE/SZSE",
        sector_classification="Shenwan_L1",
        sector_count=28,
        currency="CNY",
        price_limit_rules=PriceLimitRule(
            has_limit=True,
            regular_limit_pct=0.10,  # +/-10% for main board
            special_limit_pct=0.20,  # +/-20% for STAR Market (Shanghai) and ChiNext (Shenzhen)
            description="Main board: +/-10%; STAR/ChiNext: +/-20%; ST stocks: +/-5%",
        ),
        commission_rules=CommissionRule(
            commission_rate=0.0003,  # 0.03%
            commission_min=5.0,  # 5 CNY minimum
            stamp_duty=0.001,  # 0.1% (sell side only)
            stamp_duty_side="sell",
            transfer_fee=0.00001,  # 0.001% transfer fee
            description="Commission: 0.03% min 5 CNY; Stamp: 0.1% sell; Transfer: 0.001%",
        ),
        trading_hours=TradingHours(
            pre_market_open="09:15",  # Call auction
            market_open="09:30",
            market_close="11:30",  # Morning session
            after_market_close="15:00",  # Afternoon session ends (13:00-15:00)
            timezone="Asia/Shanghai",
        ),
    ),
    Market.US: MarketMetadata(
        path="us",
        price_precision=4,
        trading_calendar="NYSE/NASDAQ",
        sector_classification="GICS_Sector",
        sector_count=8,  # 11 GICS sectors, but often grouped to 8 for backtesting
        currency="USD",
        price_limit_rules=PriceLimitRule(
            has_limit=False,
            circuit_breaker_pct=0.07,  # Level 1: 7%, Level 2: 13%, Level 3: 20%
            description="No daily limit; Circuit breakers at 7%, 13%, 20% market-wide",
        ),
        commission_rules=CommissionRule(
            commission_rate=0.0005,  # 0.05% typical online broker
            commission_min=0.0,  # No minimum for most online brokers
            stamp_duty=0.0,  # No stamp duty in US
            stamp_duty_side="none",
            transfer_fee=0.0,  # SEC fee applies but typically absorbed
            description="Commission: typically 0.05% or flat fee; No stamp duty",
        ),
        trading_hours=TradingHours(
            pre_market_open="04:00",  # Pre-market trading
            market_open="09:30",
            market_close="16:00",
            after_market_close="20:00",  # After-hours trading
            timezone="America/New_York",
        ),
    ),
    Market.HK: MarketMetadata(
        path="hk",
        price_precision=3,
        trading_calendar="HKEX",
        sector_classification="GICS_Sector",
        sector_count=11,
        currency="HKD",
        price_limit_rules=PriceLimitRule(
            has_limit=False,
            description="No daily limit; Halt rules apply for unusual movements",
        ),
        commission_rules=CommissionRule(
            commission_rate=0.0005,
            commission_min=0.0,
            stamp_duty=0.001,  # 0.1%
            stamp_duty_side="sell",
            transfer_fee=0.0,
            description="Placeholder - HK market not yet implemented",
        ),
        trading_hours=TradingHours(
            pre_market_open="09:00",
            market_open="09:30",
            market_close="12:00",  # Morning session
            after_market_close="16:00",  # Afternoon session ends (13:00-16:00)
            timezone="Asia/Hong_Kong",
        ),
    ),
}
