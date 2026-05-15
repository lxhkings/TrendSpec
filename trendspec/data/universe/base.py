"""
Universe abstract base class for TrendSpec.

Defines the interface for PIT (point-in-time) universe tracking.
Key design rule: EVERY API MUST ACCEPT DATE PARAMETER.

PIT universe is critical for survivorship bias prevention:
- Historical windows include delisted stocks
- Universe changes over time (IPOs, delistings, index rebalancing)
- No "current universe" shortcuts
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Final


class Universe(ABC):
    """
    Abstract base class for PIT universe tracking.

    A universe represents the set of tradable instruments at any point in time.
    Examples:
    - CN_A: All listed A-shares (survivorship-free)
    - US: SP500 + Russell 1000 historical components
    - HK: Placeholder (not implemented)

    Key principle: Universe membership changes over time.
    An instrument may:
    - Not exist before IPO date
    - Be suspended during trading halts
    - Be removed after delisting

    All APIs must accept date parameter for PIT queries.
    """

    # Market identifier
    market: Final[str]

    @abstractmethod
    def tickers(self, as_of_date: date) -> list[str]:
        """
        Get all instrument_ids in the universe at a specific date.

        PIT design: as_of_date parameter is REQUIRED.
        Returns instrument_ids, not tickers (ticker can change).

        Args:
            as_of_date: Date to query

        Returns:
            List of instrument_ids in the universe at that date

        Example:
            >>> universe = CNAUniverse()
            >>> universe.tickers(date(2024, 1, 15))
            ['SH600000', 'SZ000001', ...]
        """
        pass

    @abstractmethod
    def contains(self, instrument_id: str, as_of_date: date) -> bool:
        """
        Check if an instrument is in the universe at a specific date.

        PIT design: as_of_date parameter is REQUIRED.

        Args:
            instrument_id: Instrument ID to check
            as_of_date: Date to check

        Returns:
            True if instrument is in universe at that date

        Example:
            >>> universe = CNAUniverse()
            >>> universe.contains("SH600000", date(2024, 1, 15))
            True
            >>> universe.contains("SH600000", date(1990, 1, 1))  # Before IPO
            False
        """
        pass

    @abstractmethod
    def ipo_date(self, instrument_id: str) -> date | None:
        """
        Get IPO date for an instrument.

        Args:
            instrument_id: Instrument ID

        Returns:
            IPO date or None if instrument not tracked
        """
        pass

    @abstractmethod
    def delist_date(self, instrument_id: str) -> date | None:
        """
        Get delisting date for an instrument.

        Args:
            instrument_id: Instrument ID

        Returns:
            Delisting date or None if still active or not tracked
        """
        pass

    @abstractmethod
    def is_active(self, instrument_id: str, as_of_date: date) -> bool:
        """
        Check if an instrument is active (listed and trading) at a date.

        Active = listed AND not delisted AND not halted.

        Args:
            instrument_id: Instrument ID
            as_of_date: Date to check

        Returns:
            True if active at that date
        """
        pass

    def count(self, as_of_date: date) -> int:
        """
        Count number of instruments in universe at a date.

        Args:
            as_of_date: Date to query

        Returns:
            Number of instruments in universe at that date
        """
        return len(self.tickers(as_of_date))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(market={self.market})"