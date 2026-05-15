"""
Hong Kong market universe placeholder for TrendSpec.

HK market is not implemented yet. All methods raise NotImplementedError.
"""

from datetime import date

from trendspec.data.universe.base import Universe


class HKUniverse(Universe):
    """
    Hong Kong stock universe placeholder.

    Not implemented - raises NotImplementedError for all methods.
    """

    market = "HK"

    def __init__(self, root: str | None = None) -> None:
        """
        Initialize HK universe placeholder.

        Args:
            root: Root directory (ignored)
        """
        self.root = root

    def tickers(self, as_of_date: date) -> list[str]:
        """
        Get all instrument_ids in universe at a specific date.

        Raises:
            NotImplementedError: HK market not implemented
        """
        raise NotImplementedError(
            "Hong Kong market universe not yet implemented. "
            "HK market support planned for future release."
        )

    def contains(self, instrument_id: str, as_of_date: date) -> bool:
        """
        Check if instrument is in universe at a specific date.

        Raises:
            NotImplementedError: HK market not implemented
        """
        raise NotImplementedError(
            "Hong Kong market universe not yet implemented. "
            "HK market support planned for future release."
        )

    def ipo_date(self, instrument_id: str) -> date | None:
        """
        Get IPO date for an instrument.

        Raises:
            NotImplementedError: HK market not implemented
        """
        raise NotImplementedError(
            "Hong Kong market universe not yet implemented. "
            "HK market support planned for future release."
        )

    def delist_date(self, instrument_id: str) -> date | None:
        """
        Get delisting date for an instrument.

        Raises:
            NotImplementedError: HK market not implemented
        """
        raise NotImplementedError(
            "Hong Kong market universe not yet implemented. "
            "HK market support planned for future release."
        )

    def is_active(self, instrument_id: str, as_of_date: date) -> bool:
        """
        Check if instrument is active at a date.

        Raises:
            NotImplementedError: HK market not implemented
        """
        raise NotImplementedError(
            "Hong Kong market universe not yet implemented. "
            "HK market support planned for future release."
        )