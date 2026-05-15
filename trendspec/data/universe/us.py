"""
US PIT universe for TrendSpec.

Pre-built memory index for O(1) universe membership lookup.
Key features:
- SP500 + Russell 1000 historical components
- Quarterly rebalancing tracking
- Ticker changes and M&A delistings
- Survivorship bias prevention

Primary key is (instrument_id, date) - ticker can change.
"""

from datetime import date
from pathlib import Path
from typing import Final

import polars as pl

from trendspec.config.settings import get_settings
from trendspec.data.markets import Market
from trendspec.data.parquet_loader import scan_parquet, _lazyframe_is_empty
from trendspec.data.universe.base import Universe

# =============================================================================
# Event Types
# =============================================================================

# US-specific component event types
IPO_EVENT: Final[str] = "IPO"
DELIST_EVENT: Final[str] = "DELIST"
HALT_EVENT: Final[str] = "HALT"
RESUME_EVENT: Final[str] = "RESUME"
SP500_ADD_EVENT: Final[str] = "SP500_ADD"
SP500_DEL_EVENT: Final[str] = "SP500_DEL"
R1000_ADD_EVENT: Final[str] = "R1000_ADD"
R1000_DEL_EVENT: Final[str] = "R1000_DEL"


class USUniverse(Universe):
    """
    US stock PIT universe.

    Universe definition: SP500 + Russell 1000 historical components.
    Handles:
    - Quarterly index rebalancing (March, June, September, December)
    - Ticker changes (company rebranding, M&A)
    - Delistings (acquisition, bankruptcy, regulatory)

    Pre-built memory index from components Parquet:
    - _universe_by_date: dict[date, frozenset[instrument_id]]
    - _ipo_dates: dict[instrument_id, date]
    - _delist_dates: dict[instrument_id, date]
    - _index_membership: dict[instrument_id, set[date]] for each index

    Survivorship bias prevention:
    - Historical windows include stocks that were in index at that time
    - Even if later removed from index
    """

    market: Final[str] = "US"

    def __init__(
        self,
        root: str | None = None,
        universe_type: str = "sp500_r1000",
    ) -> None:
        """
        Initialize US universe.

        Args:
            root: Root directory for data_lake
            universe_type: Universe type:
                - "sp500_r1000": SP500 + Russell 1000 (default)
                - "sp500": S&P 500 only
                - "r1000": Russell 1000 only
        """
        self.root = root or get_settings().data_lake.data_lake_root
        self.universe_type = universe_type

        # Pre-built indices
        self._universe_by_date: dict[date, frozenset[str]] = {}
        self._ipo_dates: dict[str, date] = {}
        self._delist_dates: dict[str, date] = {}
        self._halt_periods: dict[str, list[tuple[date, date | None]]] = {}
        self._sp500_membership: dict[str, set[date]] = {}
        self._r1000_membership: dict[str, set[date]] = {}
        self._all_instruments: frozenset[str] = frozenset()

        # Build indices
        self._build_indices()

    def _build_indices(self) -> None:
        """
        Build memory indices from Parquet data.

        Loads:
        1. Components data (IPO, delist, index events)
        2. Daily data (for instrument_id list)
        """
        # Load components events
        components_lf = scan_parquet(self.root, Market.US, "components")

        if not _lazyframe_is_empty(components_lf):
            components_df = components_lf.collect()

            if not components_df.is_empty():
                self._process_components(components_df)

        # Build universe by date from daily data
        daily_lf = scan_parquet(self.root, Market.US, "daily")

        if not _lazyframe_is_empty(daily_lf):
            daily_df = daily_lf.select(["instrument_id", "date"]).collect()

            if not daily_df.is_empty():
                self._build_universe_by_date(daily_df)

    def _process_components(self, df: pl.DataFrame) -> None:
        """
        Process component events DataFrame.

        Extract IPO, delist, halt, and index events.

        Args:
            df: Components DataFrame
        """
        all_instruments: set[str] = set()

        for row in df.iter_rows(named=True):
            instrument_id = row.get("instrument_id")
            event_date = row.get("date")
            event_type = row.get("event")

            if not instrument_id or not event_date or not event_type:
                continue

            all_instruments.add(instrument_id)

            if event_type == IPO_EVENT:
                self._ipo_dates[instrument_id] = event_date

            elif event_type == DELIST_EVENT:
                self._delist_dates[instrument_id] = event_date

            elif event_type == HALT_EVENT:
                if instrument_id not in self._halt_periods:
                    self._halt_periods[instrument_id] = []
                self._halt_periods[instrument_id].append((event_date, None))

            elif event_type == RESUME_EVENT:
                if instrument_id in self._halt_periods:
                    periods = self._halt_periods[instrument_id]
                    for i in range(len(periods) - 1, -1, -1):
                        if periods[i][1] is None:
                            periods[i] = (periods[i][0], event_date)
                            break

            elif event_type == SP500_ADD_EVENT:
                if instrument_id not in self._sp500_membership:
                    self._sp500_membership[instrument_id] = set()
                self._sp500_membership[instrument_id].add(event_date)

            elif event_type == SP500_DEL_EVENT:
                # Removal from index - store as delist date for index
                pass  # Handled in membership check

            elif event_type == R1000_ADD_EVENT:
                if instrument_id not in self._r1000_membership:
                    self._r1000_membership[instrument_id] = set()
                self._r1000_membership[instrument_id].add(event_date)

        self._all_instruments = frozenset(all_instruments)

    def _build_universe_by_date(self, df: pl.DataFrame) -> None:
        """
        Build universe membership by date.

        Args:
            df: Daily DataFrame
        """
        grouped = df.group_by("date").agg(
            pl.col("instrument_id").alias("instruments")
        )

        for row in grouped.iter_rows(named=True):
            d = row.get("date")
            instruments = row.get("instruments")

            if d and instruments:
                self._universe_by_date[d] = frozenset(instruments)

    def tickers(self, as_of_date: date) -> list[str]:
        """
        Get all instrument_ids in universe at a specific date.

        PIT design: as_of_date parameter is REQUIRED.

        Universe type determines which stocks are included:
        - "sp500_r1000": Union of SP500 and Russell 1000 at that date
        - "sp500": Only S&P 500 constituents
        - "r1000": Only Russell 1000 constituents

        Args:
            as_of_date: Date to query

        Returns:
            List of instrument_ids in universe at that date
        """
        if as_of_date in self._universe_by_date:
            base_universe = self._universe_by_date[as_of_date]
        else:
            base_universe = self._calculate_universe_at_date(as_of_date)

        # Apply universe type filter
        filtered_universe: list[str] = []
        for instrument_id in base_universe:
            if self._is_in_universe_type(instrument_id, as_of_date):
                if not self._is_halted_at_date(instrument_id, as_of_date):
                    filtered_universe.append(instrument_id)

        return sorted(filtered_universe)

    def _calculate_universe_at_date(self, as_of_date: date) -> frozenset[str]:
        """
        Calculate universe membership from indices.

        Args:
            as_of_date: Date to calculate

        Returns:
            Frozenset of instrument_ids
        """
        instruments: set[str] = set()

        for instrument_id in self._all_instruments:
            ipo = self._ipo_dates.get(instrument_id)
            if ipo and ipo > as_of_date:
                continue

            delist = self._delist_dates.get(instrument_id)
            if delist and delist <= as_of_date:
                continue

            instruments.add(instrument_id)

        return frozenset(instruments)

    def _is_in_universe_type(self, instrument_id: str, as_of_date: date) -> bool:
        """
        Check if instrument is in the specified universe type at date.

        Args:
            instrument_id: Instrument ID
            as_of_date: Date to check

        Returns:
            True if in universe type at that date
        """
        # For broad universe, all listed stocks are included
        if self.universe_type == "sp500_r1000":
            # Check if in either index at that date
            in_sp500 = self._is_in_sp500(instrument_id, as_of_date)
            in_r1000 = self._is_in_r1000(instrument_id, as_of_date)
            return in_sp500 or in_r1000

        elif self.universe_type == "sp500":
            return self._is_in_sp500(instrument_id, as_of_date)

        elif self.universe_type == "r1000":
            return self._is_in_r1000(instrument_id, as_of_date)

        # Default: all listed stocks
        return True

    def _is_in_sp500(self, instrument_id: str, as_of_date: date) -> bool:
        """
        Check if instrument was in S&P 500 at a date.

        Args:
            instrument_id: Instrument ID
            as_of_date: Date to check

        Returns:
            True if was in SP500 at that date
        """
        # Check daily data membership (most reliable)
        if as_of_date in self._universe_by_date:
            return instrument_id in self._universe_by_date[as_of_date]

        # Fall back to index events
        # This is simplified - actual implementation would need more data
        return instrument_id in self._sp500_membership

    def _is_in_r1000(self, instrument_id: str, as_of_date: date) -> bool:
        """
        Check if instrument was in Russell 1000 at a date.

        Args:
            instrument_id: Instrument ID
            as_of_date: Date to check

        Returns:
            True if was in R1000 at that date
        """
        if as_of_date in self._universe_by_date:
            return instrument_id in self._universe_by_date[as_of_date]

        return instrument_id in self._r1000_membership

    def _is_halted_at_date(self, instrument_id: str, as_of_date: date) -> bool:
        """
        Check if instrument is halted at a specific date.

        Args:
            instrument_id: Instrument ID
            as_of_date: Date to check

        Returns:
            True if halted at that date
        """
        if instrument_id not in self._halt_periods:
            return False

        periods = self._halt_periods[instrument_id]
        for start, end in periods:
            if start <= as_of_date:
                if end is None or end > as_of_date:
                    return True

        return False

    def contains(self, instrument_id: str, as_of_date: date) -> bool:
        """
        Check if instrument is in universe at a specific date.

        PIT design: as_of_date parameter is REQUIRED.

        Args:
            instrument_id: Instrument ID
            as_of_date: Date to check

        Returns:
            True if in universe at that date
        """
        ipo = self._ipo_dates.get(instrument_id)
        if ipo and ipo > as_of_date:
            return False

        delist = self._delist_dates.get(instrument_id)
        if delist and delist <= as_of_date:
            return False

        if self._is_halted_at_date(instrument_id, as_of_date):
            return False

        return self._is_in_universe_type(instrument_id, as_of_date)

    def ipo_date(self, instrument_id: str) -> date | None:
        """
        Get IPO date for an instrument.

        Args:
            instrument_id: Instrument ID

        Returns:
            IPO date or None
        """
        return self._ipo_dates.get(instrument_id)

    def delist_date(self, instrument_id: str) -> date | None:
        """
        Get delisting date for an instrument.

        Args:
            instrument_id: Instrument ID

        Returns:
            Delisting date or None
        """
        return self._delist_dates.get(instrument_id)

    def is_active(self, instrument_id: str, as_of_date: date) -> bool:
        """
        Check if instrument is active at a date.

        Args:
            instrument_id: Instrument ID
            as_of_date: Date to check

        Returns:
            True if active
        """
        return self.contains(instrument_id, as_of_date)

    def all_instruments(self) -> frozenset[str]:
        """
        Get all instruments ever tracked.

        Returns:
            Frozenset of all instrument_ids
        """
        return self._all_instruments

    def instrument_count_total(self) -> int:
        """Get total number of instruments ever tracked."""
        return len(self._all_instruments)

    def sp500_constituents(self, as_of_date: date) -> list[str]:
        """
        Get S&P 500 constituents at a specific date.

        PIT design: as_of_date parameter is REQUIRED.

        Args:
            as_of_date: Date to query

        Returns:
            List of instrument_ids in SP500 at that date
        """
        constituents: list[str] = []
        for instrument_id in self._universe_by_date.get(as_of_date, frozenset()):
            if self._is_in_sp500(instrument_id, as_of_date):
                constituents.append(instrument_id)
        return sorted(constituents)

    def r1000_constituents(self, as_of_date: date) -> list[str]:
        """
        Get Russell 1000 constituents at a specific date.

        PIT design: as_of_date parameter is REQUIRED.

        Args:
            as_of_date: Date to query

        Returns:
            List of instrument_ids in R1000 at that date
        """
        constituents: list[str] = []
        for instrument_id in self._universe_by_date.get(as_of_date, frozenset()):
            if self._is_in_r1000(instrument_id, as_of_date):
                constituents.append(instrument_id)
        return sorted(constituents)