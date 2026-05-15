"""
A-share PIT universe for TrendSpec.

Pre-built memory index for O(1) universe membership lookup.
Key features:
- Survivorship bias prevention: includes delisted stocks in historical windows
- IPO date tracking: filter by listing date
- Delist date tracking: include stocks before delisting
- Halting tracking: exclude suspended stocks

Memory estimate: ~150MB for full CN_A universe (~5000 stocks, 30+ years)

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

# Component event types
IPO_EVENT: Final[str] = "IPO"
DELIST_EVENT: Final[str] = "DELIST"
HALT_EVENT: Final[str] = "HALT"
RESUME_EVENT: Final[str] = "RESUME"


class CNAUniverse(Universe):
    """
    China A-share PIT universe.

    Pre-built memory index from components Parquet data.
    Structure:
    - _universe_by_date: dict[date, frozenset[instrument_id]] for O(1) lookup
    - _ipo_dates: dict[instrument_id, date]
    - _delist_dates: dict[instrument_id, date]
    - _halt_dates: dict[instrument_id, set[date]]

    Survivorship bias prevention:
    - Historical windows include stocks that were delisted later
    - Universe at date D includes all stocks listed at D
    - Delisted stocks are excluded only after their delist date
    """

    market: Final[str] = "CN_A"

    def __init__(self, root: str | None = None) -> None:
        """
        Initialize A-share universe.

        Pre-builds memory index from components Parquet.
        Memory estimate: ~50MB for CN_A (~5000 stocks, 30+ years)

        Args:
            root: Root directory for data_lake
        """
        self.root = root or get_settings().data_lake.data_lake_root

        # Pre-built indices
        self._universe_by_date: dict[date, frozenset[str]] = {}
        self._ipo_dates: dict[str, date] = {}
        self._delist_dates: dict[str, date] = {}
        self._halt_periods: dict[str, list[tuple[date, date | None]]] = {}
        self._all_instruments: frozenset[str] = frozenset()

        # Build indices
        self._build_indices()

    def _build_indices(self) -> None:
        """
        Build memory indices from Parquet data.

        Loads:
        1. Components data (IPO, delist, halt events)
        2. Daily data (for instrument_id list)

        Creates:
        - _universe_by_date: pre-calculated universe for each date
        - _ipo_dates: IPO date lookup
        - _delist_dates: delist date lookup
        - _halt_periods: halt period tracking
        """
        # Load components events
        components_lf = scan_parquet(self.root, Market.CN_A, "components")

        if not _lazyframe_is_empty(components_lf):
            components_df = components_lf.collect()

            if not components_df.is_empty():
                self._process_components(components_df)

        # Build universe by date from daily data
        daily_lf = scan_parquet(self.root, Market.CN_A, "daily")

        if not _lazyframe_is_empty(daily_lf):
            # Get all unique (instrument_id, date) pairs
            daily_df = daily_lf.select(["instrument_id", "date"]).collect()

            if not daily_df.is_empty():
                self._build_universe_by_date(daily_df)

    def _process_components(self, df: pl.DataFrame) -> None:
        """
        Process component events DataFrame.

        Extract IPO, delist, halt events and build indices.

        Args:
            df: Components DataFrame with (instrument_id, date, event) columns
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
                # Track halt start
                if instrument_id not in self._halt_periods:
                    self._halt_periods[instrument_id] = []
                # Add halt period (start date, end date=None until RESUME)
                self._halt_periods[instrument_id].append((event_date, None))

            elif event_type == RESUME_EVENT:
                # Close the most recent halt period
                if instrument_id in self._halt_periods:
                    periods = self._halt_periods[instrument_id]
                    # Find the most recent open halt period
                    for i in range(len(periods) - 1, -1, -1):
                        if periods[i][1] is None:
                            periods[i] = (periods[i][0], event_date)
                            break

        self._all_instruments = frozenset(all_instruments)

    def _build_universe_by_date(self, df: pl.DataFrame) -> None:
        """
        Build universe membership by date.

        Creates dict[date, frozenset[instrument_id]] for O(1) lookup.

        Args:
            df: Daily DataFrame with (instrument_id, date) columns
        """
        # Group by date and collect instrument_ids
        grouped = df.group_by("date").agg(
            pl.col("instrument_id").alias("instruments")
        )

        for row in grouped.iter_rows(named=True):
            d = row.get("date")
            instruments = row.get("instruments")

            if d and instruments:
                # Convert to frozenset for O(1) membership check
                self._universe_by_date[d] = frozenset(instruments)

    def tickers(self, as_of_date: date) -> list[str]:
        """
        Get all instrument_ids in universe at a specific date.

        PIT design: as_of_date parameter is REQUIRED.

        Returns instruments that:
        1. Have IPO date <= as_of_date (listed)
        2. Have no delist date OR delist date > as_of_date (not delisted)
        3. Are not halted at as_of_date (trading)

        Args:
            as_of_date: Date to query

        Returns:
            List of instrument_ids in universe at that date
        """
        # Try pre-built universe first
        if as_of_date in self._universe_by_date:
            base_universe = self._universe_by_date[as_of_date]
        else:
            # Fall back to calculating from indices
            base_universe = self._calculate_universe_at_date(as_of_date)

        # Filter out halted instruments
        active_instruments: list[str] = []
        for instrument_id in base_universe:
            if not self._is_halted_at_date(instrument_id, as_of_date):
                active_instruments.append(instrument_id)

        return sorted(active_instruments)

    def _calculate_universe_at_date(self, as_of_date: date) -> frozenset[str]:
        """
        Calculate universe membership from indices.

        Used when date is not in pre-built universe.

        Args:
            as_of_date: Date to calculate

        Returns:
            Frozenset of instrument_ids at that date
        """
        instruments: set[str] = set()

        for instrument_id in self._all_instruments:
            # Check IPO date
            ipo = self._ipo_dates.get(instrument_id)
            if ipo and ipo > as_of_date:
                continue  # Not listed yet

            # Check delist date
            delist = self._delist_dates.get(instrument_id)
            if delist and delist <= as_of_date:
                continue  # Already delisted

            instruments.add(instrument_id)

        return frozenset(instruments)

    def _is_halted_at_date(self, instrument_id: str, as_of_date: date) -> bool:
        """
        Check if instrument is halted at a specific date.

        Args:
            instrument_id: Instrument ID
            as_of_date: Date to check

        Returns:
            True if instrument is halted at that date
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
            instrument_id: Instrument ID to check
            as_of_date: Date to check

        Returns:
            True if instrument is in universe at that date
        """
        # Check IPO date
        ipo = self._ipo_dates.get(instrument_id)
        if ipo and ipo > as_of_date:
            return False

        # Check delist date
        delist = self._delist_dates.get(instrument_id)
        if delist and delist <= as_of_date:
            return False

        # Check halt status
        if self._is_halted_at_date(instrument_id, as_of_date):
            return False

        return True

    def ipo_date(self, instrument_id: str) -> date | None:
        """
        Get IPO date for an instrument.

        Args:
            instrument_id: Instrument ID

        Returns:
            IPO date or None if not tracked
        """
        return self._ipo_dates.get(instrument_id)

    def delist_date(self, instrument_id: str) -> date | None:
        """
        Get delisting date for an instrument.

        Args:
            instrument_id: Instrument ID

        Returns:
            Delisting date or None if still active or not tracked
        """
        return self._delist_dates.get(instrument_id)

    def is_active(self, instrument_id: str, as_of_date: date) -> bool:
        """
        Check if instrument is active at a date.

        Active = listed AND not delisted AND not halted.

        Args:
            instrument_id: Instrument ID
            as_of_date: Date to check

        Returns:
            True if active at that date
        """
        return self.contains(instrument_id, as_of_date)

    def all_instruments(self) -> frozenset[str]:
        """
        Get all instruments ever tracked (including delisted).

        Useful for survivorship-free historical analysis.

        Returns:
            Frozenset of all instrument_ids
        """
        return self._all_instruments

    def instrument_count_total(self) -> int:
        """Get total number of instruments ever tracked."""
        return len(self._all_instruments)

    def universe_dates(self) -> list[date]:
        """Get all dates with pre-built universe data."""
        return sorted(self._universe_by_date.keys())