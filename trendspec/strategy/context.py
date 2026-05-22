"""
Strategy context for TrendSpec strategy framework.

StrategyContext provides:
- Current bar data (date, prices, positions)
- Available capital
- Indicator cache (precomputed in init)
- Risk hooks
- PIT access methods for sector, factor, universe lookup

Key design principles:
- All PIT methods require date parameter (no "current" shortcuts)
- Indicator cache is precomputed in init() for vectorized efficiency
- Context is stateful during next() - reflects current bar's data
"""

from datetime import date as DateType
from typing import TYPE_CHECKING, Any

import polars as pl

from trendspec.data.markets import Market
from trendspec.data.sectors import SectorIndex, get_sector_index
from trendspec.data.universe import Universe, get_universe
from trendspec.strategy.signal import Signal

if TYPE_CHECKING:
    from trendspec.strategy.base import BaseStrategy


class StrategyContext:
    """
    Context provided to strategy init() and next() methods.

    Provides access to:
    - Current bar data (date, prices, positions)
    - PIT universe/sector/factor lookup
    - Indicator cache from init()
    - Signal generation methods

    The context is updated per-bar during backtesting or for latest date during screening.

    Example usage in strategy:
        >>> class MyStrategy(BaseStrategy):
        ...     def init(self, ctx: StrategyContext):
        ...         # Precompute indicators
        ...         self.ma20 = ctx.precompute_indicator("MA", period=20)
        ...
        ...     def next(self, ctx: StrategyContext):
        ...         # Access current bar data
        ...         if ctx.close > ctx.indicator_value("MA20", ctx.instrument_id):
        ...             ctx.signal("BUY", ctx.instrument_id, ctx.close)
    """

    def __init__(
        self,
        market: Market,
        strategy: "BaseStrategy",
        data: pl.DataFrame | None = None,
        root: str | None = None,
        weekly_data: pl.DataFrame | None = None,
    ) -> None:
        """
        Initialize strategy context.

        Args:
            market: Market enum (CN_A, US, HK)
            strategy: The strategy instance using this context
            data: Current bar data (DataFrame with single row per instrument)
            root: Root directory for data_lake
            weekly_data: Optional weekly OHLCV DataFrame for weekly indicators
        """
        self.market = market
        self.strategy = strategy
        self._data = data
        self._root = root
        self._weekly_data = weekly_data

        # Current bar state (updated per-bar)
        self._current_date: DateType | None = None
        self._current_instrument_id: str | None = None
        self._current_ticker: str | None = None

        # Precomputed indicator cache (populated in init())
        self._indicator_cache: dict[str, pl.DataFrame] = {}
        # Fast O(1) lookup: {cache_key: {(instrument_id, date): value}}
        self._indicator_fast: dict[str, dict[tuple, float]] = {}

        # Weekly indicator cache (populated by precompute_weekly_indicator)
        self._weekly_indicator_cache: dict[str, pl.DataFrame] = {}
        self._weekly_indicator_fast: dict[str, dict[tuple, float]] = {}
        # Sorted weekly dates per instrument for binary lookup (built lazily)
        self._weekly_dates_by_iid: dict[str, list] = {}

        # PIT access
        self._sector_index: SectorIndex | None = None
        self._universe: Universe | None = None

        # Current positions (updated by engine)
        self._positions: dict[str, float] = {}  # instrument_id -> quantity
        self._available_capital: float = 0.0

        # Set to True by ScreeningEngine — strategies skip periodic guards (e.g. weekday checks)
        self.is_screening: bool = False

        # Signals generated in current next() call
        self._pending_signals: list[Signal] = []

    # =========================================================================
    # Current Bar Data Access
    # =========================================================================

    @property
    def date(self) -> DateType:
        """Get current bar date."""
        if self._current_date is None:
            raise RuntimeError("No current bar date set")
        return self._current_date

    @property
    def instrument_id(self) -> str:
        """Get current instrument_id."""
        if self._current_instrument_id is None:
            raise RuntimeError("No current instrument_id set")
        return self._current_instrument_id

    @property
    def ticker(self) -> str:
        """Get current ticker (display symbol)."""
        if self._current_ticker is None:
            raise RuntimeError("No current ticker set")
        return self._current_ticker

    @property
    def close(self) -> float:
        """Get current close price."""
        return self._get_current_price("close")

    @property
    def open(self) -> float:
        """Get current open price."""
        return self._get_current_price("open")

    @property
    def high(self) -> float:
        """Get current high price."""
        return self._get_current_price("high")

    @property
    def low(self) -> float:
        """Get current low price."""
        return self._get_current_price("low")

    @property
    def volume(self) -> int:
        """Get current volume."""
        return self._get_current_value("volume")

    def _get_current_price(self, column: str) -> float:
        """Get price value for current instrument at current date."""
        if self._current_instrument_id is None:
            raise RuntimeError(f"No data available for {column}")

        bar = getattr(self, "_current_bar", None)
        if bar is not None and column in bar:
            return float(bar[column])

        if self._data is None:
            raise RuntimeError(f"No data available for {column}")

        filtered = self._data.filter(
            (pl.col("instrument_id") == self._current_instrument_id)
            & (pl.col("date") == self._current_date)
        )
        if filtered.is_empty():
            raise RuntimeError(f"No data for {self._current_instrument_id} at {self._current_date}")
        return filtered[column].item()

    def _get_current_value(self, column: str) -> Any:
        """Get value for current instrument at current date."""
        if self._current_instrument_id is None:
            raise RuntimeError(f"No data available for {column}")

        bar = getattr(self, "_current_bar", None)
        if bar is not None and column in bar:
            return bar[column]

        if self._data is None:
            raise RuntimeError(f"No data available for {column}")

        filtered = self._data.filter(
            (pl.col("instrument_id") == self._current_instrument_id)
            & (pl.col("date") == self._current_date)
        )
        if filtered.is_empty():
            raise RuntimeError(f"No data for {self._current_instrument_id} at {self._current_date}")
        return filtered[column].item()

    # =========================================================================
    # Position and Capital Access
    # =========================================================================

    @property
    def positions(self) -> dict[str, float]:
        """Get current positions dict (instrument_id -> quantity)."""
        return self._positions

    @property
    def available_capital(self) -> float:
        """Get available capital for new positions."""
        return self._available_capital

    def position(self, instrument_id: str | None = None) -> float:
        """
        Get position quantity for an instrument.

        Args:
            instrument_id: Instrument ID (defaults to current instrument)

        Returns:
            Position quantity (0 if not held)
        """
        target = instrument_id or self._current_instrument_id
        if target is None:
            return 0.0
        return self._positions.get(target, 0.0)

    def has_position(self, instrument_id: str | None = None) -> bool:
        """Check if position exists for an instrument."""
        return self.position(instrument_id) > 0

    # =========================================================================
    # PIT Universe and Sector Access
    # =========================================================================

    def universe(self) -> Universe:
        """Get universe instance for the market."""
        if self._universe is None:
            self._universe = get_universe(self.market, self._root)
        return self._universe

    def set_universe(self, universe: Universe) -> None:
        """Override the universe (used in tests to inject a stub without a data lake)."""
        self._universe = universe

    def index_close(self, index_id: str, as_of_date: DateType | None = None) -> float | None:
        """Get index close price at a specific date. Lazily loads and caches the indices DataFrame."""
        target_date = as_of_date or self._current_date
        if target_date is None:
            return None

        if not hasattr(self, "_indices_cache"):
            from trendspec.data.parquet_loader import read_indices
            try:
                self._indices_cache = read_indices(self.market, root=self._root)
            except Exception:
                self._indices_cache = None

        if self._indices_cache is None:
            return None

        # Build fast lookup dict on first use
        if not hasattr(self, "_indices_fast"):
            self._indices_fast: dict[tuple, float] = {
                (iid, dt): val
                for iid, dt, val in self._indices_cache.select(
                    ["instrument_id", "date", "close"]
                ).iter_rows()
            }

        return self._indices_fast.get((index_id, target_date))

    def sector_index(self) -> SectorIndex:
        """Get sector index for the market."""
        if self._sector_index is None:
            self._sector_index = get_sector_index(self.market, self._root)
        return self._sector_index

    def sector(self, instrument_id: str | None = None, as_of_date: DateType | None = None) -> str | None:
        """
        Get sector for an instrument at a specific date (PIT lookup).

        PIT design: Date parameter required. Defaults to current bar date.

        Args:
            instrument_id: Instrument ID (defaults to current)
            as_of_date: Date to check (defaults to current bar date)

        Returns:
            Sector code or None
        """
        target = instrument_id or self._current_instrument_id
        target_date = as_of_date or self._current_date

        if target is None or target_date is None:
            return None

        return self.sector_index().sector(target, target_date)

    def sector_universe(self, sector_code: str, as_of_date: DateType | None = None) -> list[str]:
        """
        Get all instruments in a sector at a specific date (PIT lookup).

        PIT design: Date parameter required. Defaults to current bar date.

        Args:
            sector_code: Sector code to filter
            as_of_date: Date to check (defaults to current bar date)

        Returns:
            List of instrument_ids in the sector at that date
        """
        target_date = as_of_date or self._current_date
        if target_date is None:
            return []

        return self.sector_index().sector_universe(sector_code, target_date)

    def pit_universe(self, as_of_date: DateType | None = None) -> list[str]:
        """
        Get all instrument_ids in the universe at a specific date (PIT lookup).

        PIT design: Date parameter required. Defaults to current bar date.

        Args:
            as_of_date: Date to check (defaults to current bar date)

        Returns:
            List of instrument_ids in the universe at that date
        """
        target_date = as_of_date or self._current_date
        if target_date is None:
            return []

        return self.universe().tickers(target_date)

    # =========================================================================
    # Factor Access
    # =========================================================================

    def factor(self, name: str, instrument_id: str | None = None, as_of_date: DateType | None = None) -> float | None:
        """
        Get factor value for an instrument at a specific date.

        Factors are computed from data and cached. Uses the factor registry.

        Args:
            name: Factor name (must be registered)
            instrument_id: Instrument ID (defaults to current)
            as_of_date: Date to check (defaults to current bar date)

        Returns:
            Factor value or None if not available
        """
        target = instrument_id or self._current_instrument_id
        target_date = as_of_date or self._current_date

        if target is None or target_date is None:
            return None

        # Look up in factor cache
        cache_key = f"{name}_{target_date.isoformat()}"
        if cache_key in self._indicator_cache:
            factor_df = self._indicator_cache[cache_key]
            filtered = factor_df.filter(pl.col("instrument_id") == target)
            if not filtered.is_empty():
                return filtered[name].item()

        return None

    # =========================================================================
    # Indicator Cache Management
    # =========================================================================

    def precompute_indicator(
        self,
        name: str,
        data: pl.DataFrame | None = None,
        **params: Any,
    ) -> pl.DataFrame:
        """
        Precompute an indicator for all instruments (vectorized).

        Called in init() to compute indicators once for entire dataset.
        Results are cached for fast lookup in next().

        Args:
            name: Indicator name (MA, EMA, RSI, etc.)
            data: DataFrame to compute on (defaults to strategy's data)
            **params: Indicator parameters (period, etc.)

        Returns:
            DataFrame with indicator column added
        """
        from trendspec.strategy.indicators import compute_indicator

        target_data = data or self._data
        if target_data is None:
            raise RuntimeError("No data available for indicator computation")

        result = compute_indicator(target_data, name, **params)
        cache_key = f"{name}_{params}"
        self._indicator_cache[cache_key] = result

        # Build fast O(1) lookup dict from the indicator column
        _col = f"{name}_{params.get('period', '')}" if params else name
        if _col not in result.columns:
            _col = name
        if _col in result.columns:
            self._indicator_fast[cache_key] = {
                (inst_id, dt): val
                for inst_id, dt, val in result.select(["instrument_id", "date", _col]).iter_rows()
                if val is not None
            }

        return result

    def indicator_value(
        self,
        name: str,
        instrument_id: str | None = None,
        as_of_date: DateType | None = None,
        **params: Any,
    ) -> float | None:
        """
        Get indicator value for an instrument at a date.

        Looks up from precomputed cache. Must have been computed in init().

        Args:
            name: Indicator name
            instrument_id: Instrument ID (defaults to current)
            as_of_date: Date to check (defaults to current bar date)
            **params: Indicator parameters used in precompute

        Returns:
            Indicator value or None if not available
        """
        target = instrument_id or self._current_instrument_id
        target_date = as_of_date or self._current_date

        if target is None or target_date is None:
            return None

        cache_key = f"{name}_{params}"
        if cache_key not in self._indicator_cache:
            # Try to compute on demand if data is available
            if self._data is not None:
                self.precompute_indicator(name, self._data, **params)
            else:
                return None

        # O(1) fast path via pre-built dict
        fast = self._indicator_fast.get(cache_key)
        if fast is not None:
            return fast.get((target, target_date))

        # Fallback: DataFrame filter (covers on-demand indicators without period)
        indicator_df = self._indicator_cache[cache_key]
        filtered = indicator_df.filter(
            (pl.col("instrument_id") == target) & (pl.col("date") == target_date)
        )

        if filtered.is_empty():
            return None

        col_name = f"{name}_{params.get('period', '')}" if params else name
        if col_name not in filtered.columns:
            col_name = name

        return filtered[col_name].item()

    # =========================================================================
    # Weekly Indicator Cache Management
    # =========================================================================

    def precompute_weekly_indicator(
        self,
        name: str,
        **params: Any,
    ) -> pl.DataFrame:
        """
        Precompute indicator on weekly data (mirrors precompute_indicator).

        Args:
            name: Indicator name (MA, EMA, RSI, ...)
            **params: Indicator parameters (period, etc.)

        Returns:
            Weekly DataFrame with indicator column added
        """
        from trendspec.strategy.indicators import compute_indicator

        if self._weekly_data is None:
            raise RuntimeError("No weekly_data; cannot precompute weekly indicator")

        result = compute_indicator(self._weekly_data, name, **params)
        cache_key = f"weekly_{name}_{params}"
        self._weekly_indicator_cache[cache_key] = result

        _col = f"{name}_{params.get('period', '')}" if params else name
        if _col not in result.columns:
            _col = name
        if _col in result.columns:
            self._weekly_indicator_fast[cache_key] = {
                (inst_id, dt): val
                for inst_id, dt, val in result.select(
                    ["instrument_id", "date", _col]
                ).iter_rows()
                if val is not None
            }

        return result

    def weekly_indicator_value(
        self,
        name: str,
        instrument_id: str | None = None,
        as_of_date: DateType | None = None,
        **params: Any,
    ) -> float | None:
        """
        Get weekly indicator value at the most recent completed weekly bar ≤ as_of_date.

        Never reads an incomplete (current) week — strict no-lookahead guarantee.

        Returns None if no weekly bar exists ≤ as_of_date, or weekly data missing.
        """
        if self._weekly_data is None:
            return None

        target_iid = instrument_id or self._current_instrument_id
        target_date = as_of_date or self._current_date
        if target_iid is None or target_date is None:
            return None

        cache_key = f"weekly_{name}_{params}"
        if cache_key not in self._weekly_indicator_fast:
            self.precompute_weekly_indicator(name, **params)

        week_end = self._resolve_week_end(target_iid, target_date)
        if week_end is None:
            return None

        return self._weekly_indicator_fast.get(cache_key, {}).get((target_iid, week_end))

    def _resolve_week_end(self, iid: str, as_of_date: DateType) -> DateType | None:
        """Binary-search largest weekly bar date ≤ as_of_date for `iid`."""
        import bisect

        if iid not in self._weekly_dates_by_iid:
            if self._weekly_data is None:
                return None
            dates = (
                self._weekly_data
                .filter(pl.col("instrument_id") == iid)
                .sort("date")["date"]
                .to_list()
            )
            self._weekly_dates_by_iid[iid] = dates

        dates = self._weekly_dates_by_iid[iid]
        if not dates:
            return None
        # bisect_right gives the insertion point AFTER all equals;
        # idx-1 → largest date ≤ as_of_date
        idx = bisect.bisect_right(dates, as_of_date)
        if idx == 0:
            return None
        return dates[idx - 1]

    # =========================================================================
    # Signal Generation
    # =========================================================================

    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Get a parameter value from the strategy.

        Args:
            key: Parameter key
            default: Default value if not found

        Returns:
            Parameter value
        """
        return self.strategy.get_param(key, default)

    def signal(
        self,
        direction: str,
        instrument_id: str | None = None,
        price: float | None = None,
        trigger_value: float | None = None,
        note: str | None = None,
    ) -> Signal:
        """
        Generate a trading signal.

        Signals are collected during next() and processed by the engine.

        Args:
            direction: "BUY" or "SELL"
            instrument_id: Instrument ID (defaults to current)
            price: Price (defaults to current close)
            trigger_value: Optional indicator/factor value that triggered
            note: Optional human-readable note

        Returns:
            The generated signal
        """
        target = instrument_id or self._current_instrument_id
        if target is None:
            raise RuntimeError("Cannot generate signal without instrument_id")

        target_price = price or self.close

        sig = Signal(
            direction=direction,
            ticker=self._current_ticker or target,
            instrument_id=target,
            price=target_price,
            trigger_value=trigger_value,
            note=note,
        )

        self._pending_signals.append(sig)
        return sig

    def pending_signals(self) -> list[Signal]:
        """Get all pending signals generated in current next() call."""
        return self._pending_signals

    def clear_signals(self) -> None:
        """Clear pending signals (called by engine after processing)."""
        self._pending_signals.clear()

    # =========================================================================
    # Context Update (Called by Engine)
    # =========================================================================

    def update_bar(
        self,
        current_date: DateType,
        instrument_id: str,
        ticker: str,
        data: pl.DataFrame,
        current_row: dict | None = None,
    ) -> None:
        """
        Update context for a new bar.

        Called by the engine before each next() call.

        Args:
            current_date: Current bar date
            instrument_id: Current instrument_id
            ticker: Current ticker
            data: Full data DataFrame (or filtered for current date)
            current_row: Pre-extracted row dict for O(1) price access
        """
        self._current_date = current_date
        self._current_instrument_id = instrument_id
        self._current_ticker = ticker
        self._data = data
        self._current_bar: dict | None = current_row

    def update_positions(self, positions: dict[str, float], available_capital: float) -> None:
        """
        Update position and capital state.

        Called by the engine after broker execution.

        Args:
            positions: Current positions dict
            available_capital: Available capital
        """
        self._positions = positions
        self._available_capital = available_capital