"""
Screening engine for TrendSpec execution.

ScreeningEngine runs a strategy for a single date (latest or specified):
- Run only latest date (or specified date)
- Call strategy.next(ctx) once for each instrument
- Collect BUY signals
- Skip matching/portfolio updates
- Output: signal list (ticker, direction, trigger value)

Key design:
- Same strategy.next() works for backtest and screening (dual-mode)
- Engine handles single-date vs loop difference
- No trade execution, portfolio, or equity curve
- Just outputs signals for further analysis

DO NOT generate market overview (top N returns, etc.)
- Screening engine outputs signals only
- Market overview generation belongs to analyzer module

Flow:
    Load universe(target_date)
    Load OHLCV data(target_date)
    Instantiate strategy
    strategy.init(ctx)  # Precompute indicators

    For each instrument in universe:
        ctx.instrument_id = instrument
        strategy.next(ctx)  # Generate signals

    Output: BUY signals list (no execution)
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import polars as pl

from trendspec.engine.base_engine import BaseEngine, EngineConfig, EngineResult
from trendspec.data.calendar import is_trading_day, previous_trading_day
from trendspec.data.markets import Market
from trendspec.strategy.base import BaseStrategy
from trendspec.strategy.signal import Signal


@dataclass
class ScreeningResult(EngineResult):
    """
    Result of screening execution.

    Extends EngineResult with screening-specific fields.

    Attributes:
        screening_date: Date screened
        universe_size: Size of universe at screening date
        buy_signals: List of BUY signals
        sell_signals: List of SELL signals
        signal_count: Total signal count
    """

    screening_date: date | None = None
    universe_size: int = 0
    buy_signals: list[Signal] = field(default_factory=list)
    sell_signals: list[Signal] = field(default_factory=list)
    signal_count: int = 0

    def buy_count(self) -> int:
        """Count BUY signals."""
        return len(self.buy_signals)

    def sell_count(self) -> int:
        """Count SELL signals."""
        return len(self.sell_signals)


@dataclass
class ScreeningConfig:
    """
    Configuration for screening engine.

    Attributes:
        market: Market to screen
        target_date: Date to screen (None = latest available)
        root: Root directory for data_lake
        adjustment_mode: Price adjustment mode
        include_sell_signals: Whether to include SELL signals
    """

    market: Market
    target_date: date | None = None
    root: str | None = None
    adjustment_mode: str = "forward"
    include_sell_signals: bool = False  # Screening typically focuses on BUYs


class ScreeningEngine(BaseEngine):
    """
    Screening execution engine.

    Runs strategy for a single date (latest or specified):
    - Load data for target date
    - Initialize strategy
    - Call next() for each instrument
    - Output signals without execution

    Key differences from BacktestEngine:
    - Single date execution (not date range loop)
    - No broker, portfolio, or trade execution
    - No equity curve or metrics
    - Outputs signals only

    DO NOT generate market overview - that's analyzer's job.

    Example:
        >>> config = EngineConfig(
        ...     market=Market.CN,
        ...     start_date=date(2024, 12, 15),
        ...     end_date=date(2024, 12, 15),
        ... )
        >>> engine = ScreeningEngine(config)
        >>> result = engine.run(MyStrategy, params={"period": 20})
        >>> result.buy_signals
        [Signal("BUY", "SH600000", ...), Signal("BUY", "SZ000001", ...)]
    """

    def __init__(self, config: EngineConfig) -> None:
        """
        Initialize screening engine.

        Args:
            config: Engine configuration
        """
        super().__init__(config)

        # Use start_date as target_date (screening is single-date)
        self._target_date = config.start_date

        # Screening-specific tracking
        self._signals: list[Signal] = []

    def get_trading_days(self) -> list[date]:
        """
        Get trading days for screening.

        Screening returns single target date.

        Returns:
            List containing target date (or nearest trading day)
        """
        # Validate target date is trading day
        if not is_trading_day(self.config.market, self._target_date):
            # Find previous trading day
            prev_date = previous_trading_day(self.config.market, self._target_date)
            return [prev_date]

        return [self._target_date]

    def run(
        self,
        strategy_class: type[BaseStrategy],
        params: dict[str, Any] | None = None,
    ) -> ScreeningResult:
        """
        Run screening with strategy.

        Execution flow:
        1. Load universe and data for target date
        2. Instantiate strategy
        3. Initialize strategy (precompute indicators)
        4. For each instrument in universe:
           - Update context with current instrument
           - Call strategy.next()
           - Collect signals
        5. Return signals (no execution)

        Args:
            strategy_class: Strategy class to run
            params: Strategy parameters

        Returns:
            ScreeningResult with signals
        """
        # Load universe and data
        self.load_universe()

        # Expand date range for indicator lookback (SMA200 needs ~200 trading days)
        original_start = self.config.start_date
        lookback_start = original_start - timedelta(days=330)  # MA200 needs ~290 calendar days + holiday buffer
        self.config.start_date = lookback_start
        self.load_data()
        self.config.start_date = original_start

        # Get target date
        target_dates = self.get_trading_days()
        if not target_dates:
            return self._create_empty_result()

        screening_date = target_dates[0]

        # Instantiate strategy
        self.instantiate_strategy(strategy_class, params)
        self.create_context()
        self.initialize_strategy()

        # Set available capital for position sizing
        self._ctx.update_positions({}, self.config.initial_capital)

        # Signal to strategy that this is a screening run (skip periodic guards like weekday checks)
        self._ctx.is_screening = True

        # Run screening for target date
        signals = self._run_screening(screening_date)

        # Create result
        buy_signals = [s for s in signals if s.is_buy()]
        sell_signals = [s for s in signals if s.is_sell()]

        universe_size = 0
        if self._universe:
            universe_size = len(self._universe.tickers(screening_date))

        return ScreeningResult(
            signals=signals,
            trades=[],  # No trades in screening
            equity_curve=[],  # No equity curve in screening
            metrics={},  # No metrics in screening
            strategy_name=self._strategy.name if self._strategy else "unknown",
            date_range=(screening_date, screening_date),
            market=self.config.market,
            screening_date=screening_date,
            universe_size=universe_size,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            signal_count=len(signals),
        )

    def _run_screening(self, screening_date: date) -> list[Signal]:
        """
        Run screening for a single date.

        Args:
            screening_date: Date to screen

        Returns:
            List of signals generated
        """
        data = self._data
        ctx = self._ctx
        strategy = self._strategy

        if data is None or ctx is None or strategy is None:
            return []

        # Filter data for screening date
        day_data = data.filter(pl.col("date") == screening_date)

        if day_data.is_empty():
            return []

        # Get universe for screening date
        universe_instruments = self._universe.tickers(screening_date) if self._universe else []

        # Update context with screening date
        ctx._current_date = screening_date

        # Clear signals
        ctx.clear_signals()
        signals: list[Signal] = []

        # Call strategy.next() for each instrument in universe
        for instrument_id in universe_instruments:
            # Check if instrument has data for this date
            instrument_data = day_data.filter(pl.col("instrument_id") == instrument_id)
            if instrument_data.is_empty():
                continue

            # Get ticker
            row = instrument_data.row(0, named=True)
            ticker = row.get("ticker", instrument_id)

            # Update context for this instrument
            ctx.update_bar(screening_date, instrument_id, ticker, data, current_row=row)

            # Call strategy.next()
            strategy.next(ctx)

            # Collect signals
            signals.extend(ctx.pending_signals())

            # Clear for next instrument
            ctx.clear_signals()

        return signals

    def _create_empty_result(self) -> ScreeningResult:
        """Create empty screening result."""
        return ScreeningResult(
            signals=[],
            trades=[],
            equity_curve=[],
            metrics={},
            strategy_name=self._strategy.name if self._strategy else "unknown",
            date_range=(self.config.start_date, self.config.end_date),
            market=self.config.market,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def signals_dataframe(self) -> pl.DataFrame:
        """
        Get signals as Polars DataFrame.

        Returns:
            DataFrame with signal data
        """
        if not self._signals:
            return pl.DataFrame()

        records = []
        for signal in self._signals:
            records.append({
                "direction": signal.direction,
                "ticker": signal.ticker,
                "instrument_id": signal.instrument_id,
                "price": signal.price,
                "trigger_value": signal.trigger_value,
                "note": signal.note,
            })

        return pl.DataFrame(records)

    def summary(self) -> str:
        """Get screening summary."""
        base_summary = super().summary()

        lines = [
            base_summary,
            "",
            "Screening Configuration:",
            f"  Target date: {self._target_date}",
            f"  Include sell signals: {self.config.costs_model}",
        ]

        return "\n".join(lines)


# =============================================================================
# Convenience Function
# =============================================================================


def screen(
    market: Market,
    strategy_class: type[BaseStrategy],
    target_date: date | None = None,
    params: dict[str, Any] | None = None,
    root: str | None = None,
) -> ScreeningResult:
    """
    Convenience function for one-shot screening.

    Args:
        market: Market to screen
        strategy_class: Strategy class to run
        target_date: Date to screen (None = latest)
        params: Strategy parameters
        root: Data lake root directory

    Returns:
        ScreeningResult with signals

    Example:
        >>> result = screen(Market.CN, MyStrategy, date(2024, 12, 15))
        >>> len(result.buy_signals)
        5
    """
    from datetime import datetime
    from trendspec.config.settings import get_settings

    # Use today if no target date
    if target_date is None:
        target_date = date.today()

    # Use default root if not specified
    if root is None:
        root = get_settings().data_lake.data_lake_root

    # Create config
    config = EngineConfig(
        market=market,
        start_date=target_date,
        end_date=target_date,
        root=root,
    )

    # Run screening
    engine = ScreeningEngine(config)
    return engine.run(strategy_class, params)