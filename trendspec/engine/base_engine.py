"""
Abstract base engine for TrendSpec execution.

BaseEngine provides the orchestration framework for both backtest and screening.
Subclasses implement the scheduling loop differences.

Key design:
- Engine orchestrates: Load universe -> Load data -> Init strategy -> Execute
- Same strategy.next() works for backtest and screening (dual-mode)
- Engine handles the loop (backtest: per-day, screening: single call)
- Context provides PIT access for universe/sector/factor

Flow:
    1. Load universe for date range (PIT)
    2. Load OHLCV data for date range
    3. Instantiate strategy with params
    4. Call strategy.init(ctx) to precompute indicators
    5. Subclass.run() implements the execution loop
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import polars as pl

from trendspec.config.settings import get_settings
from trendspec.data.markets import Market
from trendspec.data.fundamentals import enrich_daily_panel
from trendspec.data.parquet_loader import bars
from trendspec.data.universe import Universe, get_universe
from trendspec.data.calendar import trading_days_between
from trendspec.strategy.base import BaseStrategy
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.signal import Signal
from trendspec.risk.pipeline import RiskPipeline


@dataclass
class EngineResult:
    """
    Result of engine execution.

    Contains all outputs from backtest or screening run.

    Attributes:
        signals: List of signals generated (screening) or all signals (backtest)
        trades: List of trades executed (backtest only)
        equity_curve: NAV over time (backtest only)
        metrics: Performance metrics (backtest only)
        strategy_name: Name of strategy that ran
        date_range: Date range of execution
        market: Market that was run
    """

    signals: list[Signal] = field(default_factory=list)
    trades: list[Any] = field(default_factory=list)  # Trade from broker
    equity_curve: list[Any] = field(default_factory=list)  # EquityCurvePoint
    metrics: dict[str, Any] = field(default_factory=dict)
    strategy_name: str = ""
    date_range: tuple[date, date] | None = None
    market: Market | None = None

    def is_empty(self) -> bool:
        """Check if result has any output."""
        return len(self.signals) == 0 and len(self.trades) == 0


@dataclass
class EngineConfig:
    """
    Configuration for engine execution.

    Attributes:
        market: Market to run (CN_A, US, HK)
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Initial capital (default: 100000)
        order_size: Default order size in shares
        risk_pipeline: Risk pipeline configuration
        costs_model: Transaction costs model
        adjustment_mode: Price adjustment mode
        root: Root directory for data_lake
    """

    market: Market
    start_date: date
    end_date: date
    initial_capital: float = 100000.0
    order_size: int = 100  # Default shares per order
    risk_pipeline: RiskPipeline | None = None
    costs_model: str = "default"  # "default", "none", or specific model
    adjustment_mode: str = "forward"  # "forward", "backward", "raw"
    root: str | None = None


class BaseEngine(ABC):
    """
    Abstract base class for execution engines.

    Provides the common orchestration for both backtest and screening:
    - Load universe (PIT)
    - Load OHLCV data
    - Instantiate strategy
    - Call strategy.init()
    - Subclass implements execution loop

    Subclasses:
    - BacktestEngine: Loop over trading days, execute trades
    - ScreeningEngine: Single call for latest date, output signals

    Usage:
        >>> config = EngineConfig(
        ...     market=Market.CN,
        ...     start_date=date(2024, 1, 1),
        ...     end_date=date(2024, 12, 31),
        ... )
        >>> engine = BacktestEngine(config)
        >>> result = engine.run(MyStrategy, params={"period": 20})

    Attributes:
        config: Engine configuration
        universe: PIT universe for the market
        data: OHLCV data DataFrame
        strategy: Strategy instance
        ctx: Strategy context
    """

    def __init__(self, config: EngineConfig) -> None:
        """
        Initialize engine with configuration.

        Args:
            config: Engine configuration
        """
        self.config = config
        self.root = config.root or get_settings().data_lake.data_lake_root

        # Lazy-loaded components
        self._universe: Universe | None = None
        self._data: pl.DataFrame | None = None
        self._weekly_data: pl.DataFrame | None = None
        self._strategy: BaseStrategy | None = None
        self._ctx: StrategyContext | None = None

    def inject(self, data=None, universe=None) -> None:
        """注入预加载数据 / universe,短路 load_data / load_universe 的重复读盘。

        data: 预 load 的 OHLCV DataFrame;universe: 预构建的 Universe 实例。
        """
        if data is not None:
            self._data = data
        if universe is not None:
            self._universe = universe

    # =========================================================================
    # Component Loading
    # =========================================================================

    def load_universe(self) -> Universe:
        """
        Load universe for the market.

        Returns:
            Universe instance for PIT lookups
        """
        if self._universe is None:
            self._universe = get_universe(self.config.market, self.root)
        return self._universe

    def load_data(self) -> pl.DataFrame:
        """
        Load OHLCV data for the date range.

        Returns:
            DataFrame with OHLCV data
        """
        if self._data is None:
            self._data = bars(
                market=self.config.market,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                adjustment_mode=self.config.adjustment_mode,
                root=self.root,
            )
            self._data = enrich_daily_panel(
                self._data, self.config.market, self.root
            )
            # Best-effort weekly load (may be empty if weekly ingest not run yet)
            try:
                self._weekly_data = bars(
                    market=self.config.market,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    adjustment_mode=self.config.adjustment_mode,
                    root=self.root,
                    frequency="weekly",
                )
                if self._weekly_data.is_empty():
                    self._weekly_data = None
            except Exception:
                self._weekly_data = None
        return self._data

    def instantiate_strategy(
        self,
        strategy_class: type[BaseStrategy],
        params: dict[str, Any] | None = None,
    ) -> BaseStrategy:
        """
        Instantiate strategy with parameters.

        Args:
            strategy_class: Strategy class to instantiate
            params: Strategy parameters

        Returns:
            Strategy instance
        """
        # Merge default order_size into params if not specified
        merged_params = params or {}
        if "order_size" not in merged_params:
            merged_params["order_size"] = self.config.order_size

        self._strategy = strategy_class(params=merged_params)
        return self._strategy

    def create_context(self) -> StrategyContext:
        """
        Create strategy context with data access.

        Returns:
            StrategyContext instance
        """
        data = self.load_data()
        strategy = self._strategy

        if strategy is None:
            raise RuntimeError("Strategy must be instantiated before creating context")

        self._ctx = StrategyContext(
            market=self.config.market,
            strategy=strategy,
            data=data,
            root=self.root,
            weekly_data=self._weekly_data,
        )
        return self._ctx

    # =========================================================================
    # Strategy Initialization
    # =========================================================================

    def initialize_strategy(self) -> None:
        """
        Initialize strategy by calling init() with context.

        Precomputes indicators and sets up strategy state.
        """
        ctx = self._ctx
        strategy = self._strategy

        if ctx is None or strategy is None:
            raise RuntimeError("Context and strategy must be created before init")

        strategy.set_context(ctx)
        strategy.init(ctx)
        strategy.mark_initialized()

    # =========================================================================
    # Abstract Methods (Subclass Implementation)
    # =========================================================================

    @abstractmethod
    def run(
        self,
        strategy_class: type[BaseStrategy],
        params: dict[str, Any] | None = None,
    ) -> EngineResult:
        """
        Run the engine with a strategy.

        Subclasses implement the execution loop:
        - BacktestEngine: Loop over trading days
        - ScreeningEngine: Single call for latest date

        Args:
            strategy_class: Strategy class to run
            params: Strategy parameters

        Returns:
            EngineResult with signals, trades, metrics
        """
        pass

    @abstractmethod
    def get_trading_days(self) -> list[date]:
        """
        Get trading days for execution.

        Backtest: All trading days in date range.
        Screening: Single target date.

        Returns:
            List of trading dates
        """
        pass

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_risk_pipeline(self) -> RiskPipeline:
        """
        Get risk pipeline for signal filtering.

        Returns configured pipeline or creates default.
        """
        if self.config.risk_pipeline is not None:
            return self.config.risk_pipeline

        # Create default pipeline
        from trendspec.risk.pipeline import default_pipeline

        return default_pipeline(
            max_positions=10,
            max_position_pct=0.10,
            min_capital=self.config.initial_capital * 0.05,  # 5% min reserve
        )

    def summary(self) -> str:
        """
        Get engine configuration summary.

        Returns:
            Human-readable summary string
        """
        lines = [
            f"Engine Configuration:",
            f"  Market: {self.config.market}",
            f"  Date range: {self.config.start_date} to {self.config.end_date}",
            f"  Initial capital: {self.config.initial_capital}",
            f"  Order size: {self.config.order_size}",
            f"  Adjustment mode: {self.config.adjustment_mode}",
            f"  Data loaded: {self._data is not None}",
            f"  Universe loaded: {self._universe is not None}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(market={self.config.market})"