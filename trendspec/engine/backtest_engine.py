"""
Backtest engine for TrendSpec execution.

BacktestEngine runs a strategy over historical data:
- Daily loop: For each trading day in range
- Feed bar to strategy.next(ctx)
- Collect signals
- Pass to RiskPipeline
- Submit to Broker for matching
- Update Portfolio
- Record to TradeLog, EquityCurve

Key design:
- Same strategy.next() works for backtest and screening (dual-mode)
- Engine handles the loop, strategy just generates signals
- Risk pipeline filters signals before execution
- Broker simulates realistic execution with slippage
- Portfolio tracks positions and NAV

Flow:
    Load universe(date_range)
    Load OHLCV data(date_range)
    Instantiate strategy
    strategy.init(ctx)  # Precompute indicators

    For each trading_day:
        ctx.date = trading_day
        ctx.universe = universe.tickers(trading_day)

        For each instrument in universe:
            ctx.instrument_id = instrument
            strategy.next(ctx)  # Generate signals

        For each signal:
            result = risk_pipeline.check(signal)
            if allowed:
                trade = broker.submit(signal)
                portfolio.update(trade)
                trade_log.append(trade)
                equity_curve.append(portfolio.nav())

    Output: metrics, trades, equity curve
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import polars as pl

from trendspec.engine.base_engine import BaseEngine, EngineConfig, EngineResult
from trendspec.engine.broker import Broker, Trade
from trendspec.engine.costs import CostsModel, get_costs_model, NoCostsModel
from trendspec.engine.portfolio import Portfolio, EquityCurvePoint, TradeLog
from trendspec.data.calendar import trading_days_between, next_trading_day
from trendspec.data.markets import Market
from trendspec.strategy.base import BaseStrategy
from trendspec.strategy.signal import Signal
from trendspec.risk.base import Portfolio as RiskPortfolio
from trendspec.risk.pipeline import RiskPipeline


@dataclass
class BacktestMetrics:
    """
    Performance metrics for backtest.

    Attributes:
        total_return: Total return percentage
        annualized_return: Annualized return
        max_drawdown: Maximum drawdown percentage
        sharpe_ratio: Sharpe ratio (if enough data)
        total_trades: Total number of trades
        win_rate: Win rate percentage
        avg_win: Average winning trade return
        avg_loss: Average losing trade return
        profit_factor: Profit factor (gross_win / gross_loss)
        total_costs: Total transaction costs
        initial_capital: Initial capital
        final_nav: Final NAV
    """

    total_return: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    total_costs: float = 0.0
    initial_capital: float = 0.0
    final_nav: float = 0.0
    trading_days: int = 0
    total_pnl: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "total_costs": self.total_costs,
            "initial_capital": self.initial_capital,
            "final_nav": self.final_nav,
            "trading_days": self.trading_days,
            "total_pnl": self.total_pnl,
        }


class BacktestEngine(BaseEngine):
    """
    Backtest execution engine.

    Runs strategy over historical data with:
    - Daily loop over trading days
    - Signal generation per bar
    - Risk pipeline filtering
    - Broker execution simulation
    - Portfolio tracking
    - Equity curve recording

    Example:
        >>> config = EngineConfig(
        ...     market=Market.CN,
        ...     start_date=date(2024, 1, 1),
        ...     end_date=date(2024, 12, 31),
        ...     initial_capital=100000,
        ... )
        >>> engine = BacktestEngine(config)
        >>> result = engine.run(MyStrategy, params={"period": 20})
        >>> result.metrics["total_return"]
        0.25  # 25% return
    """

    def __init__(self, config: EngineConfig) -> None:
        """
        Initialize backtest engine.

        Args:
            config: Engine configuration
        """
        super().__init__(config)

        # Execution components
        self._portfolio: Portfolio | None = None
        self._broker: Broker | None = None
        self._costs_model: CostsModel | None = None
        self._trade_log: TradeLog = TradeLog()
        self._equity_curve: list[EquityCurvePoint] = []
        self._all_signals: list[Signal] = []

        # Metrics tracking
        self._prev_nav: float = 0.0
        self._peak_nav: float = 0.0
        self._daily_returns: list[float] = []

    def get_trading_days(self) -> list[date]:
        """
        Get all trading days in the date range.

        Returns:
            List of trading dates
        """
        return trading_days_between(
            self.config.market,
            self.config.start_date,
            self.config.end_date,
        )

    def run(
        self,
        strategy_class: type[BaseStrategy],
        params: dict[str, Any] | None = None,
    ) -> EngineResult:
        """
        Run backtest with strategy.

        Execution flow:
        1. Load universe and data
        2. Instantiate strategy
        3. Initialize strategy (precompute indicators)
        4. For each trading day:
           - Update context with current bar
           - Call strategy.next() for each instrument
           - Collect signals
           - Process through risk pipeline
           - Execute orders via broker
           - Update portfolio
           - Record equity curve
        5. Calculate metrics

        Args:
            strategy_class: Strategy class to run
            params: Strategy parameters

        Returns:
            EngineResult with trades, equity curve, metrics
        """
        # Initialize components
        self._initialize_execution_components()

        # Load universe and data
        self.load_universe()
        self.load_data()
        self.instantiate_strategy(strategy_class, params)
        self.create_context()
        self.initialize_strategy()

        # Get trading days
        trading_days = self.get_trading_days()

        if not trading_days:
            return self._create_empty_result()

        # Initialize portfolio
        self._portfolio = Portfolio(initial_capital=self.config.initial_capital)
        self._prev_nav = self.config.initial_capital
        self._peak_nav = self.config.initial_capital

        # Main execution loop
        for trading_day in trading_days:
            self._run_day(trading_day)

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Create result
        return EngineResult(
            signals=self._all_signals,
            trades=self._trade_log.trades,
            equity_curve=self._equity_curve,
            metrics=metrics,
            strategy_name=self._strategy.name if self._strategy else "unknown",
            date_range=(self.config.start_date, self.config.end_date),
            market=self.config.market,
        )

    def _initialize_execution_components(self) -> None:
        """Initialize broker, costs model, and other components."""
        # Get costs model for market
        if self.config.costs_model == "none":
            self._costs_model = NoCostsModel()
        elif self.config.costs_model == "default":
            self._costs_model = get_costs_model(self.config.market)
        else:
            self._costs_model = get_costs_model(self.config.market)

        # Create broker
        self._broker = Broker(
            slippage_bps=0.0,  # Default no slippage
            execution_mode="next_open",
            costs_model=self._costs_model,
        )

        # Clear logs
        self._trade_log = TradeLog()
        self._equity_curve = []
        self._all_signals = []
        self._daily_returns = []

    def _run_day(self, trading_day: date) -> None:
        """
        Run backtest for a single trading day.

        Args:
            trading_day: Current trading date
        """
        data = self._data
        ctx = self._ctx
        strategy = self._strategy
        portfolio = self._portfolio

        if data is None or ctx is None or strategy is None or portfolio is None:
            return

        # Filter data for current date
        day_data = data.filter(pl.col("date") == trading_day)

        if day_data.is_empty():
            return

        # Get universe for current date
        universe_instruments = self._universe.tickers(trading_day) if self._universe else []

        # Update context with current date
        ctx._current_date = trading_day

        # Sync portfolio state into context so strategies can access positions and capital
        ctx.update_positions(
            {iid: pos.shares for iid, pos in portfolio.active_positions().items()},
            portfolio.available_cash(),
        )

        # Clear signals from previous day
        ctx.clear_signals()

        # Pre-build dict for O(1) per-instrument lookup
        day_rows = {r["instrument_id"]: r for r in day_data.iter_rows(named=True)}

        # Call strategy.next() for each instrument in universe
        for instrument_id in universe_instruments:
            row = day_rows.get(instrument_id)
            if row is None:
                continue

            # Get ticker (from data)
            ticker = row.get("ticker", instrument_id)

            # Update context for this instrument
            ctx.update_bar(trading_day, instrument_id, ticker, data)

            # Update portfolio prices for this instrument
            current_price = row.get("close", 0.0)
            portfolio.update_prices({instrument_id: current_price})

            # Call strategy.next()
            strategy.next(ctx)

        # Get signals generated
        signals = ctx.pending_signals()
        self._all_signals.extend(signals)

        # Process signals through risk pipeline
        if signals:
            allowed_signals = self._process_signals(signals, trading_day)

            # Submit orders to broker (use signal.shares if set, else order_size)
            for signal in allowed_signals:
                order_shares = (
                    int(signal.shares)
                    if signal.shares is not None
                    else ctx.get_param("order_size", 100)
                )
                self._broker.submit(signal, shares=order_shares)

        # Execute orders at next day's open (T+1)
        # For simplicity, execute at current close for now
        executed_trades = self._broker.execute_orders(trading_day, day_data)

        # Update portfolio for each trade
        for trade in executed_trades:
            sector = ctx.sector(trade.instrument_id, trading_day) if ctx else None
            portfolio.update_position(
                instrument_id=trade.instrument_id,
                ticker=trade.ticker,
                direction=trade.direction,
                shares=trade.shares,
                price=trade.price,
                cost=trade.cost,
                trade_date=trade.execution_date or trading_day,
                sector=sector,
            )
            self._trade_log.append(trade)

        # Update all position prices with current day's close prices
        price_updates = {}
        for instrument_id in portfolio.positions().keys():
            inst_data = day_data.filter(pl.col("instrument_id") == instrument_id)
            if not inst_data.is_empty():
                price_updates[instrument_id] = inst_data.row(0, named=True).get("close", 0.0)
        portfolio.update_prices(price_updates)

        # Record equity curve point
        self._record_equity_curve(trading_day)

        # Call strategy.on_bar_end()
        strategy.on_bar_end(ctx)

        # Clear signals after processing
        ctx.clear_signals()

    def _process_signals(
        self,
        signals: list[Signal],
        trading_day: date,
    ) -> list[Signal]:
        """
        Process signals through risk pipeline.

        Args:
            signals: Signals to process
            trading_day: Current trading date

        Returns:
            List of allowed signals
        """
        allowed = []
        pipeline = self.get_risk_pipeline()
        ctx = self._ctx
        portfolio = self._portfolio

        if ctx is None or portfolio is None:
            return signals  # No filtering if components not ready

        # Create risk portfolio state
        risk_portfolio = RiskPortfolio(
            positions={id: pos.shares for id, pos in portfolio.positions().items()},
            cash=portfolio.cash,
            equity=portfolio.nav(),
            positions_value=portfolio.total_position_value(),
            sector_weights=portfolio.sector_weights(),
            position_prices={id: pos.current_price for id, pos in portfolio.positions().items()},
        )

        for signal in signals:
            result = pipeline.run(signal, risk_portfolio, ctx)
            if result.is_allowed():
                allowed.append(result.signal)

        return allowed

    def _record_equity_curve(self, trading_day: date) -> None:
        """
        Record equity curve point for the day.

        Args:
            trading_day: Current trading date
        """
        portfolio = self._portfolio
        if portfolio is None:
            return

        nav = portfolio.nav()
        cash = portfolio.cash
        position_value = portfolio.total_position_value()
        position_count = portfolio.position_count()

        # Calculate daily return
        daily_return = 0.0
        if self._prev_nav > 0:
            daily_return = (nav - self._prev_nav) / self._prev_nav
        self._daily_returns.append(daily_return)

        # Calculate cumulative return
        cumulative_return = 0.0
        if self.config.initial_capital > 0:
            cumulative_return = (nav - self.config.initial_capital) / self.config.initial_capital

        # Update peak for drawdown calculation
        if nav > self._peak_nav:
            self._peak_nav = nav

        point = EquityCurvePoint(
            date=trading_day,
            nav=nav,
            cash=cash,
            position_value=position_value,
            position_count=position_count,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
        )

        self._equity_curve.append(point)
        self._prev_nav = nav

    def _calculate_metrics(self) -> dict[str, Any]:
        """
        Calculate performance metrics from backtest results.

        Returns:
            Dict with performance metrics
        """
        portfolio = self._portfolio
        if portfolio is None:
            return {}

        trading_days = len(self._equity_curve)
        final_nav = portfolio.nav()
        initial_capital = self.config.initial_capital

        # Total return
        total_return = 0.0
        if initial_capital > 0:
            total_return = (final_nav - initial_capital) / initial_capital

        # Annualized return
        annualized_return = 0.0
        if trading_days > 0:
            # Assume 252 trading days per year
            years = trading_days / 252
            if years > 0:
                annualized_return = (1 + total_return) ** (1 / years) - 1

        # Max drawdown
        max_drawdown = 0.0
        peak = initial_capital
        for point in self._equity_curve:
            if point.nav > peak:
                peak = point.nav
            drawdown = (peak - point.nav) / peak if peak > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Sharpe ratio (simplified, using daily returns)
        sharpe_ratio = 0.0
        if len(self._daily_returns) > 10:
            import statistics
            avg_return = statistics.mean(self._daily_returns)
            std_return = statistics.stdev(self._daily_returns)
            if std_return > 0:
                # Annualized Sharpe (252 trading days)
                sharpe_ratio = (avg_return * 252) / (std_return * (252 ** 0.5))

        # Trade statistics
        trades = self._trade_log.trades
        total_trades = len(trades)

        # Win rate (based on realized P&L from closed positions)
        total_pnl = portfolio.total_pnl()
        total_costs = portfolio.total_costs()

        metrics = BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_trades=total_trades,
            total_costs=total_costs,
            initial_capital=initial_capital,
            final_nav=final_nav,
            trading_days=trading_days,
            total_pnl=total_pnl,
        )

        return metrics.to_dict()

    def _create_empty_result(self) -> EngineResult:
        """Create empty result when no trading days."""
        return EngineResult(
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

    def equity_curve_dataframe(self) -> pl.DataFrame:
        """
        Get equity curve as Polars DataFrame.

        Returns:
            DataFrame with equity curve data
        """
        if not self._equity_curve:
            return pl.DataFrame()

        records = [point.to_dict() for point in self._equity_curve]
        return pl.DataFrame(records)

    def trades_dataframe(self) -> pl.DataFrame:
        """
        Get trades as Polars DataFrame.

        Returns:
            DataFrame with trade data
        """
        return self._trade_log.to_dataframe()

    def summary(self) -> str:
        """Get backtest summary."""
        base_summary = super().summary()

        if self._portfolio is None:
            return base_summary

        metrics = self._calculate_metrics()

        lines = [
            base_summary,
            "",
            "Backtest Results:",
            f"  Final NAV: {metrics.get('final_nav', 0):.2f}",
            f"  Total Return: {metrics.get('total_return', 0):.2%}",
            f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}",
            f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}",
            f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"  Total Trades: {metrics.get('total_trades', 0)}",
            f"  Total Costs: {metrics.get('total_costs', 0):.2f}",
        ]

        return "\n".join(lines)