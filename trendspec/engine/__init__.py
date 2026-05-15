"""
TrendSpec engine module.

Provides execution engines for backtesting and screening strategies.
Key components:
- BaseEngine: Abstract base class for all engines
- BacktestEngine: Historical backtest execution
- ScreeningEngine: Current date signal screening
- Broker: Simulated broker for order execution
- Portfolio: Position and cash management
- CostsModel: Transaction cost modeling

Design principles:
- Dual-mode: Same strategy.next() works for backtest and screening
- Engine orchestrates: Load universe -> Load data -> Init strategy -> Execute
- Risk pipeline: Signals filtered before broker submission
- PIT data: All lookups use date parameter

Engine flow:
    BacktestEngine:
        Load universe(date_range)
        Load OHLCV data(date_range)
        Instantiate strategy
        strategy.init(ctx)  # Precompute indicators

        For each trading_day:
            ctx.date = trading_day
            ctx.universe = universe.tickers(trading_day)
            strategy.next(ctx)  # Generate signals

            For each signal:
                result = risk_pipeline.check(signal)
                if allowed:
                    trade = broker.submit(signal)
                    portfolio.update(trade)
                    trade_log.append(trade)
                    equity_curve.append(portfolio.nav())

    ScreeningEngine:
        Load universe(target_date)
        Load OHLCV data(target_date)
        Instantiate strategy
        strategy.init(ctx)
        strategy.next(ctx)  # Single call
        Output: BUY signals list (no execution)
"""

from trendspec.engine.base_engine import BaseEngine
from trendspec.engine.backtest_engine import BacktestEngine
from trendspec.engine.screening_engine import ScreeningEngine
from trendspec.engine.broker import Broker, Order, Trade
from trendspec.engine.portfolio import (
    Portfolio as EnginePortfolio,
    Position,
    EquityCurvePoint,
)
from trendspec.engine.costs import (
    CostsModel,
    CNACostsModel,
    USCostsModel,
    NoCostsModel,
)

__all__ = [
    # Engines
    "BaseEngine",
    "BacktestEngine",
    "ScreeningEngine",
    # Broker
    "Broker",
    "Order",
    "Trade",
    # Portfolio
    "EnginePortfolio",
    "Position",
    "EquityCurvePoint",
    # Costs
    "CostsModel",
    "CNACostsModel",
    "USCostsModel",
    "NoCostsModel",
]