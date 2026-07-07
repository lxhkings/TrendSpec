"""
Portfolio management for TrendSpec execution engines.

Portfolio tracks positions, cash, and NAV throughout backtest execution.
Provides position updates after trade execution and sector weight calculations.

Key design:
- Position tracking: Dict of instrument_id -> Position
- Cash management: Track available capital
- NAV calculation: Positions value + cash
- Sector weights: For risk pipeline sector concentration checks

Integration:
- BacktestEngine updates portfolio after each trade
- Risk pipeline checks portfolio state before allowing trades
- Equity curve records portfolio.nav() at each bar end
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import polars as pl


@dataclass
class Position:
    """
    Single position in a portfolio.

    Tracks the holding for one instrument including entry cost,
    current value, and profit/loss.

    Attributes:
        instrument_id: Immutable instrument ID (primary key)
        ticker: Display ticker (mutable, for readability)
        shares: Number of shares held
        avg_cost: Average cost per share
        current_price: Current market price
        current_value: Current market value (shares * current_price)
        entry_date: Date position was opened
        cost_basis: Total cost of position (shares * avg_cost)
        unrealized_pnl: Unrealized profit/loss
        unrealized_pnl_pct: Unrealized P&L as percentage
        sector: Sector classification (for risk checks)
    """

    instrument_id: str
    ticker: str
    shares: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0
    entry_date: date | None = None
    sector: str | None = None

    @property
    def current_value(self) -> float:
        """Calculate current market value."""
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        """Calculate total cost basis."""
        return self.shares * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        return self.current_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost

    def is_active(self) -> bool:
        """Check if position is active (shares > 0)."""
        return self.shares > 0

    def update_price(self, price: float) -> None:
        """Update current market price."""
        self.current_price = price

    def add_shares(self, shares: float, cost: float) -> None:
        """
        Add shares to position (buy).

        Updates average cost weighted by shares.

        Args:
            shares: Number of shares to add
            cost: Cost per share for this addition
        """
        if shares <= 0 or cost <= 0:
            return

        total_shares = self.shares + shares
        if total_shares <= 0:
            return

        # Weighted average cost
        new_cost_basis = self.cost_basis + (shares * cost)
        self.avg_cost = new_cost_basis / total_shares
        self.shares = total_shares

    def remove_shares(self, shares: float) -> float:
        """
        Remove shares from position (sell).

        Returns realized P&L for the sold shares.

        Args:
            shares: Number of shares to remove

        Returns:
            Realized profit/loss for sold shares
        """
        if shares <= 0:
            return 0.0

        sell_shares = min(shares, self.shares)
        if sell_shares <= 0:
            return 0.0

        # Calculate realized P&L
        realized_pnl = sell_shares * (self.current_price - self.avg_cost)

        self.shares -= sell_shares

        # Reset avg_cost if position closed
        if self.shares <= 0:
            self.shares = 0.0
            self.avg_cost = 0.0

        return realized_pnl

    def to_dict(self) -> dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "instrument_id": self.instrument_id,
            "ticker": self.ticker,
            "shares": self.shares,
            "avg_cost": self.avg_cost,
            "current_price": self.current_price,
            "current_value": self.current_value,
            "cost_basis": self.cost_basis,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "entry_date": self.entry_date.isoformat() if self.entry_date else None,
            "sector": self.sector,
        }


@dataclass
class EquityCurvePoint:
    """
    Single point on the equity curve.

    Records portfolio state at a specific date for performance tracking.

    Attributes:
        date: Date of this point
        nav: Net asset value (positions + cash)
        cash: Cash balance
        position_value: Total value of all positions
        position_count: Number of active positions
        daily_return: Daily return percentage
        cumulative_return: Cumulative return from start
    """

    date: date
    nav: float = 0.0
    cash: float = 0.0
    position_value: float = 0.0
    position_count: int = 0
    daily_return: float = 0.0
    cumulative_return: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "nav": self.nav,
            "cash": self.cash,
            "position_value": self.position_value,
            "position_count": self.position_count,
            "daily_return": self.daily_return,
            "cumulative_return": self.cumulative_return,
        }


@dataclass
class TradeLog:
    """
    Trade log for recording all executed trades.

    Attributes:
        trades: List of executed trades
    """

    trades: list[Any] = field(default_factory=list)  # Trade from broker

    def append(self, trade: Any) -> None:
        """Append a trade to the log."""
        self.trades.append(trade)

    def to_dataframe(self) -> pl.DataFrame:
        """Convert trade log to Polars DataFrame."""
        if not self.trades:
            return pl.DataFrame()

        records = [t.to_dict() for t in self.trades]
        return pl.DataFrame(records)

    def buy_trades(self) -> list[Any]:
        """Get all buy trades."""
        return [t for t in self.trades if t.direction == "BUY"]

    def sell_trades(self) -> list[Any]:
        """Get all sell trades."""
        return [t for t in self.trades if t.direction == "SELL"]

    def count(self) -> int:
        """Count total trades."""
        return len(self.trades)

    def total_volume(self) -> float:
        """Calculate total trading volume."""
        return sum(t.shares * t.price for t in self.trades)

    def total_costs(self) -> float:
        """Calculate total transaction costs."""
        return sum(t.cost for t in self.trades)


class Portfolio:
    """
    Portfolio manager for backtest execution.

    Tracks positions, cash, and provides NAV calculations.
    Updates positions after each trade execution.

    Key methods:
    - nav(): Net asset value (positions + cash)
    - available_cash(): Cash for new positions
    - position(instrument_id): Get position for instrument
    - update_position(trade): Update after trade execution
    - sector_weights(): Calculate sector concentration

    Integration:
    - BacktestEngine updates portfolio after broker execution
    - Risk pipeline checks portfolio state (positions, cash, sector weights)
    - Equity curve records portfolio.nav() at each bar end

    Example:
        >>> portfolio = Portfolio(initial_capital=100000)
        >>> portfolio.update_position(trade)
        >>> portfolio.nav()
        102500.0
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
    ) -> None:
        """
        Initialize portfolio with starting capital.

        Args:
            initial_capital: Initial cash balance
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self._positions: dict[str, Position] = {}

        # Track realized P&L
        self._realized_pnl: float = 0.0
        self._total_costs: float = 0.0

    # =========================================================================
    # Position Management
    # =========================================================================

    def positions(self) -> dict[str, Position]:
        """Get all positions."""
        return self._positions

    def active_positions(self) -> dict[str, Position]:
        """Get active positions (shares > 0)."""
        return {
            id: pos for id, pos in self._positions.items() if pos.is_active()
        }

    def position(self, instrument_id: str) -> Position | None:
        """Get position for an instrument."""
        return self._positions.get(instrument_id)

    def has_position(self, instrument_id: str) -> bool:
        """Check if position exists."""
        pos = self._positions.get(instrument_id)
        return pos is not None and pos.is_active()

    def position_count(self) -> int:
        """Count active positions."""
        return len(self.active_positions())

    def position_value(self, instrument_id: str) -> float:
        """Get position value for an instrument."""
        pos = self._positions.get(instrument_id)
        if pos is None or not pos.is_active():
            return 0.0
        return pos.current_value

    def total_position_value(self) -> float:
        """Calculate total value of all positions."""
        return sum(pos.current_value for pos in self.active_positions().values())

    # =========================================================================
    # NAV and Cash
    # =========================================================================

    def nav(self) -> float:
        """
        Calculate net asset value.

        NAV = Cash + Total position value
        """
        return self.cash + self.total_position_value()

    def available_cash(self) -> float:
        """
        Calculate available cash for new positions.

        This is the cash balance minus any reserves.
        For now, returns full cash balance.
        """
        return self.cash

    def equity(self) -> float:
        """Alias for nav()."""
        return self.nav()

    # =========================================================================
    # Position Updates
    # =========================================================================

    def update_position(
        self,
        instrument_id: str,
        ticker: str,
        direction: str,
        shares: float,
        price: float,
        cost: float,
        trade_date: date,
        sector: str | None = None,
    ) -> float | None:
        """
        Update portfolio after trade execution.

        Args:
            instrument_id: Instrument ID
            ticker: Display ticker
            direction: "BUY" or "SELL"
            shares: Number of shares
            price: Execution price
            cost: Transaction cost
            trade_date: Date of trade
            sector: Sector classification

        Returns:
            Realized P&L (for sells), or 0.0 for a successful BUY. Returns
            None if the trade was rejected (insufficient cash on BUY, or no
            matching position on SELL) and no portfolio state changed.
        """
        realized_pnl = 0.0

        if direction == "BUY":
            # Deduct cash (shares * price + cost). Reject if insufficient cash —
            # cash must never go negative (no margin/leverage in this engine).
            total_cost = shares * price + cost
            if total_cost > self.cash:
                return None
            self.cash -= total_cost
            self._total_costs += cost

            # Update or create position
            pos = self._positions.get(instrument_id)
            if pos is None:
                pos = Position(
                    instrument_id=instrument_id,
                    ticker=ticker,
                    shares=shares,
                    avg_cost=price,
                    current_price=price,
                    entry_date=trade_date,
                    sector=sector,
                )
                self._positions[instrument_id] = pos
            else:
                pos.add_shares(shares, price)

        elif direction == "SELL":
            # Get position
            pos = self._positions.get(instrument_id)
            if pos is None:
                return None

            # Remove shares and get realized P&L
            realized_pnl = pos.remove_shares(shares)

            # Add cash (shares * price - cost)
            total_cash = shares * price - cost
            self.cash += total_cash
            self._total_costs += cost

            # Record realized P&L
            self._realized_pnl += realized_pnl

            # Clean up closed positions
            if not pos.is_active():
                del self._positions[instrument_id]

        return realized_pnl

    def update_prices(self, prices: dict[str, float]) -> None:
        """
        Update all position prices.

        Args:
            prices: Dict of instrument_id -> current price
        """
        for instrument_id, pos in self._positions.items():
            if instrument_id in prices:
                pos.update_price(prices[instrument_id])

    # =========================================================================
    # Sector Analysis
    # =========================================================================

    def sector_weights(self) -> dict[str, float]:
        """
        Calculate sector weights.

        Returns:
            Dict of sector -> weight (percentage of NAV)
        """
        weights: dict[str, float] = {}
        nav = self.nav()

        if nav == 0:
            return weights

        for pos in self.active_positions().values():
            sector = pos.sector or "Unknown"
            weight = pos.current_value / nav
            weights[sector] = weights.get(sector, 0.0) + weight

        return weights

    def sector_weight(self, sector: str) -> float:
        """
        Get weight for a specific sector.

        Args:
            sector: Sector code

        Returns:
            Sector weight as percentage of NAV
        """
        return self.sector_weights().get(sector, 0.0)

    # =========================================================================
    # Statistics
    # =========================================================================

    def realized_pnl(self) -> float:
        """Get total realized P&L."""
        return self._realized_pnl

    def total_costs(self) -> float:
        """Get total transaction costs."""
        return self._total_costs

    def unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions."""
        return sum(pos.unrealized_pnl for pos in self.active_positions().values())

    def total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self._realized_pnl + self.unrealized_pnl()

    def total_return_pct(self) -> float:
        """Calculate total return percentage."""
        if self.initial_capital == 0:
            return 0.0
        return (self.nav() - self.initial_capital) / self.initial_capital

    def largest_position_pct(self) -> float:
        """Get largest position as percentage of NAV."""
        nav = self.nav()
        if nav == 0:
            return 0.0

        max_value = max(
            (pos.current_value for pos in self.active_positions().values()),
            default=0.0,
        )
        return max_value / nav

    # =========================================================================
    # Conversion and Export
    # =========================================================================

    def to_risk_portfolio(self) -> dict[str, Any]:
        """
        Convert to risk pipeline Portfolio format.

        Returns portfolio state for risk rule checks.
        """
        return {
            "positions": {
                id: pos.shares for id, pos in self.active_positions().items()
            },
            "cash": self.cash,
            "equity": self.nav(),
            "positions_value": self.total_position_value(),
            "sector_weights": self.sector_weights(),
            "position_prices": {
                id: pos.current_price for id, pos in self._positions.items()
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert portfolio to dictionary."""
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "nav": self.nav(),
            "position_count": self.position_count(),
            "total_position_value": self.total_position_value(),
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": self.unrealized_pnl(),
            "total_pnl": self.total_pnl(),
            "total_costs": self._total_costs,
            "total_return_pct": self.total_return_pct(),
            "positions": {
                id: pos.to_dict() for id, pos in self.active_positions().items()
            },
            "sector_weights": self.sector_weights(),
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Portfolio(nav={self.nav():.2f}, "
            f"cash={self.cash:.2f}, "
            f"positions={self.position_count()})"
        )