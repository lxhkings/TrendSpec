"""
Simulated broker for TrendSpec execution engines.

Broker handles order submission and execution simulation.
Provides realistic execution modeling:
- Order book management
- Execution price logic (open/close/limit)
- Slippage model (configurable)
- Transaction cost integration

Key design:
- Orders are signals with execution parameters
- Broker simulates realistic execution
- Execution price depends on execution mode
- Slippage can be configured per market

Execution modes:
- "next_open": Execute at next day's open (default for T+1 markets)
- "same_close": Execute at same day's close (intraday)
- "limit": Execute at limit price if reached
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import polars as pl


@dataclass
class Order:
    """
    Order submitted to broker.

    Contains the signal and execution parameters.

    Attributes:
        instrument_id: Instrument ID
        ticker: Display ticker
        direction: "BUY" or "SELL"
        shares: Number of shares to trade
        signal_price: Price from signal (typically close)
        limit_price: Optional limit price for limit orders
        execution_mode: "next_open", "same_close", or "limit"
        timestamp: Order submission timestamp
        order_id: Unique order ID
        signal_note: Note from original signal
        trigger_value: Trigger value from signal
    """

    instrument_id: str
    ticker: str
    direction: Literal["BUY", "SELL"]
    shares: float
    signal_price: float
    limit_price: float | None = None
    execution_mode: str = "next_open"
    timestamp: float | None = None
    order_id: str = ""
    signal_note: str | None = None
    trigger_value: float | None = None

    def __post_init__(self) -> None:
        """Generate order ID if not provided."""
        if not self.order_id:
            self.order_id = f"{self.instrument_id}_{self.direction}_{id(self)}"

    def is_buy(self) -> bool:
        """Check if buy order."""
        return self.direction == "BUY"

    def is_sell(self) -> bool:
        """Check if sell order."""
        return self.direction == "SELL"

    def is_limit(self) -> bool:
        """Check if limit order."""
        return self.execution_mode == "limit" and self.limit_price is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "instrument_id": self.instrument_id,
            "ticker": self.ticker,
            "direction": self.direction,
            "shares": self.shares,
            "signal_price": self.signal_price,
            "limit_price": self.limit_price,
            "execution_mode": self.execution_mode,
            "timestamp": self.timestamp,
            "signal_note": self.signal_note,
            "trigger_value": self.trigger_value,
        }


@dataclass
class Trade:
    """
    Executed trade.

    Contains execution details including actual price and costs.

    Attributes:
        instrument_id: Instrument ID
        ticker: Display ticker
        direction: "BUY" or "SELL"
        shares: Number of shares executed
        price: Execution price
        signal_price: Original signal price
        slippage: Slippage in price
        cost: Transaction cost
        total_value: Total trade value (shares * price)
        execution_date: Date of execution
        order_id: Original order ID
        trade_id: Unique trade ID
        note: Execution note
    """

    instrument_id: str
    ticker: str
    direction: Literal["BUY", "SELL"]
    shares: float
    price: float
    signal_price: float
    slippage: float = 0.0
    cost: float = 0.0
    execution_date: date | None = None
    order_id: str = ""
    trade_id: str = ""
    note: str | None = None

    def __post_init__(self) -> None:
        """Generate trade ID if not provided."""
        if not self.trade_id:
            self.trade_id = f"trade_{self.instrument_id}_{self.direction}_{id(self)}"

    @property
    def total_value(self) -> float:
        """Calculate total trade value."""
        return self.shares * self.price

    def is_buy(self) -> bool:
        """Check if buy trade."""
        return self.direction == "BUY"

    def is_sell(self) -> bool:
        """Check if sell trade."""
        return self.direction == "SELL"

    def pnl_impact(self) -> float:
        """
        Calculate P&L impact.

        For buys: Cost of acquisition
        For sells: Revenue from sale minus cost
        """
        if self.is_buy():
            return -self.total_value - self.cost
        else:
            return self.total_value - self.cost

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "instrument_id": self.instrument_id,
            "ticker": self.ticker,
            "direction": self.direction,
            "shares": self.shares,
            "price": self.price,
            "signal_price": self.signal_price,
            "slippage": self.slippage,
            "cost": self.cost,
            "total_value": self.total_value,
            "execution_date": self.execution_date.isoformat() if self.execution_date else None,
            "note": self.note,
        }


@dataclass
class OrderRejection:
    """
    Rejected order with reason.

    Attributes:
        order: The rejected order
        reason: Rejection reason
        details: Additional rejection details
    """

    order: Order
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


class Broker:
    """
    Simulated broker for order execution.

    Handles order submission, execution simulation, and trade recording.

    Execution logic:
    - next_open: Execute at next trading day's open
    - same_close: Execute at current day's close
    - limit: Execute at limit price if reached during the day

    Slippage model:
    - Basis points (bps) slippage added to execution price
    - BUY: Price + slippage (pay more)
    - SELL: Price - slippage (receive less)

    Integration with costs:
    - CostsModel calculates transaction costs
    - Costs added to trade execution

    Example:
        >>> broker = Broker(slippage_bps=5)
        >>> order = Order("SH600000", "BUY", 100, 10.5)
        >>> trade = broker.submit(order, execution_date, prices)
        >>> trade.price
        10.5005  # With slippage
    """

    def __init__(
        self,
        slippage_bps: float = 0.0,  # Basis points
        execution_mode: str = "next_open",
        costs_model: Any | None = None,  # CostsModel from costs.py
    ) -> None:
        """
        Initialize broker.

        Args:
            slippage_bps: Slippage in basis points (0.01 = 1 bps = 0.0001)
            execution_mode: Default execution mode
            costs_model: Transaction costs model
        """
        self.slippage_bps = slippage_bps
        self.default_execution_mode = execution_mode
        self.costs_model = costs_model

        # Order and trade tracking
        self._pending_orders: list[Order] = []
        self._executed_trades: list[Trade] = []
        self._rejected_orders: list[OrderRejection] = []

    # =========================================================================
    # Order Submission
    # =========================================================================

    def submit(
        self,
        signal: Any,  # Signal from strategy
        shares: float | None = None,
        execution_mode: str | None = None,
        limit_price: float | None = None,
    ) -> Order:
        """
        Submit order from signal.

        Creates order and adds to pending queue.
        Does not execute immediately - execution happens on next bar.

        Args:
            signal: Signal from strategy
            shares: Number of shares (default: from signal or 100)
            execution_mode: Execution mode override
            limit_price: Limit price for limit orders

        Returns:
            Order object
        """
        # Default shares from signal price or standard size
        order_shares = shares or 100

        order = Order(
            instrument_id=signal.instrument_id,
            ticker=signal.ticker,
            direction=signal.direction,
            shares=order_shares,
            signal_price=signal.price,
            limit_price=limit_price,
            execution_mode=execution_mode or self.default_execution_mode,
            timestamp=signal.timestamp,
            signal_note=signal.note,
            trigger_value=signal.trigger_value,
        )

        self._pending_orders.append(order)
        return order

    def pending_orders(self) -> list[Order]:
        """Get pending orders."""
        return self._pending_orders

    def clear_pending(self) -> None:
        """Clear pending orders."""
        self._pending_orders.clear()

    # =========================================================================
    # Order Execution
    # =========================================================================

    def execute_orders(
        self,
        execution_date: date,
        prices_df: pl.DataFrame,
        sector: str | None = None,
    ) -> list[Trade]:
        """
        Execute all pending orders.

        Uses prices_df to get execution prices for the date.

        Args:
            execution_date: Date of execution
            prices_df: DataFrame with prices for the date
            sector: Sector classification (for costs)

        Returns:
            List of executed trades
        """
        executed: list[Trade] = []
        pending = self._pending_orders.copy()
        self._pending_orders.clear()

        for order in pending:
            trade = self._execute_order(order, execution_date, prices_df, sector)
            if trade is not None:
                executed.append(trade)
                self._executed_trades.append(trade)
            else:
                rejection = OrderRejection(
                    order=order,
                    reason="No price data available",
                    details={"date": execution_date.isoformat()},
                )
                self._rejected_orders.append(rejection)

        return executed

    def _execute_order(
        self,
        order: Order,
        execution_date: date,
        prices_df: pl.DataFrame,
        sector: str | None = None,
    ) -> Trade | None:
        """
        Execute a single order.

        Args:
            order: Order to execute
            execution_date: Execution date
            prices_df: DataFrame with prices
            sector: Sector classification

        Returns:
            Trade or None if cannot execute
        """
        # Get prices for instrument at execution date
        instrument_prices = prices_df.filter(
            (pl.col("instrument_id") == order.instrument_id)
            & (pl.col("date") == execution_date)
        )

        if instrument_prices.is_empty():
            return None

        # Determine execution price
        execution_price = self._get_execution_price(
            order, instrument_prices, execution_date
        )

        if execution_price is None:
            return None

        # Apply slippage
        slippage_amount = self._calculate_slippage(execution_price, order.direction)
        final_price = execution_price + slippage_amount

        # Calculate costs
        cost = self._calculate_costs(order, final_price, sector)

        # Create trade
        trade = Trade(
            instrument_id=order.instrument_id,
            ticker=order.ticker,
            direction=order.direction,
            shares=order.shares,
            price=final_price,
            signal_price=order.signal_price,
            slippage=slippage_amount,
            cost=cost,
            execution_date=execution_date,
            order_id=order.order_id,
            note=f"Executed at {final_price:.4f} (slippage: {slippage_amount:.4f})",
        )

        return trade

    def _get_execution_price(
        self,
        order: Order,
        prices_df: pl.DataFrame,
        execution_date: date,
    ) -> float | None:
        """
        Determine execution price based on mode.

        Args:
            order: Order to execute
            prices_df: Filtered DataFrame for instrument/date
            execution_date: Execution date

        Returns:
            Execution price or None if limit not reached
        """
        row = prices_df.row(0, named=True)

        if order.execution_mode == "next_open":
            # Execute at open price
            return row.get("open", row.get("close"))

        elif order.execution_mode == "same_close":
            # Execute at close price
            return row.get("close")

        elif order.execution_mode == "limit":
            # Check if limit price was reached
            high = row.get("high", row.get("close"))
            low = row.get("low", row.get("close"))
            close = row.get("close")

            if order.limit_price is None:
                return close  # No limit, use close

            if order.is_buy():
                # Buy limit: execute if price dropped to limit
                if low <= order.limit_price:
                    return order.limit_price
                return None  # Limit not reached
            else:
                # Sell limit: execute if price rose to limit
                if high >= order.limit_price:
                    return order.limit_price
                return None  # Limit not reached

        # Default to close
        return row.get("close")

    def _calculate_slippage(self, price: float, direction: str) -> float:
        """
        Calculate slippage amount.

        BUY: Slippage increases price (pay more)
        SELL: Slippage decreases price (receive less)

        Args:
            price: Base price
            direction: "BUY" or "SELL"

        Returns:
            Slippage amount
        """
        if self.slippage_bps == 0:
            return 0.0

        # Convert bps to decimal (1 bps = 0.0001)
        slippage_pct = self.slippage_bps * 0.0001
        slippage_amount = price * slippage_pct

        if direction == "BUY":
            return slippage_amount  # Pay more
        else:
            return -slippage_amount  # Receive less

    def _calculate_costs(
        self,
        order: Order,
        price: float,
        sector: str | None = None,
    ) -> float:
        """
        Calculate transaction costs.

        Args:
            order: Order to calculate costs for
            price: Execution price
            sector: Sector classification

        Returns:
            Transaction cost
        """
        if self.costs_model is None:
            return 0.0

        trade_value = order.shares * price

        return self.costs_model.calculate(
            direction=order.direction,
            value=trade_value,
        )

    # =========================================================================
    # Trade History
    # =========================================================================

    def executed_trades(self) -> list[Trade]:
        """Get all executed trades."""
        return self._executed_trades

    def rejected_orders(self) -> list[OrderRejection]:
        """Get rejected orders."""
        return self._rejected_orders

    def trade_count(self) -> int:
        """Count executed trades."""
        return len(self._executed_trades)

    def total_volume(self) -> float:
        """Calculate total trading volume."""
        return sum(t.total_value for t in self._executed_trades)

    def total_costs(self) -> float:
        """Calculate total transaction costs."""
        return sum(t.cost for t in self._executed_trades)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def create_order_from_signal(
        self,
        signal: Any,
        shares: float | None = None,
    ) -> Order:
        """
        Create order from signal.

        Args:
            signal: Signal from strategy
            shares: Number of shares

        Returns:
            Order object
        """
        return self.submit(signal, shares)

    def reset(self) -> None:
        """Reset broker state."""
        self._pending_orders.clear()
        self._executed_trades.clear()
        self._rejected_orders.clear()

    def summary(self) -> str:
        """Get broker summary."""
        lines = [
            f"Broker Summary:",
            f"  Slippage (bps): {self.slippage_bps}",
            f"  Execution mode: {self.default_execution_mode}",
            f"  Pending orders: {len(self._pending_orders)}",
            f"  Executed trades: {len(self._executed_trades)}",
            f"  Rejected orders: {len(self._rejected_orders)}",
            f"  Total volume: {self.total_volume():.2f}",
            f"  Total costs: {self.total_costs():.2f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Broker(slippage_bps={self.slippage_bps}, trades={len(self._executed_trades)})"