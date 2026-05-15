"""
Trade log analysis for TrendSpec.

Provides trade statistics and summary from broker trade records.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl


@dataclass
class TradeSummary:
    """
    Trade summary statistics.

    Attributes:
        total_trades: Total number of trades
        buy_trades: Number of buy trades
        sell_trades: Number of sell trades
        total_volume: Total trading volume
        total_costs: Total transaction costs
        avg_trade_size: Average trade size
        largest_trade: Largest trade value
        smallest_trade: Smallest trade value
    """

    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    total_volume: float = 0.0
    total_costs: float = 0.0
    avg_trade_size: float = 0.0
    largest_trade: float = 0.0
    smallest_trade: float = 0.0


class TradeLogAnalyzer:
    """
    Trade log analysis class.

    Provides winning/losing trade analysis and summary from broker trades.

    Attributes:
        trades: List of Trade objects from broker

    Example:
        >>> analyzer = TradeLogAnalyzer(trades)
        >>> summary = analyzer.trade_summary()
        >>> winners = analyzer.winning_trades(avg_costs)
        >>> losers = analyzer.losing_trades(avg_costs)
    """

    def __init__(
        self,
        trades: list[Any],  # List of Trade objects from broker
        position_costs: dict[str, float] | None = None,  # instrument_id -> avg_cost
    ) -> None:
        """
        Initialize trade log analyzer.

        Args:
            trades: List of Trade objects
            position_costs: Dict of instrument_id -> average cost for P&L calculation
        """
        self.trades = trades
        self._position_costs = position_costs or {}

    def winning_trades(self) -> list[Any]:
        """
        Get winning trades.

        Winning trades are sells where price > average cost.

        Returns:
            List of winning Trade objects
        """
        winners: list[Any] = []

        for trade in self.trades:
            if trade.is_sell():
                avg_cost = self._position_costs.get(trade.instrument_id, trade.signal_price)
                if trade.price > avg_cost:
                    winners.append(trade)

        return winners

    def losing_trades(self) -> list[Any]:
        """
        Get losing trades.

        Losing trades are sells where price < average cost.

        Returns:
            List of losing Trade objects
        """
        losers: list[Any] = []

        for trade in self.trades:
            if trade.is_sell():
                avg_cost = self._position_costs.get(trade.instrument_id, trade.signal_price)
                if trade.price < avg_cost:
                    losers.append(trade)

        return losers

    def buy_trades(self) -> list[Any]:
        """
        Get all buy trades.

        Returns:
            List of buy Trade objects
        """
        return [t for t in self.trades if t.is_buy()]

    def sell_trades(self) -> list[Any]:
        """
        Get all sell trades.

        Returns:
            List of sell Trade objects
        """
        return [t for t in self.trades if t.is_sell()]

    def trade_summary(self) -> TradeSummary:
        """
        Get trade summary statistics.

        Returns:
            TradeSummary object
        """
        summary = TradeSummary()

        if not self.trades:
            return summary

        summary.total_trades = len(self.trades)
        summary.buy_trades = len(self.buy_trades())
        summary.sell_trades = len(self.sell_trades())

        trade_values = [t.total_value for t in self.trades]

        summary.total_volume = sum(trade_values)
        summary.total_costs = sum(t.cost for t in self.trades)

        if trade_values:
            summary.avg_trade_size = sum(trade_values) / len(trade_values)
            summary.largest_trade = max(trade_values)
            summary.smallest_trade = min(trade_values)

        return summary

    def trade_pnl(self) -> dict[str, float]:
        """
        Calculate realized P&L per instrument.

        Returns:
            Dict of instrument_id -> realized P&L
        """
        pnl: dict[str, float] = {}

        for trade in self.trades:
            if trade.instrument_id not in pnl:
                pnl[trade.instrument_id] = 0.0

            if trade.is_sell():
                avg_cost = self._position_costs.get(trade.instrument_id, trade.signal_price)
                realized_pnl = (trade.price - avg_cost) * trade.shares
                pnl[trade.instrument_id] += realized_pnl

        return pnl

    def trades_by_instrument(self) -> dict[str, list[Any]]:
        """
        Group trades by instrument.

        Returns:
            Dict of instrument_id -> list of trades
        """
        grouped: dict[str, list[Any]] = {}

        for trade in self.trades:
            if trade.instrument_id not in grouped:
                grouped[trade.instrument_id] = []
            grouped[trade.instrument_id].append(trade)

        return grouped

    def trades_by_date(self) -> dict[date, list[Any]]:
        """
        Group trades by date.

        Returns:
            Dict of date -> list of trades
        """
        grouped: dict[date, list[Any]] = {}

        for trade in self.trades:
            exec_date = trade.execution_date
            if exec_date:
                if exec_date not in grouped:
                    grouped[exec_date] = []
                grouped[exec_date].append(trade)

        return grouped

    def to_dataframe(self) -> pl.DataFrame:
        """
        Convert trades to Polars DataFrame.

        Returns:
            DataFrame with trade data
        """
        if not self.trades:
            return pl.DataFrame()

        records = []
        for trade in self.trades:
            record = {
                "trade_id": trade.trade_id,
                "instrument_id": trade.instrument_id,
                "ticker": trade.ticker,
                "direction": trade.direction,
                "shares": trade.shares,
                "price": trade.price,
                "signal_price": trade.signal_price,
                "slippage": trade.slippage,
                "cost": trade.cost,
                "total_value": trade.total_value,
                "execution_date": trade.execution_date.isoformat() if trade.execution_date else None,
                "note": trade.note,
            }
            records.append(record)

        return pl.DataFrame(records)

    def summary_dict(self) -> dict[str, Any]:
        """
        Get summary as dictionary with Chinese names.

        Returns:
            Dict with Chinese keys
        """
        summary = self.trade_summary()

        CHINESE_NAMES = {
            "total_trades": "交易次数",
            "buy_trades": "买入次数",
            "sell_trades": "卖出次数",
            "total_volume": "总成交额",
            "total_costs": "总交易成本",
            "avg_trade_size": "平均交易规模",
            "largest_trade": "最大交易",
            "smallest_trade": "最小交易",
        }

        return {
            CHINESE_NAMES.get(k, k): v
            for k, v in {
                "total_trades": summary.total_trades,
                "buy_trades": summary.buy_trades,
                "sell_trades": summary.sell_trades,
                "total_volume": summary.total_volume,
                "total_costs": summary.total_costs,
                "avg_trade_size": summary.avg_trade_size,
                "largest_trade": summary.largest_trade,
                "smallest_trade": summary.smallest_trade,
            }.items()
        }