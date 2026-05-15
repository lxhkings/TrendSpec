"""
Performance metrics calculation for TrendSpec.

Calculates standard trading performance metrics with Chinese output names.
All metrics are calculated from equity curve and trade data.

Metrics calculated:
- Total return (总收益率)
- Annualized return (年化收益率)
- Max drawdown (最大回撤)
- Drawdown duration (回撤持续时间)
- Sharpe ratio (夏普比率)
- Win rate (胜率)
- Trade count (交易次数)
- Profit/Loss ratio (盈亏比)
"""

from dataclasses import dataclass
from datetime import date
from typing import Any
import statistics


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for backtest results.

    All values are in decimal form (e.g., 0.25 = 25%).
    Chinese names are provided for output formatting.

    Attributes:
        total_return: Total return (总收益率)
        annualized_return: Annualized return (年化收益率)
        max_drawdown: Maximum drawdown (最大回撤)
        drawdown_duration_days: Max drawdown duration in days (回撤持续天数)
        sharpe_ratio: Sharpe ratio (夏普比率)
        win_rate: Win rate (胜率)
        total_trades: Total trade count (交易次数)
        profit_loss_ratio: Avg profit / avg loss ratio (盈亏比)
        avg_profit: Average profit per winning trade (平均盈利)
        avg_loss: Average loss per losing trade (平均亏损)
        initial_capital: Initial capital (初始资金)
        final_nav: Final NAV (最终净值)
        trading_days: Number of trading days (交易日数)
        total_costs: Total transaction costs (总交易成本)
    """

    total_return: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration_days: int = 0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_loss_ratio: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    initial_capital: float = 0.0
    final_nav: float = 0.0
    trading_days: int = 0
    total_costs: float = 0.0

    # Chinese names for output
    CHINESE_NAMES = {
        "total_return": "总收益率",
        "annualized_return": "年化收益率",
        "max_drawdown": "最大回撤",
        "drawdown_duration_days": "回撤持续天数",
        "sharpe_ratio": "夏普比率",
        "win_rate": "胜率",
        "total_trades": "交易次数",
        "profit_loss_ratio": "盈亏比",
        "avg_profit": "平均盈利",
        "avg_loss": "平均亏损",
        "initial_capital": "初始资金",
        "final_nav": "最终净值",
        "trading_days": "交易日数",
        "total_costs": "总交易成本",
    }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "max_drawdown": self.max_drawdown,
            "drawdown_duration_days": self.drawdown_duration_days,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "profit_loss_ratio": self.profit_loss_ratio,
            "avg_profit": self.avg_profit,
            "avg_loss": self.avg_loss,
            "initial_capital": self.initial_capital,
            "final_nav": self.final_nav,
            "trading_days": self.trading_days,
            "total_costs": self.total_costs,
        }

    def to_chinese_dict(self) -> dict[str, Any]:
        """Convert to dictionary with Chinese keys."""
        return {
            self.CHINESE_NAMES.get(k, k): v
            for k, v in self.to_dict().items()
        }

    def format_percentage(self, value: float) -> str:
        """Format value as percentage string."""
        return f"{value:.2%}"

    def format_money(self, value: float) -> str:
        """Format value as money string."""
        return f"{value:,.2f}"

    def format_number(self, value: float | int) -> str:
        """Format value as number."""
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)


def calculate_metrics(
    equity_curve: list[Any],
    trades: list[Any],
    initial_capital: float,
    risk_free_rate: float = 0.03,
    trading_days_per_year: int = 252,
) -> PerformanceMetrics:
    """
    Calculate performance metrics from equity curve and trades.

    Args:
        equity_curve: List of EquityCurvePoint objects
        trades: List of Trade objects
        initial_capital: Initial capital
        risk_free_rate: Risk-free rate for Sharpe (default 3%)
        trading_days_per_year: Trading days per year (default 252)

    Returns:
        PerformanceMetrics object
    """
    metrics = PerformanceMetrics(
        initial_capital=initial_capital,
    )

    if not equity_curve:
        return metrics

    # Basic stats
    trading_days = len(equity_curve)
    final_nav = equity_curve[-1].nav
    metrics.trading_days = trading_days
    metrics.final_nav = final_nav

    # Total return
    if initial_capital > 0:
        total_return = (final_nav - initial_capital) / initial_capital
        metrics.total_return = total_return

    # Annualized return
    if trading_days > 0:
        years = trading_days / trading_days_per_year
        if years > 0 and total_return != 0:
            metrics.annualized_return = (1 + total_return) ** (1 / years) - 1

    # Max drawdown and duration
    peak = initial_capital
    peak_date = None
    max_dd = 0.0
    dd_start_date = None
    dd_duration = 0
    current_dd_start = None

    for point in equity_curve:
        if point.nav > peak:
            peak = point.nav
            peak_date = point.date
            current_dd_start = None
        else:
            dd = (peak - point.nav) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
                dd_start_date = peak_date
                current_dd_start = point.date

            if current_dd_start:
                # Count days in drawdown
                duration = (point.date - current_dd_start).days
                if duration > dd_duration:
                    dd_duration = duration

    metrics.max_drawdown = max_dd
    metrics.drawdown_duration_days = dd_duration

    # Sharpe ratio
    if len(equity_curve) > 10:
        daily_returns = [p.daily_return for p in equity_curve if p.daily_return != 0]
        if len(daily_returns) > 10:
            avg_return = statistics.mean(daily_returns)
            std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0.0

            if std_return > 0:
                # Annualized Sharpe
                excess_return = avg_return * trading_days_per_year - risk_free_rate
                annualized_std = std_return * (trading_days_per_year ** 0.5)
                metrics.sharpe_ratio = excess_return / annualized_std

    # Trade statistics
    metrics.total_trades = len(trades)
    metrics.total_costs = sum(t.cost for t in trades) if trades else 0.0

    if trades:
        # Calculate realized P&L per trade (simplified)
        winning_trades = []
        losing_trades = []

        for trade in trades:
            if trade.is_sell():
                # Estimate P&L (would need avg_cost for exact)
                # For simplicity, use signal_price vs execution_price
                pnl_estimate = (trade.price - trade.signal_price) * trade.shares
                if pnl_estimate > 0:
                    winning_trades.append(pnl_estimate)
                else:
                    losing_trades.append(abs(pnl_estimate))

        if winning_trades or losing_trades:
            total_wins = len(winning_trades)
            total_losses = len(losing_trades)
            total_closed = total_wins + total_losses

            if total_closed > 0:
                metrics.win_rate = total_wins / total_closed

            if winning_trades:
                metrics.avg_profit = statistics.mean(winning_trades)

            if losing_trades:
                metrics.avg_loss = statistics.mean(losing_trades)

            if metrics.avg_loss > 0 and metrics.avg_profit > 0:
                metrics.profit_loss_ratio = metrics.avg_profit / metrics.avg_loss

    return metrics