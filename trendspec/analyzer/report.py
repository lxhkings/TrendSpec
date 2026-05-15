"""
Backtest report with Chinese output for TrendSpec.

Generates rich.Table output for terminal and exports results to files.
Exports: metrics.json + equity_curve.csv + trades.csv
Path: results/backtest/<run_id>/
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from trendspec.analyzer.metrics import PerformanceMetrics, calculate_metrics
from trendspec.analyzer.equity_curve import EquityCurve
from trendspec.analyzer.trade_log import TradeLogAnalyzer
from trendspec.config.settings import get_settings


class BacktestReport:
    """
    Backtest report with Chinese output formatting.

    Generates:
    - Terminal output with rich.Table (Chinese column names)
    - metrics.json file
    - equity_curve.csv file
    - trades.csv file

    Output path: results/backtest/<run_id>/

    Example:
        >>> report = BacktestReport(
        ...     equity_curve=points,
        ...     trades=trades,
        ...     initial_capital=100000,
        ...     strategy_name="ma_cross",
        ...     date_range=(start, end),
        ... )
        >>> report.output()
        >>> report.export()
    """

    def __init__(
        self,
        equity_curve: list[Any],
        trades: list[Any],
        initial_capital: float,
        strategy_name: str = "unknown",
        date_range: tuple[Any, Any] | None = None,
        market: str = "CN_A",
        position_costs: dict[str, float] | None = None,
        risk_free_rate: float = 0.03,
    ) -> None:
        """
        Initialize backtest report.

        Args:
            equity_curve: List of EquityCurvePoint objects
            trades: List of Trade objects
            initial_capital: Initial capital
            strategy_name: Strategy name
            date_range: (start_date, end_date) tuple
            market: Market code
            position_costs: Dict of instrument_id -> avg_cost for P&L
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.equity_curve = equity_curve
        self.trades = trades
        self.initial_capital = initial_capital
        self.strategy_name = strategy_name
        self.date_range = date_range
        self.market = market
        self._position_costs = position_costs or {}
        self._risk_free_rate = risk_free_rate

        # Generate run ID
        self.run_id = self._generate_run_id()

        # Calculate metrics
        self._metrics = calculate_metrics(
            equity_curve=equity_curve,
            trades=trades,
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate,
        )

        # Analyzers
        self._equity_analyzer = EquityCurve(equity_curve, initial_capital)
        self._trade_analyzer = TradeLogAnalyzer(trades, position_costs)

        # Console for output
        self._console = Console()

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        return f"{self.strategy_name}_{timestamp}_{short_uuid}"

    def output(self) -> None:
        """
        Output report to terminal.

        Prints metrics table, trade summary, and equity curve summary.
        """
        self._console.print(self._create_header())
        self._console.print(self._create_metrics_table())
        self._console.print(self._create_trade_table())
        self._console.print(self._create_equity_summary_table())

    def export(self, output_dir: str | Path | None = None) -> Path:
        """
        Export report to files.

        Creates:
        - metrics.json
        - equity_curve.csv
        - trades.csv

        Args:
            output_dir: Output directory (default: results/backtest/<run_id>)

        Returns:
            Path to output directory
        """
        if output_dir is None:
            # Use settings for default path
            settings = get_settings()
            base_path = Path(settings.data_lake.data_lake_root).parent / "results" / "backtest"
            output_dir = base_path / self.run_id

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export metrics.json
        metrics_path = output_path / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self._metrics.to_chinese_dict(), f, ensure_ascii=False, indent=2)

        # Export equity_curve.csv
        equity_path = output_path / "equity_curve.csv"
        equity_df = self._equity_analyzer.to_dataframe()
        if not equity_df.is_empty():
            equity_df.write_csv(equity_path)

        # Export trades.csv
        trades_path = output_path / "trades.csv"
        trades_df = self._trade_analyzer.to_dataframe()
        if not trades_df.is_empty():
            trades_df.write_csv(trades_path)

        return output_path

    def _create_header(self) -> Panel:
        """Create report header panel."""
        start_date = self.date_range[0] if self.date_range else "N/A"
        end_date = self.date_range[1] if self.date_range else "N/A"

        title = f"回测报告 - {self.strategy_name}"
        content = f"""
策略: {self.strategy_name}
市场: {self.market}
日期范围: {start_date} 至 {end_date}
初始资金: {self.initial_capital:,.2f}
运行ID: {self.run_id}
        """

        return Panel(content.strip(), title=title, border_style="blue")

    def _create_metrics_table(self) -> Table:
        """Create performance metrics table."""
        table = Table(title="绩效指标", show_header=True, header_style="bold cyan")
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="green")

        # Format metrics
        metrics = self._metrics

        table.add_row("总收益率", f"{metrics.total_return:.2%}")
        table.add_row("年化收益率", f"{metrics.annualized_return:.2%}")
        table.add_row("最大回撤", f"{metrics.max_drawdown:.2%}")
        table.add_row("回撤持续天数", f"{metrics.drawdown_duration_days} 天")
        table.add_row("夏普比率", f"{metrics.sharpe_ratio:.2f}")
        table.add_row("胜率", f"{metrics.win_rate:.2%}")
        table.add_row("交易次数", f"{metrics.total_trades}")
        table.add_row("盈亏比", f"{metrics.profit_loss_ratio:.2f}")
        table.add_row("平均盈利", f"{metrics.avg_profit:,.2f}")
        table.add_row("平均亏损", f"{metrics.avg_loss:,.2f}")
        table.add_row("最终净值", f"{metrics.final_nav:,.2f}")
        table.add_row("交易日数", f"{metrics.trading_days}")
        table.add_row("总交易成本", f"{metrics.total_costs:,.2f}")

        return table

    def _create_trade_table(self) -> Table:
        """Create trade summary table."""
        table = Table(title="交易统计", show_header=True, header_style="bold cyan")
        table.add_column("统计项", style="cyan")
        table.add_column("数值", style="green")

        summary = self._trade_analyzer.trade_summary()

        table.add_row("交易次数", f"{summary.total_trades}")
        table.add_row("买入次数", f"{summary.buy_trades}")
        table.add_row("卖出次数", f"{summary.sell_trades}")
        table.add_row("总成交额", f"{summary.total_volume:,.2f}")
        table.add_row("总交易成本", f"{summary.total_costs:,.2f}")
        table.add_row("平均交易规模", f"{summary.avg_trade_size:,.2f}")

        return table

    def _create_equity_summary_table(self) -> Table:
        """Create equity curve summary table."""
        table = Table(title="净值曲线摘要", show_header=True, header_style="bold cyan")
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="green")

        summary = self._equity_analyzer.summary()

        if summary:
            table.add_row("起始日期", summary.get("start_date", "N/A"))
            table.add_row("结束日期", summary.get("end_date", "N/A"))
            table.add_row("起始净值", f"{summary.get('initial_nav', 0):,.2f}")
            table.add_row("最终净值", f"{summary.get('final_nav', 0):,.2f}")
            table.add_row("最高净值", f"{summary.get('max_nav', 0):,.2f}")
            table.add_row("最低净值", f"{summary.get('min_nav', 0):,.2f}")
            table.add_row("最大回撤", f"{summary.get('max_drawdown', 0):.2%}")
            table.add_row("当前回撤", f"{summary.get('current_drawdown', 0):.2%}")

        return table

    def get_metrics(self) -> PerformanceMetrics:
        """Get calculated metrics."""
        return self._metrics

    def get_run_id(self) -> str:
        """Get run ID."""
        return self.run_id

    def to_dict(self) -> dict[str, Any]:
        """Get full report as dictionary."""
        return {
            "run_id": self.run_id,
            "strategy_name": self.strategy_name,
            "market": self.market,
            "date_range": {
                "start": str(self.date_range[0]) if self.date_range else None,
                "end": str(self.date_range[1]) if self.date_range else None,
            },
            "initial_capital": self.initial_capital,
            "metrics": self._metrics.to_dict(),
            "trade_summary": self._trade_analyzer.trade_summary().__dict__,
            "equity_summary": self._equity_analyzer.summary(),
        }