"""
Screening report with Chinese output for TrendSpec.

Generates rich.Table output for terminal and exports signals to CSV.
Exports: signals_YYYYMMDD.csv
Path: results/screening/
"""

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from trendspec.config.settings import get_settings


class ScreeningReport:
    """
    Screening report with Chinese output formatting.

    Generates:
    - Terminal output with rich.Table (Chinese column names)
    - signals_YYYYMMDD.csv file

    Output path: results/screening/

    Example:
        >>> report = ScreeningReport(
        ...     signals=signals,
        ...     screening_date=date(2024, 5, 15),
        ...     strategy_name="ma_cross",
        ...     market="CN",
        ... )
        >>> report.output()
        >>> report.export()
    """

    def __init__(
        self,
        signals: list[Any],
        screening_date: date,
        strategy_name: str = "unknown",
        market: str = "CN",
        universe_size: int = 0,
    ) -> None:
        """
        Initialize screening report.

        Args:
            signals: List of Signal objects
            screening_date: Date screened
            strategy_name: Strategy name
            market: Market code
            universe_size: Size of universe at screening date
        """
        self.signals = signals
        self.screening_date = screening_date
        self.strategy_name = strategy_name
        self.market = market
        self.universe_size = universe_size

        # Console for output
        self._console = Console()

    def output(self) -> None:
        """
        Output report to terminal.

        Prints signals table with Chinese column names.
        """
        self._console.print(self._create_header())

        buy_signals = [s for s in self.signals if s.is_buy()]
        sell_signals = [s for s in self.signals if s.is_sell()]

        if buy_signals:
            self._console.print(self._create_signals_table(buy_signals, "买入信号"))

        if sell_signals:
            self._console.print(self._create_signals_table(sell_signals, "卖出信号"))

        if not buy_signals and not sell_signals:
            self._console.print("[yellow]未发现信号[/yellow]")

    def export(self, output_dir: str | Path | None = None) -> Path:
        """
        Export signals to CSV.

        Creates: signals_YYYYMMDD.csv

        Args:
            output_dir: Output directory (default: results/screening/)

        Returns:
            Path to output directory
        """
        if output_dir is None:
            settings = get_settings()
            base_path = Path(settings.data_lake.data_lake_root).parent / "results" / "screening"
            output_dir = base_path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        date_str = self.screening_date.strftime("%Y%m%d")
        signals_path = output_path / f"signals_{date_str}.csv"

        # Export to CSV
        df = self._signals_to_dataframe()
        if not df.is_empty():
            df.write_csv(signals_path)

        return signals_path

    def _create_header(self) -> Panel:
        """Create report header panel."""
        title = f"选股报告 - {self.strategy_name}"
        content = f"""
策略: {self.strategy_name}
市场: {self.market}
日期: {self.screening_date}
股票池大小: {self.universe_size}
信号总数: {len(self.signals)}
买入信号: {len([s for s in self.signals if s.is_buy()])}
卖出信号: {len([s for s in self.signals if s.is_sell()])}
        """

        return Panel(content.strip(), title=title, border_style="green")

    def _create_signals_table(self, signals: list[Any], title: str) -> Table:
        """Create signals table with Chinese column names."""
        table = Table(title=title, show_header=True, header_style="bold green")

        # Chinese column names
        table.add_column("股票代码", style="cyan")
        table.add_column("日期", style="cyan")
        table.add_column("方向", style="yellow")
        table.add_column("价格", style="green")
        table.add_column("触发指标值", style="blue")
        table.add_column("备注", style="white")

        for signal in signals:
            table.add_row(
                signal.ticker,
                self.screening_date.isoformat(),
                signal.direction,
                f"{signal.price:.2f}",
                f"{signal.trigger_value:.2f}" if signal.trigger_value else "N/A",
                signal.note or "",
            )

        return table

    def _signals_to_dataframe(self) -> pl.DataFrame:
        """Convert signals to Polars DataFrame."""
        if not self.signals:
            return pl.DataFrame()

        records = []
        for signal in self.signals:
            record = {
                "股票代码": signal.ticker,
                "instrument_id": signal.instrument_id,
                "日期": self.screening_date.isoformat(),
                "方向": signal.direction,
                "价格": signal.price,
                "触发指标值": signal.trigger_value,
                "备注": signal.note or "",
            }
            records.append(record)

        return pl.DataFrame(records)

    def buy_signals(self) -> list[Any]:
        """Get buy signals."""
        return [s for s in self.signals if s.is_buy()]

    def sell_signals(self) -> list[Any]:
        """Get sell signals."""
        return [s for s in self.signals if s.is_sell()]

    def buy_count(self) -> int:
        """Count buy signals."""
        return len(self.buy_signals())

    def sell_count(self) -> int:
        """Count sell signals."""
        return len(self.sell_signals())

    def to_dict(self) -> dict[str, Any]:
        """Get report as dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "market": self.market,
            "screening_date": self.screening_date.isoformat(),
            "universe_size": self.universe_size,
            "total_signals": len(self.signals),
            "buy_signals": self.buy_count(),
            "sell_signals": self.sell_count(),
        }

    def summary(self) -> str:
        """Get summary string."""
        return (
            f"Screening Report: {self.strategy_name} @ {self.screening_date}\n"
            f"  Market: {self.market}\n"
            f"  Universe size: {self.universe_size}\n"
            f"  Buy signals: {self.buy_count()}\n"
            f"  Sell signals: {self.sell_count()}"
        )