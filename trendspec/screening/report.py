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
from rich.panel import Panel
from rich.table import Table

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
        signals_path = output_path / f"signals_{self.strategy_name}_{date_str}.csv"

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
        if title == "买入信号" and self.strategy_name == "clenow_momentum":
            return self._create_clenow_buy_table(signals)
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
        """Convert signals to Polars DataFrame, schema varies by strategy."""
        if not self.signals:
            return pl.DataFrame()

        if self.strategy_name == "clenow_momentum":
            return self._signals_to_clenow_dataframe()

        # Original 7-column schema
        records = []
        for signal in self.signals:
            records.append({
                "股票代码": signal.ticker,
                "instrument_id": signal.instrument_id,
                "日期": self.screening_date.isoformat(),
                "方向": signal.direction,
                "价格": signal.price,
                "触发指标值": signal.trigger_value,
                "备注": signal.note or "",
            })
        return pl.DataFrame(records)

    def _signals_to_clenow_dataframe(self) -> pl.DataFrame:
        """13-column schema: BUY rows fully populated, SELL rows blank display cols."""
        records = []
        for s in self.signals:
            if s.is_buy():
                e = s.extras or {}
                alerts = e.get("alerts") or []
                note = "[警报] " + "，".join(alerts) if alerts else "正常"
                records.append({
                    "股票代码": s.ticker,
                    "instrument_id": s.instrument_id,
                    "日期": self.screening_date.isoformat(),
                    "方向": "BUY",
                    "行业": e.get("sector") or "",
                    "选股排名": e.get("rank"),
                    "建议买入价": s.price,
                    "初始止损线": e.get("stop_loss"),
                    "趋势质量 (R²)": f"{e.get('r2', 0.0):.4f}",
                    "乖离率 (距 MA200)": f"{e.get('deviation_pct', 0.0):.2f}",
                    "回撤 (距 63 日高点)": f"{e.get('drawdown_pct', 0.0):.2f}",
                    "放量倍数": f"{e.get('vol_mult', 0.0):.4f}",
                    "备注/预警": note,
                })
            else:
                records.append({
                    "股票代码": s.ticker,
                    "instrument_id": s.instrument_id,
                    "日期": self.screening_date.isoformat(),
                    "方向": "SELL",
                    "行业": "",
                    "选股排名": None,
                    "建议买入价": s.price,
                    "初始止损线": None,
                    "趋势质量 (R²)": "",
                    "乖离率 (距 MA200)": "",
                    "回撤 (距 63 日高点)": "",
                    "放量倍数": "",
                    "备注/预警": s.note or "",
                })
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

    @staticmethod
    def _r2_label(r2: float) -> str:
        if r2 >= 0.85:
            return "极平稳"
        if r2 >= 0.75:
            return "优秀"
        if r2 >= 0.65:
            return "良好"
        return "一般"

    def _iter_clenow_buy_rows(self, signals: list[Any]):
        """Yield formatted row tuples (10 items) for clenow BUY signals."""
        for s in signals:
            e = s.extras or {}
            sector = e.get("sector") or "-"
            rank = e.get("rank")
            r2 = e.get("r2", 0.0)
            deviation = e.get("deviation_pct", 0.0)
            drawdown = e.get("drawdown_pct", 0.0)
            vol_mult = e.get("vol_mult", 0.0)
            stop_loss = e.get("stop_loss", 0.0)
            alerts = e.get("alerts") or []
            note = "[警报] " + "，".join(alerts) if alerts else "正常"
            yield (
                s.ticker,
                sector,
                f"#{rank}" if rank is not None else "-",
                f"${s.price:.2f}",
                f"${stop_loss:.2f}",
                f"{r2:.2f} ({self._r2_label(r2)})",
                f"{deviation:+.1f}%",
                f"{drawdown:+.1f}%",
                f"{vol_mult:.1f}x",
                note,
            )

    def _create_clenow_buy_table(self, signals: list[Any]) -> Table:
        table = Table(title="买入信号", show_header=True, header_style="bold green")
        table.add_column("股票代码", style="cyan")
        table.add_column("行业", style="cyan")
        table.add_column("选股排名", style="magenta")
        table.add_column("建议买入价", style="green")
        table.add_column("初始止损线", style="red")
        table.add_column("趋势质量 (R²)", style="blue")
        table.add_column("乖离率 (距 MA200)", style="yellow")
        table.add_column("回撤 (距 63 日高点)", style="yellow")
        table.add_column("放量倍数", style="blue")
        table.add_column("备注/预警", style="white")

        for row, s in zip(self._iter_clenow_buy_rows(signals), signals, strict=True):
            alerts = (s.extras or {}).get("alerts") or []
            style = "red" if alerts else "white"
            table.add_row(*row, style=style)
        return table
