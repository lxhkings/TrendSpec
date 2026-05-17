"""
Multi-strategy backtest comparison report.

ComparisonRow holds metrics for one strategy's backtest run.
ComparisonReport renders a sorted rich.Table and supports CSV/JSON/Markdown export.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.table import Table


@dataclass
class ComparisonRow:
    strategy_name: str
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    final_nav: float
    elapsed_seconds: float
    error: str | None = None


class ComparisonReport:
    """Renders comparison of multiple strategy backtests."""

    def __init__(
        self,
        rows: list[ComparisonRow],
        market: str,
        date_range: tuple[date, date],
    ) -> None:
        self.rows = rows
        self.market = market
        self.date_range = date_range

    def _sorted_rows(self, sort_key: str = "sharpe") -> list[ComparisonRow]:
        key_map = {
            "return": lambda r: r.total_return,
            "annual": lambda r: r.annualized_return,
            "mdd": lambda r: -r.max_drawdown,
            "sharpe": lambda r: r.sharpe_ratio,
            "trades": lambda r: r.total_trades,
        }
        fn = key_map.get(sort_key, key_map["sharpe"])
        ok = [r for r in self.rows if r.error is None]
        err = [r for r in self.rows if r.error is not None]
        return sorted(ok, key=fn, reverse=True) + err

    def output(self, sort_key: str = "sharpe", console: Console | None = None) -> None:
        con = console or Console()
        sorted_rows = self._sorted_rows(sort_key)

        title = f"策略回测对比 — {self.market.upper()} {self.date_range[0]} → {self.date_range[1]}"
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("策略", style="cyan", min_width=20)
        table.add_column("总收益", style="green", justify="right")
        table.add_column("年化收益", justify="right")
        table.add_column("最大回撤", style="red", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("交易次数", justify="right")
        table.add_column("耗时(s)", justify="right")

        best = sorted_rows[0].strategy_name if sorted_rows and sorted_rows[0].error is None else None

        for _i, row in enumerate(sorted_rows):
            style = "bold green" if row.strategy_name == best else ""
            if row.error:
                table.add_row(
                    row.strategy_name, "[red]ERROR[/red]", "—", "—", "—", "—", "—",
                    style="dim",
                )
                continue
            table.add_row(
                row.strategy_name,
                f"{row.total_return:+.1%}",
                f"{row.annualized_return:+.1%}",
                f"{row.max_drawdown:.1%}",
                f"{row.sharpe_ratio:.2f}",
                str(row.total_trades),
                f"{row.elapsed_seconds:.1f}",
                style=style,
            )

        con.print(table)
        if best:
            con.print(f"\n[bold green]最优策略: {best} (按 {sort_key} 排序)[/bold green]")

        for row in [r for r in sorted_rows if r.error]:
            con.print(f"[red]  {row.strategy_name}: {row.error}[/red]")

    def export(self, format: str, output_dir: Path | str | None = None) -> Path:
        output_dir = Path(output_dir or "results")
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = date.today().isoformat()
        stem = f"comparison_{self.market}_{ts}"

        if format == "csv":
            path = output_dir / f"{stem}.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["strategy", "total_return", "annualized_return",
                                 "max_drawdown", "sharpe_ratio", "total_trades",
                                 "final_nav", "elapsed_seconds", "error"])
                for r in self._sorted_rows():
                    writer.writerow([r.strategy_name, r.total_return, r.annualized_return,
                                     r.max_drawdown, r.sharpe_ratio, r.total_trades,
                                     r.final_nav, r.elapsed_seconds, r.error or ""])
        elif format == "json":
            path = output_dir / f"{stem}.json"
            data = [vars(r) for r in self._sorted_rows()]
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        elif format == "markdown":
            path = output_dir / f"{stem}.md"
            lines = ["| 策略 | 总收益 | 年化 | 最大回撤 | Sharpe | 交易数 |",
                     "|------|--------|------|---------|--------|--------|"]
            for r in self._sorted_rows():
                if r.error:
                    lines.append(f"| {r.strategy_name} | ERROR | — | — | — | — |")
                else:
                    lines.append(f"| {r.strategy_name} | {r.total_return:+.1%} | "
                                 f"{r.annualized_return:+.1%} | {r.max_drawdown:.1%} | "
                                 f"{r.sharpe_ratio:.2f} | {r.total_trades} |")
            path.write_text("\n".join(lines), encoding="utf-8")
        else:
            raise ValueError(f"Unknown export format: {format}. Use csv, json, or markdown.")

        return path
