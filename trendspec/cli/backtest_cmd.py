"""
Backtest command for TrendSpec CLI.

Run backtest with strategy and output Chinese report.

Command:
    trendspec backtest --strategy ma_cross --market cn --start 2020-01-01 --end 2024-12-31
"""

from datetime import date
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="运行回测")
console = Console()


@app.command("run")
def backtest_run(
    strategy: str = typer.Option(
        ...,
        "--strategy",
        "-s",
        help="策略名称",
    ),
    market: str = typer.Option(
        "cn",
        "--market",
        "-m",
        help="市场代码 (cn, us)",
    ),
    start: str = typer.Option(
        "2020-01-01",
        "--start",
        help="起始日期 (YYYY-MM-DD)",
    ),
    end: str = typer.Option(
        "2024-12-31",
        "--end",
        help="结束日期 (YYYY-MM-DD)",
    ),
    capital: float = typer.Option(
        100000.0,
        "--capital",
        "-c",
        help="初始资金",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="输出目录",
    ),
) -> None:
    """
    运行回测并输出中文报告.

    加载策略类，运行回测引擎，输出绩效报告.

    示例:
        trendspec backtest run --strategy ma_cross --market cn --start 2020-01-01 --end 2024-12-31
    """
    from trendspec.data.markets import Market
    from trendspec.engine.base_engine import EngineConfig
    from trendspec.engine.backtest_engine import BacktestEngine
    from trendspec.strategy.base import get_strategy, create_strategy
    from trendspec.analyzer.report import BacktestReport
    import trendspec.strategy.examples  # noqa: F401 — triggers @register_strategy decorators

    # Parse dates
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError:
        console.print("[red]日期格式错误，请使用 YYYY-MM-DD 格式[/red]")
        raise typer.Exit(1)

    # Get market
    try:
        market_enum = Market(market.upper())
    except ValueError:
        console.print(f"[red]不支持的市场: {market}[/red]")
        raise typer.Exit(1)

    # Load strategy
    strategy_class = get_strategy(strategy)
    if strategy_class is None:
        console.print(f"[red]未找到策略: {strategy}[/red]")
        console.print("[yellow]可用策略列表:[/yellow]")
        from trendspec.strategy.base import list_strategies
        for name in list_strategies():
            console.print(f"  - {name}")
        raise typer.Exit(1)

    console.print(f"[cyan]运行回测[/cyan]")
    console.print(f"  策略: {strategy}")
    console.print(f"  市场: {market}")
    console.print(f"  日期范围: {start_date} 至 {end_date}")
    console.print(f"  初始资金: {capital:,.2f}")

    try:
        # Create config
        config = EngineConfig(
            market=market_enum,
            start_date=start_date,
            end_date=end_date,
            initial_capital=capital,
        )

        # Create engine
        engine = BacktestEngine(config)

        # Run backtest
        result = engine.run(strategy_class)

        # Create report
        report = BacktestReport(
            equity_curve=result.equity_curve,
            trades=result.trades,
            initial_capital=capital,
            strategy_name=strategy,
            date_range=(start_date, end_date),
            market=market,
        )

        # Output to terminal
        report.output()

        # Export to files
        output_path = report.export(output)
        console.print(f"\n[green]报告已保存至: {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]回测失败: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command("list")
def backtest_list() -> None:
    """
    列出可用策略.

    显示所有注册的策略名称.
    """
    from trendspec.strategy.base import list_strategies
    from rich.table import Table
    import trendspec.strategy.examples  # noqa: F401

    strategies = list_strategies()

    table = Table(title="可用策略")
    table.add_column("策略名称", style="cyan")

    if strategies:
        for name in strategies:
            table.add_row(name)
    else:
        table.add_row("(无注册策略)")

    console.print(table)
    console.print("\n[yellow]提示: 使用 --strategy 参数选择策略[/yellow]")

@app.command("compare")
def backtest_compare(
    market: str = typer.Option("cn", "--market", "-m", help="市场 (cn, us)"),
    start: str = typer.Option("2020-01-01", "--start", help="起始日期"),
    end: str = typer.Option("2024-12-31", "--end", help="结束日期"),
    capital: float = typer.Option(100000.0, "--capital", "-c", help="初始资金"),
    sort: str = typer.Option("sharpe", "--sort", help="排序: return|annual|mdd|sharpe|trades"),
    export: Optional[str] = typer.Option(None, "--export", help="导出: csv|json|markdown"),
    exclude: Optional[str] = typer.Option(None, "--exclude", help="排除策略(逗号分隔)"),
) -> None:
    """运行全部策略回测并对比绩效."""
    import time
    from trendspec.data.markets import Market
    from trendspec.engine.base_engine import EngineConfig
    from trendspec.engine.backtest_engine import BacktestEngine
    from trendspec.strategy.base import get_strategy, list_strategies
    from trendspec.analyzer.strategy_comparison import ComparisonRow, ComparisonReport
    import trendspec.strategy.examples  # noqa: F401

    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
        market_enum = Market(market.upper())
    except ValueError as e:
        console.print(f"[red]参数错误: {e}[/red]")
        raise typer.Exit(1)

    excluded = {x.strip() for x in (exclude or "").split(",") if x.strip()}
    strategy_names = [n for n in list_strategies() if n not in excluded]

    console.print(f"[cyan]对比 {len(strategy_names)} 个策略 — {market.upper()} "
                  f"{start_date} → {end_date}[/cyan]\n")

    rows: list[ComparisonRow] = []
    for name in strategy_names:
        strategy_class = get_strategy(name)
        if strategy_class is None:
            continue
        console.print(f"  运行 [cyan]{name}[/cyan]...")
        t0 = time.perf_counter()
        try:
            config = EngineConfig(
                market=market_enum,
                start_date=start_date,
                end_date=end_date,
                initial_capital=capital,
            )
            result = BacktestEngine(config).run(strategy_class)
            m = result.metrics
            elapsed = time.perf_counter() - t0
            rows.append(ComparisonRow(
                strategy_name=name,
                total_return=m.get("total_return", 0.0),
                annualized_return=m.get("annualized_return", 0.0),
                max_drawdown=m.get("max_drawdown", 0.0),
                sharpe_ratio=m.get("sharpe_ratio", 0.0),
                total_trades=m.get("total_trades", 0),
                final_nav=m.get("final_nav", capital),
                elapsed_seconds=elapsed,
            ))
        except Exception as e:
            elapsed = time.perf_counter() - t0
            rows.append(ComparisonRow(
                strategy_name=name, total_return=0, annualized_return=0,
                max_drawdown=0, sharpe_ratio=0, total_trades=0,
                final_nav=0, elapsed_seconds=elapsed, error=str(e),
            ))

    report = ComparisonReport(rows, market, (start_date, end_date))
    report.output(sort_key=sort)

    if export:
        path = report.export(export)
        console.print(f"\n[green]已导出至: {path}[/green]")
