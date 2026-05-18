"""
Screen command for TrendSpec CLI.

Run screening with strategy and output Chinese signal table.

Command:
    trendspec screen --strategy ma_cross --market cn --date 2024-05-15
"""

from datetime import date
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="运行选股")
console = Console()


@app.command("run")
def screen_run(
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
    date_str: Optional[str] = typer.Option(
        None,
        "--date",
        "-d",
        help="筛选日期 (YYYY-MM-DD，默认为今日)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="输出目录",
    ),
) -> None:
    """
    运行选股并输出信号列表.

    加载策略类，运行选股引擎，输出买入/卖出信号.

    示例:
        trendspec screen run --strategy ma_cross --market cn --date 2024-05-15
        trendspec screen run --strategy ma_cross --market cn  # 使用今日日期
    """
    from trendspec.data.markets import Market
    from trendspec.engine.base_engine import EngineConfig
    from trendspec.engine.screening_engine import ScreeningEngine
    from trendspec.strategy.base import get_strategy
    import trendspec.strategy.examples  # noqa: F401 — triggers @register_strategy decorators
    from trendspec.screening.report import ScreeningReport

    # Parse date (default to today)
    if date_str:
        try:
            target_date = date.fromisoformat(date_str)
        except ValueError:
            console.print("[red]日期格式错误，请使用 YYYY-MM-DD 格式[/red]")
            raise typer.Exit(1)
    else:
        target_date = date.today()

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

    console.print(f"[cyan]运行选股[/cyan]")
    console.print(f"  策略: {strategy}")
    console.print(f"  市场: {market}")
    console.print(f"  日期: {target_date}")

    try:
        # Create config
        config = EngineConfig(
            market=market_enum,
            start_date=target_date,
            end_date=target_date,
        )

        # Create engine
        engine = ScreeningEngine(config)

        # Run screening
        result = engine.run(strategy_class)

        # Create report
        report = ScreeningReport(
            signals=result.signals,
            screening_date=target_date,
            strategy_name=strategy,
            market=market,
            universe_size=result.universe_size,
        )

        # Output to terminal
        report.output()

        # Export to files
        if result.signals:
            output_path = report.export(output)
            console.print(f"\n[green]信号已保存至: {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]选股失败: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command("list")
def screen_list() -> None:
    """
    列出可用策略.

    显示所有注册的策略名称.
    """
    from trendspec.strategy.base import list_strategies
    from rich.table import Table

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