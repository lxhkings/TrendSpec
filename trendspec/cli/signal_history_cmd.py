"""
Signal-history command for TrendSpec CLI.

Build and inspect cached signal history for strategies.

Commands:
    trendspec signal-history build --strategy clenow_momentum --market us [--years 10] [--rebuild]
    trendspec signal-history status --strategy clenow_momentum --market us
"""


import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="构建和查看信号历史缓存")
console = Console()


@app.command("build")
def build_history(
    strategy: str = typer.Option(
        ...,
        "--strategy",
        "-s",
        help="策略名称",
    ),
    market: str = typer.Option(
        "us",
        "--market",
        "-m",
        help="市场代码 (us, cn)",
    ),
    years: int = typer.Option(
        10,
        "--years",
        "-y",
        help="回看年数（默认 10）",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        "-r",
        help="忽略缓存，强制全量重建",
    ),
) -> None:
    """
    构建策略信号历史缓存.

    回看指定年数的交易日，重放策略信号，计算远期收益率并聚合。
    结果缓存为 Parquet 文件，供 screening report 快速查询。

    示例:
        trendspec signal-history build --strategy clenow_momentum --market us
        trendspec signal-history build --strategy clenow_momentum --market us --years 5
        trendspec signal-history build --strategy clenow_momentum --market us --rebuild
    """
    import trendspec.strategy.examples  # noqa: F401 — triggers @register_strategy decorators
    from trendspec.analyzer.signal_history import SignalHistoryBuilder
    from trendspec.data.markets import Market
    from trendspec.strategy.base import get_strategy, list_strategies

    # Validate market
    try:
        market_enum = Market(market.upper())
    except ValueError:
        console.print(f"[red]不支持的市场: {market}（可选: us, cn）[/red]")
        raise typer.Exit(1) from None

    # Validate strategy exists
    strategy_class = get_strategy(strategy)
    if strategy_class is None:
        console.print(f"[red]未找到策略: {strategy}[/red]")
        console.print("[yellow]可用策略:[/yellow]")
        for name in list_strategies():
            console.print(f"  - {name}")
        raise typer.Exit(1) from None

    # Determine if incremental or full
    mode = "全量重建" if rebuild else "增量/首次构建"

    console.print("[cyan]构建信号历史[/cyan]")
    console.print(f"  策略: {strategy}")
    console.print(f"  市场: {market}")
    console.print(f"  回看: {years} 年")
    console.print(f"  模式: {mode}")
    console.print()

    try:
        builder = SignalHistoryBuilder()
        result = builder.build(
            strategy_name=strategy,
            market=market_enum,
            lookback_years=years,
            rebuild=rebuild,
        )

        if result.is_empty():
            console.print("[yellow]未产生任何信号，缓存为空[/yellow]")
            return

        n_instruments = result.height
        total_signals = result["n_signals"].sum()

        console.print()
        console.print("[green]构建完成[/green]")
        console.print(f"  覆盖标的: {n_instruments}")
        console.print(f"  总信号数: {total_signals}")

    except Exception as e:
        console.print(f"[red]构建失败: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(1) from None


@app.command("status")
def status_history(
    strategy: str = typer.Option(
        ...,
        "--strategy",
        "-s",
        help="策略名称",
    ),
    market: str = typer.Option(
        "us",
        "--market",
        "-m",
        help="市场代码 (us, cn)",
    ),
) -> None:
    """
    查看信号历史缓存状态.

    显示缓存是否存在、记录数、最后构建时间、信号日期范围。

    示例:
        trendspec signal-history status --strategy clenow_momentum --market us
    """
    from trendspec.analyzer.signal_history import SignalHistoryStore
    from trendspec.data.markets import Market

    # Validate market
    try:
        market_enum = Market(market.upper())
    except ValueError:
        console.print(f"[red]不支持的市场: {market}（可选: us, cn）[/red]")
        raise typer.Exit(1) from None

    console.print("[cyan]信号历史缓存状态[/cyan]")
    console.print(f"  策略: {strategy}")
    console.print(f"  市场: {market}")
    console.print()

    df = SignalHistoryStore.load(strategy, market_enum)

    if df is None or df.is_empty() or "last_built_at" not in df.columns:
        if df is not None and "last_built_at" not in df.columns:
            console.print("[yellow]缓存格式不匹配，可能需要重建[/yellow]")
        else:
            console.print("[yellow]缓存不存在[/yellow]")
        console.print("\n提示: 使用 build 命令构建缓存")
        console.print(f"  trendspec signal-history build --strategy {strategy} --market {market}")
        return

    n_rows = df.height
    last_built = df["last_built_at"].max()
    last_signal = df["last_signal_date"].max()
    first_signal = df["last_signal_date"].min()

    # Format dates
    if hasattr(last_built, "strftime"):
        last_built_str = last_built.strftime("%Y-%m-%d %H:%M:%S")
    else:
        last_built_str = str(last_built)

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("属性", style="dim")
    table.add_column("值")

    table.add_row("缓存状态", "[green]存在[/green]")
    table.add_row("记录数", str(n_rows))
    table.add_row("最后构建", last_built_str)
    table.add_row("信号日期范围", f"{first_signal} ~ {last_signal}")

    console.print(table)
