"""
TrendSpec data ingest CLI.

Commands:
    trendspec ingest daily --market us
    trendspec ingest weekly --market us
    trendspec ingest daily --market cn
    trendspec ingest components --market us
    trendspec ingest sectors --market us
    trendspec ingest status
"""

import typer
from rich.console import Console

app = typer.Typer(name="ingest", help="导入市场数据")
console = Console()


@app.command("daily")
def ingest_daily(
    market: str = typer.Option(
        "us",
        "--market",
        help="市场代码 (cn, us)",
    ),
    since: str = typer.Option(
        "2000-01-01",
        "--since",
        help="起始日期 YYYY-MM-DD（仅用于显示，实际由 manifest 控制增量）",
    ),
    incremental: bool = typer.Option(
        True,
        "--incremental/--full",
        help="增量同步 (默认) 或全量同步",
    ),
) -> None:
    """
    从群辉 stocks DB 导入 OHLCV 日线数据.

    示例:
        trendspec ingest daily --market us
        trendspec ingest daily --market cn --full
    """
    from trendspec.data.markets import Market
    from trendspec.config.settings import get_settings
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.mariadb_client import create_engine_from_settings
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_daily, ingest_us_daily

    market_enum = Market(market.upper())
    settings = get_settings()
    engine = create_engine_from_settings(settings.db)
    root = settings.data_lake.data_lake_root
    manifest = Manifest(market_enum, root)
    full_sync = not incremental

    console.print(f"[cyan]导入 {market} 日线数据...[/cyan]")
    try:
        if market_enum == Market.US:
            result = ingest_us_daily(engine, manifest, root, full_sync=full_sync)
        elif market_enum == Market.CN:
            result = ingest_cn_daily(engine, manifest, root, full_sync=full_sync)
        else:
            console.print(f"[red]不支持的市场: {market}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]完成: {result['row_count']} 行, {result['instrument_count']} 只股票[/green]")
    except Exception as e:
        console.print(f"[red]导入失败: {e}[/red]")
        raise typer.Exit(1)


@app.command("weekly")
def ingest_weekly(
    market: str = typer.Option("us", "--market", help="市场代码 (cn, us)"),
    incremental: bool = typer.Option(
        True, "--incremental/--full",
        help="增量同步 (默认) 或全量同步",
    ),
) -> None:
    """
    从群辉 stocks DB 导入 OHLCV 周线数据.

    示例:
        trendspec ingest weekly --market us
        trendspec ingest weekly --market cn --full
    """
    from trendspec.data.markets import Market
    from trendspec.config.settings import get_settings
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.mariadb_client import create_engine_from_settings
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_weekly, ingest_us_weekly

    market_enum = Market(market.upper())
    settings = get_settings()
    engine = create_engine_from_settings(settings.db)
    root = settings.data_lake.data_lake_root
    manifest = Manifest(market_enum, root)
    full_sync = not incremental

    console.print(f"[cyan]导入 {market} 周线数据...[/cyan]")
    try:
        if market_enum == Market.US:
            result = ingest_us_weekly(engine, manifest, root, full_sync=full_sync)
        elif market_enum == Market.CN:
            result = ingest_cn_weekly(engine, manifest, root, full_sync=full_sync)
        else:
            console.print(f"[red]不支持的市场: {market}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]完成: {result['row_count']} 行, {result['instrument_count']} 只股票[/green]")
    except Exception as e:
        console.print(f"[red]导入失败: {e}[/red]")
        raise typer.Exit(1)


@app.command("intraday")
def ingest_intraday(
    market: str = typer.Option("us", "--market", help="市场代码 (目前仅 us)"),
    incremental: bool = typer.Option(
        True, "--incremental/--full", help="增量 / 全量"
    ),
) -> None:
    """
    导入 1h K 线（prices_intraday）。

    示例:
        trendspec ingest intraday --market us
        trendspec ingest intraday --market us --full
    """
    from trendspec.data.markets import Market
    from trendspec.config.settings import get_settings
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.mariadb_client import create_engine_from_settings
    from trendspec.ingest.stocks_db_ingestor import ingest_us_intraday

    market_enum = Market(market.upper())
    if market_enum != Market.US:
        console.print(f"[red]intraday 目前仅支持 us，收到: {market}[/red]")
        raise typer.Exit(1)

    settings = get_settings()
    engine = create_engine_from_settings(settings.db)
    root = settings.data_lake.data_lake_root
    manifest = Manifest(market_enum, root)
    full_sync = not incremental

    console.print(f"[cyan]导入 {market} 1h 数据...[/cyan]")
    result = ingest_us_intraday(engine, manifest, root, full_sync=full_sync)
    console.print(
        f"[green]完成: {result['row_count']:,} 行, "
        f"{result['instrument_count']} 只, {result['date_range']}[/green]"
    )


@app.command("components")
def ingest_components(
    market: str = typer.Option("us", "--market", help="市场代码 (cn, us)"),
) -> None:
    """导入成分变动数据."""
    from trendspec.data.markets import Market
    from trendspec.config.settings import get_settings
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.mariadb_client import create_engine_from_settings
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_components, ingest_us_components

    market_enum = Market(market.upper())
    settings = get_settings()
    engine = create_engine_from_settings(settings.db)
    root = settings.data_lake.data_lake_root
    manifest = Manifest(market_enum, root)

    console.print(f"[cyan]导入 {market} 成分数据...[/cyan]")
    try:
        if market_enum == Market.US:
            result = ingest_us_components(engine, manifest, root)
        elif market_enum == Market.CN:
            result = ingest_cn_components(engine, manifest, root)
        else:
            console.print(f"[red]不支持的市场: {market}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]完成: {result['row_count']} 行[/green]")
    except Exception as e:
        console.print(f"[red]导入失败: {e}[/red]")
        raise typer.Exit(1)


@app.command("sectors")
def ingest_sectors(
    market: str = typer.Option("us", "--market", help="市场代码 (cn, us)"),
) -> None:
    """导入行业分类数据."""
    from trendspec.data.markets import Market
    from trendspec.config.settings import get_settings
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.mariadb_client import create_engine_from_settings
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_sectors, ingest_us_sectors

    market_enum = Market(market.upper())
    settings = get_settings()
    engine = create_engine_from_settings(settings.db)
    root = settings.data_lake.data_lake_root
    manifest = Manifest(market_enum, root)

    console.print(f"[cyan]导入 {market} 行业数据...[/cyan]")
    try:
        if market_enum == Market.US:
            result = ingest_us_sectors(engine, manifest, root)
        elif market_enum == Market.CN:
            result = ingest_cn_sectors(engine, manifest, root)
        else:
            console.print(f"[red]不支持的市场: {market}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]完成: {result['row_count']} 行[/green]")
    except Exception as e:
        console.print(f"[red]导入失败: {e}[/red]")
        raise typer.Exit(1)


@app.command("indices")
def ingest_indices(
    market: str = typer.Option("us", "--market", "-m", help="市场代码 (cn, us)"),
    full: bool = typer.Option(False, "--full", help="全量同步"),
) -> None:
    """摄入大盘指数数据 (SP500 / CSI800 等)."""
    from trendspec.data.markets import Market
    from trendspec.config.settings import get_settings
    from trendspec.ingest.mariadb_client import create_engine_from_settings
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.stocks_db_ingestor import ingest_us_indices, ingest_cn_indices

    try:
        market_enum = Market(market.upper())
    except ValueError:
        console.print(f"[red]不支持的市场: {market}[/red]")
        raise typer.Exit(1)

    settings = get_settings()
    db_engine = create_engine_from_settings(settings.db)
    root = settings.data_lake.data_lake_root
    mf = Manifest(market_enum, root)

    try:
        if market_enum == Market.US:
            result = ingest_us_indices(db_engine, mf, root, full_sync=full)
        else:
            result = ingest_cn_indices(db_engine, mf, root, full_sync=full)
        console.print(f"[green]摄入完成[/green] — {result['row_count']} 行, "
                      f"指数数 {result['instrument_count']}, "
                      f"日期: {result['date_range']}")
    except Exception as e:
        console.print(f"[red]摄入失败: {e}[/red]")
        raise typer.Exit(1)


@app.command("status")
def ingest_status(
    market: str = typer.Option("us", "--market", help="市场代码 (cn, us)"),
) -> None:
    """显示摄入状态."""
    from trendspec.data.markets import Market
    from trendspec.config.settings import get_settings
    from trendspec.ingest.manifest import Manifest

    market_enum = Market(market.upper())
    settings = get_settings()
    root = settings.data_lake.data_lake_root
    manifest = Manifest(market_enum, root)

    for dataset in ("daily", "components", "sectors"):
        state = manifest.get_dataset_state(dataset)
        if state:
            console.print(f"[green]{dataset}:[/green] {state.get('row_count', 0)} 行, "
                          f"日期: {state.get('date_range', {})}")
        else:
            console.print(f"[yellow]{dataset}:[/yellow] 未同步")
