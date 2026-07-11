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
        None,
        "--since",
        help="起始日期 YYYY-MM-DD（含当天）；覆盖 manifest 增量与 --full 起点",
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

    from datetime import date as date_type

    if since is not None:
        try:
            date_type.fromisoformat(since)
        except ValueError:
            console.print("[red]日期格式错误，请使用 YYYY-MM-DD 格式[/red]")
            raise typer.Exit(1)

    market_enum = Market(market.upper())
    settings = get_settings()
    engine = create_engine_from_settings(settings.db)
    root = settings.data_lake.data_lake_root
    manifest = Manifest(market_enum, root)
    full_sync = not incremental

    console.print(f"[cyan]导入 {market} 日线数据...[/cyan]")
    try:
        if market_enum == Market.US:
            result = ingest_us_daily(engine, manifest, root, full_sync=full_sync, since=since)
        elif market_enum == Market.CN:
            result = ingest_cn_daily(engine, manifest, root, full_sync=full_sync, since=since)
        else:
            console.print(f"[red]不支持的市场: {market}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]完成: {result['row_count']} 行, {result['instrument_count']} 只股票[/green]")
    except Exception as e:
        console.print(f"[red]导入失败: {e}[/red]")
        raise typer.Exit(1)


@app.command("fundamentals")
def ingest_fundamentals(
    market: str = typer.Option(
        "us",
        "--market",
        help="市场代码 (us, cn)",
    ),
    incremental: bool = typer.Option(
        True,
        "--incremental/--full",
        help="保留参数兼容；基本面始终全量重算（TTM/YoY 需历史）",
    ),
) -> None:
    """从群辉 stocks DB 导入季度基本面数据（US/CN）。

    示例:
        trendspec ingest fundamentals --market us
        trendspec ingest fundamentals --market cn
    """
    from trendspec.data.markets import Market
    from trendspec.config.settings import get_settings
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.mariadb_client import create_engine_from_settings
    from trendspec.ingest.fundamentals_ingestor import (
        ingest_cn_fundamentals,
        ingest_us_fundamentals,
    )

    market_enum = Market(market.upper())
    settings = get_settings()
    engine = create_engine_from_settings(settings.db)
    root = settings.data_lake.data_lake_root
    manifest = Manifest(market_enum, root)
    full_sync = not incremental

    console.print(f"[cyan]导入 {market} 基本面数据...[/cyan]")
    try:
        if market_enum == Market.US:
            result = ingest_us_fundamentals(engine, manifest, root, full_sync=full_sync)
        elif market_enum == Market.CN:
            result = ingest_cn_fundamentals(engine, manifest, root, full_sync=full_sync)
        else:
            console.print(f"[red]基本面目前仅支持 us/cn，收到: {market}[/red]")
            raise typer.Exit(1)
        console.print(
            f"[green]完成: {result['row_count']} 行, "
            f"{result['instrument_count']} 只股票[/green]"
        )
    except Exception as e:
        console.print(f"[red]导入失败: {e}[/red]")
        raise typer.Exit(1)


@app.command("valuation")
def ingest_valuation(
    market: str = typer.Option(
        "cn",
        "--market",
        help="市场代码 (目前仅 cn)",
    ),
    since: str = typer.Option(
        None,
        "--since",
        help="起始日期 YYYY-MM-DD（含当天）；覆盖 manifest 增量与 --full 起点",
    ),
    incremental: bool = typer.Option(
        True,
        "--incremental/--full",
        help="增量同步 (默认) 或全量同步",
    ),
) -> None:
    """从群辉 stocks DB 导入每日估值快照（CN, PE/PB/PS）。

    示例:
        trendspec ingest valuation --market cn
        trendspec ingest valuation --market cn --full
    """
    from trendspec.data.markets import Market
    from trendspec.config.settings import get_settings
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.mariadb_client import create_engine_from_settings
    from trendspec.ingest.fundamentals_ingestor import ingest_cn_valuation

    from datetime import date as date_type

    if since is not None:
        try:
            date_type.fromisoformat(since)
        except ValueError:
            console.print("[red]日期格式错误，请使用 YYYY-MM-DD 格式[/red]")
            raise typer.Exit(1)

    market_enum = Market(market.upper())
    settings = get_settings()
    engine = create_engine_from_settings(settings.db)
    root = settings.data_lake.data_lake_root
    manifest = Manifest(market_enum, root)
    full_sync = not incremental

    console.print(f"[cyan]导入 {market} 估值数据...[/cyan]")
    try:
        if market_enum == Market.CN:
            result = ingest_cn_valuation(engine, manifest, root, full_sync=full_sync, since=since)
        else:
            console.print(f"[red]估值目前仅支持 cn，收到: {market}[/red]")
            raise typer.Exit(1)
        console.print(
            f"[green]完成: {result['row_count']} 行, "
            f"{result['instrument_count']} 只股票[/green]"
        )
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
    from trendspec.ingest.stocks_db_ingestor import (
        ingest_cn_components,
        ingest_cn_full_universe_events,
        ingest_us_components,
    )

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
        if market_enum == Market.CN:
            console.print("[cyan]导入 cn 全A股 IPO/DELIST 事件...[/cyan]")
            events_result = ingest_cn_full_universe_events(engine, manifest, root)
            console.print(f"[green]完成: {events_result['row_count']} 行[/green]")
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


_STATUS_DATASETS = (
    "daily", "weekly", "intraday", "components", "sectors",
    "fundamentals", "valuation", "indices",
)


@app.command("status")
def ingest_status(
    market: str = typer.Option("us", "--market", help="市场代码 (cn, us)"),
) -> None:
    """显示各数据集在 data_lake 中的实际全量统计（非最近一次增量批次）。"""
    import polars as pl
    from rich.table import Table

    from trendspec.config.settings import get_settings
    from trendspec.data.markets import Market
    from trendspec.data.parquet_loader import _lazyframe_is_empty, scan_parquet

    market_enum = Market(market.upper())
    settings = get_settings()
    root = settings.data_lake.data_lake_root

    table = Table(title=f"摄入状态 — {market.upper()}")
    table.add_column("数据集")
    table.add_column("行数", justify="right")
    table.add_column("日期范围")
    table.add_column("标的数", justify="right")

    for dataset in _STATUS_DATASETS:
        lf = scan_parquet(root, market_enum, dataset)
        if _lazyframe_is_empty(lf):
            table.add_row(dataset, "-", "未同步", "-")
            continue
        stats = lf.select([
            pl.len().alias("row_count"),
            pl.col("date").min().alias("min_date"),
            pl.col("date").max().alias("max_date"),
            pl.col("instrument_id").n_unique().alias("instrument_count"),
        ]).collect().row(0, named=True)
        table.add_row(
            dataset,
            f"{stats['row_count']:,}",
            f"{stats['min_date']} ~ {stats['max_date']}",
            f"{stats['instrument_count']:,}",
        )

    console.print(table)
