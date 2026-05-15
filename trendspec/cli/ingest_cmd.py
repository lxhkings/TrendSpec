<arg_value>
"""
Ingest command for TrendSpec CLI.

Import market data from MariaDB to Parquet data lake.

Commands:
    trendspec ingest --market cn_a --dataset daily --since 2020-01-01
    trendspec ingest --market cn_a --dataset components --since 2020-01-01
    trendspec ingest --market cn_a --dataset sectors --since 2020-01-01
    trendspec ingest --market us --dataset daily --since 2020-01-01
    trendspec ingest --status
"""

from datetime import date
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="导入市场数据")
console = Console()


@app.command("daily")
def ingest_daily(
    market: str = typer.Option(
        "cn_a",
        "--market",
        "-m",
        help="市场代码 (cn_a, us)",
    ),
    since: str = typer.Option(
        "2020-01-01",
        "--since",
        "-s",
        help="起始日期 (YYYY-MM-DD)",
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental",
        "-i",
        help="增量导入 (仅导入新数据)",
    ),
) -> None:
    """
    导入日线行情数据.

    从MariaDB导入OHLCV日线数据到Parquet数据湖.

    示例:
        trendspec ingest daily --market cn_a --since 2020-01-01
        trendspec ingest daily --market us --since 2020-01-01
    """
    from trendspec.data.markets import Market
    from trendspec.ingest.cn_a_ingestor import CNAIngestor
    from trendspec.ingest.us_ingestor import USIngestor
    from trendspec.config.settings import get_settings

    # Parse date
    try:
        since_date = date.fromisoformat(since)
    except ValueError:
        console.print("[red]日期格式错误，请使用 YYYY-MM-DD 格式[/red]")
        raise typer.Exit(1)

    # Get market
    market_enum = Market(market)

    # Get settings
    settings = get_settings()

    console.print(f"[cyan]导入 {market} 日线数据，起始日期: {since_date}[/cyan]")

    try:
        if market_enum == Market.CN_A:
            ingestor = CNAIngestor(
                db_url=settings.db.connection_url,
                root=settings.data_lake.data_lake_root,
            )
            result = ingestor.ingest_daily(
                since=since_date,
                incremental=incremental,
            )
        elif market_enum == Market.US:
            ingestor = USIngestor(
                db_url=settings.db.connection_url,
                root=settings.data_lake.data_lake_root,
            )
            result = ingestor.ingest_daily(
                since=since_date,
                incremental=incremental,
            )
        else:
            console.print(f"[red]不支持的市场: {market}[/red]")
            raise typer.Exit(1)

        console.print(f"[green]导入完成: {result}[/green]")

    except Exception as e:
        console.print(f"[red]导入失败: {e}[/red]")
        raise typer.Exit(1)


@app.command("components")
def ingest_components(
    market: str = typer.Option(
        "cn_a",
        "--market",
        "-m",
        help="市场代码 (cn_a)",
    ),
    since: str = typer.Option(
        "2020-01-01",
        "--since",
        "-s",
        help="起始日期 (YYYY-MM-DD)",
    ),
) -> None:
    """
    导入指数成分股数据.

    从MariaDB导入指数成分股变更数据到Parquet数据湖.

    示例:
        trendspec ingest components --market cn_a --since 2020-01-01
    """
    from trendspec.data.markets import Market
    from trendspec.ingest.components_ingestor import ComponentsIngestor
    from trendspec.config.settings import get_settings

    try:
        since_date = date.fromisoformat(since)
    except ValueError:
        console.print("[red]日期格式错误，请使用 YYYY-MM-DD 格式[/red]")
        raise typer.Exit(1)

    market_enum = Market(market)
    settings = get_settings()

    console.print(f"[cyan]导入 {market} 指数成分股数据，起始日期: {since_date}[/cyan]")

    try:
        ingestor = ComponentsIngestor(
            db_url=settings.db.connection_url,
            root=settings.data_lake.data_lake_root,
        )
        result = ingestor.ingest(since=since_date)

        console.print(f"[green]导入完成: {result}[/green]")

    except Exception as e:
        console.print(f"[red]导入失败: {e}[/red]")
        raise typer.Exit(1)


@app.command("sectors")
def ingest_sectors(
    market: str = typer.Option(
        "cn_a",
        "--market",
        "-m",
        help="市场代码 (cn_a)",
    ),
    since: str = typer.Option(
        "2020-01-01",
        "--since",
        "-s",
        help="起始日期 (YYYY-MM-DD)",
    ),
) -> None:
    """
    导入板块分类数据.

    从MariaDB导入板块分类数据到Parquet数据湖.

    示例:
        trendspec ingest sectors --market cn_a --since 2020-01-01
    """
    from trendspec.data.markets import Market
    from trendspec.ingest.sectors_ingestor import SectorsIngestor
    from trendspec.config.settings import get_settings

    try:
        since_date = date.fromisoformat(since)
    except ValueError:
        console.print("[red]日期格式错误，请使用 YYYY-MM-DD 格式[/red]")
        raise typer.Exit(1)

    market_enum = Market(market)
    settings = get_settings()

    console.print(f"[cyan]导入 {market} 板块分类数据，起始日期: {since_date}[/cyan]")

    try:
        ingestor = SectorsIngestor(
            db_url=settings.db.connection_url,
            root=settings.data_lake.data_lake_root,
        )
        result = ingestor.ingest(since=since_date)

        console.print(f"[green]导入完成: {result}[/green]")

    except Exception as e:
        console.print(f"[red]导入失败: {e}[/red]")
        raise typer.Exit(1)


@app.command("status")
def ingest_status() -> None:
    """
    查看数据导入状态.

    显示数据湖中各数据集的状态信息.
    """
    from trendspec.ingest.manifest import Manifest
    from trendspec.config.settings import get_settings

    settings = get_settings()
    manifest = Manifest(root=settings.data_lake.data_lake_root)

    table = Table(title="数据导入状态")
    table.add_column("数据集", style="cyan")
    table.add_column("市场", style="cyan")
    table.add_column("最后更新", style="green")
    table.add_column("记录数", style="yellow")
    table.add_column("状态", style="white")

    status = manifest.get_status()

    for dataset, info in status.items():
        table.add_row(
            dataset,
            info.get("market", "N/A"),
            info.get("last_update", "N/A"),
            str(info.get("records", 0)),
            info.get("status", "未知"),
        )

    console.print(table)


@app.command("all")
def ingest_all(
    market: str = typer.Option(
        "cn_a",
        "--market",
        "-m",
        help="市场代码 (cn_a, us)",
    ),
    since: str = typer.Option(
        "2020-01-01",
        "--since",
        "-s",
        help="起始日期 (YYYY-MM-DD)",
    ),
) -> None:
    """
    导入所有数据集.

    一次性导入日线、成分股、板块数据.

    示例:
        trendspec ingest all --market cn_a --since 2020-01-01
    """
    console.print("[cyan]导入所有数据集...[/cyan]")

    # Run each ingest command
    ingest_daily(market=market, since=since, incremental=False)

    if market == "cn_a":
        ingest_components(market=market, since=since)
        ingest_sectors(market=market, since=since)

    console.print("[green]所有数据导入完成[/green]")