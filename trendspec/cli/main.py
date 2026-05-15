"""
TrendSpec CLI main entry point.

Typer application with subcommands for ingest, backtest, and screen.
"""

import typer

# Create main app
app = typer.Typer(
    name="trendspec",
    help="TrendSpec - 量化回测与选股系统",
    add_completion=False,
)

# Import subcommand apps
from trendspec.cli.ingest_cmd import app as ingest_app
from trendspec.cli.backtest_cmd import app as backtest_app
from trendspec.cli.screen_cmd import app as screen_app

# Add subcommands
app.add_typer(ingest_app, name="ingest", help="导入市场数据")
app.add_typer(backtest_app, name="backtest", help="运行回测")
app.add_typer(screen_app, name="screen", help="运行选股")


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="显示版本信息",
    ),
) -> None:
    """
    TrendSpec CLI - 量化回测与选股系统.

    用法:
        trendspec ingest --market cn_a --dataset daily
        trendspec backtest --strategy ma_cross --market cn_a --start 2020-01-01
        trendspec screen --strategy ma_cross --market cn_a --date 2024-05-15
    """
    if version:
        from trendspec import __version__
        typer.echo(f"TrendSpec version: {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()