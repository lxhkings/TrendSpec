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
from trendspec.cli.signal_history_cmd import app as signal_history_app
from trendspec.cli.research_cmd import app as research_app
from trendspec.cli.winrate_cmd import app as winrate_app

# Add subcommands
app.add_typer(ingest_app, name="ingest", help="导入市场数据")
app.add_typer(backtest_app, name="backtest", help="运行回测")
app.add_typer(screen_app, name="screen", help="运行选股")
app.add_typer(signal_history_app, name="signal-history", help="构建和查看信号历史缓存")
app.add_typer(research_app, name="research", help="AI 自动因子研究闭环")
app.add_typer(winrate_app, name="winrate", help="信号胜率研究")


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
        trendspec ingest --market cn --dataset daily
        trendspec backtest --strategy ma_cross --market cn --start 2020-01-01
        trendspec screen --strategy ma_cross --market cn --date 2024-05-15
    """
    if version:
        from trendspec import __version__
        typer.echo(f"TrendSpec version: {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()