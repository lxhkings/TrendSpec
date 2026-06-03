"""
trendspec winrate ema-cross — 1h EMA 金叉胜率 + 选股。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command("ema-cross")
def winrate_ema_cross(
    market: str = typer.Option("us", "--market", help="市场代码 (目前仅 us)"),
    ema_short: int = typer.Option(60, "--ema-short", help="短 EMA 周期"),
    ema_long: int = typer.Option(120, "--ema-long", help="长 EMA 周期"),
    start: Optional[str] = typer.Option(None, "--start", help="起始 YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, "--end", help="结束 YYYY-MM-DD"),
    max_bars_since: int = typer.Option(20, "--max-bars-since", help="新金叉信号 bars_since 上限"),
    min_adv_us: float = typer.Option(50_000_000, "--min-adv-us", help="日均成交额阈值（美元）"),
    csv: Optional[str] = typer.Option(
        None, "--csv", help="CSV 输出前缀，写 <csv>_trades/_summary/_screen/_recent.csv"
    ),
) -> None:
    """
    EMA 金叉进/死叉出 胜率报告 + 当前金叉态选股 + 新金叉信号。

    示例:
        trendspec winrate ema-cross --market us --csv ./winrate_out
        trendspec winrate ema-cross --market us --max-bars-since 10 --min-adv-us 100000000
    """
    from trendspec.data.markets import Market
    from trendspec.research.ema_cross_winrate import run_winrate

    market_enum = Market(market.upper())
    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None

    console.print(
        f"[cyan]EMA{ema_short}/{ema_long} 金叉胜率 ({market})...[/cyan]"
    )
    res = run_winrate(
        market_enum, ema_short=ema_short, ema_long=ema_long,
        start=start_dt, end=end_dt,
        max_bars_since=max_bars_since,
        min_adv=min_adv_us,
    )
    s = res["summary"]

    # 汇总表
    t = Table(title=f"EMA{ema_short}/{ema_long} 金叉胜率汇总")
    t.add_column("指标"); t.add_column("值", justify="right")
    t.add_row("总交易数", f"{s['total_trades']:,}")
    t.add_row("胜率", f"{s['win_rate']:.2%}")
    t.add_row("平均盈利", f"{s['avg_win']:.2%}")
    t.add_row("平均亏损", f"{s['avg_loss']:.2%}")
    t.add_row("盈亏比", f"{s['profit_factor']:.2f}")
    t.add_row("平均持有(1h根)", f"{s['avg_bars_held']:.1f}")
    console.print(t)
    console.print("[dim]注: 毛收益, raw 未复权价[/dim]")

    # 当前金叉态选股表（前 20）
    screen = res["screen"]
    st = Table(title="当前金叉态 (按浮动收益降序, 前 20)")
    for c in ["ticker", "金叉时间", "持有根数", "浮动收益", "现价"]:
        st.add_column(c)
    for r in screen.head(20).iter_rows(named=True):
        st.add_row(
            r["instrument_id"], str(r["cross_dt"]),
            str(r["bars_since"]), f"{r['unrealized_ret']:.2%}",
            f"{r['last_close']:.2f}",
        )
    console.print(st)

    # 新金叉信号表（≤ max_bars_since）
    recent = res["recent_screen"]
    rt = Table(title=f"新金叉信号 (bars_since ≤ {max_bars_since})")
    for c in ["ticker", "金叉时间", "持有根数", "浮动收益", "现价"]:
        rt.add_column(c)
    for r in recent.iter_rows(named=True):
        rt.add_row(
            r["instrument_id"], str(r["cross_dt"]),
            str(r["bars_since"]), f"{r['unrealized_ret']:.2%}",
            f"{r['last_close']:.2f}",
        )
    console.print(rt)

    if csv:
        res["trades"].write_csv(f"{csv}_trades.csv")
        pl.DataFrame([s]).write_csv(f"{csv}_summary.csv")
        screen.write_csv(f"{csv}_screen.csv")
        recent.write_csv(f"{csv}_recent.csv")
        console.print(f"[green]CSV 已写: {csv}_trades/_summary/_screen/_recent.csv[/green]")