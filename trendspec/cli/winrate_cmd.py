"""
trendspec winrate ema-cross — 1h EMA 金叉胜率 + 选股。
"""

from __future__ import annotations

import os
from datetime import date, datetime

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
    start: str | None = typer.Option(None, "--start", help="起始 YYYY-MM-DD"),
    end: str | None = typer.Option(None, "--end", help="结束 YYYY-MM-DD"),
    max_bars_since: int = typer.Option(60, "--max-bars-since", help="新金叉信号 bars_since 上限"),
    min_adv_us: float = typer.Option(50_000_000, "--min-adv-us", help="日均成交额阈值（美元）"),
    min_samples: int = typer.Option(3, "--min-samples", help="历史金叉样本下限（0<N<阈值 剔除，N=0 保留标灰）"),
    csv: bool = typer.Option(True, "--csv/--no-csv", help="导出 CSV（默认导出到 results/winrate/）"),
) -> None:
    """
    EMA 金叉进/死叉出 胜率报告 + 当前金叉态选股 + 新金叉信号。

    示例:
        trendspec winrate ema-cross --market us
        trendspec winrate ema-cross --market us --max-bars-since 10 --min-adv-us 100000000
        trendspec winrate ema-cross --market us --min-samples 5
        trendspec winrate ema-cross --market us --no-csv  # 不导出 CSV
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
        min_samples=min_samples,
    )
    s = res["summary"]

    # 汇总表
    t = Table(title=f"EMA{ema_short}/{ema_long} 金叉胜率汇总")
    t.add_column("指标")
    t.add_column("值", justify="right")
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

    # 新金叉信号表（≤ max_bars_since），附历史金叉→死叉统计
    def _pct(v):
        return "-" if v is None else f"{v:.2%}"

    def _ratio(v):
        return "-" if v is None else f"{v:.0%}"

    recent = res["recent_screen"]
    rt = Table(title=f"新金叉信号 (bars_since ≤ {max_bars_since}, 按历史中位收益排序)")
    for c in ["ticker", "金叉时间", "持有根数", "浮动收益", "现价",
              "N", "总收益", "中位收益", "进度%", "过热%"]:
        rt.add_column(c)
    for r in recent.iter_rows(named=True):
        rt.add_row(
            r["instrument_id"], str(r["cross_dt"]),
            str(r["bars_since"]), f"{r['unrealized_ret']:.2%}",
            f"{r['last_close']:.2f}",
            str(int(r["N"])),
            _pct(r.get("total_return")),
            _pct(r["median_ret"]),
            _ratio(r["progress_pct"]),
            _ratio(r["overheat_pct"]),
        )
    console.print(rt)
    console.print("[dim]进度%=持有根数÷历史中位根数; 过热%=浮动÷历史中位MFE; N=0 为历史无死叉强势股; 总收益=历史金叉累计收益[/dim]")

    if csv:
        # 输出到 results/winrate/，文件名加日期
        today = date.today().isoformat()
        out_dir = "results/winrate"
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"{out_dir}/ema{ema_short}_{ema_long}_{today}"

        res["trades"].write_csv(f"{prefix}_trades.csv")
        pl.DataFrame([s]).write_csv(f"{prefix}_summary.csv")
        screen.write_csv(f"{prefix}_screen.csv")
        recent.write_csv(f"{prefix}_recent.csv")
        console.print(f"[green]CSV 已写: {prefix}_*.csv[/green]")
