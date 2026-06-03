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


def _ascii_histogram(values: list[float], bins: int = 20, width: int = 40) -> str:
    """终值分布 ASCII 直方图，固定 bins 档。"""
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    if hi == lo:
        return f"{lo:,.0f} | {'#' * width} ({len(values)})"
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        b = min(int((v - lo) / step), bins - 1)
        counts[b] += 1
    peak = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        edge = lo + i * step
        bar = "#" * round(c / peak * width)
        lines.append(f"{edge:>14,.0f} | {bar} {c}")
    return "\n".join(lines)


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
    rt = Table(title=f"新金叉信号 (bars_since ≤ {max_bars_since}, 按总收益排序)")
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


@app.command("montecarlo")
def winrate_montecarlo(
    market: str = typer.Option("us", "--market", help="市场代码 (目前仅 us)"),
    ema_short: int = typer.Option(60, "--ema-short", help="短 EMA 周期"),
    ema_long: int = typer.Option(120, "--ema-long", help="长 EMA 周期"),
    start: str | None = typer.Option(None, "--start", help="起始 YYYY-MM-DD"),
    end: str | None = typer.Option(None, "--end", help="结束 YYYY-MM-DD"),
    sims: int = typer.Option(100, "--sims", help="模拟次数"),
    capital: float = typer.Option(1_000_000, "--capital", help="每次全仓本金（美元）"),
    seed: int | None = typer.Option(None, "--seed", help="随机种子（复现用）"),
    csv: bool = typer.Option(True, "--csv/--no-csv", help="导出明细 CSV"),
) -> None:
    """
    EMA 金叉→死叉 蒙特卡洛随机回测：bootstrap 抽样历史交易，
    每次全仓 capital 抽一笔、记单笔 P&L，跑 sims 次看分布。

    示例:
        trendspec winrate montecarlo --market us
        trendspec winrate montecarlo --market us --sims 500 --seed 42
    """
    from trendspec.data.markets import Market
    from trendspec.research.ema_cross_winrate import monte_carlo, run_winrate

    market_enum = Market(market.upper())
    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None

    console.print(
        f"[cyan]EMA{ema_short}/{ema_long} 蒙特卡洛 ({market}, sims={sims})...[/cyan]"
    )
    wr = run_winrate(
        market_enum, ema_short=ema_short, ema_long=ema_long,
        start=start_dt, end=end_dt,
    )
    try:
        res = monte_carlo(wr["trades"], sims=sims, capital=capital, seed=seed)
    except RuntimeError as e:
        console.print(f"[red]错误: {e}[/red]")
        raise typer.Exit(code=1) from None
    s = res["summary"]
    pct = res["percentiles"]

    # 汇总表
    t = Table(title=f"EMA{ema_short}/{ema_long} 蒙特卡洛汇总 ({sims} 次)")
    t.add_column("指标")
    t.add_column("值", justify="right")
    t.add_row("模拟次数", f"{s['sims']:,}")
    t.add_row("本金", f"${s['capital']:,.0f}")
    t.add_row("终值均值", f"${s['mean_equity']:,.0f}")
    t.add_row("终值中位", f"${s['median_equity']:,.0f}")
    t.add_row("终值最好", f"${s['best_equity']:,.0f}")
    t.add_row("终值最差", f"${s['worst_equity']:,.0f}")
    t.add_row("胜率", f"{s['win_rate']:.2%}")
    t.add_row("终值标准差", f"${s['std_equity']:,.0f}")
    t.add_row("总 P&L", f"${s['total_pnl']:,.0f}")
    t.add_row("平均收益", f"{s['mean_ret']:.2%}")
    console.print(t)

    # 百分位表
    pt = Table(title="百分位 (终值 / 收益)")
    pt.add_column("分位")
    pt.add_column("终值", justify="right")
    pt.add_column("收益", justify="right")
    for k in ["p5", "p25", "p50", "p75", "p95"]:
        pt.add_row(k.upper(), f"${pct['equity'][k]:,.0f}", f"{pct['ret'][k]:.2%}")
    console.print(pt)

    # ASCII 直方图（终值分布）
    console.print("[bold]终值分布[/bold]")
    console.print(_ascii_histogram(res["detail"]["final_equity"].to_list(), bins=20))
    console.print("[dim]注: 毛收益, raw 未复权价, 各次独立不复利[/dim]")

    if csv:
        today = date.today().isoformat()
        out_dir = "results/montecarlo"
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/ema{ema_short}_{ema_long}_{today}_montecarlo.csv"
        res["detail"].write_csv(path)
        console.print(f"[green]CSV 已写: {path}[/green]")


@app.command("novice-sim")
def winrate_novice_sim(
    market: str = typer.Option("us", "--market", help="市場代碼 (目前僅 us)"),
    ema_short: int = typer.Option(60, "--ema-short", help="短 EMA 週期"),
    ema_long: int = typer.Option(120, "--ema-long", help="長 EMA 週期"),
    start: str | None = typer.Option(None, "--start", help="起始 YYYY-MM-DD"),
    end: str | None = typer.Option(None, "--end", help="結束 YYYY-MM-DD"),
    sims: int = typer.Option(100, "--sims", help="模擬小白人數"),
    capital: float = typer.Option(1_000_000, "--capital", help="每個小白初始資金（美元）"),
    seed: int | None = typer.Option(None, "--seed", help="隨機種子（復現用）"),
    csv: bool = typer.Option(True, "--csv/--no-csv", help="導出明細 CSV"),
) -> None:
    """
    小白交易員時間軸模擬：每個小白跑完完整 1h 數據，
    看到金叉隨機全倉買入，死叉賣出，複利積累。跑 sims 個小白看分佈。

    示例:
        trendspec winrate novice-sim --market us
        trendspec winrate novice-sim --market us --sims 200 --seed 42
    """
    from trendspec.data.markets import Market
    from trendspec.data.parquet_loader import read_intraday
    from trendspec.research.ema_cross_winrate import compute_ema_cross, run_novice_simulations

    market_enum = Market(market.upper())
    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None

    console.print(
        f"[cyan]小白交易員模擬 EMA{ema_short}/{ema_long} ({market}, sims={sims})...[/cyan]"
    )

    bars_df = read_intraday(market_enum, root=None, start=start_dt, end=end_dt)
    if bars_df.is_empty():
        console.print(
            f"[red]錯誤: 無 intraday 數據。先執行 trendspec ingest intraday --market {market}[/red]"
        )
        raise typer.Exit(code=1) from None

    cross = compute_ema_cross(bars_df, ema_short, ema_long)

    try:
        res = run_novice_simulations(cross, sims=sims, capital=capital, seed=seed)
    except RuntimeError as e:
        console.print(f"[red]錯誤: {e}[/red]")
        raise typer.Exit(code=1) from None

    s = res["summary"]
    pct = res["percentiles"]

    # 汇总表
    t = Table(title=f"小白模擬 EMA{ema_short}/{ema_long} ({sims} 人)")
    t.add_column("指標")
    t.add_column("值", justify="right")
    t.add_row("模擬人數", f"{s['sims']:,}")
    t.add_row("初始資金", f"${s['capital']:,.0f}")
    t.add_row("終值均值", f"${s['mean_equity']:,.0f}")
    t.add_row("終值中位", f"${s['median_equity']:,.0f}")
    t.add_row("終值最好", f"${s['best_equity']:,.0f}")
    t.add_row("終值最差", f"${s['worst_equity']:,.0f}")
    t.add_row("勝率(終值>本金)", f"{s['win_rate']:.2%}")
    t.add_row("終值標準差", f"${s['std_equity']:,.0f}")
    t.add_row("平均收益", f"{s['mean_ret']:.2%}")
    t.add_row("中位交易次數", f"{s['median_n_trades']:.1f}")
    console.print(t)

    # 百分位表
    pt = Table(title="百分位 (終值 / 總收益)")
    pt.add_column("分位")
    pt.add_column("終值", justify="right")
    pt.add_column("總收益", justify="right")
    for k in ["p5", "p25", "p50", "p75", "p95"]:
        pt.add_row(k.upper(), f"${pct['equity'][k]:,.0f}", f"{pct['ret'][k]:.2%}")
    console.print(pt)

    # ASCII 直方圖
    console.print("[bold]終值分佈[/bold]")
    console.print(_ascii_histogram(res["detail"]["final_equity"].to_list(), bins=20))
    console.print("[dim]注: 複利模擬，含強制平倉，毛收益未扣交易成本[/dim]")

    if csv:
        today = date.today().isoformat()
        out_dir = "results/novice_sim"
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/ema{ema_short}_{ema_long}_{today}_novice.csv"
        res["detail"].write_csv(path)
        console.print(f"[green]CSV 已寫: {path}[/green]")
