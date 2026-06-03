"""
EMA 金叉胜率事件研究（1h）。

金叉 = EMA_short 上穿 EMA_long；死叉 = 下穿。
一笔交易 = 金叉进、之后第一个死叉出，ret = exit/entry - 1（毛收益，未复权）。
最后一个金叉无后续死叉 = open trade，不入胜率，转选股。
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from trendspec.data.markets import Market
from trendspec.data.parquet_loader import bars


def compute_adv20_daily(
    market: Market,
    root: str | None = None,  # noqa: ARG001 — preserved for future override
    instrument_ids: list[str] | None = None,
) -> dict[str, float]:
    """
    从 daily parquet 计算每只股票的 20 日平均成交额（美元）。

    返回：{instrument_id: adv20_usd}
    """
    # 取最近 30 日数据（留 buffer）
    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    df = bars(
        market,
        start_date=start_date,
        end_date=end_date,
        instrument_ids=instrument_ids,
        columns=["instrument_id", "date", "close", "volume"],
    )

    if df.is_empty():
        return {}

    # 按 instrument_id 分组取最近 20 日
    adv = (
        df.sort("date")
        .group_by("instrument_id")
        .agg([
            pl.col("date").last().alias("_last_date"),
            pl.col("close").last().alias("_last_close"),
            pl.col("volume").tail(20).mean().alias("_avg_volume"),
        ])
        .with_columns(
            (pl.col("_avg_volume") * pl.col("_last_close")).alias("adv20")
        )
        .select(["instrument_id", "adv20"])
    )

    return {row["instrument_id"]: row["adv20"] for row in adv.iter_rows(named=True)}


def _ema_expr(period: int) -> pl.Expr:
    alpha = 2.0 / (1.0 + period)
    return (
        pl.col("close")
        .ewm_mean(alpha=alpha, adjust=False)
        .over("instrument_id")
    )


def compute_ema_cross(df: pl.DataFrame, ema_short: int, ema_long: int) -> pl.DataFrame:
    """
    返回 [instrument_id, datetime, close, ema_s, ema_l, signal]。
    signal ∈ {"golden", "death", None}。剔除预热期（每股前 ema_long 根）。
    """
    out = (
        df.sort(["instrument_id", "datetime"])
        .with_columns([
            _ema_expr(ema_short).alias("ema_s"),
            _ema_expr(ema_long).alias("ema_l"),
        ])
        .with_columns(
            pl.col("datetime").cum_count().over("instrument_id").alias("_bar")
        )
        .filter(pl.col("_bar") > ema_long + 1)  # 预热期剔除 + 保留 prev bar
        .drop("_bar")
    )
    prev_s = pl.col("ema_s").shift(1).over("instrument_id")
    prev_l = pl.col("ema_l").shift(1).over("instrument_id")
    golden = (prev_s <= prev_l) & (pl.col("ema_s") > pl.col("ema_l"))
    death = (prev_s >= prev_l) & (pl.col("ema_s") < pl.col("ema_l"))
    return out.with_columns(
        pl.when(golden).then(pl.lit("golden"))
        .when(death).then(pl.lit("death"))
        .otherwise(None)
        .alias("signal")
    ).select(["instrument_id", "datetime", "close", "ema_s", "ema_l", "signal"])


def pair_trades(cross: pl.DataFrame, mfe_window: int = 20) -> pl.DataFrame:
    """
    配对成交：每个 golden 配之后第一个 death。
    每笔多记 mfe = 进场后 mfe_window 根内 max(close)/entry-1（窗口末尾封顶）。
    返回 [instrument_id, entry_dt, entry_close, exit_dt, exit_close, ret, bars_held, mfe, win]。
    """
    rows = []
    for iid, g in cross.filter(pl.col("signal").is_not_null()).group_by(
        "instrument_id", maintain_order=True
    ):
        events = g.sort("datetime").select(["datetime", "close", "signal"]).iter_rows()
        events = list(events)
        # 同时需要 bar 索引算 bars_held + MFE → 用全序列定位
        seq = cross.filter(pl.col("instrument_id") == iid[0]).sort("datetime")
        seq_closes = seq["close"].to_list()
        idx = {dt: i for i, dt in enumerate(seq["datetime"].to_list())}
        open_entry = None
        for dt, close, sig in events:
            if sig == "golden" and open_entry is None:
                open_entry = (dt, close)
            elif sig == "death" and open_entry is not None:
                e_dt, e_close = open_entry
                ret = float(close) / float(e_close) - 1.0
                e_idx = idx[e_dt]
                window = seq_closes[e_idx : e_idx + mfe_window + 1]
                mfe = float(max(window)) / float(e_close) - 1.0
                rows.append({
                    "instrument_id": iid[0],
                    "entry_dt": e_dt, "entry_close": float(e_close),
                    "exit_dt": dt, "exit_close": float(close),
                    "ret": ret,
                    "bars_held": idx[dt] - e_idx,
                    "mfe": mfe,
                    "win": ret > 0,
                })
                open_entry = None
    if not rows:
        return pl.DataFrame(schema={
            "instrument_id": pl.Utf8, "entry_dt": pl.Datetime,
            "entry_close": pl.Float64, "exit_dt": pl.Datetime,
            "exit_close": pl.Float64, "ret": pl.Float64,
            "bars_held": pl.Int64, "mfe": pl.Float64, "win": pl.Boolean,
        })
    return pl.DataFrame(rows)


def aggregate(trades: pl.DataFrame) -> dict:
    """整体聚合指标。空 trades 返回零值。"""
    if trades.is_empty():
        return {"total_trades": 0, "win_rate": 0.0, "avg_win": 0.0,
                "avg_loss": 0.0, "profit_factor": 0.0, "avg_bars_held": 0.0}
    wins = trades.filter(pl.col("win"))
    losses = trades.filter(~pl.col("win"))
    gross_win = wins["ret"].sum() if wins.height else 0.0
    gross_loss = abs(losses["ret"].sum()) if losses.height else 0.0
    return {
        "total_trades": trades.height,
        "win_rate": wins.height / trades.height,
        "avg_win": wins["ret"].mean() if wins.height else 0.0,
        "avg_loss": losses["ret"].mean() if losses.height else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_bars_held": trades["bars_held"].mean(),
    }


def monte_carlo(
    trades: pl.DataFrame,
    sims: int = 100,
    capital: float = 1_000_000,
    seed: int | None = None,
) -> dict:
    """
    Bootstrap 随机回测：放回抽样 sims 笔历史交易，每次 capital 全仓，记单笔 P&L。
    各次独立、不复利。返回 detail / summary / percentiles。
    """
    if trades.is_empty():
        raise RuntimeError(
            "Empty trade pool. Need golden→death trades first "
            "(run winrate ema-cross)."
        )

    rng = np.random.default_rng(seed)
    n = trades.height
    idx = rng.integers(0, n, size=sims)

    sampled = trades[idx.tolist()]
    rets = sampled["ret"].to_numpy()
    pnl = capital * rets
    equity = capital * (1.0 + rets)

    detail = pl.DataFrame({
        "sim_id": list(range(1, sims + 1)),
        "instrument_id": sampled["instrument_id"].to_list(),
        "entry_dt": sampled["entry_dt"].to_list(),
        "exit_dt": sampled["exit_dt"].to_list(),
        "ret": rets.tolist(),
        "pnl_usd": pnl.tolist(),
        "final_equity": equity.tolist(),
    })

    summary = {
        "sims": sims,
        "capital": capital,
        "mean_equity": float(equity.mean()),
        "median_equity": float(np.median(equity)),
        "best_equity": float(equity.max()),
        "worst_equity": float(equity.min()),
        "win_rate": float((rets > 0).mean()),
        "std_equity": float(equity.std()),
        "total_pnl": float(pnl.sum()),
        "mean_ret": float(rets.mean()),
    }

    def _pcts(arr):
        ps = np.percentile(arr, [5, 25, 50, 75, 95])
        return {"p5": float(ps[0]), "p25": float(ps[1]), "p50": float(ps[2]),
                "p75": float(ps[3]), "p95": float(ps[4])}

    percentiles = {"equity": _pcts(equity), "ret": _pcts(rets)}

    return {"detail": detail, "summary": summary, "percentiles": percentiles}


def simulate_novice(
    cross: pl.DataFrame,
    capital: float,
    rng: np.random.Generator,
) -> dict:
    """
    模擬一個小白交易員跑完完整時間軸。
    看到金叉隨機選股全倉買入，死叉賣出，複利累積。
    末尾仍持倉則按最後收盤價強制平倉。
    """
    events = (
        cross.filter(pl.col("signal").is_not_null())
        .sort(["datetime", "instrument_id"])
    )

    current_capital = capital
    position = None  # (instrument_id, entry_close, entry_capital, entry_dt)
    trades = []

    # 取每個 instrument_id 的最後收盤（強制平倉用），從完整 cross 取
    last_closes = {
        row["instrument_id"]: row["close"]
        for row in cross.sort("datetime").group_by("instrument_id").agg(
            pl.col("close").last().alias("close"),
        ).iter_rows(named=True)
    }

    for dt_val in events["datetime"].unique(maintain_order=False).sort().to_list():
        group = events.filter(pl.col("datetime") == dt_val)

        # 步驟1：持倉股有死叉 → 賣出
        if position is not None:
            held_id, entry_close, entry_cap, entry_dt = position
            death = group.filter(
                (pl.col("instrument_id") == held_id) & (pl.col("signal") == "death")
            )
            if death.height > 0:
                exit_close = float(death["close"][0])
                ret = exit_close / entry_close - 1.0
                current_capital = entry_cap * (1.0 + ret)
                trades.append({
                    "instrument_id": held_id,
                    "entry_dt": entry_dt,
                    "exit_dt": dt_val,
                    "entry_close": entry_close,
                    "exit_close": exit_close,
                    "ret": ret,
                    "capital_after": current_capital,
                })
                position = None

        # 步驟2：無持倉且本 bar 有金叉 → 隨機選一支買入
        if position is None:
            golden = group.filter(pl.col("signal") == "golden")
            if golden.height > 0:
                idx = int(rng.integers(0, golden.height))
                chosen = golden.row(idx, named=True)
                position = (
                    chosen["instrument_id"],
                    float(chosen["close"]),
                    current_capital,
                    dt_val,
                )

    # 強制平倉
    if position is not None:
        held_id, entry_close, entry_cap, entry_dt = position
        exit_close = float(last_closes.get(held_id, entry_close))
        ret = exit_close / entry_close - 1.0
        current_capital = entry_cap * (1.0 + ret)
        trades.append({
            "instrument_id": held_id,
            "entry_dt": entry_dt,
            "exit_dt": None,
            "entry_close": entry_close,
            "exit_close": exit_close,
            "ret": ret,
            "capital_after": current_capital,
        })

    return {
        "final_capital": current_capital,
        "total_ret": current_capital / capital - 1.0,
        "n_trades": len(trades),
        "trades": trades,
    }


def run_novice_simulations(
    cross: pl.DataFrame,
    sims: int = 100,
    capital: float = 1_000_000,
    seed: int | None = None,
) -> dict:
    """
    跑 sims 個獨立小白交易員模擬，返回 detail / summary / percentiles。
    隨機性僅在同一 bar 多支股票同時金叉時選哪支。
    """
    golden_count = cross.filter(pl.col("signal") == "golden").height
    if golden_count == 0:
        raise RuntimeError(
            "No golden cross signals in data. "
            "Novice trader has nothing to buy."
        )

    master_rng = np.random.default_rng(seed)
    seeds = master_rng.integers(0, 2**31, size=sims)

    rows = []
    for i, s in enumerate(seeds):
        r = simulate_novice(cross, capital, np.random.default_rng(int(s)))
        rows.append({
            "sim_id": i + 1,
            "n_trades": r["n_trades"],
            "final_equity": r["final_capital"],
            "total_ret": r["total_ret"],
        })

    detail = pl.DataFrame(rows)
    equities = detail["final_equity"].to_numpy()
    rets = detail["total_ret"].to_numpy()
    n_trades_arr = detail["n_trades"].to_numpy()

    def _pcts(arr):
        ps = np.percentile(arr, [5, 25, 50, 75, 95])
        return {
            "p5": float(ps[0]), "p25": float(ps[1]), "p50": float(ps[2]),
            "p75": float(ps[3]), "p95": float(ps[4]),
        }

    summary = {
        "sims": sims,
        "capital": capital,
        "mean_equity": float(equities.mean()),
        "median_equity": float(np.median(equities)),
        "best_equity": float(equities.max()),
        "worst_equity": float(equities.min()),
        "win_rate": float((equities > capital).mean()),
        "std_equity": float(equities.std()),
        "mean_ret": float(rets.mean()),
        "median_n_trades": float(np.median(n_trades_arr)),
    }

    return {
        "detail": detail,
        "summary": summary,
        "percentiles": {"equity": _pcts(equities), "ret": _pcts(rets)},
    }


def per_ticker(trades: pl.DataFrame) -> pl.DataFrame:
    """每 ticker 一行聚合（含中位数统计）。"""
    if trades.is_empty():
        return pl.DataFrame()
    return (
        trades.group_by("instrument_id")
        .agg([
            pl.len().alias("total_trades"),
            pl.col("win").mean().alias("win_rate"),
            pl.col("ret").filter(pl.col("win")).mean().alias("avg_win"),
            pl.col("ret").filter(~pl.col("win")).mean().alias("avg_loss"),
            pl.col("bars_held").mean().alias("avg_bars_held"),
            pl.col("ret").median().alias("median_ret"),
            pl.col("ret").min().alias("worst_ret"),
            pl.col("bars_held").median().alias("median_bars"),
            pl.col("mfe").median().alias("median_mfe"),
            pl.col("ret").sum().alias("total_return"),
        ])
        .sort("win_rate", descending=True)
    )


def current_screen(cross: pl.DataFrame) -> pl.DataFrame:
    """
    当前金叉态：每股最新 bar ema_s>ema_l 且最近一次穿越是 golden。
    返回 [instrument_id, cross_dt, bars_since, unrealized_ret, last_close]。
    """
    rows = []
    for iid, g in cross.group_by("instrument_id", maintain_order=True):
        seq = g.sort("datetime")
        last = seq.row(-1, named=True)
        if last["ema_s"] <= last["ema_l"]:
            continue
        sigs = seq.filter(pl.col("signal").is_not_null())
        if sigs.is_empty() or sigs.row(-1, named=True)["signal"] != "golden":
            continue
        cross_row = sigs.row(-1, named=True)
        dts = seq["datetime"].to_list()
        bars_since = (len(dts) - 1) - dts.index(cross_row["datetime"])
        rows.append({
            "instrument_id": iid[0],
            "cross_dt": cross_row["datetime"],
            "bars_since": bars_since,
            "unrealized_ret": float(last["close"]) / float(cross_row["close"]) - 1.0,
            "last_close": float(last["close"]),
        })
    if not rows:
        return pl.DataFrame(schema={
            "instrument_id": pl.Utf8, "cross_dt": pl.Datetime,
            "bars_since": pl.Int64, "unrealized_ret": pl.Float64,
            "last_close": pl.Float64,
        })
    return pl.DataFrame(rows).sort("unrealized_ret", descending=True)


def recent_golden_cross(
    cross: pl.DataFrame,
    max_bars_since: int = 20,
    min_adv: float = 0,
    adv_dict: dict[str, float] | None = None,
    stats: pl.DataFrame | None = None,
    min_samples: int = 3,
) -> pl.DataFrame:
    """
    最近 N 根 bar 内发生金叉且仍金叉态。

    条件：ema_s > ema_l + bars_since ≤ max_bars_since + adv20 ≥ min_adv。

    stats 非空时：left-join 历史金叉→死叉统计，加衍生列 progress_pct/overheat_pct，
    过滤 0<N<min_samples（N=0 无历史保留标灰），按 median_ret 降序、N=0 排尾。
    stats=None 时返回未 enrich 的旧结构。
    """
    screen = current_screen(cross)
    if screen.is_empty():
        return screen

    # bars_since 过滤
    screen = screen.filter(pl.col("bars_since") <= max_bars_since)

    # ADV 过滤
    if min_adv > 0 and adv_dict is not None:
        valid_ids = [iid for iid, adv in adv_dict.items() if adv >= min_adv]
        screen = screen.filter(pl.col("instrument_id").is_in(valid_ids))

    if stats is None or stats.is_empty():
        return screen

    enriched = (
        screen.join(stats, on="instrument_id", how="left")
        .rename({"total_trades": "N"})
        .with_columns(pl.col("N").fill_null(0))
        .with_columns([
            (pl.col("bars_since") / pl.col("median_bars")).alias("progress_pct"),
            (pl.col("unrealized_ret") / pl.col("median_mfe")).alias("overheat_pct"),
        ])
        .filter((pl.col("N") == 0) | (pl.col("N") >= min_samples))
        .sort("total_return", descending=True, nulls_last=True)
    )
    return enriched


def run_winrate(
    market,
    root: str | None = None,
    ema_short: int = 60,
    ema_long: int = 120,
    start=None,
    end=None,
    max_bars_since: int = 20,
    min_adv: float = 0,
    min_samples: int = 3,
) -> dict:
    """编排：读 intraday → 算金叉 → 配对 → 聚合 + 选股。"""
    from trendspec.data.parquet_loader import read_intraday

    bars = read_intraday(market, root=root, start=start, end=end)
    if bars.is_empty():
        raise RuntimeError(
            f"No intraday data for {market.value}. "
            f"Run `trendspec ingest intraday --market {market.value.lower()}` first."
        )

    # 预计算 ADV（仅当需要过滤时）
    adv_dict = None
    if min_adv > 0:
        instrument_ids = bars["instrument_id"].unique().to_list()
        adv_dict = compute_adv20_daily(market, root=root, instrument_ids=instrument_ids)

    cross = compute_ema_cross(bars, ema_short, ema_long)
    trades = pair_trades(cross, mfe_window=max_bars_since)
    stats = per_ticker(trades)
    screen = current_screen(cross)
    return {
        "summary": aggregate(trades),
        "trades": trades,
        "per_ticker": stats,
        "screen": screen,
        "recent_screen": recent_golden_cross(
            cross, max_bars_since=max_bars_since,
            min_adv=min_adv, adv_dict=adv_dict,
            stats=stats, min_samples=min_samples,
        ),
    }
