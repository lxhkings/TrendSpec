"""
EMA 金叉胜率事件研究（1h）。

金叉 = EMA_short 上穿 EMA_long；死叉 = 下穿。
一笔交易 = 金叉进、之后第一个死叉出，ret = exit/entry - 1（毛收益，未复权）。
最后一个金叉无后续死叉 = open trade，不入胜率，转选股。
"""

from __future__ import annotations

import polars as pl


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


def pair_trades(cross: pl.DataFrame) -> pl.DataFrame:
    """
    配对成交：每个 golden 配之后第一个 death。
    返回 [instrument_id, entry_dt, entry_close, exit_dt, exit_close, ret, bars_held, win]。
    """
    rows = []
    for iid, g in cross.filter(pl.col("signal").is_not_null()).group_by(
        "instrument_id", maintain_order=True
    ):
        events = g.sort("datetime").select(["datetime", "close", "signal"]).iter_rows()
        events = list(events)
        # 同时需要 bar 索引算 bars_held → 用全序列定位
        seq = cross.filter(pl.col("instrument_id") == iid[0]).sort("datetime")
        idx = {dt: i for i, dt in enumerate(seq["datetime"].to_list())}
        open_entry = None
        for dt, close, sig in events:
            if sig == "golden" and open_entry is None:
                open_entry = (dt, close)
            elif sig == "death" and open_entry is not None:
                e_dt, e_close = open_entry
                ret = close / e_close - 1.0
                rows.append({
                    "instrument_id": iid[0],
                    "entry_dt": e_dt, "entry_close": e_close,
                    "exit_dt": dt, "exit_close": close,
                    "ret": ret,
                    "bars_held": idx[dt] - idx[e_dt],
                    "win": ret > 0,
                })
                open_entry = None
    if not rows:
        return pl.DataFrame(schema={
            "instrument_id": pl.Utf8, "entry_dt": pl.Datetime,
            "entry_close": pl.Float64, "exit_dt": pl.Datetime,
            "exit_close": pl.Float64, "ret": pl.Float64,
            "bars_held": pl.Int64, "win": pl.Boolean,
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


def per_ticker(trades: pl.DataFrame) -> pl.DataFrame:
    """每 ticker 一行聚合。"""
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
            "unrealized_ret": last["close"] / cross_row["close"] - 1.0,
            "last_close": last["close"],
        })
    if not rows:
        return pl.DataFrame(schema={
            "instrument_id": pl.Utf8, "cross_dt": pl.Datetime,
            "bars_since": pl.Int64, "unrealized_ret": pl.Float64,
            "last_close": pl.Float64,
        })
    return pl.DataFrame(rows).sort("unrealized_ret", descending=True)


def run_winrate(
    market,
    root: str | None = None,
    ema_short: int = 60,
    ema_long: int = 120,
    start=None,
    end=None,
) -> dict:
    """编排：读 intraday → 算金叉 → 配对 → 聚合 + 选股。"""
    from trendspec.data.parquet_loader import read_intraday

    bars = read_intraday(market, root=root, start=start, end=end)
    if bars.is_empty():
        raise RuntimeError(
            f"No intraday data for {market.value}. "
            f"Run `trendspec ingest intraday --market {market.value.lower()}` first."
        )
    cross = compute_ema_cross(bars, ema_short, ema_long)
    trades = pair_trades(cross)
    return {
        "summary": aggregate(trades),
        "trades": trades,
        "per_ticker": per_ticker(trades),
        "screen": current_screen(cross),
    }