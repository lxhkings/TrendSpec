"""
EMA 金叉胜率事件研究（1h）。

金叉 = EMA_short 上穿 EMA_long；死叉 = 下穿。
一笔交易 = 金叉进、之后第一个死叉出，ret = exit/entry - 1（毛收益，未复权）。
最后一个金叉无后续死叉 = open trade，不入胜率，转选股。
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from trendspec.data.markets import Market
from trendspec.data.parquet_loader import bars


def compute_adv20_daily(
    market: Market,
    root: str | None = None,
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
                mfe = max(window) / float(e_close) - 1.0
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
        .sort("median_ret", descending=True, nulls_last=True)
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
    trades = pair_trades(cross)
    screen = current_screen(cross)
    return {
        "summary": aggregate(trades),
        "trades": trades,
        "per_ticker": per_ticker(trades),
        "screen": screen,
        "recent_screen": recent_golden_cross(
            cross, max_bars_since=max_bars_since,
            min_adv=min_adv, adv_dict=adv_dict
        ),
    }