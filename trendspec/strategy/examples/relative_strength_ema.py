"""
Relative Strength Top-N rotation strategy (rs_ema_cross).

每周调仓：在相对基准 (QQQ) 处于金叉态 (ratio EMA60 > EMA120) 且流动性达标
(ADV20 >= min_adv) 的股票中，按相对强度 e60/e120-1 排序取 Top-N 等权持有。
跌出 Top-N 或死叉则卖出。组合级 next()，沿用 clenow_momentum 模式。
"""

from datetime import date as DateType
from typing import Any

import polars as pl

from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext

_DEFAULTS = {
    "benchmark_id": "QQQ",
    "ema_short": 60,
    "ema_long": 120,
    "top_n": 20,
    "rebalance_weekday": 0,  # Monday
    "min_adv_us": 1e8,
    "min_adv_cn": 0.0,
    "regime_ma": 200,  # 基准跌破 N 日均线则清仓停开仓；0 关闭
}


@register_strategy("rs_ema_cross")
class RelativeStrengthEMACross(BaseStrategy):
    """相对强度 Top-N 周度轮动。"""

    name = "rs_ema_cross"
    version = "2.0.0"
    params: dict[str, Any] = dict(_DEFAULTS)

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged = dict(_DEFAULTS)
        if params:
            merged.update(params)
        super().__init__(params=merged)

    def init(self, ctx: StrategyContext) -> None:
        from trendspec.data.parquet_loader import read_indices
        from trendspec.strategy.indicators import compute_indicator

        self._rs_short: dict[tuple, float] = {}
        self._rs_long: dict[tuple, float] = {}
        self._last_rebalance_date: DateType | None = None
        self._full_data = ctx._data

        bench_id = self.get_param("benchmark_id")
        short = self.get_param("ema_short")
        long = self.get_param("ema_long")

        bench = read_indices(ctx.market, root=ctx._root, instrument_ids=[bench_id])
        if bench.is_empty():
            raise RuntimeError(
                f"No index data for {bench_id}. "
                f"Run `trendspec ingest indices --market {ctx.market.value}` first."
            )

        # ADV20 流动性指标（成交额）
        ctx.precompute_indicator("ADV", period=20)

        # 大盘 regime 闸：基准收盘 vs N 日均线（date -> bool 在线）
        self._regime_ok: dict[DateType, bool] = {}
        regime_ma = self.get_param("regime_ma")
        if regime_ma and regime_ma > 0:
            reg = bench.sort("date").with_columns(
                pl.col("close").rolling_mean(window_size=regime_ma).alias("_ma")
            )
            for dt, close, ma in reg.select(["date", "close", "_ma"]).iter_rows():
                if ma is not None:
                    self._regime_ok[dt] = close > ma

        bench = bench.select(["date", pl.col("close").alias("_bench_close")])
        data = ctx._data
        if data is None or data.is_empty():
            return

        ratio_df = (
            data.join(bench, on="date", how="inner")
            .with_columns((pl.col("close") / pl.col("_bench_close")).alias("close"))
            .drop("_bench_close")
        )
        if ratio_df.is_empty():
            return

        short_col = f"EMA_{short}"
        long_col = f"EMA_{long}"
        ema_s = compute_indicator(ratio_df, "EMA", period=short).select(
            ["instrument_id", "date", short_col]
        )
        ema_l = compute_indicator(ratio_df, "EMA", period=long).select(
            ["instrument_id", "date", long_col]
        )
        merged = (
            ema_s.join(ema_l, on=["instrument_id", "date"], how="inner")
            .sort(["instrument_id", "date"])
            .with_columns(pl.col("date").cum_count().over("instrument_id").alias("_bar"))
            .filter(pl.col("_bar") >= long)  # 剔除预热期
        )
        for iid, dt, es, el in merged.select(
            ["instrument_id", "date", short_col, long_col]
        ).iter_rows():
            if es is not None and el is not None:
                self._rs_short[(iid, dt)] = es
                self._rs_long[(iid, dt)] = el

    def next(self, ctx: StrategyContext) -> None:
        d = ctx.date
        if not ctx.is_screening and d.weekday() != self.get_param("rebalance_weekday"):
            return
        if d == self._last_rebalance_date:
            return
        self._last_rebalance_date = d

        top_n = self.get_param("top_n")
        if ctx.market.value.upper() == "US":
            min_adv = self.get_param("min_adv_us")
        else:
            min_adv = self.get_param("min_adv_cn")

        day_data = self._full_data.filter(pl.col("date") == d)
        day_rows = {r["instrument_id"]: r for r in day_data.iter_rows(named=True)}

        # 0 大盘 regime 闸：基准跌破均线 → 全平 + 不开新仓
        if not self._regime_ok.get(d, True):
            for iid, sh in list(ctx.positions.items()):
                if sh > 0:
                    row = day_rows.get(iid)
                    if row:
                        sig = ctx.signal("SELL", iid, row["close"], note="regime off liquidate")
                        sig.shares = float(sh)
            return

        # 1+2 候选 + 排序（金叉态 ∧ ADV 达标）
        cand: list[tuple[str, float]] = []
        for iid in ctx.pit_universe(d):
            es = self._rs_short.get((iid, d))
            el = self._rs_long.get((iid, d))
            if es is None or el is None or es <= el:
                continue
            adv = ctx.indicator_value("ADV", iid, d, period=20)
            if adv is None or adv < min_adv:
                continue
            cand.append((iid, es / el - 1.0))
        ranked = [iid for iid, _ in sorted(cand, key=lambda x: x[1], reverse=True)]
        top = ranked[:top_n]
        top_set = set(top)

        # 3 NAV = 现金 + 持仓市值
        positions = ctx.positions
        nav = ctx.available_capital
        for iid, sh in positions.items():
            row = day_rows.get(iid)
            if row and sh:
                nav += sh * row["close"]

        # 4 SELL 跌出 top（显式全平，避免默认 order_size=100 部分平仓）
        for iid, sh in list(positions.items()):
            if sh > 0 and iid not in top_set:
                row = day_rows.get(iid)
                if row:
                    sig = ctx.signal("SELL", iid, row["close"], note="rs_ema rotation exit")
                    sig.shares = float(sh)

        # 5 BUY 新进 top（等权 NAV/top_n）
        if nav > 0 and top_n > 0:
            per = nav / top_n
            for iid in top:
                if positions.get(iid, 0.0) > 0:
                    continue
                row = day_rows.get(iid)
                if not row or row["close"] <= 0:
                    continue
                shares = int(per / row["close"])
                if shares < 1:
                    continue
                es = self._rs_short[(iid, d)]
                el = self._rs_long[(iid, d)]
                sig = ctx.signal(
                    "BUY",
                    iid,
                    row["close"],
                    trigger_value=es / el - 1.0,
                    note=f"rs_ema rotation top{top_n}",
                )
                sig.shares = float(shares)
