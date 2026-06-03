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
        return