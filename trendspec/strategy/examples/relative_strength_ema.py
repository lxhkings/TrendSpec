"""
Relative Strength EMA Cross strategy (rs_ema_cross).

对每只股票计算其相对基准 (QQQ) 的比值 ratio = close / benchmark_close，
在 ratio 序列上取 EMA60/EMA120：
  BUY  = EMA_short > EMA_long  且 空仓
  SELL = EMA_short <= EMA_long 且 持仓
状态型语义，全美股池，仓位走引擎默认。
"""

from typing import Any

from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext

_DEFAULTS = {
    "benchmark_id": "QQQ",
    "ema_short": 60,
    "ema_long": 120,
}


@register_strategy("rs_ema_cross")
class RelativeStrengthEMACross(BaseStrategy):
    """股票/基准比值的 EMA 短长周期交叉。"""

    name = "rs_ema_cross"
    version = "1.0.0"
    params: dict[str, Any] = dict(_DEFAULTS)

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged = dict(_DEFAULTS)
        if params:
            merged.update(params)
        super().__init__(params=merged)

    def init(self, ctx: StrategyContext) -> None:
        import polars as pl

        from trendspec.data.parquet_loader import read_indices
        from trendspec.strategy.indicators import compute_indicator

        self._rs_short = {}
        self._rs_long = {}

        bench_id = self.get_param("benchmark_id")
        short = self.get_param("ema_short")
        long = self.get_param("ema_long")

        bench = read_indices(ctx.market, root=ctx._root, instrument_ids=[bench_id])
        if bench.is_empty():
            raise RuntimeError(
                f"No index data for {bench_id}. "
                f"Run `trendspec ingest indices --market {ctx.market.value}` first."
            )

        bench = bench.select(["date", pl.col("close").alias("_bench_close")])
        data = ctx._data
        if data is None or data.is_empty():
            return

        ratio_df = (
            data.join(bench, on="date", how="inner")
            .with_columns(
                # 比值替换 close 列，其他列保持不变（compute_indicator 需要 OHLCV）
                (pl.col("close") / pl.col("_bench_close")).alias("close")
            )
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
            .filter(pl.col("_bar") >= long)  # 剔除预热期（不足 ema_long 个 bar）
        )

        for iid, dt, es, el in merged.select(
            ["instrument_id", "date", short_col, long_col]
        ).iter_rows():
            if es is not None and el is not None:
                self._rs_short[(iid, dt)] = es
                self._rs_long[(iid, dt)] = el

    def next(self, ctx: StrategyContext) -> None:
        iid = ctx.instrument_id
        d = ctx.date
        es = self._rs_short.get((iid, d))
        el = self._rs_long.get((iid, d))
        if es is None or el is None:
            return

        if not ctx.has_position() and es > el:
            ctx.signal("BUY", iid, ctx.close, trigger_value=es - el, note="rs_ema_cross BUY")
        elif ctx.has_position() and es <= el:
            ctx.signal("SELL", iid, ctx.close, trigger_value=es - el, note="rs_ema_cross SELL")
