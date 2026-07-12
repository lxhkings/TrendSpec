"""
EMA alignment factor for TrendSpec.

Factors:
- EMAAlignmentFactor: bullish-alignment strength across fast/mid/slow EMAs
"""

from typing import ClassVar

import polars as pl

from trendspec.factors.base import Factor, FactorResult
from trendspec.factors.registry import register


@register("ema_alignment")
class EMAAlignmentFactor(Factor):
    """
    多头排列强度：close > EMA(fast) > EMA(mid) > EMA(slow)（如 EMA20/60/120，
    俗称"2点钟方向"）。

    对相邻两层取相对差 (a/b - 1)，再取三者最小值。任一层排列被破坏（如
    EMA20 跌破 EMA60）时，最小值会被那一层拖成负数，天然把排列被破坏的
    股票压到排名底部；三层都排好时，取值等于最弱一层的差距，即整体排列的
    瓶颈强度，而不会被最强的一层掩盖。

    Note: 这里用标准 span 定义 EMA（alpha = 2/(N+1)），对齐看盘软件里的
    EMA20/60/120；不同于 ma_bias.py 用 half_life 定义的 EMA。

    Parameters:
        fast: 快线周期 (默认 20)
        mid: 中线周期 (默认 60)
        slow: 慢线周期 (默认 120)
    """

    name: ClassVar[str] = "ema_alignment"
    description: ClassVar[str] = (
        "Bullish EMA alignment strength (close > EMA_fast > EMA_mid > EMA_slow)"
    )
    category: ClassVar[str] = "technical"

    def __init__(self, fast: int = 20, mid: int = 60, slow: int = 120) -> None:
        self.params = {"fast": fast, "mid": mid, "slow": slow}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        fast = self.params.get("fast", 20)
        mid = self.params.get("mid", 60)
        slow = self.params.get("slow", 120)

        ema_fast = pl.col("close").ewm_mean(span=fast).over("instrument_id")
        ema_mid = pl.col("close").ewm_mean(span=mid).over("instrument_id")
        ema_slow = pl.col("close").ewm_mean(span=slow).over("instrument_id")

        gap_price_fast = pl.col("close") / ema_fast - 1
        gap_fast_mid = ema_fast / ema_mid - 1
        gap_mid_slow = ema_mid / ema_slow - 1

        return pl.min_horizontal(gap_price_fast, gap_fast_mid, gap_mid_slow)

    def compute_full(self, df: pl.DataFrame) -> FactorResult:
        df_sorted = df.sort("date")
        fast = self.params.get("fast", 20)
        mid = self.params.get("mid", 60)
        slow = self.params.get("slow", 120)
        col_name = f"ema_alignment_{fast}_{mid}_{slow}"

        expr = self.compute(df_sorted)
        df_result = df_sorted.with_columns(expr.alias(col_name))

        result_df = df_result.select(["instrument_id", "date", col_name])

        return FactorResult(
            values=result_df,
            name=col_name,
            metadata={
                "description": self.description,
                "category": self.category,
                "params": self.params,
            },
        )
