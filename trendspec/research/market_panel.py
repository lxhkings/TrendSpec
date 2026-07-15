"""市场面板：一次性 load OHLCV + universe，按窗口内存切片，避免重复读盘。"""

from datetime import date as DateType

import polars as pl

from trendspec.data.fundamentals import enrich_daily_panel
from trendspec.data.markets import Market
from trendspec.data.parquet_loader import bars
from trendspec.data.universe import Universe, get_universe


class MarketPanel:
    def __init__(self, data: pl.DataFrame, universe: Universe | None = None) -> None:
        self.data = data
        self.universe = universe

    @classmethod
    def load(
        cls,
        market: str,
        start: DateType,
        end: DateType,
        root: str | None = None,
        adjustment_mode: str = "forward",
    ) -> "MarketPanel":
        m = Market(market.upper())
        df = bars(market=m, start_date=start, end_date=end,
                  adjustment_mode=adjustment_mode, root=root)
        df = enrich_daily_panel(df, m, root)
        uni = get_universe(m, root)
        return cls(data=df, universe=uni)

    def slice(self, start: DateType, end: DateType) -> pl.DataFrame:
        return self.data.filter((pl.col("date") >= start) & (pl.col("date") <= end))
