"""Forward-fill (PIT) merge of the fundamentals dataset into the daily frame.

join_asof backward: for each daily (instrument_id, date) row, attach the most
recent fundamentals row whose date (= ann_date) <= the trading date. This makes
each quarterly report visible only on/after its announcement — no lookahead.
"""

import polars as pl

from trendspec.data.markets import Market
from trendspec.data.parquet_loader import _lazyframe_is_empty, scan_parquet


def merge_fundamentals_frame(daily: pl.DataFrame, fund: pl.DataFrame) -> pl.DataFrame:
    """Backward as-of join `fund` onto `daily` by instrument_id over date."""
    if fund.is_empty() or daily.is_empty():
        return daily
    daily_sorted = daily.sort(["instrument_id", "date"])
    fund_sorted = fund.sort(["instrument_id", "date"])
    return daily_sorted.join_asof(
        fund_sorted, on="date", by="instrument_id", strategy="backward"
    )


def merge_fundamentals(daily: pl.DataFrame, market: Market, root: str | None) -> pl.DataFrame:
    """Load the fundamentals dataset for `market` and PIT-merge into `daily`.

    Best-effort: returns `daily` unchanged if the dataset is absent/empty.
    """
    lf = scan_parquet(root, market, "fundamentals")
    if _lazyframe_is_empty(lf):
        return daily
    return merge_fundamentals_frame(daily, lf.collect())


def merge_valuation(daily: pl.DataFrame, market: Market, root: str | None) -> pl.DataFrame:
    """Load the valuation dataset for `market` and PIT-merge into `daily`.

    Same backward as-of join as merge_fundamentals — valuation updates daily
    (date = trade_date) so the join is effectively an exact match on trading
    days, falling back to the prior available snapshot otherwise.

    Kept as a separate dataset/merge from fundamentals (not unioned) because
    the two update at different cadences (quarterly vs daily): unioning them
    would null out fundamentals columns on valuation-only rows, and the
    as-of join would then pick up those nulls instead of the true last
    reported quarter.

    Best-effort: returns `daily` unchanged if the dataset is absent/empty.
    """
    lf = scan_parquet(root, market, "valuation")
    if _lazyframe_is_empty(lf):
        return daily
    return merge_fundamentals_frame(daily, lf.collect())
