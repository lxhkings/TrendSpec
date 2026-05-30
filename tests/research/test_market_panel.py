import datetime as dt
import polars as pl
from trendspec.research.market_panel import MarketPanel


def test_slice_matches_subrange():
    rows = []
    for i in range(30):
        d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
        rows.append({"instrument_id": "A", "date": d, "open": 1.0, "high": 1.0,
                     "low": 1.0, "close": 1.0 + i, "volume": 100, "ticker": "A"})
    df = pl.DataFrame(rows)
    panel = MarketPanel(data=df)
    lo, hi = dt.date(2020, 1, 5), dt.date(2020, 1, 10)
    sliced = panel.slice(lo, hi)
    expected = df.filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
    assert sliced.sort("date").equals(expected.sort("date"))
