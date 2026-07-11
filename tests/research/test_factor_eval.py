import datetime as dt

import polars as pl
import pytest

from trendspec.research.factor_eval import _attach_forward_returns


def _panel() -> pl.DataFrame:
    """2支股票，20天，close = 10 + i（等差数列，方便手算前瞻收益）。"""
    rows = []
    for iid, base in [("A", 10.0), ("B", 100.0)]:
        for i in range(20):
            d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
            rows.append({"instrument_id": iid, "date": d, "close": base + i})
    return pl.DataFrame(rows)


def test_attach_forward_returns_computes_shifted_ratio():
    df = _panel()
    out = _attach_forward_returns(df, horizon=5)
    row = out.filter((pl.col("instrument_id") == "A") & (pl.col("date") == dt.date(2020, 1, 1)))
    close_t0 = 10.0
    close_t5 = 10.0 + 5
    expected = close_t5 / close_t0 - 1
    assert row["fwd_ret_5d"][0] == pytest.approx(expected)


def test_attach_forward_returns_tail_is_null():
    df = _panel()
    out = _attach_forward_returns(df, horizon=5)
    last_row = out.filter(
        (pl.col("instrument_id") == "A") & (pl.col("date") == dt.date(2020, 1, 1) + dt.timedelta(days=19))
    )
    assert last_row["fwd_ret_5d"][0] is None


def test_attach_forward_returns_does_not_cross_instruments():
    """A 的最后一行不应该拿 B 的 close 算前瞻收益。"""
    df = _panel()
    out = _attach_forward_returns(df, horizon=1)
    second_last_a = out.filter(
        (pl.col("instrument_id") == "A") & (pl.col("date") == dt.date(2020, 1, 1) + dt.timedelta(days=18))
    )
    # A 第19天(index18) close=28, 第20天(index19) close=29, 不应等于 B 的 close
    assert second_last_a["fwd_ret_1d"][0] == pytest.approx(29.0 / 28.0 - 1)
