"""PIT merge of fundamentals into the daily frame (no lookahead)."""

from datetime import date

import polars as pl

from trendspec.data.fundamentals import merge_fundamentals_frame


def _daily():
    return pl.DataFrame({
        "instrument_id": ["AAPL"] * 4,
        "date": [date(2026, 1, 28), date(2026, 1, 29),
                 date(2026, 1, 30), date(2026, 4, 30)],
        "close": [100.0, 101.0, 102.0, 110.0],
    })


def _fund():
    return pl.DataFrame({
        "instrument_id": ["AAPL", "AAPL"],
        "date": [date(2026, 1, 29), date(2026, 4, 30)],  # ann_dates
        "roe": [33.0, 34.0],
        "eps_ttm": [4.0, 4.2],
    })


def test_merge_is_pit_backward():
    merged = merge_fundamentals_frame(_daily(), _fund()).sort("date")
    by_date = {r["date"]: r for r in merged.iter_rows(named=True)}
    # before first announcement -> null (no lookahead)
    assert by_date[date(2026, 1, 28)]["roe"] is None
    # on/after announcement -> visible
    assert by_date[date(2026, 1, 29)]["roe"] == 33.0
    assert by_date[date(2026, 1, 30)]["roe"] == 33.0  # forward-filled
    # next report visible on its ann_date
    assert by_date[date(2026, 4, 30)]["roe"] == 34.0
    assert by_date[date(2026, 4, 30)]["eps_ttm"] == 4.2


def test_merge_empty_fundamentals_returns_daily_unchanged():
    daily = _daily()
    merged = merge_fundamentals_frame(daily, pl.DataFrame())
    assert merged.equals(daily)
