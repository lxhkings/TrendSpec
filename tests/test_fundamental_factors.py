"""Fundamental factor compute + registration."""

from datetime import date

import polars as pl

import trendspec.factors  # noqa: F401  (triggers registration)
from trendspec.factors.registry import get_factor, list_factors


def _df():
    return pl.DataFrame({
        "instrument_id": ["AAPL", "MSFT"],
        "date": [date(2026, 4, 30), date(2026, 4, 30)],
        "close": [110.0, 200.0],
        "roe": [34.0, 25.0],
        "roic": [30.0, 20.0],
        "net_margin": [10.0, 12.0],
        "op_margin": [12.0, 14.0],
        "revenue_yoy": [0.5, 0.1],
        "net_income_yoy": [0.5, 0.05],
        "eps_ttm": [4.0, 8.0],
    })


def test_fundamental_factors_registered():
    names = list_factors()
    for n in ("fund_roe", "fund_roic", "fund_net_margin", "fund_op_margin",
              "fund_revenue_yoy", "fund_net_income_yoy", "fund_pe_ttm"):
        assert n in names


def test_fund_roe_passthrough():
    res = get_factor("fund_roe").compute_full(_df())
    vals = res.values.sort("instrument_id")
    assert vals["fund_roe"].to_list() == [34.0, 25.0]


def test_fund_pe_ttm_is_close_over_eps():
    res = get_factor("fund_pe_ttm").compute_full(_df())
    vals = res.values.sort("instrument_id")
    # AAPL 110/4 = 27.5 ; MSFT 200/8 = 25.0
    assert vals["fund_pe_ttm"].to_list() == [27.5, 25.0]


def test_fund_pe_ttm_null_when_eps_nonpositive():
    df = _df().with_columns(pl.Series("eps_ttm", [0.0, -1.0]))
    res = get_factor("fund_pe_ttm").compute_full(df)
    assert res.values["fund_pe_ttm"].null_count() == 2


def test_fundamental_factor_missing_column_yields_null():
    bare = pl.DataFrame({
        "instrument_id": ["AAPL"], "date": [date(2026, 4, 30)], "close": [110.0],
    })
    res = get_factor("fund_roe").compute_full(bare)
    assert res.values["fund_roe"].null_count() == 1
