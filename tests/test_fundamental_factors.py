"""Fundamental factor compute + registration."""

from datetime import date

import polars as pl

import trendspec.factors  # noqa: F401  (triggers registration)
from trendspec.data.markets import Market
from trendspec.factors.registry import get_factor, list_factors
from trendspec.research.spec import FactorSpec
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.factor_strategy import FactorStrategy


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


def test_factor_combo_accepts_fundamental_factor():
    # spec validation rejects unregistered factors; this confirms registration
    spec = FactorSpec(
        market="us",
        factors=[
            {"name": "fund_roe", "direction": "high", "weight": 1.0},
            {"name": "fund_pe_ttm", "direction": "low", "weight": 1.0},
        ],
        top_k=1,
        rebalance=5,
    )
    assert len(spec.factors) == 2

    # init() must score without raising on a daily frame carrying fundamental cols
    df = pl.DataFrame({
        "instrument_id": ["AAPL", "MSFT", "AAPL", "MSFT"],
        "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
        "date": [date(2026, 4, 30), date(2026, 4, 30),
                 date(2026, 5, 1), date(2026, 5, 1)],
        "close": [110.0, 200.0, 111.0, 201.0],
        "roe": [34.0, 25.0, 34.0, 25.0],
        "eps_ttm": [4.0, 8.0, 4.0, 8.0],
    })
    # BaseStrategy takes params via constructor; StrategyContext needs market + strategy.
    strat = FactorStrategy(params={"spec": spec.model_dump()})
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)
    # AAPL: high roe (34) + low pe (27.5) ; MSFT: roe 25, pe 25.
    # combo z-scores: confirm a ranking was produced for the date.
    ranked = strat._ranked_by_date[date(2026, 4, 30)]
    assert set(ranked) == {"AAPL", "MSFT"}
