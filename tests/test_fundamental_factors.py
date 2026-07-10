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
              "fund_revenue_yoy", "fund_net_income_yoy", "fund_pe_ttm", "fund_pb"):
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


def test_fund_pe_ttm_prefers_direct_pe_ttm_column():
    # CN Tushare daily_basic supplies pe_ttm directly — must win over the
    # close/eps_ttm formula even when eps_ttm is also present.
    df = _df().with_columns(pl.Series("pe_ttm", [18.0, 22.0]))
    res = get_factor("fund_pe_ttm").compute_full(df)
    vals = res.values.sort("instrument_id")
    assert vals["fund_pe_ttm"].to_list() == [18.0, 22.0]


def test_fund_pe_ttm_null_when_direct_pe_ttm_nonpositive():
    df = _df().with_columns(pl.Series("pe_ttm", [0.0, -5.0]))
    res = get_factor("fund_pe_ttm").compute_full(df)
    assert res.values["fund_pe_ttm"].null_count() == 2


def test_fund_pb_passthrough():
    df = _df().with_columns(pl.Series("pb", [3.5, 9.2]))
    res = get_factor("fund_pb").compute_full(df)
    vals = res.values.sort("instrument_id")
    assert vals["fund_pb"].to_list() == [3.5, 9.2]


def test_fund_pb_null_when_missing_or_nonpositive():
    assert get_factor("fund_pb").compute_full(_df()).values["fund_pb"].null_count() == 2
    df = _df().with_columns(pl.Series("pb", [0.0, -1.0]))
    assert get_factor("fund_pb").compute_full(df).values["fund_pb"].null_count() == 2


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
    ranked = strat._ranked_by_group_date[(date(2026, 4, 30), "_all")]
    assert set(ranked) == {"AAPL", "MSFT"}


class _FakeSectorIndex:
    """Minimal stand-in for SectorIndex — maps instrument_id -> sector code."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def sector(self, instrument_id: str, _as_of_date) -> str | None:
        return self._mapping.get(instrument_id)


def test_factor_strategy_sector_filter_restricts_universe():
    spec = FactorSpec(
        market="cn",
        factors=[{"name": "fund_roe", "direction": "high", "weight": 1.0}],
        top_k=5,
        rebalance=1,
        sector_filter=["08"],  # 食品饮料 only
    )
    df = pl.DataFrame({
        "instrument_id": ["A", "B", "C"],
        "ticker": ["A", "B", "C"],
        "date": [date(2026, 4, 30)] * 3,
        "close": [10.0, 20.0, 30.0],
        "roe": [5.0, 20.0, 30.0],
    })
    strat = FactorStrategy(params={"spec": spec.model_dump()})
    ctx = StrategyContext(market=Market.CN, strategy=strat, data=df)
    strat.init(ctx)
    ctx._sector_index = _FakeSectorIndex({"A": "08", "B": "07", "C": "08"})
    ctx._universe = type("U", (), {"tickers": staticmethod(lambda _d: ["A", "B", "C"])})()
    ctx.update_positions({}, 1_000_000.0)
    ctx.update_bar(date(2026, 4, 30), "A", "A", df)
    strat.next(ctx)

    buys = {s.instrument_id for s in ctx.pending_signals() if s.is_buy()}
    # B (sector "07") is excluded even though it has the 2nd-highest ROE.
    assert buys == {"A", "C"}


def test_factor_strategy_no_sector_filter_keeps_full_universe():
    spec = FactorSpec(
        market="cn",
        factors=[{"name": "fund_roe", "direction": "high", "weight": 1.0}],
        top_k=5,
        rebalance=1,
    )
    assert spec.sector_filter is None

    df = pl.DataFrame({
        "instrument_id": ["A", "B"],
        "ticker": ["A", "B"],
        "date": [date(2026, 4, 30)] * 2,
        "close": [10.0, 20.0],
        "roe": [5.0, 20.0],
    })
    strat = FactorStrategy(params={"spec": spec.model_dump()})
    ctx = StrategyContext(market=Market.CN, strategy=strat, data=df)
    strat.init(ctx)
    ctx._universe = type("U", (), {"tickers": staticmethod(lambda _d: ["A", "B"])})()
    ctx.update_positions({}, 1_000_000.0)
    ctx.update_bar(date(2026, 4, 30), "A", "A", df)
    strat.next(ctx)

    buys = {s.instrument_id for s in ctx.pending_signals() if s.is_buy()}
    assert buys == {"A", "B"}


def test_fund_debt_to_assets_passthrough():
    df = _df().with_columns(pl.Series("debt_to_assets", [45.2, 60.1]))
    res = get_factor("fund_debt_to_assets").compute_full(df)
    vals = res.values.sort("instrument_id")
    assert vals["fund_debt_to_assets"].to_list() == [45.2, 60.1]


def test_fund_current_ratio_passthrough():
    df = _df().with_columns(pl.Series("current_ratio", [1.8, 0.9]))
    res = get_factor("fund_current_ratio").compute_full(df)
    vals = res.values.sort("instrument_id")
    assert vals["fund_current_ratio"].to_list() == [1.8, 0.9]


def test_fund_quick_ratio_passthrough():
    df = _df().with_columns(pl.Series("quick_ratio", [1.2, 0.5]))
    res = get_factor("fund_quick_ratio").compute_full(df)
    vals = res.values.sort("instrument_id")
    assert vals["fund_quick_ratio"].to_list() == [1.2, 0.5]


def test_fund_debt_to_eqt_passthrough():
    df = _df().with_columns(pl.Series("debt_to_eqt", [82.5, 150.0]))
    res = get_factor("fund_debt_to_eqt").compute_full(df)
    vals = res.values.sort("instrument_id")
    assert vals["fund_debt_to_eqt"].to_list() == [82.5, 150.0]


def test_leverage_factors_missing_column_yield_null():
    bare = pl.DataFrame({
        "instrument_id": ["AAPL"], "date": [date(2026, 4, 30)], "close": [110.0],
    })
    for name in ("fund_debt_to_assets", "fund_current_ratio",
                 "fund_quick_ratio", "fund_debt_to_eqt"):
        res = get_factor(name).compute_full(bare)
        assert res.values[name].null_count() == 1


def test_fund_ocf_to_debt_passthrough():
    df = _df().with_columns(pl.Series("ocf_to_debt", [0.35, 0.12]))
    res = get_factor("fund_ocf_to_debt").compute_full(df)
    vals = res.values.sort("instrument_id")
    assert vals["fund_ocf_to_debt"].to_list() == [0.35, 0.12]


def test_fund_ocf_to_shortdebt_passthrough():
    df = _df().with_columns(pl.Series("ocf_to_shortdebt", [1.6, 0.8]))
    res = get_factor("fund_ocf_to_shortdebt").compute_full(df)
    vals = res.values.sort("instrument_id")
    assert vals["fund_ocf_to_shortdebt"].to_list() == [1.6, 0.8]


def test_fund_q_ocf_to_sales_passthrough():
    df = _df().with_columns(pl.Series("q_ocf_to_sales", [0.22, 0.05]))
    res = get_factor("fund_q_ocf_to_sales").compute_full(df)
    vals = res.values.sort("instrument_id")
    assert vals["fund_q_ocf_to_sales"].to_list() == [0.22, 0.05]


def test_fund_fcff_passthrough():
    df = _df().with_columns(pl.Series("fcff", [5.0e9, -1.2e9]))
    res = get_factor("fund_fcff").compute_full(df)
    vals = res.values.sort("instrument_id")
    assert vals["fund_fcff"].to_list() == [5.0e9, -1.2e9]


def test_cashflow_quality_factors_missing_column_yield_null():
    bare = pl.DataFrame({
        "instrument_id": ["AAPL"], "date": [date(2026, 4, 30)], "close": [110.0],
    })
    for name in ("fund_ocf_to_debt", "fund_ocf_to_shortdebt",
                 "fund_q_ocf_to_sales", "fund_fcff"):
        res = get_factor(name).compute_full(bare)
        assert res.values[name].null_count() == 1


def test_fund_ps_ttm_passthrough():
    df = _df().with_columns(pl.Series("ps_ttm", [4.5, 8.1]))
    res = get_factor("fund_ps_ttm").compute_full(df)
    vals = res.values.sort("instrument_id")
    assert vals["fund_ps_ttm"].to_list() == [4.5, 8.1]


def test_fund_ps_ttm_null_when_missing_or_nonpositive():
    assert get_factor("fund_ps_ttm").compute_full(_df()).values["fund_ps_ttm"].null_count() == 2
    df = _df().with_columns(pl.Series("ps_ttm", [0.0, -1.0]))
    assert get_factor("fund_ps_ttm").compute_full(df).values["fund_ps_ttm"].null_count() == 2
