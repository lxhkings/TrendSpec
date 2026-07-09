from datetime import date, timedelta

import polars as pl

import trendspec.strategy.factor_strategy as factor_strategy_module
from trendspec.data.markets import Market
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.factor_strategy import FactorStrategy


def _make_bars(iid: str, n: int, start_close: float, drift: float) -> pl.DataFrame:
    rows = []
    close = start_close
    ticker = iid.split("_")[0]
    for i in range(n):
        d = date(2024, 1, 1) + timedelta(days=i)
        rows.append({
            "instrument_id": iid, "date": d, "ticker": ticker,
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": 1_000_000, "adj_factor": 1.0,
        })
        close *= drift
    return pl.DataFrame(rows)


def _two_stock_data() -> pl.DataFrame:
    # FAST 强动量, SLOW 弱动量 —— 同一日截面可比较
    fast = _make_bars("FAST_US", 120, 100.0, 1.004)
    slow = _make_bars("SLOW_US", 120, 100.0, 1.0005)
    return pl.concat([fast, slow])


def _spec_dict():
    return {
        "spec": {
            "market": "us",
            "factors": [{"name": "momentum", "params": {"period": 60},
                         "direction": "high", "weight": 1.0}],
            "top_k": 1,
            "rebalance": 5,
        }
    }


def test_init_builds_score_cache():
    df = _two_stock_data()
    strat = FactorStrategy(params=_spec_dict())
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)

    last_date = df["date"].max()
    ranked = strat._ranked_by_group_date[(last_date, "_all")]
    # 强动量股排第一
    assert ranked[0] == "FAST_US"
    assert strat._score_by_date[(last_date, "FAST_US")] > strat._score_by_date[(last_date, "SLOW_US")]


def test_direction_low_inverts_rank():
    df = _two_stock_data()
    d = _spec_dict()
    d["spec"]["factors"][0]["direction"] = "low"
    strat = FactorStrategy(params=d)
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)

    last_date = df["date"].max()
    # direction=low → 弱动量股反而排第一
    assert strat._ranked_by_group_date[(last_date, "_all")][0] == "SLOW_US"


def _run_next_once(strat, ctx, df, target_date):
    """模拟引擎：对某交易日逐 instrument 调 next()，收集信号。"""
    universe = df["instrument_id"].unique().to_list()
    ctx.set_universe(_StubUniverse(universe))
    day = df.filter(pl.col("date") == target_date)
    rows = {r["instrument_id"]: r for r in day.iter_rows(named=True)}
    ctx.clear_signals()
    for iid in universe:
        row = rows.get(iid)
        if row is None:
            continue
        ctx.update_bar(target_date, iid, row["ticker"], df, current_row=row)
        strat.next(ctx)
    return ctx.pending_signals()


class _StubUniverse:
    def __init__(self, ids):
        self._ids = ids

    def tickers(self, _as_of_date):
        return self._ids


def test_next_emits_buy_for_top_k_on_rebalance():
    df = _two_stock_data()
    strat = FactorStrategy(params=_spec_dict())  # top_k=1
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)

    last_date = df["date"].max()
    sigs = _run_next_once(strat, ctx, df, last_date)
    buys = [s for s in sigs if s.direction == "BUY"]
    # top_k=1 → 只买强动量股
    assert len(buys) == 1
    assert buys[0].instrument_id == "FAST_US"


def test_next_respects_rebalance_interval():
    df = _two_stock_data()
    strat = FactorStrategy(params=_spec_dict())  # rebalance=5
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)

    all_dates = sorted(df["date"].unique().to_list())
    # 第一次调仓日（day index 60）出信号
    sigs1 = _run_next_once(strat, ctx, df, all_dates[60])
    assert len(sigs1) >= 1
    # 紧邻下一日（间隔 1 < 5）不再调仓
    sigs2 = _run_next_once(strat, ctx, df, all_dates[61])
    assert sigs2 == []


def test_init_passes_spec_market_into_get_factor_with_market(monkeypatch):
    """spec.market 必须原样传给 get_factor_with_market，而不是被丢弃或写死。"""
    df = _two_stock_data()
    spec = {
        "spec": {
            "market": "us",
            "factors": [{"name": "rank_within_sector",
                         "params": {"factor_name": "returns"},
                         "direction": "low", "weight": 1.0}],
            "top_k": 1,
            "rebalance": 5,
        }
    }

    captured: dict = {}
    original = factor_strategy_module.get_factor_with_market

    def spy(name, params, market):
        captured["name"] = name
        captured["params"] = params
        captured["market"] = market
        return original(name, params, market)

    monkeypatch.setattr(factor_strategy_module, "get_factor_with_market", spy)

    strat = FactorStrategy(params=spec)
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)

    assert captured["name"] == "rank_within_sector"
    assert captured["market"] == "us"
    assert captured["params"] == {"factor_name": "returns"}


def _make_bars_cn(iid: str, ticker: str, n: int, start_close: float, drift: float) -> pl.DataFrame:
    rows = []
    close = start_close
    for i in range(n):
        d = date(2024, 1, 1) + timedelta(days=i)
        rows.append({
            "instrument_id": iid, "date": d, "ticker": ticker,
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": 1_000_000, "adj_factor": 1.0,
        })
        close *= drift
    return pl.DataFrame(rows)


def _two_group_data() -> pl.DataFrame:
    # 金融组：BANK_FAST(强动量) vs BANK_SLOW(弱动量)
    # 能源组：ENERGY_FAST(强动量) vs ENERGY_SLOW(弱动量)
    return pl.concat([
        _make_bars_cn("SH600000", "600000", 120, 100.0, 1.004),
        _make_bars_cn("SZ000001", "000001", 120, 100.0, 1.0005),
        _make_bars_cn("SH600900", "600900", 120, 100.0, 1.004),
        _make_bars_cn("SZ000002", "000002", 120, 100.0, 1.0005),
    ])


def _sectors_df_for_two_groups() -> pl.DataFrame:
    return pl.DataFrame({
        "instrument_id": ["SH600000", "SZ000001", "SH600900", "SZ000002"],
        "date": [date(2000, 1, 1)] * 4,
        "sector": ["银行", "银行", "煤炭开采", "煤炭开采"],
        "sector_name": [""] * 4,
    })


def test_init_group_by_ranks_within_group_not_globally(tmp_path, monkeypatch):
    """group_by 设置时，每组内部各自排名，组间互不影响。"""
    import trendspec.strategy.factor_strategy as fs_module
    from trendspec.ingest.writer import write_parquet
    from trendspec.data.markets import Market as MarketEnum

    write_parquet(_sectors_df_for_two_groups(), MarketEnum.CN, "sectors", str(tmp_path))

    df = _two_group_data()
    spec_dict = {
        "spec": {
            "market": "cn",
            "factors": [{"name": "momentum", "params": {"period": 60},
                         "direction": "high", "weight": 1.0}],
            "top_k": 1,
            "rebalance": 5,
            "group_by": {"金融": ["银行"], "能源": ["煤炭开采"]},
        }
    }
    strat = FactorStrategy(params=spec_dict)
    ctx = StrategyContext(market=Market.CN, strategy=strat, data=df, root=str(tmp_path))
    strat.init(ctx)

    last_date = df["date"].max()
    assert strat._ranked_by_group_date[(last_date, "金融")][0] == "SH600000"
    assert strat._ranked_by_group_date[(last_date, "能源")][0] == "SH600900"


def test_init_winsorize_caps_extreme_values(tmp_path):
    """极端值被 winsorize 截断，不再无限主导 combo_score。"""
    from trendspec.ingest.writer import write_parquet
    from trendspec.data.markets import Market as MarketEnum

    write_parquet(_sectors_df_for_two_groups(), MarketEnum.CN, "sectors", str(tmp_path))

    df = _two_group_data()
    # 极端拉高 SH600000 最后一天的收盘价，制造离群值
    last_date = df["date"].max()
    df = df.with_columns(
        pl.when((pl.col("instrument_id") == "SH600000") & (pl.col("date") == last_date))
          .then(pl.col("close") * 1000)
          .otherwise(pl.col("close"))
          .alias("close")
    )
    spec_dict = {
        "spec": {
            "market": "cn",
            "factors": [{"name": "momentum", "params": {"period": 60},
                         "direction": "high", "weight": 1.0}],
            "top_k": 1, "rebalance": 5,
            "group_by": {"金融": ["银行"], "能源": ["煤炭开采"]},
            "winsorize_pct": 0.01,
        }
    }
    strat = FactorStrategy(params=spec_dict)
    ctx = StrategyContext(market=Market.CN, strategy=strat, data=df, root=str(tmp_path))
    strat.init(ctx)

    score = strat._score_by_date.get((last_date, "SH600000"))
    assert score is not None
    assert abs(score) < 100  # 未截断时该股 z-score 会是天文数字，截断后应在合理范围


def test_init_missing_factor_excludes_from_ranking(tmp_path):
    """任一因子缺失（null）的股票不参与当期排名。"""
    from trendspec.ingest.writer import write_parquet
    from trendspec.data.markets import Market as MarketEnum

    write_parquet(_sectors_df_for_two_groups(), MarketEnum.CN, "sectors", str(tmp_path))

    df = _two_group_data()
    last_date = df["date"].max()
    # SH600000 最后一天 close 设为 null，动量因子在该日算不出来
    df = df.with_columns(
        pl.when((pl.col("instrument_id") == "SH600000") & (pl.col("date") == last_date))
          .then(None)
          .otherwise(pl.col("close"))
          .alias("close")
    )
    spec_dict = {
        "spec": {
            "market": "cn",
            "factors": [{"name": "momentum", "params": {"period": 60},
                         "direction": "high", "weight": 1.0}],
            "top_k": 5, "rebalance": 5,
            "group_by": {"金融": ["银行"], "能源": ["煤炭开采"]},
        }
    }
    strat = FactorStrategy(params=spec_dict)
    ctx = StrategyContext(market=Market.CN, strategy=strat, data=df, root=str(tmp_path))
    strat.init(ctx)

    ranked = strat._ranked_by_group_date.get((last_date, "金融"), [])
    assert "SH600000" not in ranked
