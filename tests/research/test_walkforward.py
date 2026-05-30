from datetime import date

from trendspec.research.walkforward import run_walkforward


def _fake_backtest(canned):
    """返回一个 backtest_fn：按调用顺序吐 canned metrics。"""
    calls = {"n": 0}

    def _fn(spec_dict, market, start, end, capital):
        m = canned[calls["n"]]
        calls["n"] += 1
        return m

    return _fn, calls


def test_splits_into_n_windows_and_aggregates():
    canned = [
        {"sharpe_ratio": 1.2, "max_drawdown": 0.10, "total_return": 0.15, "total_trades": 8},
        {"sharpe_ratio": 0.8, "max_drawdown": 0.18, "total_return": 0.05, "total_trades": 6},
        {"sharpe_ratio": 1.0, "max_drawdown": 0.12, "total_return": 0.09, "total_trades": 7},
    ]
    fn, calls = _fake_backtest(canned)
    result = run_walkforward(
        spec_dict={"market": "us"}, market="us",
        start=date(2018, 1, 1), end=date(2023, 12, 31),
        n_windows=3, capital=100000.0, backtest_fn=fn,
    )
    assert calls["n"] == 3
    assert len(result.windows) == 3
    assert result.window_sharpes == [1.2, 0.8, 1.0]
    assert result.oos_sharpe == 1.0          # 均值
    assert result.worst_window_sharpe == 0.8  # 最差
    assert result.oos_max_drawdown == 0.18    # 最差(最大)回撤


def test_window_date_ranges_are_contiguous_and_ordered():
    fn, _ = _fake_backtest([{"sharpe_ratio": 1.0, "max_drawdown": 0.1,
                             "total_return": 0.1, "total_trades": 1}] * 2)
    result = run_walkforward(
        spec_dict={"market": "us"}, market="us",
        start=date(2020, 1, 1), end=date(2021, 12, 31),
        n_windows=2, capital=100000.0, backtest_fn=fn,
    )
    w0, w1 = result.windows
    assert w0.start < w0.end <= w1.start < w1.end
