import polars as pl
import pytest

from trendspec.research.ema_cross_winrate import monte_carlo


def _pool(rets):
    """手造抽样池：每个 ret 一行 trade（仅填 monte_carlo 用到的列）。"""
    n = len(rets)
    return pl.DataFrame({
        "instrument_id": [f"T{i}" for i in range(n)],
        "entry_dt": [None] * n,
        "exit_dt": [None] * n,
        "ret": [float(r) for r in rets],
    })


def test_reproducible_with_seed():
    pool = _pool([0.1, -0.05, 0.2, -0.1, 0.3])
    a = monte_carlo(pool, sims=50, capital=1_000_000, seed=42)
    b = monte_carlo(pool, sims=50, capital=1_000_000, seed=42)
    assert a["detail"]["pnl_usd"].to_list() == b["detail"]["pnl_usd"].to_list()


def test_sims_row_count():
    pool = _pool([0.1, -0.05, 0.2])
    res = monte_carlo(pool, sims=100, capital=1_000_000, seed=1)
    assert res["detail"].height == 100
    assert res["summary"]["sims"] == 100


def test_pnl_and_equity_math():
    pool = _pool([0.25])  # 单一 ret → 每次必抽 0.25
    res = monte_carlo(pool, sims=10, capital=1_000_000, seed=7)
    detail = res["detail"]
    assert all(abs(p - 250_000.0) < 1e-6 for p in detail["pnl_usd"].to_list())
    assert all(abs(e - 1_250_000.0) < 1e-6 for e in detail["final_equity"].to_list())
    assert abs(res["summary"]["mean_equity"] - 1_250_000.0) < 1e-6
    assert res["summary"]["win_rate"] == 1.0


def test_with_replacement_pool_smaller_than_sims():
    pool = _pool([0.1, -0.1])  # 池仅 2 笔
    res = monte_carlo(pool, sims=100, capital=1_000_000, seed=3)
    assert res["detail"].height == 100  # 放回抽样可超池容量


def test_empty_pool_raises():
    empty = _pool([])
    with pytest.raises(RuntimeError):
        monte_carlo(empty, sims=10)


def test_percentiles_present():
    pool = _pool([0.1, -0.05, 0.2, -0.1, 0.3])
    res = monte_carlo(pool, sims=100, capital=1_000_000, seed=9)
    pct = res["percentiles"]
    for k in ["p5", "p25", "p50", "p75", "p95"]:
        assert k in pct["equity"]
        assert k in pct["ret"]
    # 单调非降
    eq = [pct["equity"][k] for k in ["p5", "p25", "p50", "p75", "p95"]]
    assert eq == sorted(eq)
