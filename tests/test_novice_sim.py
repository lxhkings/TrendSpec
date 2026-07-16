from datetime import datetime

import numpy as np
import polars as pl

from trendspec.analyzer.ema_cross_winrate import run_novice_simulations, simulate_novice


def _dt(day, hour):
    return datetime(2024, 1, day, hour, 0)


def _cross(rows):
    """構造 cross DataFrame。rows = list of (instrument_id, day, hour, close, signal)"""
    return pl.DataFrame({
        "instrument_id": [r[0] for r in rows],
        "datetime": [_dt(r[1], r[2]) for r in rows],
        "close": [float(r[3]) for r in rows],
        "signal": [r[4] for r in rows],
    })


def test_single_trade_pnl():
    """單一金叉→死叉：final_capital = capital × (exit/entry)"""
    cross = _cross([
        ("A", 1, 9, 100.0, "golden"),
        ("A", 2, 9, 120.0, "death"),
    ])
    rng = np.random.default_rng(0)
    res = simulate_novice(cross, capital=1_000_000, rng=rng)
    assert res["n_trades"] == 1
    assert abs(res["final_capital"] - 1_200_000.0) < 1e-6
    assert abs(res["total_ret"] - 0.2) < 1e-6


def test_force_close_at_end():
    """無死叉時用最後收盤價強制平倉"""
    cross = _cross([
        ("A", 1, 9, 100.0, "golden"),
        ("A", 2, 9, 150.0, None),
        ("A", 3, 9, 130.0, None),
    ])
    rng = np.random.default_rng(0)
    res = simulate_novice(cross, capital=1_000_000, rng=rng)
    assert res["n_trades"] == 1
    assert abs(res["final_capital"] - 1_300_000.0) < 1e-6


def test_compound_two_trades():
    """兩筆交易複利：final = capital × r1 × r2"""
    cross = _cross([
        ("A", 1, 9,  100.0, "golden"),
        ("A", 2, 9,  110.0, "death"),
        ("B", 3, 9,  200.0, "golden"),
        ("B", 4, 9,  220.0, "death"),
    ])
    rng = np.random.default_rng(0)
    res = simulate_novice(cross, capital=1_000_000, rng=rng)
    assert res["n_trades"] == 2
    expected = 1_000_000 * 1.1 * 1.1
    assert abs(res["final_capital"] - expected) < 1e-4


def test_ignore_golden_while_in_position():
    """持倉期間其他金叉忽略（不同 bar）"""
    cross = _cross([
        ("A", 1, 9,  100.0, "golden"),
        ("B", 1, 10, 50.0,  "golden"),   # 不同 bar，A 已持倉，B 忽略
        ("A", 3, 9,  90.0,  "death"),
    ])
    rng = np.random.default_rng(0)
    res = simulate_novice(cross, capital=1_000_000, rng=rng)
    assert res["n_trades"] == 1
    assert res["trades"][0]["instrument_id"] == "A"


def test_same_bar_golden_only_one_chosen():
    """同一 bar 多支金叉：選一支後另一支被忽略（持倉鎖定）"""
    # A、B 同 bar 金叉，noice 選其中一支後，另一支在同 bar 被忽略
    cross = _cross([
        ("A", 1, 9, 100.0, "golden"),
        ("B", 1, 9, 200.0, "golden"),   # 同一 bar (day=1, hour=9)
        ("A", 2, 9, 110.0, "death"),
        ("B", 2, 9, 180.0, "death"),
    ])
    rng = np.random.default_rng(0)
    res = simulate_novice(cross, capital=1_000_000, rng=rng)
    # 只能持1支 → 只有1筆交易
    assert res["n_trades"] == 1
    # 持倉股的死叉觸發賣出
    assert res["trades"][0]["instrument_id"] in ("A", "B")


def test_no_signal_no_trade():
    """無金叉信號 → 不交易，本金不變"""
    cross = _cross([
        ("A", 1, 9, 100.0, None),
        ("A", 2, 9, 110.0, None),
    ])
    rng = np.random.default_rng(0)
    res = simulate_novice(cross, capital=1_000_000, rng=rng)
    assert res["n_trades"] == 0
    assert res["final_capital"] == 1_000_000.0


def test_reproducible_with_seed():
    """固定 seed → 結果完全一致"""
    cross = _cross([
        ("A", 1, 9,  100.0, "golden"),
        ("B", 1, 9,  200.0, "golden"),
        ("A", 2, 9,  110.0, "death"),
        ("B", 2, 9,  180.0, "death"),
    ])
    r1 = simulate_novice(cross, capital=1_000_000, rng=np.random.default_rng(42))
    r2 = simulate_novice(cross, capital=1_000_000, rng=np.random.default_rng(42))
    assert r1["final_capital"] == r2["final_capital"]
    assert r1["trades"][0]["instrument_id"] == r2["trades"][0]["instrument_id"]


def test_run_novice_simulations_row_count():
    """run_novice_simulations 返回 detail 行數 == sims"""
    cross = _cross([
        ("A", 1, 9, 100.0, "golden"),
        ("A", 2, 9, 110.0, "death"),
    ])
    res = run_novice_simulations(cross, sims=50, capital=1_000_000, seed=7)
    assert res["detail"].height == 50
    assert res["summary"]["sims"] == 50


def test_run_novice_simulations_reproducible():
    """固定 seed → run_novice_simulations 結果一致"""
    cross = _cross([
        ("A", 1, 9, 100.0, "golden"),
        ("B", 1, 9, 200.0, "golden"),
        ("A", 2, 9, 115.0, "death"),
        ("B", 2, 9, 190.0, "death"),
    ])
    a = run_novice_simulations(cross, sims=20, capital=1_000_000, seed=99)
    b = run_novice_simulations(cross, sims=20, capital=1_000_000, seed=99)
    assert a["detail"]["final_equity"].to_list() == b["detail"]["final_equity"].to_list()


def test_percentiles_monotone():
    """百分位單調非降"""
    cross = _cross([
        ("A", 1, 9, 100.0, "golden"),
        ("B", 1, 9, 200.0, "golden"),
        ("A", 2, 9, 110.0, "death"),
        ("B", 2, 9, 180.0, "death"),
    ])
    res = run_novice_simulations(cross, sims=100, capital=1_000_000, seed=1)
    eq = [res["percentiles"]["equity"][k] for k in ["p5", "p25", "p50", "p75", "p95"]]
    assert eq == sorted(eq)
