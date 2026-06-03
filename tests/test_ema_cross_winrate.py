from datetime import datetime

import polars as pl

from trendspec.research.ema_cross_winrate import (
    aggregate,
    compute_ema_cross,
    current_screen,
    pair_trades,
)


def _dt(h):
    return datetime(2024, 6, 4, h, 30)


def test_compute_ema_cross_detects_golden_and_death():
    """先跌后涨再跌的序列：短 EMA 先下穿后上穿再下穿长 EMA。"""
    # 构造价格：下跌 → 拉升（金叉）→ 跳水（死叉）
    closes = [180] * 5 + [180, 160, 140, 120, 100] + [100, 120, 140, 160, 180] + [180, 120, 90, 70, 60]
    n = len(closes)
    df = pl.DataFrame({
        "instrument_id": ["X"] * n,
        "datetime": [datetime(2024, 6, 4 + i // 7, 13 + i % 7, 30) for i in range(n)],
        "close": [float(c) for c in closes],
    })
    out = compute_ema_cross(df, ema_short=3, ema_long=6)
    signals = out.filter(pl.col("signal").is_not_null()) \
                 .select(["signal"]).to_series().to_list()
    assert "golden" in signals
    assert "death" in signals
    # golden 在 death 之前
    sig_rows = out.filter(pl.col("signal").is_not_null())
    first_two = sig_rows["signal"].to_list()[:2]
    assert first_two[0] == "golden"


def test_pair_trades_golden_to_next_death():
    """手工金叉/死叉事件 → 配对成交。"""
    cross = pl.DataFrame({
        "instrument_id": ["X", "X", "X", "X", "X"],
        "datetime": [_dt(1), _dt(2), _dt(3), _dt(4), _dt(5)],
        "close": [100.0, 110.0, 121.0, 90.0, 95.0],
        "signal": ["golden", None, "death", "golden", None],
    })
    trades = pair_trades(cross)
    assert trades.height == 1  # 第二个 golden 无后续 death → open，不入
    row = trades.row(0, named=True)
    assert row["entry_close"] == 100.0
    assert row["exit_close"] == 121.0
    assert abs(row["ret"] - 0.21) < 1e-9
    assert row["bars_held"] == 2
    assert row["win"] is True


def test_aggregate_metrics():
    trades = pl.DataFrame({
        "instrument_id": ["X", "X", "Y"],
        "ret": [0.20, -0.10, 0.05],
        "bars_held": [4, 2, 6],
        "win": [True, False, True],
    })
    summary = aggregate(trades)
    assert summary["total_trades"] == 3
    assert abs(summary["win_rate"] - 2 / 3) < 1e-9
    assert abs(summary["avg_win"] - 0.125) < 1e-9   # (0.20+0.05)/2
    assert abs(summary["avg_loss"] - (-0.10)) < 1e-9
    assert abs(summary["profit_factor"] - 0.25 / 0.10) < 1e-9  # (0.20+0.05)/0.10
    assert abs(summary["avg_bars_held"] - 4.0) < 1e-9


def test_current_screen_open_golden():
    """最近穿越是 golden 且尾部仍 ema_s>ema_l → 入选。"""
    cross = pl.DataFrame({
        "instrument_id": ["X", "X", "X"],
        "datetime": [_dt(1), _dt(2), _dt(3)],
        "close": [100.0, 110.0, 121.0],
        "ema_s": [99.0, 105.0, 112.0],
        "ema_l": [100.0, 104.0, 108.0],
        "signal": ["golden", None, None],
    })
    screen = current_screen(cross)
    assert screen.height == 1
    row = screen.row(0, named=True)
    assert row["instrument_id"] == "X"
    assert row["cross_dt"] == _dt(1)
    assert row["bars_since"] == 2
    assert abs(row["unrealized_ret"] - 0.21) < 1e-9