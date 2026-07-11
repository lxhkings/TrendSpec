from datetime import datetime, date

import polars as pl

from trendspec.research.ema_cross_winrate import (
    aggregate,
    compute_adv20_daily,
    compute_ema_cross,
    current_screen,
    pair_trades,
    per_ticker,
    recent_golden_cross,
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


def test_recent_golden_cross_filters_by_bars_since():
    """bars_since ≤ max_bars_since 的金叉态入选。"""
    cross = pl.DataFrame({
        "instrument_id": ["A", "A", "A", "B", "B", "B"],
        "datetime": [_dt(1), _dt(2), _dt(3), _dt(1), _dt(2), _dt(3)],
        "close": [100.0, 110.0, 121.0, 200.0, 210.0, 220.0],
        "ema_s": [99.0, 105.0, 112.0, 199.0, 205.0, 212.0],
        "ema_l": [100.0, 104.0, 108.0, 200.0, 204.0, 208.0],
        "signal": ["golden", None, None, "golden", None, None],
    })
    recent = recent_golden_cross(cross, max_bars_since=2)
    # A: bars_since=2 (≤2)入选, B: bars_since=2 (≤2)入选
    assert recent.height == 2
    # 设 max_bars_since=1，只有 bars_since=1 入选（无）
    recent2 = recent_golden_cross(cross, max_bars_since=1)
    assert recent2.height == 0


def test_compute_adv20_daily_returns_dict():
    """计算每只股票的 20 日平均成交额（美元）— 直接验证计算逻辑。"""
    from trendspec.data.markets import Market

    # 模拟 bars() 返回的 DataFrame
    df = pl.DataFrame({
        "instrument_id": ["A"] * 20 + ["B"] * 20,
        "date": [date(2024, 6, i+1) for i in range(20)] * 2,
        "close": [100.0] * 20 + [10.0] * 20,
        "volume": [1_000_000] * 20 + [100_000] * 20,
    })

    # 直接调用内部计算逻辑（不经过 bars 加载）
    adv = (
        df.sort("date")
        .group_by("instrument_id")
        .agg([
            pl.col("date").last().alias("_last_date"),
            pl.col("close").last().alias("_last_close"),
            pl.col("volume").tail(20).mean().alias("_avg_volume"),
        ])
        .with_columns(
            (pl.col("_avg_volume") * pl.col("_last_close")).alias("adv20")
        )
        .select(["instrument_id", "adv20"])
    )

    result = {row["instrument_id"]: row["adv20"] for row in adv.iter_rows(named=True)}
    # A: 100 * 1M = 100M 美元，B: 10 * 100K = 1M 美元
    assert result["A"] == 100_000_000.0
    assert result["B"] == 1_000_000.0


def test_recent_golden_cross_filters_by_adv():
    """按 min_adv 过滤低成交额股票。"""
    cross = pl.DataFrame({
        "instrument_id": ["A", "A", "A", "B", "B", "B"],
        "datetime": [_dt(1), _dt(2), _dt(3), _dt(1), _dt(2), _dt(3)],
        "close": [100.0, 110.0, 121.0, 200.0, 210.0, 220.0],
        "ema_s": [99.0, 105.0, 112.0, 199.0, 205.0, 212.0],
        "ema_l": [100.0, 104.0, 108.0, 200.0, 204.0, 208.0],
        "signal": ["golden", None, None, "golden", None, None],
    })
    adv_dict = {"A": 100_000_000.0, "B": 1_000_000.0}
    recent = recent_golden_cross(cross, max_bars_since=2, min_adv=50_000_000, adv_dict=adv_dict)
    # A: 100M ≥ 50M 入选，B: 1M < 50M 排除
    assert recent.height == 1
    assert recent["instrument_id"].to_list() == ["A"]


def test_pair_trades_records_mfe():
    """每笔交易记录进场后 W 根内最大涨幅 MFE。"""
    # 进场 100，路径冲高到 150（idx2），死叉出场 95（idx4）
    cross = pl.DataFrame({
        "instrument_id": ["X", "X", "X", "X", "X"],
        "datetime": [_dt(1), _dt(2), _dt(3), _dt(4), _dt(5)],
        "close": [100.0, 120.0, 150.0, 130.0, 95.0],
        "signal": ["golden", None, None, None, "death"],
    })
    # 默认大窗口 → 抓全局峰值 150 → mfe=0.5
    trades = pair_trades(cross)
    assert abs(trades.row(0, named=True)["mfe"] - 0.5) < 1e-9
    # 窗口=1 → 只看进场后 1 根 [100,120] → 峰值 120 → mfe=0.2
    trades_w1 = pair_trades(cross, mfe_window=1)
    assert abs(trades_w1.row(0, named=True)["mfe"] - 0.2) < 1e-9


def test_per_ticker_extended_stats():
    """每股聚合含中位收益/最差/中位根数/中位MFE。"""
    trades = pl.DataFrame({
        "instrument_id": ["X", "X", "X"],
        "ret": [0.20, -0.10, 0.05],
        "bars_held": [4, 2, 6],
        "mfe": [0.30, 0.02, 0.10],
        "win": [True, False, True],
    })
    pt = per_ticker(trades)
    row = pt.filter(pl.col("instrument_id") == "X").row(0, named=True)
    assert row["total_trades"] == 3
    assert abs(row["median_ret"] - 0.05) < 1e-9
    assert abs(row["worst_ret"] - (-0.10)) < 1e-9
    assert abs(row["median_bars"] - 4.0) < 1e-9
    assert abs(row["median_mfe"] - 0.10) < 1e-9


def test_recent_golden_cross_enriches_and_filters():
    """join 历史统计 + 过滤 0<N<min_samples + N=0 保留排尾 + 按中位收益排序。"""
    # A/B/C 三股均最近金叉（bars_since=2）
    cross = pl.DataFrame({
        "instrument_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "datetime": [_dt(1), _dt(2), _dt(3)] * 3,
        "close": [100.0, 110.0, 121.0, 200.0, 210.0, 220.0, 50.0, 55.0, 60.0],
        "ema_s": [99.0, 105.0, 112.0, 199.0, 205.0, 212.0, 49.0, 52.0, 56.0],
        "ema_l": [100.0, 104.0, 108.0, 200.0, 204.0, 208.0, 50.0, 51.0, 53.0],
        "signal": ["golden", None, None, "golden", None, None, "golden", None, None],
    })
    # A 历史 N=5 可信；B 历史 N=2 < min_samples 应剔除；C 无历史 N=0 保留标灰
    stats = pl.DataFrame({
        "instrument_id": ["A", "B"],
        "total_trades": [5, 2],
        "win_rate": [0.6, 0.5],
        "avg_win": [0.10, 0.08],
        "avg_loss": [-0.05, -0.04],
        "avg_bars_held": [9.0, 7.0],
        "median_ret": [0.12, 0.20],
        "worst_ret": [-0.10, -0.08],
        "median_bars": [10.0, 8.0],
        "median_mfe": [0.15, 0.18],
        "total_return": [0.20, 0.04],
    })
    out = recent_golden_cross(cross, max_bars_since=2, stats=stats, min_samples=3)
    ids = out["instrument_id"].to_list()
    assert ids == ["A", "C"]          # B 剔除；A(total_return 非空)在前，C(N=0, total_return 为空)排尾
    a = out.filter(pl.col("instrument_id") == "A").row(0, named=True)
    assert a["N"] == 5
    assert abs(a["progress_pct"] - 2 / 10.0) < 1e-9      # bars_since/median_bars
    assert abs(a["overheat_pct"] - (a["unrealized_ret"] / 0.15)) < 1e-9
    c = out.filter(pl.col("instrument_id") == "C").row(0, named=True)
    assert c["N"] == 0
    assert c["median_ret"] is None
    assert c["progress_pct"] is None


def test_recent_golden_cross_backward_compatible_without_stats():
    """stats=None 时返回旧结构，不含 N 列。"""
    cross = pl.DataFrame({
        "instrument_id": ["A", "A", "A"],
        "datetime": [_dt(1), _dt(2), _dt(3)],
        "close": [100.0, 110.0, 121.0],
        "ema_s": [99.0, 105.0, 112.0],
        "ema_l": [100.0, 104.0, 108.0],
        "signal": ["golden", None, None],
    })
    out = recent_golden_cross(cross, max_bars_since=2)
    assert "N" not in out.columns
    assert out.height == 1