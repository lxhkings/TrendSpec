"""Tests for ScreeningReport: filename + clenow-specific rendering."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from trendspec.screening.report import ScreeningReport
from trendspec.strategy.signal import Signal


def _buy_signal(ticker: str, price: float, extras: dict | None = None) -> Signal:
    return Signal(
        direction="BUY",
        ticker=ticker,
        instrument_id=ticker,
        price=price,
        trigger_value=1.0,
        note="",
        extras=extras or {},
    )


class TestCSVFilename:
    def test_csv_filename_contains_strategy_name(self, tmp_path: Path) -> None:
        signals = [_buy_signal("AAPL", 100.0)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="ma_cross",
            market="us",
        )
        out = report.export(tmp_path)
        assert "ma_cross" in out.name
        assert "20260518" in out.name
        assert out.name.endswith(".csv")
        assert out.exists()

    def test_csv_filename_distinct_per_strategy(self, tmp_path: Path) -> None:
        sigs = [_buy_signal("AAPL", 100.0)]
        ScreeningReport(
            signals=sigs, screening_date=date(2026, 5, 18),
            strategy_name="ma_cross", market="us",
        ).export(tmp_path)
        ScreeningReport(
            signals=sigs, screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum", market="us",
        ).export(tmp_path)
        files = sorted(p.name for p in tmp_path.glob("signals_*.csv"))
        assert len(files) == 2
        assert any("ma_cross" in f for f in files)
        assert any("clenow_momentum" in f for f in files)


class TestClenowBuyTableRendering:
    def _clenow_signal(self, ticker: str, price: float, **extras_override) -> Signal:
        extras = {
            "sector": "Technology",
            "rank": 1,
            "r2": 0.85,
            "deviation_pct": 32.5,
            "drawdown_pct": -2.1,
            "vol_mult": 1.5,
            "stop_loss": price * 0.77,
            "alerts": [],
        }
        extras.update(extras_override)
        return _buy_signal(ticker, price, extras)

    def test_clenow_buy_table_has_10_columns(self) -> None:
        signals = [self._clenow_signal("LITE", 1001.81)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        table = report._create_signals_table(signals, "买入信号")
        assert len(table.columns) == 10
        col_headers = [c.header for c in table.columns]
        assert col_headers == [
            "股票代码", "行业", "选股排名", "建议买入价", "初始止损线",
            "趋势质量 (R²)", "乖离率 (距 MA200)", "回撤 (距 63 日高点)",
            "放量倍数", "备注/预警",
        ]

    def test_non_clenow_buy_table_keeps_6_columns(self) -> None:
        signals = [_buy_signal("AAPL", 100.0)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="ma_cross",
            market="us",
        )
        table = report._create_signals_table(signals, "买入信号")
        assert len(table.columns) == 6

    def test_clenow_sell_table_uses_default_6_columns(self) -> None:
        """SELL signals still use 6 columns even when strategy_name is clenow_momentum."""
        sell = Signal(
            direction="SELL", ticker="LITE", instrument_id="LITE",
            price=900.0, note="below SMA200",
        )
        report = ScreeningReport(
            signals=[sell],
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        table = report._create_signals_table([sell], "卖出信号")
        assert len(table.columns) == 6

    def test_clenow_sector_none_renders_dash(self) -> None:
        signals = [self._clenow_signal("LITE", 1000.0, sector=None)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        rows = list(report._iter_clenow_buy_rows(signals))
        assert rows[0][1] == "-"  # sector column

    def test_clenow_alerts_renders_with_prefix(self) -> None:
        signals = [self._clenow_signal(
            "CIEN", 591.57,
            alerts=["均线乖离过大", "量能萎缩"],
        )]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        rows = list(report._iter_clenow_buy_rows(signals))
        assert rows[0][9] == "[警报] 均线乖离过大，量能萎缩"

    def test_clenow_no_alerts_renders_normal(self) -> None:
        signals = [self._clenow_signal("LITE", 1001.81, alerts=[])]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        rows = list(report._iter_clenow_buy_rows(signals))
        assert rows[0][9] == "正常"

    def test_clenow_r2_label_buckets(self) -> None:
        report = ScreeningReport(
            signals=[], screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum", market="us",
        )
        for r2, label in [(0.90, "极平稳"), (0.80, "优秀"), (0.70, "良好"), (0.50, "一般")]:
            sig = self._clenow_signal("X", 100.0, r2=r2)
            rows = list(report._iter_clenow_buy_rows([sig]))
            assert label in rows[0][5]
