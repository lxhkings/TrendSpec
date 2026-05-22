"""Tests for ScreeningReport: filename + clenow-specific rendering."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from trendspec.analyzer.signal_history import SignalHistoryStore
from trendspec.data.markets import Market
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
            strategy_name="ema_cluster_pullback",
            market="us",
        )
        out = report.export(tmp_path)
        assert "ema_cluster_pullback" in out.name
        assert "20260518" in out.name
        assert out.name.endswith(".csv")
        assert out.exists()

    def test_csv_filename_distinct_per_strategy(self, tmp_path: Path) -> None:
        sigs = [_buy_signal("AAPL", 100.0)]
        ScreeningReport(
            signals=sigs,
            screening_date=date(2026, 5, 18),
            strategy_name="ema_cluster_pullback",
            market="us",
        ).export(tmp_path)
        ScreeningReport(
            signals=sigs,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        ).export(tmp_path)
        files = sorted(p.name for p in tmp_path.glob("signals_*.csv"))
        assert len(files) == 2
        assert any("ema_cluster_pullback" in f for f in files)
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

    def test_clenow_buy_table_has_12_columns(self) -> None:
        signals = [self._clenow_signal("LITE", 1001.81)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        table = report._create_signals_table(signals, "买入信号")
        assert len(table.columns) == 12
        col_headers = [c.header for c in table.columns]
        assert col_headers == [
            "股票代码",
            "行业",
            "选股排名",
            "建议买入价",
            "初始止损线",
            "趋势质量 (R²)",
            "乖离率 (距 MA200)",
            "回撤 (距 63 日高点)",
            "放量倍数",
            "备注/预警",
            "历史 5d 收益 %",
            "信号置信度",
        ]

    def test_non_clenow_buy_table_keeps_6_columns(self) -> None:
        signals = [_buy_signal("AAPL", 100.0)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="ema_cluster_pullback",
            market="us",
        )
        table = report._create_signals_table(signals, "买入信号")
        assert len(table.columns) == 6

    def test_clenow_sell_table_uses_default_6_columns(self) -> None:
        """SELL signals still use 6 columns even when strategy_name is clenow_momentum."""
        sell = Signal(
            direction="SELL",
            ticker="LITE",
            instrument_id="LITE",
            price=900.0,
            note="below SMA200",
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
        signals = [
            self._clenow_signal(
                "CIEN",
                591.57,
                alerts=["均线乖离过大", "量能萎缩"],
            )
        ]
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
            signals=[],
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        for r2, label in [(0.90, "极平稳"), (0.80, "优秀"), (0.70, "良好"), (0.50, "一般")]:
            sig = self._clenow_signal("X", 100.0, r2=r2)
            rows = list(report._iter_clenow_buy_rows([sig]))
            assert label in rows[0][5]


class TestClenowCSVSchema:
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

    def test_clenow_csv_has_19_columns(self, tmp_path: Path) -> None:
        signals = [self._clenow_signal("LITE", 1001.81)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        out = report.export(tmp_path)
        df = pl.read_csv(out)
        assert df.columns == [
            "股票代码",
            "instrument_id",
            "日期",
            "方向",
            "行业",
            "选股排名",
            "建议买入价",
            "初始止损线",
            "趋势质量 (R²)",
            "乖离率 (距 MA200)",
            "回撤 (距 63 日高点)",
            "放量倍数",
            "备注/预警",
            "历史样本数",
            "历史 1d 均值收益 %",
            "历史 5d 均值收益 %",
            "历史 20d 均值收益 %",
            "历史 5d 胜率 %",
            "信号置信度",
        ]

    def test_clenow_csv_buy_row_fully_populated(self, tmp_path: Path) -> None:
        signals = [self._clenow_signal("CIEN", 591.57, alerts=["量能萎缩"])]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        out = report.export(tmp_path)
        df = pl.read_csv(out)
        row = df.row(0, named=True)
        assert row["股票代码"] == "CIEN"
        assert row["方向"] == "BUY"
        assert row["行业"] == "Technology"
        assert row["选股排名"] == 1
        assert row["建议买入价"] == pytest.approx(591.57)
        assert "量能萎缩" in str(row["备注/预警"])

    def test_clenow_csv_sell_row_blanks_display_cols(self, tmp_path: Path) -> None:
        buy = self._clenow_signal("LITE", 1001.81)
        sell = Signal(
            direction="SELL",
            ticker="OLD",
            instrument_id="OLD",
            price=50.0,
            note="below SMA200",
        )
        report = ScreeningReport(
            signals=[buy, sell],
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        out = report.export(tmp_path)
        df = pl.read_csv(out)
        sell_row = df.filter(pl.col("方向") == "SELL").row(0, named=True)
        assert sell_row["股票代码"] == "OLD"
        assert sell_row["建议买入价"] == pytest.approx(50.0)
        assert sell_row["备注/预警"] == "below SMA200"
        for col in [
            "行业",
            "选股排名",
            "初始止损线",
            "趋势质量 (R²)",
            "乖离率 (距 MA200)",
            "回撤 (距 63 日高点)",
            "放量倍数",
        ]:
            v = sell_row[col]
            assert v is None or v == "" or v == 0

    def test_non_clenow_csv_keeps_7_columns(self, tmp_path: Path) -> None:
        signals = [_buy_signal("AAPL", 100.0)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="ema_cluster_pullback",
            market="us",
        )
        out = report.export(tmp_path)
        df = pl.read_csv(out)
        assert df.columns == [
            "股票代码",
            "instrument_id",
            "日期",
            "方向",
            "价格",
            "触发指标值",
            "备注",
        ]


class TestSignalHistoryIntegration:
    """Tests for SignalHistoryStore integration in the clenow report."""

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

    def test_cache_miss_csv_has_defaults(self, tmp_path: Path) -> None:
        """When SignalHistoryStore returns None, new columns have default values."""
        signals = [self._clenow_signal("LITE", 1001.81)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        with patch.object(report, "_load_signal_history", return_value=None):
            out = report.export(tmp_path)
        df = pl.read_csv(out)
        assert df.shape[0] == 1
        # Default values on cache miss
        assert df["历史样本数"][0] == 0
        assert df["信号置信度"][0] == "-"
        # Return columns are null
        assert df["历史 1d 均值收益 %"][0] is None
        assert df["历史 5d 均值收益 %"][0] is None
        assert df["历史 20d 均值收益 %"][0] is None
        assert df["历史 5d 胜率 %"][0] is None

    def test_cache_miss_terminal_table_still_works(self) -> None:
        """Terminal table renders with 12 columns even when history cache is empty."""
        signals = [self._clenow_signal("LITE", 1000.0)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        # hist_cache=None simulates cache miss
        rows = list(report._iter_clenow_buy_rows(signals, hist_cache=None))
        assert len(rows) == 1
        assert len(rows[0]) == 12
        # hist_5d and conf should be "-"
        assert rows[0][10] == "-"  # 历史 5d 收益 %
        assert rows[0][11] == "-"  # 信号置信度

    def test_signal_history_csv_columns_populated(self, tmp_path: Path) -> None:
        """When SignalHistoryStore returns data, new CSV columns are filled."""
        hist_df = pl.DataFrame(
            {
                "instrument_id": ["LITE"],
                "n_signals": [8],
                "mean_ret_1d": [0.005],
                "mean_ret_5d": [0.012],
                "mean_ret_20d": [0.03],
                "hit_rate_5d": [0.75],
            }
        )

        signals = [self._clenow_signal("LITE", 1001.81)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )

        # Bypass actual _load_signal_history by providing cache to _iter_clenow_buy_rows
        # and directly to _signals_to_clenow_dataframe via mock
        from unittest.mock import patch

        with patch.object(report, "_load_signal_history", return_value=hist_df):
            out = report.export(tmp_path)

        df = pl.read_csv(out)
        row = df.row(0, named=True)
        assert row["历史样本数"] == 8
        assert row["历史 1d 均值收益 %"] == pytest.approx(0.5)
        assert row["历史 5d 均值收益 %"] == pytest.approx(1.2)
        assert row["历史 20d 均值收益 %"] == pytest.approx(3.0)
        assert row["历史 5d 胜率 %"] == pytest.approx(75.0)
        # 8 signals => 5 <= n < 10 => "★★"
        assert row["信号置信度"] == "★★"

    def test_signal_history_terminal_table_populated(self) -> None:
        """Terminal table shows historical data and confidence stars."""
        hist_df = pl.DataFrame(
            {
                "instrument_id": ["LITE"],
                "n_signals": [15],
                "mean_ret_1d": [0.003],
                "mean_ret_5d": [0.015],
                "mean_ret_20d": [0.02],
                "hit_rate_5d": [0.80],
            }
        )

        signals = [self._clenow_signal("LITE", 1000.0)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )

        rows = list(report._iter_clenow_buy_rows(signals, hist_cache=hist_df))
        assert len(rows) == 1
        assert len(rows[0]) == 12
        # 历史 5d 收益 % => "+1.50%"
        assert rows[0][10] == "+1.50%"
        # 15 signals => "★★★"
        assert rows[0][11] == "★★★"

    def test_signal_history_small_sample_one_star(self) -> None:
        """n_signals < 5 => one star."""
        hist_df = pl.DataFrame(
            {
                "instrument_id": ["X"],
                "n_signals": [3],
                "mean_ret_5d": [0.01],
            }
        )
        signals = [self._clenow_signal("X", 50.0)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        rows = list(report._iter_clenow_buy_rows(signals, hist_cache=hist_df))
        assert rows[0][11] == "★"

    def test_signal_history_zero_signals_dash(self) -> None:
        """n_signals == 0 => dash for confidence."""
        hist_df = pl.DataFrame(
            {
                "instrument_id": ["Y"],
                "n_signals": [0],
                "mean_ret_5d": [0.0],
            }
        )
        signals = [self._clenow_signal("Y", 50.0)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        rows = list(report._iter_clenow_buy_rows(signals, hist_cache=hist_df))
        assert rows[0][11] == "-"

    def test_load_signal_history_lowercase_market_does_not_raise(self) -> None:
        """_load_signal_history must not raise for lowercase market strings.

        screen_cmd.py passes market as lowercase ("us"/"cn"), which previously
        caused Market("us") to raise ValueError, silently swallowed and returning
        None for every real CLI invocation.
        """
        signals = [self._clenow_signal("LITE", 1000.0)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",  # lowercase — what screen_cmd.py actually passes
        )
        # Patch Store.load directly so we don't touch the filesystem,
        # but DO NOT mock _load_signal_history itself (that's the code under test)
        with patch.object(SignalHistoryStore, "load", return_value=None) as mock_load:
            result = report._load_signal_history()

        assert result is None  # Store returned None, no exception
        # Verify the Market enum was constructed with uppercase "US"
        mock_load.assert_called_once_with("clenow_momentum", Market.US)
