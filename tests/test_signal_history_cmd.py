"""Tests for trendspec/cli/signal_history_cmd.py."""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import polars as pl
from typer.testing import CliRunner

from trendspec.cli.signal_history_cmd import app

runner = CliRunner()


def _make_agg_df(n_instruments: int = 2, n_signals_each: int = 5) -> pl.DataFrame:
    """Minimal valid aggregate DataFrame for mocking build/status."""
    return pl.DataFrame({
        "instrument_id": [f"SH60000{i}" for i in range(n_instruments)],
        "n_signals": [n_signals_each] * n_instruments,
        "mean_ret_5d": [0.005] * n_instruments,
        "hit_rate_5d": [0.7] * n_instruments,
        "last_signal_date": [date(2024, 10, 1)] * n_instruments,
        "last_built_at": [datetime(2024, 10, 2, 8, 0, 0)] * n_instruments,
    })


# =============================================================================
# build_history tests
# =============================================================================


class TestBuildHistory:
    def test_invalid_market_exits_with_code_1(self):
        result = runner.invoke(app, ["build", "--strategy", "clenow_momentum", "--market", "xx"])
        assert result.exit_code == 1
        assert "不支持的市场" in result.output

    def test_unknown_strategy_shows_available_list(self):
        result = runner.invoke(
            app, ["build", "--strategy", "does_not_exist_xyz", "--market", "us"]
        )
        assert result.exit_code == 1
        assert "未找到策略" in result.output

    def test_builder_exception_exits_with_code_1(self):
        mock_strategy = MagicMock()
        mock_strategy.name = "clenow_momentum"

        with patch("trendspec.cli.signal_history_cmd.build_history", wraps=None), \
             patch("trendspec.strategy.base.get_strategy", return_value=mock_strategy), \
             patch("trendspec.analyzer.signal_history.SignalHistoryBuilder.build",
                   side_effect=RuntimeError("data source unavailable")):
            result = runner.invoke(
                app, ["build", "--strategy", "clenow_momentum", "--market", "us"]
            )
        assert result.exit_code == 1

    def test_empty_result_prints_no_signals_message(self):
        mock_strategy = MagicMock()
        mock_strategy.name = "clenow_momentum"
        empty_df = pl.DataFrame()

        with patch("trendspec.strategy.base.get_strategy", return_value=mock_strategy), \
             patch("trendspec.analyzer.signal_history.SignalHistoryBuilder.build",
                   return_value=empty_df):
            result = runner.invoke(
                app, ["build", "--strategy", "clenow_momentum", "--market", "us"]
            )
        assert result.exit_code == 0
        assert "未产生任何信号" in result.output

    def test_success_prints_instrument_and_signal_counts(self):
        mock_strategy = MagicMock()
        mock_strategy.name = "clenow_momentum"
        agg_df = _make_agg_df(n_instruments=3, n_signals_each=7)

        with patch("trendspec.strategy.base.get_strategy", return_value=mock_strategy), \
             patch("trendspec.analyzer.signal_history.SignalHistoryBuilder.build",
                   return_value=agg_df):
            result = runner.invoke(
                app, ["build", "--strategy", "clenow_momentum", "--market", "us"]
            )
        assert result.exit_code == 0
        assert "构建完成" in result.output
        assert "3" in result.output   # n_instruments
        assert "21" in result.output  # total_signals = 3 * 7


# =============================================================================
# status_history tests
# =============================================================================


class TestStatusHistory:
    def test_invalid_market_exits_with_code_1(self):
        result = runner.invoke(
            app, ["status", "--strategy", "clenow_momentum", "--market", "xx"]
        )
        assert result.exit_code == 1
        assert "不支持的市场" in result.output

    def test_cache_none_shows_not_found(self):
        with patch("trendspec.analyzer.signal_history.SignalHistoryStore.load",
                   return_value=None):
            result = runner.invoke(
                app, ["status", "--strategy", "clenow_momentum", "--market", "us"]
            )
        assert result.exit_code == 0
        assert "缓存不存在" in result.output

    def test_cache_empty_shows_not_found_or_format_mismatch(self):
        # Empty DataFrame has no columns → "last_built_at" missing → shows format-mismatch
        # (not "缓存不存在" — which only fires when df is None)
        with patch("trendspec.analyzer.signal_history.SignalHistoryStore.load",
                   return_value=pl.DataFrame()):
            result = runner.invoke(
                app, ["status", "--strategy", "clenow_momentum", "--market", "us"]
            )
        assert result.exit_code == 0
        # Either message means "cache unavailable" — both show the build hint
        assert "trendspec signal-history build" in result.output

    def test_cache_missing_last_built_at_shows_format_mismatch(self):
        df_no_col = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "n_signals": [5],
        })
        with patch("trendspec.analyzer.signal_history.SignalHistoryStore.load",
                   return_value=df_no_col):
            result = runner.invoke(
                app, ["status", "--strategy", "clenow_momentum", "--market", "us"]
            )
        assert result.exit_code == 0
        assert "缓存格式不匹配" in result.output

    def test_success_path_renders_table(self):
        agg_df = _make_agg_df(n_instruments=5, n_signals_each=10)
        with patch("trendspec.analyzer.signal_history.SignalHistoryStore.load",
                   return_value=agg_df):
            result = runner.invoke(
                app, ["status", "--strategy", "clenow_momentum", "--market", "us"]
            )
        assert result.exit_code == 0
        assert "缓存状态" in result.output
        assert "5" in result.output    # n_rows

    def test_last_built_formatted_via_strftime(self):
        """last_built_at datetime → strftime string in output."""
        agg_df = _make_agg_df()
        with patch("trendspec.analyzer.signal_history.SignalHistoryStore.load",
                   return_value=agg_df):
            result = runner.invoke(
                app, ["status", "--strategy", "clenow_momentum", "--market", "us"]
            )
        assert result.exit_code == 0
        # 2024-10-02 should appear in the formatted last_built_at
        assert "2024-10-02" in result.output
