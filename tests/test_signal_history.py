# ruff: noqa: ARG002, ARG001
"""
Tests for trendspec/analyzer/signal_history.py.

Tests SignalHistoryBuilder and SignalHistoryStore with mocked builder methods
and synthetic price data.
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from trendspec.analyzer.signal_history import (
    SignalHistoryBuilder,
    SignalHistoryStore,
)
from trendspec.data.markets import Market
from trendspec.strategy.signal import Signal

# =============================================================================
# Helpers
# =============================================================================


def _make_mock_signal(
    instrument_id: str = "SH600000",
    ticker: str = "600000",
    price: float = 10.0,
    rank: float = 1.0,
) -> Signal:
    return Signal(
        direction="BUY",
        ticker=ticker,
        instrument_id=instrument_id,
        price=price,
        extras={"rank": rank},
    )


def _make_mock_screening_result(signals: list[Signal]) -> MagicMock:
    result = MagicMock()
    result.buy_signals = [s for s in signals if s.is_buy()]
    result.sell_signals = [s for s in signals if s.is_sell()]
    return result


def _make_price_bars(
    instrument_id: str,
    start: date,
    n_days: int = 60,
    base_close: float = 100.0,
) -> pl.DataFrame:
    """Generate synthetic daily bars with ascending close prices."""
    dates = []
    d = start
    for _ in range(n_days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        dates.append(d)
        d += timedelta(days=1)

    return pl.DataFrame({
        "instrument_id": [instrument_id] * n_days,
        "date": dates,
        "ticker": [instrument_id.replace("SH", "").replace("SZ", "")] * n_days,
        "open": [base_close + i * 0.1 for i in range(n_days)],
        "high": [base_close + i * 0.1 + 0.5 for i in range(n_days)],
        "low": [base_close + i * 0.1 - 0.3 for i in range(n_days)],
        "close": [base_close + i * 0.1 for i in range(n_days)],
        "volume": [1_000_000] * n_days,
        "adj_factor": [1.0] * n_days,
    })


# =============================================================================
# SignalHistoryStore tests
# =============================================================================


class TestSignalHistoryStore:
    """Test cache save/load."""

    def test_load_returns_none_when_no_cache(self, mock_settings):
        # Use a unique name to avoid pollution from other tests
        result = SignalHistoryStore.load("__nonexistent_strategy_xyz__", Market.CN)
        assert result is None

    def test_save_and_load_roundtrip(self, mock_settings):
        df = pl.DataFrame({
            "instrument_id": ["SH600000", "SZ000001"],
            "n_signals": [10, 5],
            "mean_ret_1d": [0.001, -0.002],
            "mean_ret_5d": [0.005, -0.003],
            "hit_rate_5d": [0.6, 0.4],
            "last_signal_date": [date(2024, 1, 15), date(2024, 1, 15)],
            "last_built_at": [date(2024, 1, 16), date(2024, 1, 16)],
        })

        path = SignalHistoryStore.save(df, "roundtrip_test", Market.CN)
        assert path.exists()
        assert "signal_history" in str(path)
        assert "strategy=roundtrip_test" in str(path)
        assert "market=CN" in str(path)
        assert path.name == "agg.parquet"

        loaded = SignalHistoryStore.load("roundtrip_test", Market.CN)
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded["instrument_id"].to_list() == ["SH600000", "SZ000001"]

    def test_save_creates_parent_dirs(self, mock_settings):
        df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "n_signals": [1],
        })
        path = SignalHistoryStore.save(df, "new_strategy_xyz", Market.US)
        assert path.parent.exists()

    def test_different_markets_separate_files(self, mock_settings):
        df_cn = pl.DataFrame({"instrument_id": ["SH600000"], "n_signals": [5]})
        df_us = pl.DataFrame({"instrument_id": ["AAPL"], "n_signals": [3]})

        SignalHistoryStore.save(df_cn, "multi_market_test", Market.CN)
        SignalHistoryStore.save(df_us, "multi_market_test", Market.US)

        loaded_cn = SignalHistoryStore.load("multi_market_test", Market.CN)
        loaded_us = SignalHistoryStore.load("multi_market_test", Market.US)

        assert loaded_cn["instrument_id"].to_list() == ["SH600000"]
        assert loaded_us["instrument_id"].to_list() == ["AAPL"]


# =============================================================================
# SignalHistoryBuilder tests
# =============================================================================


class TestSignalHistoryBuilder:
    """Test builder with mocked dependencies via patch.object."""

    @pytest.fixture
    def builder(self):
        return SignalHistoryBuilder()

    @pytest.fixture
    def mock_strategy_class(self):
        cls = MagicMock()
        cls.name = "test_strategy"
        return cls

    def test_build_raises_on_unknown_strategy(self, builder):
        with pytest.raises(ValueError, match="Unknown strategy"):
            builder.build("nonexistent_strategy", Market.CN)

    def test_replay_signals_collects_buy_signals(
        self, builder, mock_strategy_class, mock_settings,
    ):
        test_date = date(2024, 1, 15)  # Monday
        signals = [
            _make_mock_signal("SH600000", rank=1.0),
            _make_mock_signal("SZ000001", rank=2.0),
        ]

        with patch.object(
            builder, "_get_trading_days", return_value=[test_date],
        ), patch.object(
            builder, "_run_screen",
            return_value=_make_mock_screening_result(signals),
        ):
            records = builder._replay_signals(
                mock_strategy_class,
                Market.CN,
                start=test_date,
                end=test_date,
            )

        assert len(records) == 2
        assert records[0]["instrument_id"] == "SH600000"
        assert records[0]["rank"] == 1.0
        assert records[1]["instrument_id"] == "SZ000001"
        assert records[1]["rank"] == 2.0

    def test_replay_signals_skips_days_with_errors(
        self, builder, mock_strategy_class, mock_settings,
    ):
        test_date = date(2024, 1, 15)

        with patch.object(
            builder, "_get_trading_days", return_value=[test_date],
        ), patch.object(
            builder, "_run_screen",
            side_effect=RuntimeError("no data"),
        ):
            records = builder._replay_signals(
                mock_strategy_class,
                Market.CN,
                start=test_date,
                end=test_date,
            )

        assert len(records) == 0

    def test_replay_signals_only_buy_signals(
        self, builder, mock_strategy_class, mock_settings,
    ):
        test_date = date(2024, 1, 15)

        buy_sig = _make_mock_signal("SH600000")
        sell_sig = Signal(
            direction="SELL",
            ticker="600000",
            instrument_id="SH600000",
            price=10.0,
        )
        result = _make_mock_screening_result([buy_sig, sell_sig])

        with patch.object(
            builder, "_get_trading_days", return_value=[test_date],
        ), patch.object(
            builder, "_run_screen",
            return_value=result,
        ):
            records = builder._replay_signals(
                mock_strategy_class,
                Market.CN,
                start=test_date,
                end=test_date,
            )

        assert len(records) == 1
        assert records[0]["instrument_id"] == "SH600000"

    def test_attach_forward_returns(self, builder, mock_settings):
        start = date(2024, 1, 1)
        bars = _make_price_bars("SH600000", start, n_days=60, base_close=100.0)

        signal_df = pl.DataFrame({
            "signal_date": [start],
            "instrument_id": ["SH600000"],
            "rank": [1.0],
        })

        with patch.object(builder, "_load_bars", return_value=bars):
            rets = builder._attach_forward_returns(signal_df, Market.CN)

        assert len(rets) == 1
        assert "ret_1d" in rets.columns
        assert "ret_5d" in rets.columns
        assert "ret_20d" in rets.columns
        # ret_1d for price series 100.0, 100.1 -> ~0.001
        assert rets["ret_1d"][0] > 0

    def test_attach_forward_returns_missing_data(self, builder, mock_settings):
        signal_df = pl.DataFrame({
            "signal_date": [date(2024, 1, 15)],
            "instrument_id": ["SH999999"],
            "rank": [1.0],
        })

        with patch.object(builder, "_load_bars", return_value=pl.DataFrame()):
            rets = builder._attach_forward_returns(signal_df, Market.CN)

        assert rets.is_empty()

    def test_aggregate_per_instrument(self, builder):
        rets_df = pl.DataFrame({
            "signal_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "instrument_id": ["SH600000", "SH600000", "SH600000"],
            "rank": [1.0, 2.0, 3.0],
            "ret_1d": [0.01, 0.02, -0.01],
            "ret_3d": [0.03, 0.04, -0.02],
            "ret_5d": [0.05, 0.06, 0.07],
            "ret_10d": [0.10, 0.11, 0.12],
            "ret_20d": [0.20, 0.21, -0.05],
        })

        agg = builder._aggregate_per_instrument(rets_df)

        assert len(agg) == 1
        row = agg.row(0, named=True)
        assert row["instrument_id"] == "SH600000"
        assert row["n_signals"] == 3
        assert abs(row["mean_ret_1d"] - 0.006666) < 0.001
        assert abs(row["hit_rate_5d"] - 1.0) < 0.001  # All 3 positive
        assert "last_built_at" in row
        assert "last_signal_date" in row

    def test_aggregate_multiple_instruments(self, builder):
        rets_df = pl.DataFrame({
            "signal_date": [date(2024, 1, 1), date(2024, 1, 1)],
            "instrument_id": ["SH600000", "SZ000001"],
            "rank": [1.0, 1.0],
            "ret_1d": [0.01, -0.01],
            "ret_3d": [0.03, -0.02],
            "ret_5d": [0.05, -0.03],
            "ret_10d": [0.10, -0.04],
            "ret_20d": [0.20, -0.05],
        })

        agg = builder._aggregate_per_instrument(rets_df)

        assert len(agg) == 2
        inst_ids = set(agg["instrument_id"].to_list())
        assert inst_ids == {"SH600000", "SZ000001"}

    def test_aggregate_filters_low_sample(self, builder):
        rets_df = pl.DataFrame({
            "signal_date": [date(2024, 1, 1)],
            "instrument_id": ["SH600000"],
            "rank": [1.0],
            "ret_1d": [0.01],
            "ret_3d": [0.03],
            "ret_5d": [0.05],
            "ret_10d": [0.10],
            "ret_20d": [0.20],
        })

        agg = builder._aggregate_per_instrument(rets_df)
        assert len(agg) == 1

    def test_empty_aggregate_schema(self, builder):
        empty = builder._empty_aggregate()
        expected_cols = [
            "instrument_id", "n_signals",
            "mean_ret_1d", "mean_ret_3d", "mean_ret_5d",
            "mean_ret_10d", "mean_ret_20d",
            "hit_rate_5d", "hit_rate_20d",
            "last_signal_date", "last_built_at",
        ]
        for col in expected_cols:
            assert col in empty.columns
        assert len(empty) == 0

    def test_end_to_end_with_mocks(
        self, builder, mock_strategy_class, mock_settings,
    ):
        """Full build pipeline with all builder methods mocked."""
        test_start = date(2024, 1, 15)  # Monday

        def mock_screen_fn(_market, _strategy_cls, _target_date):
            return _make_mock_screening_result([
                _make_mock_signal("SH600000", rank=1.0, price=100.0),
                _make_mock_signal("SZ000001", rank=2.0, price=50.0),
            ])

        def mock_bars_fn(_market, instrument_id, _start_date, _end_date):
            return _make_price_bars(
                instrument_id,
                start=test_start,
                n_days=60,
                base_close=100.0,
            )

        with patch.object(
            builder, "_get_trading_days",
            return_value=[test_start],
        ), patch.object(
            builder, "_run_screen",
            side_effect=mock_screen_fn,
        ), patch.object(
            builder, "_load_bars",
            side_effect=mock_bars_fn,
        ), patch(
            "trendspec.analyzer.signal_history.get_strategy",
            return_value=mock_strategy_class,
        ):
            result = builder.build(
                "test_strategy",
                Market.CN,
                lookback_years=1,
                rebuild=True,
            )

        assert len(result) == 2
        inst_ids = set(result["instrument_id"].to_list())
        assert inst_ids == {"SH600000", "SZ000001"}

        sh_row = result.filter(pl.col("instrument_id") == "SH600000").row(0, named=True)
        assert sh_row["n_signals"] >= 1

        cached = SignalHistoryStore.load("test_strategy", Market.CN)
        assert cached is not None
        assert len(cached) == 2


# =============================================================================
# Incremental build tests
# =============================================================================


class TestIncrementalBuild:
    """Test incremental update behavior."""

    @pytest.fixture
    def builder(self):
        return SignalHistoryBuilder()

    @pytest.fixture
    def mock_strategy_class(self):
        cls = MagicMock()
        cls.name = "test_strategy"
        return cls

    def test_incremental_uses_cache_last_signal_date(
        self, builder, mock_strategy_class, mock_settings,
    ):
        """When cache exists, start from last_signal_date."""
        cache_df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "n_signals": [5],
            "mean_ret_1d": [0.001],
            "mean_ret_3d": [0.003],
            "mean_ret_5d": [0.005],
            "mean_ret_10d": [0.010],
            "mean_ret_20d": [0.020],
            "hit_rate_5d": [0.8],
            "hit_rate_20d": [0.6],
            "last_signal_date": [date(2024, 10, 1)],
            "last_built_at": [date(2024, 10, 2)],
        })
        SignalHistoryStore.save(cache_df, "incr_test_strategy", Market.CN)

        screened_dates = []

        def mock_screen_fn(_market, _strategy_cls, target_date):
            screened_dates.append(target_date)
            return _make_mock_screening_result([
                _make_mock_signal("SH600000", rank=1.0, price=100.0),
            ])

        def mock_bars_fn(_market, instrument_id, start_date, _end_date):
            return _make_price_bars(
                instrument_id,
                start=start_date or date(2024, 10, 1),
                n_days=60,
            )

        with patch.object(
            builder, "_get_trading_days",
            return_value=[date(2024, 10, 1), date(2024, 10, 2), date(2024, 10, 3)],
        ), patch.object(
            builder, "_run_screen",
            side_effect=mock_screen_fn,
        ), patch.object(
            builder, "_load_bars",
            side_effect=mock_bars_fn,
        ), patch(
            "trendspec.analyzer.signal_history.get_strategy",
            return_value=mock_strategy_class,
        ):
            builder.build("incr_test_strategy", Market.CN, lookback_years=10)

        # All screened dates should be >= 2024-10-01 (incremental)
        for d in screened_dates:
            assert d >= date(2024, 10, 1), f"Screened unexpected date: {d}"

    def test_rebuild_ignores_cache(
        self, builder, mock_strategy_class, mock_settings,
    ):
        """With rebuild=True, ignore existing cache and use full lookback."""
        cache_df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "n_signals": [5],
            "mean_ret_1d": [0.001],
            "mean_ret_3d": [0.003],
            "mean_ret_5d": [0.005],
            "mean_ret_10d": [0.010],
            "mean_ret_20d": [0.020],
            "hit_rate_5d": [0.8],
            "hit_rate_20d": [0.6],
            "last_signal_date": [date(2024, 10, 1)],
            "last_built_at": [date(2024, 10, 2)],
        })
        SignalHistoryStore.save(cache_df, "rebuild_test_strategy", Market.CN)

        start_date_used = []

        def mock_get_days(_market, start, _end):
            start_date_used.append(start)
            return [date(2024, 1, 15)]

        def mock_screen_fn(_market, _strategy_cls, _target_date):
            return _make_mock_screening_result([])

        def mock_bars_fn(_market, instrument_id, start_date, _end_date):
            return _make_price_bars(
                instrument_id,
                start=start_date or date(2020, 1, 1),
                n_days=60,
            )

        with patch.object(
            builder, "_get_trading_days",
            side_effect=mock_get_days,
        ), patch.object(
            builder, "_run_screen",
            side_effect=mock_screen_fn,
        ), patch.object(
            builder, "_load_bars",
            side_effect=mock_bars_fn,
        ), patch(
            "trendspec.analyzer.signal_history.get_strategy",
            return_value=mock_strategy_class,
        ):
            builder.build("rebuild_test_strategy", Market.CN, lookback_years=5, rebuild=True)

        # With rebuild=True, start should be ~5 years ago from today, not 2024-10-01
        assert len(start_date_used) == 1
        rebuild_start = start_date_used[0]
        assert rebuild_start < date(2024, 10, 1), (
            f"Rebuild should start before cache date, got {rebuild_start}"
        )
