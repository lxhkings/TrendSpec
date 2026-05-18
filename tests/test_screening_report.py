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
