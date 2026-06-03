# tests/test_relative_strength_ema.py
"""Tests for rs_ema_cross relative-strength EMA cross strategy."""

from datetime import date, timedelta

import polars as pl
import pytest

from trendspec.data.markets import Market
from trendspec.ingest.writer import write_parquet
from trendspec.strategy.base import get_strategy
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.examples.relative_strength_ema import RelativeStrengthEMACross


def test_strategy_registered() -> None:
    """rs_ema_cross is discoverable via the registry."""
    cls = get_strategy("rs_ema_cross")
    assert cls is RelativeStrengthEMACross


def test_default_params() -> None:
    """Defaults present even when constructed with no params."""
    strat = RelativeStrengthEMACross()
    assert strat.get_param("benchmark_id") == "QQQ"
    assert strat.get_param("ema_short") == 60
    assert strat.get_param("ema_long") == 120