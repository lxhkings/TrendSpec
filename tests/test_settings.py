"""Tests for TrendSpec settings configuration."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from trendspec.config import Settings
from trendspec.config.settings import (
    BacktestSettings,
    DatabaseSettings,
    RiskSettings,
)


class TestDatabaseSettings:
    """Tests for database settings."""

    def test_database_settings_requires_host(self) -> None:
        """Database host should be required."""
        with pytest.raises(ValidationError):
            DatabaseSettings()

    def test_database_settings_requires_user(self) -> None:
        """Database user should be required."""
        with patch.dict(os.environ, {"DB_HOST": "localhost"}, clear=False), pytest.raises(
            ValidationError
        ):
            DatabaseSettings()

    def test_database_settings_requires_password(self) -> None:
        """Database password should be required."""
        with patch.dict(
            os.environ, {"DB_HOST": "localhost", "DB_USER": "testuser"}, clear=False
        ), pytest.raises(ValidationError):
            DatabaseSettings()

    def test_database_settings_rejects_root_user(self) -> None:
        """Root user should be rejected for security."""
        with patch.dict(
            os.environ,
            {"DB_HOST": "localhost", "DB_USER": "root", "DB_PASSWORD": "test"},
            clear=False,
        ), pytest.raises(ValueError, match="cannot be 'root'"):
            DatabaseSettings()

    def test_database_settings_accepts_valid_config(self) -> None:
        """Valid database configuration should be accepted."""
        with patch.dict(
            os.environ,
            {
                "DB_HOST": "192.168.8.9",
                "DB_PORT": "3306",
                "DB_USER": "trendspec",
                "DB_PASSWORD": "securepassword",
                "DB_NAME": "stocks",
            },
            clear=False,
        ):
            settings = DatabaseSettings()
            assert settings.host == "192.168.8.9"
            assert settings.port == 3306
            assert settings.user == "trendspec"
            assert settings.password == "securepassword"
            assert settings.name == "stocks"

    def test_connection_url_format(self) -> None:
        """Connection URL should be properly formatted."""
        with patch.dict(
            os.environ,
            {
                "DB_HOST": "localhost",
                "DB_USER": "testuser",
                "DB_PASSWORD": "testpass",
                "DB_NAME": "testdb",
            },
            clear=False,
        ):
            settings = DatabaseSettings()
            url = settings.connection_url
            assert url.startswith("mysql+pymysql://")
            assert "testuser:testpass@localhost:3306/testdb" in url


class TestBacktestSettings:
    """Tests for backtest settings."""

    def test_default_values(self) -> None:
        """Default backtest values should be sensible."""
        settings = BacktestSettings()
        assert settings.initial_capital == 1_000_000.0
        assert settings.commission_rate_cn == 0.0003
        assert settings.commission_rate_us == 0.0005
        assert settings.stamp_duty_cn == 0.001
        assert settings.slippage_bps == 2

    def test_slippage_rate_conversion(self) -> None:
        """Slippage rate should convert basis points to decimal."""
        settings = BacktestSettings(slippage_bps=10)
        assert settings.slippage_rate == 0.001  # 10 bps = 0.1% = 0.001

    def test_initial_capital_must_be_positive(self) -> None:
        """Initial capital must be greater than zero."""
        with pytest.raises(ValidationError):
            BacktestSettings(initial_capital=-100)


class TestRiskSettings:
    """Tests for risk settings."""

    def test_default_values(self) -> None:
        """Default risk values should be sensible."""
        settings = RiskSettings()
        assert settings.risk_free_rate == 0.03
        assert settings.max_position_pct == 0.10
        assert settings.max_sector_pct == 0.25
        assert settings.drawdown_halt_pct == 0.20


class TestSettings:
    """Tests for main settings aggregation."""

    def test_settings_aggregates_all_groups(self) -> None:
        """Settings should contain all settings groups."""
        with patch.dict(
            os.environ,
            {
                "DB_HOST": "localhost",
                "DB_USER": "testuser",
                "DB_PASSWORD": "testpass",
            },
            clear=False,
        ):
            settings = Settings()
            assert hasattr(settings, "db")
            assert hasattr(settings, "data_lake")
            assert hasattr(settings, "backtest")
            assert hasattr(settings, "risk")

    def test_settings_is_cached(self) -> None:
        """Settings.get() should return cached instance."""
        with patch.dict(
            os.environ,
            {
                "DB_HOST": "localhost",
                "DB_USER": "testuser",
                "DB_PASSWORD": "testpass",
            },
            clear=False,
        ):
            # Clear cache first
            Settings.get.cache_clear()
            settings1 = Settings.get()
            settings2 = Settings.get()
            assert settings1 is settings2


class TestLoggingConfig:
    """Tests for logging configuration."""

    def test_get_console_returns_rich_console(self) -> None:
        """get_console should return a Rich Console."""
        from rich.console import Console

        from trendspec.config.logging_config import get_console

        console = get_console()
        assert console is not None
        assert isinstance(console, Console)

    def test_log_messages_exist(self) -> None:
        """LogMessages should have Chinese log templates."""
        from trendspec.config.logging_config import LogMessages

        assert hasattr(LogMessages, "DB_CONNECTING")
        assert hasattr(LogMessages, "DB_CONNECTED")
        assert hasattr(LogMessages, "BACKTEST_START")
        assert "数据库" in LogMessages.DB_CONNECTED  # Contains Chinese
        assert "回测" in LogMessages.BACKTEST_START  # Contains Chinese
