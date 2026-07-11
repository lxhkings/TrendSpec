"""
TrendSpec settings module.

Centralized configuration management using pydantic-settings.
All credentials are loaded from environment variables.
No hardcoded credentials allowed.
"""

from functools import lru_cache
from urllib.parse import quote_plus

import warnings

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """MariaDB/MySQL database connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(..., description="Database host address")
    port: int = Field(default=3306, description="Database port")
    user: str = Field(..., description="Database username (must be read-only)")
    password: str = Field(..., description="Database password")
    name: str = Field(default="stocks", description="Database name")
    charset: str = Field(default="utf8mb4", description="Database charset")
    # Read directly from env/dotenv (no DB_ prefix — uses validation_alias)
    allow_root_db_user: bool = Field(
        default=False,
        validation_alias="ALLOW_ROOT_DB_USER",
        description="Allow root DB user (development only)",
    )

    @model_validator(mode="after")
    def check_not_root(self) -> "DatabaseSettings":
        """Ensure database user is not root unless ALLOW_ROOT_DB_USER=true."""
        if self.user.lower() == "root":
            if not self.allow_root_db_user:
                raise ValueError(
                    "DB_USER cannot be 'root'. Use a read-only account for security. "
                    "Create one with: CREATE USER 'trendspec'@'%' IDENTIFIED BY '<password>'; "
                    "GRANT SELECT ON stocks.* TO 'trendspec'@'%'; "
                    "Or set ALLOW_ROOT_DB_USER=true in .env for development."
                )
            warnings.warn("DB_USER=root is insecure. Development only.", UserWarning, stacklevel=2)
        return self

    @property
    def connection_url(self) -> str:
        """Build SQLAlchemy connection URL with URL-encoded credentials.

        connect_timeout/read_timeout bound how long a hung/unresponsive NAS
        connection can block the caller — without them, a stalled TCP socket
        blocks forever with no way to recover short of killing the process.
        """
        return (
            f"mysql+pymysql://{quote_plus(self.user)}:{quote_plus(self.password)}"
            f"@{self.host}:{self.port}/{self.name}?charset={self.charset}"
            f"&connect_timeout=10&read_timeout=120"
        )


class DataLakeSettings(BaseSettings):
    """Local data lake (Parquet cache) settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_lake_root: str = Field(
        default="./data_lake",
        description="Root directory for Parquet cache",
    )


class BacktestSettings(BaseSettings):
    """Backtest parameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Field names automatically map to uppercase env vars
    # e.g., initial_capital -> INITIAL_CAPITAL
    initial_capital: float = Field(
        default=1_000_000.0,
        gt=0,
        description="Initial capital for backtesting",
    )
    commission_rate_cn: float = Field(
        default=0.0003,
        ge=0,
        le=0.01,
        description="Commission rate for China A-shares",
    )
    commission_rate_us: float = Field(
        default=0.0005,
        ge=0,
        le=0.01,
        description="Commission rate for US stocks",
    )
    stamp_duty_cn: float = Field(
        default=0.001,
        ge=0,
        le=0.01,
        description="Stamp duty for China A-shares (sell only)",
    )
    slippage_bps: int = Field(
        default=2,
        ge=0,
        le=100,
        description="Slippage in basis points",
    )

    @property
    def slippage_rate(self) -> float:
        """Convert basis points to decimal rate."""
        return self.slippage_bps / 10000.0


class RiskSettings(BaseSettings):
    """Risk management parameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Field names automatically map to uppercase env vars
    risk_free_rate: float = Field(
        default=0.03,
        ge=0,
        le=0.2,
        description="Risk-free rate for Sharpe ratio calculation",
    )
    max_position_pct: float = Field(
        default=0.10,
        gt=0,
        le=1.0,
        description="Maximum percentage of capital in single position",
    )
    max_sector_pct: float = Field(
        default=0.25,
        gt=0,
        le=1.0,
        description="Maximum percentage of capital in single sector",
    )
    drawdown_halt_pct: float = Field(
        default=0.20,
        gt=0,
        le=1.0,
        description="Drawdown threshold for halting trading",
    )


class Settings(BaseSettings):
    """
    Main TrendSpec configuration.

    Aggregates all settings groups and provides centralized access.
    All sensitive values are loaded from environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested settings groups
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    data_lake: DataLakeSettings = Field(default_factory=DataLakeSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)

    @classmethod
    @lru_cache
    def get(cls) -> "Settings":
        """
        Get cached settings instance.

        Settings are loaded once and cached for the application lifetime.
        """
        return cls()


# Convenience function for accessing settings
def get_settings() -> Settings:
    """Get the application settings instance."""
    return Settings.get()
