"""
Transaction costs model for TrendSpec execution engines.

Provides realistic transaction cost modeling for different markets:
- CN_A: A-shares with commission 0.03%, stamp duty 0.1% (sell only)
- US: US stocks with commission 0.05%, no stamp duty
- HK: Hong Kong stocks (placeholder)

Key design:
- CostsModel interface for extensibility
- Per-market configurable via Market enum
- Slippage model (basis points)
- Commission, stamp duty, transfer fees

Usage:
    >>> costs = CNACostsModel()
    >>> cost = costs.calculate("BUY", 10000)  # 10000 CNY trade value
    >>> cost
    3.0  # Commission only for buy
    >>> cost = costs.calculate("SELL", 10000)
    >>> cost
    13.0  # Commission + stamp duty for sell
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from trendspec.data.markets import Market


@dataclass
class CostsConfig:
    """
    Configuration for transaction costs.

    Attributes:
        commission_rate: Commission rate (e.g., 0.0003 = 0.03%)
        commission_min: Minimum commission per trade
        stamp_duty_rate: Stamp duty rate (e.g., 0.001 = 0.1%)
        stamp_duty_side: Side that has stamp duty ("buy", "sell", "both", "none")
        transfer_fee_rate: Transfer fee rate
        slippage_bps: Slippage in basis points
    """

    commission_rate: float = 0.0003  # 0.03%
    commission_min: float = 5.0  # Minimum commission
    stamp_duty_rate: float = 0.001  # 0.1%
    stamp_duty_side: Literal["buy", "sell", "both", "none"] = "sell"
    transfer_fee_rate: float = 0.00001  # 0.001%
    slippage_bps: float = 0.0  # Basis points


class CostsModel(ABC):
    """
    Abstract base class for transaction cost models.

    Each market has its own cost model with specific rates.

    Methods to implement:
    - calculate(direction, value): Calculate total cost for a trade

    Example:
        >>> costs = CNACostsModel()
        >>> costs.calculate("BUY", 10000)
        3.0
        >>> costs.calculate("SELL", 10000)
        13.0
    """

    @abstractmethod
    def calculate(
        self,
        direction: Literal["BUY", "SELL"],
        value: float,
    ) -> float:
        """
        Calculate transaction cost for a trade.

        Args:
            direction: "BUY" or "SELL"
            value: Trade value (shares * price)

        Returns:
            Total transaction cost
        """
        pass

    @abstractmethod
    def breakdown(
        self,
        direction: Literal["BUY", "SELL"],
        value: float,
    ) -> dict[str, float]:
        """
        Get breakdown of costs by component.

        Args:
            direction: "BUY" or "SELL"
            value: Trade value

        Returns:
            Dict with component costs (commission, stamp_duty, etc.)
        """
        pass

    @abstractmethod
    def get_config(self) -> CostsConfig:
        """Get costs configuration."""
        pass


class CNACostsModel(CostsModel):
    """
    China A-shares transaction cost model.

    Rates:
    - Commission: 0.03%, minimum 5 CNY
    - Stamp duty: 0.1% (sell side only)
    - Transfer fee: 0.001%

    Example:
        >>> costs = CNACostsModel()
        >>> # Buy 1000 shares at 10 CNY = 10000 value
        >>> costs.calculate("BUY", 10000)
        5.0  # Commission: max(3, 5) = 5
        >>> # Sell 1000 shares at 10 CNY = 10000 value
        >>> costs.calculate("SELL", 10000)
        15.0  # Commission: 5 + Stamp duty: 10 = 15
    """

    def __init__(
        self,
        commission_rate: float = 0.0003,
        commission_min: float = 5.0,
        stamp_duty_rate: float = 0.001,
        transfer_fee_rate: float = 0.00001,
    ) -> None:
        """
        Initialize CN_A costs model.

        Args:
            commission_rate: Commission rate (default: 0.03%)
            commission_min: Minimum commission (default: 5 CNY)
            stamp_duty_rate: Stamp duty rate (default: 0.1%)
            transfer_fee_rate: Transfer fee rate (default: 0.001%)
        """
        self._config = CostsConfig(
            commission_rate=commission_rate,
            commission_min=commission_min,
            stamp_duty_rate=stamp_duty_rate,
            stamp_duty_side="sell",
            transfer_fee_rate=transfer_fee_rate,
        )

    def calculate(
        self,
        direction: Literal["BUY", "SELL"],
        value: float,
    ) -> float:
        """
        Calculate CN_A transaction costs.

        Components:
        - Commission: rate * value, min commission_min
        - Stamp duty: rate * value (sell only)
        - Transfer fee: rate * value

        Args:
            direction: "BUY" or "SELL"
            value: Trade value (shares * price)

        Returns:
            Total transaction cost in CNY
        """
        if value <= 0:
            return 0.0

        costs = 0.0

        # Commission (both sides)
        commission = value * self._config.commission_rate
        commission = max(commission, self._config.commission_min)
        costs += commission

        # Stamp duty (sell side only)
        if direction == "SELL":
            stamp_duty = value * self._config.stamp_duty_rate
            costs += stamp_duty

        # Transfer fee (both sides)
        transfer_fee = value * self._config.transfer_fee_rate
        costs += transfer_fee

        return costs

    def breakdown(
        self,
        direction: Literal["BUY", "SELL"],
        value: float,
    ) -> dict[str, float]:
        """
        Get breakdown of CN_A costs.

        Args:
            direction: "BUY" or "SELL"
            value: Trade value

        Returns:
            Dict with commission, stamp_duty, transfer_fee
        """
        if value <= 0:
            return {
                "commission": 0.0,
                "stamp_duty": 0.0,
                "transfer_fee": 0.0,
                "total": 0.0,
            }

        commission = value * self._config.commission_rate
        commission = max(commission, self._config.commission_min)

        stamp_duty = 0.0
        if direction == "SELL":
            stamp_duty = value * self._config.stamp_duty_rate

        transfer_fee = value * self._config.transfer_fee_rate

        return {
            "commission": commission,
            "stamp_duty": stamp_duty,
            "transfer_fee": transfer_fee,
            "total": commission + stamp_duty + transfer_fee,
        }

    def get_config(self) -> CostsConfig:
        """Get costs configuration."""
        return self._config


class USCostsModel(CostsModel):
    """
    US stocks transaction cost model.

    Rates:
    - Commission: 0.05% (typical online broker), no minimum
    - Stamp duty: None (no stamp duty in US)
    - SEC fee: Very small, typically absorbed by broker

    Example:
        >>> costs = USCostsModel()
        >>> # Buy 100 shares at $100 = $10000 value
        >>> costs.calculate("BUY", 10000)
        5.0  # Commission: 0.05% * 10000
        >>> # Sell 100 shares at $100 = $10000 value
        >>> costs.calculate("SELL", 10000)
        5.0  # Commission only
    """

    def __init__(
        self,
        commission_rate: float = 0.0005,
        commission_min: float = 0.0,
    ) -> None:
        """
        Initialize US costs model.

        Args:
            commission_rate: Commission rate (default: 0.05%)
            commission_min: Minimum commission (default: 0)
        """
        self._config = CostsConfig(
            commission_rate=commission_rate,
            commission_min=commission_min,
            stamp_duty_rate=0.0,
            stamp_duty_side="none",
            transfer_fee_rate=0.0,
        )

    def calculate(
        self,
        direction: Literal["BUY", "SELL"],
        value: float,
    ) -> float:
        """
        Calculate US transaction costs.

        Components:
        - Commission: rate * value

        Args:
            direction: "BUY" or "SELL"
            value: Trade value (shares * price)

        Returns:
            Total transaction cost in USD
        """
        if value <= 0:
            return 0.0

        # Commission (both sides)
        commission = value * self._config.commission_rate
        if self._config.commission_min > 0:
            commission = max(commission, self._config.commission_min)

        return commission

    def breakdown(
        self,
        direction: Literal["BUY", "SELL"],
        value: float,
    ) -> dict[str, float]:
        """
        Get breakdown of US costs.

        Args:
            direction: "BUY" or "SELL"
            value: Trade value

        Returns:
            Dict with commission, total
        """
        if value <= 0:
            return {
                "commission": 0.0,
                "stamp_duty": 0.0,
                "total": 0.0,
            }

        commission = value * self._config.commission_rate
        if self._config.commission_min > 0:
            commission = max(commission, self._config.commission_min)

        return {
            "commission": commission,
            "stamp_duty": 0.0,
            "total": commission,
        }

    def get_config(self) -> CostsConfig:
        """Get costs configuration."""
        return self._config


class HKCostsModel(CostsModel):
    """
    Hong Kong stocks transaction cost model.

    Placeholder - HK market not fully implemented.

    Rates (placeholder):
    - Commission: 0.05%
    - Stamp duty: 0.1% (sell side only)
    """

    def __init__(
        self,
        commission_rate: float = 0.0005,
        stamp_duty_rate: float = 0.001,
    ) -> None:
        """
        Initialize HK costs model.

        Args:
            commission_rate: Commission rate
            stamp_duty_rate: Stamp duty rate
        """
        self._config = CostsConfig(
            commission_rate=commission_rate,
            commission_min=0.0,
            stamp_duty_rate=stamp_duty_rate,
            stamp_duty_side="sell",
            transfer_fee_rate=0.0,
        )

    def calculate(
        self,
        direction: Literal["BUY", "SELL"],
        value: float,
    ) -> float:
        """Calculate HK transaction costs."""
        if value <= 0:
            return 0.0

        costs = 0.0

        # Commission
        commission = value * self._config.commission_rate
        costs += commission

        # Stamp duty (sell only)
        if direction == "SELL":
            stamp_duty = value * self._config.stamp_duty_rate
            costs += stamp_duty

        return costs

    def breakdown(
        self,
        direction: Literal["BUY", "SELL"],
        value: float,
    ) -> dict[str, float]:
        """Get breakdown of HK costs."""
        if value <= 0:
            return {
                "commission": 0.0,
                "stamp_duty": 0.0,
                "total": 0.0,
            }

        commission = value * self._config.commission_rate

        stamp_duty = 0.0
        if direction == "SELL":
            stamp_duty = value * self._config.stamp_duty_rate

        return {
            "commission": commission,
            "stamp_duty": stamp_duty,
            "total": commission + stamp_duty,
        }

    def get_config(self) -> CostsConfig:
        """Get costs configuration."""
        return self._config


class NoCostsModel(CostsModel):
    """
    No transaction costs model.

    Used for testing or idealized scenarios.

    Example:
        >>> costs = NoCostsModel()
        >>> costs.calculate("BUY", 10000)
        0.0
    """

    def calculate(
        self,
        direction: Literal["BUY", "SELL"],
        value: float,
    ) -> float:
        """Return zero costs."""
        return 0.0

    def breakdown(
        self,
        direction: Literal["BUY", "SELL"],
        value: float,
    ) -> dict[str, float]:
        """Return empty breakdown."""
        return {
            "commission": 0.0,
            "stamp_duty": 0.0,
            "transfer_fee": 0.0,
            "total": 0.0,
        }

    def get_config(self) -> CostsConfig:
        """Get empty costs configuration."""
        return CostsConfig(
            commission_rate=0.0,
            commission_min=0.0,
            stamp_duty_rate=0.0,
            stamp_duty_side="none",
            transfer_fee_rate=0.0,
        )


# =============================================================================
# Factory Function
# =============================================================================


def get_costs_model(market: Market) -> CostsModel:
    """
    Get costs model for a market.

    Args:
        market: Market enum

    Returns:
        CostsModel instance for the market

    Example:
        >>> costs = get_costs_model(Market.CN_A)
        >>> costs.calculate("BUY", 10000)
        5.0
    """
    if market == Market.CN_A:
        return CNACostsModel()
    elif market == Market.US:
        return USCostsModel()
    elif market == Market.HK:
        return HKCostsModel()
    else:
        return NoCostsModel()


def get_costs_model_from_config(config: CostsConfig) -> CostsModel:
    """
    Create costs model from configuration.

    Args:
        config: Costs configuration

    Returns:
        Generic costs model with configured rates
    """
    class ConfiguredCostsModel(CostsModel):
        """Costs model with custom configuration."""

        def __init__(self, config: CostsConfig) -> None:
            self._config = config

        def calculate(
            self,
            direction: Literal["BUY", "SELL"],
            value: float,
        ) -> float:
            if value <= 0:
                return 0.0

            costs = 0.0

            # Commission
            commission = value * self._config.commission_rate
            if self._config.commission_min > 0:
                commission = max(commission, self._config.commission_min)
            costs += commission

            # Stamp duty
            if self._config.stamp_duty_side == "both":
                costs += value * self._config.stamp_duty_rate
            elif self._config.stamp_duty_side == "sell" and direction == "SELL":
                costs += value * self._config.stamp_duty_rate
            elif self._config.stamp_duty_side == "buy" and direction == "BUY":
                costs += value * self._config.stamp_duty_rate

            # Transfer fee
            costs += value * self._config.transfer_fee_rate

            return costs

        def breakdown(
            self,
            direction: Literal["BUY", "SELL"],
            value: float,
        ) -> dict[str, float]:
            if value <= 0:
                return {
                    "commission": 0.0,
                    "stamp_duty": 0.0,
                    "transfer_fee": 0.0,
                    "total": 0.0,
                }

            commission = value * self._config.commission_rate
            if self._config.commission_min > 0:
                commission = max(commission, self._config.commission_min)

            stamp_duty = 0.0
            if self._config.stamp_duty_side == "both":
                stamp_duty = value * self._config.stamp_duty_rate
            elif self._config.stamp_duty_side == "sell" and direction == "SELL":
                stamp_duty = value * self._config.stamp_duty_rate
            elif self._config.stamp_duty_side == "buy" and direction == "BUY":
                stamp_duty = value * self._config.stamp_duty_rate

            transfer_fee = value * self._config.transfer_fee_rate

            return {
                "commission": commission,
                "stamp_duty": stamp_duty,
                "transfer_fee": transfer_fee,
                "total": commission + stamp_duty + transfer_fee,
            }

        def get_config(self) -> CostsConfig:
            return self._config

    return ConfiguredCostsModel(config)