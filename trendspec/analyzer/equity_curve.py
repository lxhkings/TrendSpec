"""
Equity curve analysis for TrendSpec.

Provides drawdown analysis and returns series from equity curve points.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl


@dataclass
class DrawdownPoint:
    """
    Single drawdown point.

    Attributes:
        date: Date of the point
        drawdown: Drawdown percentage (0.10 = 10% drawdown)
        peak_date: Date when peak was reached
        duration_days: Days since peak
    """

    date: date
    drawdown: float = 0.0
    peak_date: date | None = None
    duration_days: int = 0


class EquityCurve:
    """
    Equity curve analysis class.

    Provides drawdown and returns analysis from equity curve points.

    Attributes:
        points: List of EquityCurvePoint objects
        initial_capital: Initial capital

    Example:
        >>> equity_curve = EquityCurve(points, initial_capital=100000)
        >>> drawdown = equity_curve.drawdown_series()
        >>> max_dd = equity_curve.max_drawdown()
        >>> returns = equity_curve.returns_series()
    """

    def __init__(
        self,
        points: list[Any],  # List of EquityCurvePoint
        initial_capital: float = 100000.0,
    ) -> None:
        """
        Initialize equity curve.

        Args:
            points: List of EquityCurvePoint objects
            initial_capital: Initial capital
        """
        self.points = points
        self.initial_capital = initial_capital

    def drawdown_series(self) -> list[DrawdownPoint]:
        """
        Calculate drawdown at each point.

        Returns:
            List of DrawdownPoint objects
        """
        if not self.points:
            return []

        drawdowns: list[DrawdownPoint] = []
        peak = self.initial_capital
        peak_date = None

        for point in self.points:
            if point.nav > peak:
                peak = point.nav
                peak_date = point.date

            dd = 0.0
            duration = 0

            if peak > 0:
                dd = (peak - point.nav) / peak

            if peak_date:
                duration = (point.date - peak_date).days

            drawdowns.append(DrawdownPoint(
                date=point.date,
                drawdown=dd,
                peak_date=peak_date,
                duration_days=duration,
            ))

        return drawdowns

    def returns_series(self) -> list[float]:
        """
        Calculate daily returns series.

        Returns:
            List of daily return percentages
        """
        if not self.points:
            return []

        return [p.daily_return for p in self.points]

    def cumulative_returns_series(self) -> list[float]:
        """
        Calculate cumulative returns series.

        Returns:
            List of cumulative return percentages
        """
        if not self.points:
            return []

        return [p.cumulative_return for p in self.points]

    def max_drawdown(self) -> float:
        """
        Get maximum drawdown.

        Returns:
            Maximum drawdown percentage
        """
        drawdowns = self.drawdown_series()
        if not drawdowns:
            return 0.0

        return max(d.drawdown for d in drawdowns)

    def max_drawdown_date(self) -> date | None:
        """
        Get date of maximum drawdown.

        Returns:
            Date when max drawdown occurred
        """
        drawdowns = self.drawdown_series()
        if not drawdowns:
            return None

        max_dd = self.max_drawdown()
        for d in drawdowns:
            if d.drawdown == max_dd:
                return d.date

        return None

    def current_drawdown(self) -> float:
        """
        Get current drawdown.

        Returns:
            Current drawdown percentage
        """
        if not self.points:
            return 0.0

        peak = self.initial_capital
        for point in self.points:
            if point.nav > peak:
                peak = point.nav

        final_nav = self.points[-1].nav
        if peak > 0:
            return (peak - final_nav) / peak

        return 0.0

    def underwater_periods(self) -> list[tuple[date, date]]:
        """
        Find periods where equity is below peak.

        Returns:
            List of (start_date, end_date) tuples for underwater periods
        """
        if not self.points:
            return []

        drawdowns = self.drawdown_series()
        periods: list[tuple[date, date]] = []

        in_underwater = False
        start_date = None

        for d in drawdowns:
            if d.drawdown > 0 and not in_underwater:
                in_underwater = True
                start_date = d.date
            elif d.drawdown == 0 and in_underwater:
                in_underwater = False
                if start_date:
                    periods.append((start_date, d.date))
                    start_date = None

        # If still underwater at end
        if in_underwater and start_date:
            periods.append((start_date, drawdowns[-1].date))

        return periods

    def to_dataframe(self) -> pl.DataFrame:
        """
        Convert equity curve to Polars DataFrame.

        Returns:
            DataFrame with equity curve data
        """
        if not self.points:
            return pl.DataFrame()

        records = []
        drawdowns = self.drawdown_series()

        for i, point in enumerate(self.points):
            dd = drawdowns[i] if i < len(drawdowns) else DrawdownPoint(date=point.date)
            records.append({
                "date": point.date.isoformat(),
                "nav": point.nav,
                "cash": point.cash,
                "position_value": point.position_value,
                "position_count": point.position_count,
                "daily_return": point.daily_return,
                "cumulative_return": point.cumulative_return,
                "drawdown": dd.drawdown,
                "drawdown_duration": dd.duration_days,
            })

        return pl.DataFrame(records)

    def summary(self) -> dict[str, Any]:
        """
        Get equity curve summary statistics.

        Returns:
            Dict with summary statistics
        """
        if not self.points:
            return {}

        returns = self.returns_series()

        return {
            "total_points": len(self.points),
            "start_date": self.points[0].date.isoformat(),
            "end_date": self.points[-1].date.isoformat(),
            "initial_nav": self.initial_capital,
            "final_nav": self.points[-1].nav,
            "max_nav": max(p.nav for p in self.points),
            "min_nav": min(p.nav for p in self.points),
            "max_drawdown": self.max_drawdown(),
            "current_drawdown": self.current_drawdown(),
            "avg_daily_return": sum(returns) / len(returns) if returns else 0.0,
        }