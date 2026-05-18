"""
Common indicators for TrendSpec strategy framework.

All indicators are implemented as Polars expressions for vectorized computation.
This allows efficient batch computation in init() instead of per-bar calculation.

Key design:
- Indicators return Polars expressions (not values)
- Computed once in init() via precompute_indicator()
- Cached for fast lookup in next()

Supported indicators:
- MA (Simple Moving Average)
- EMA (Exponential Moving Average)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- ATR (Average True Range)
- Bollinger Bands
"""

from typing import Any, Callable

import polars as pl

from trendspec.data.schema import REQUIRED_COLUMNS


# =============================================================================
# Indicator Registry
# =============================================================================

_INDICATOR_REGISTRY: dict[str, Callable] = {}


def register_indicator(name: str) -> Callable:
    """
    Decorator to register an indicator function.

    Args:
        name: Indicator name for lookup

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        _INDICATOR_REGISTRY[name] = func
        return func
    return decorator


def get_indicator(name: str) -> Callable | None:
    """Get indicator function by name."""
    return _INDICATOR_REGISTRY.get(name)


def compute_indicator(df: pl.DataFrame, name: str, **params: Any) -> pl.DataFrame:
    """
    Compute an indicator on a DataFrame.

    Args:
        df: DataFrame with OHLCV data
        name: Indicator name
        **params: Indicator parameters

    Returns:
        DataFrame with indicator column(s) added
    """
    func = get_indicator(name)
    if func is None:
        raise ValueError(f"Unknown indicator: {name}")

    # Check required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return func(df, **params)


# =============================================================================
# Moving Averages
# =============================================================================


@register_indicator("MA")
def ma(df: pl.DataFrame, period: int = 20, column: str = "close") -> pl.DataFrame:
    """
    Simple Moving Average.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period
        column: Column to average (default: close)

    Returns:
        DataFrame with MA column added
    """
    col_name = f"MA_{period}"

    return df.sort("date").with_columns(
        pl.col(column)
        .rolling_mean(window_size=period)
        .over("instrument_id")
        .alias(col_name)
    )


@register_indicator("SMA")
def sma(df: pl.DataFrame, period: int = 20, column: str = "close") -> pl.DataFrame:
    """
    Simple Moving Average (alias for MA).

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period
        column: Column to average

    Returns:
        DataFrame with SMA column added
    """
    return ma(df, period, column)


@register_indicator("EMA")
def ema(df: pl.DataFrame, period: int = 20, column: str = "close", smoothing: float = 2.0) -> pl.DataFrame:
    """
    Exponential Moving Average.

    EMA = (Close - EMA_prev) * smoothing_factor + EMA_prev
    smoothing_factor = smoothing / (1 + period)

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period
        column: Column to average
        smoothing: Smoothing factor (default: 2)

    Returns:
        DataFrame with EMA column added
    """
    col_name = f"EMA_{period}"
    smoothing_factor = smoothing / (1 + period)

    return df.sort("date").with_columns(
        pl.col(column)
        .ewm_mean(alpha=smoothing_factor, adjust=False)
        .over("instrument_id")
        .alias(col_name)
    )


@register_indicator("WMA")
def wma(df: pl.DataFrame, period: int = 20, column: str = "close") -> pl.DataFrame:
    """
    Weighted Moving Average.

    Weighted average where weights decrease linearly.
    WMA = (n*P_n + (n-1)*P_{n-1} + ... + 1*P_1) / (n + n-1 + ... + 1)

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period
        column: Column to average

    Returns:
        DataFrame with WMA column added
    """
    col_name = f"WMA_{period}"

    # Weight numerator: n + n-1 + ... + 1 = n*(n+1)/2
    weight_sum = period * (period + 1) / 2

    # Weights: period, period-1, ..., 1
    weights = [period - i for i in range(period)]

    return df.sort("date").with_columns(
        pl.col(column)
        .rolling_mean(window_size=period, weights=weights)
        .over("instrument_id")
        .alias(col_name)
    )


# =============================================================================
# RSI (Relative Strength Index)
# =============================================================================


@register_indicator("RSI")
def rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """
    Relative Strength Index.

    RSI = 100 - 100 / (1 + RS)
    RS = Average Gain / Average Loss

    Uses Wilder's smoothing for average gain/loss.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period (default: 14)

    Returns:
        DataFrame with RSI column added
    """
    col_name = f"RSI_{period}"

    # Calculate price changes
    df_sorted = df.sort("date")

    # Calculate change per instrument
    df_with_change = df_sorted.with_columns(
        (pl.col("close") - pl.col("close").shift(1))
        .over("instrument_id")
        .alias("change")
    )

    # Separate gains and losses
    df_gains_losses = df_with_change.with_columns([
        pl.when(pl.col("change") > 0)
          .then(pl.col("change"))
          .otherwise(0.0)
          .alias("gain"),
        pl.when(pl.col("change") < 0)
          .then(-pl.col("change"))
          .otherwise(0.0)
          .alias("loss"),
    ])

    # Calculate average gain and loss using EMA (Wilder's smoothing)
    # Wilder's smoothing: alpha = 1/period
    alpha = 1.0 / period

    df_avg = df_gains_losses.with_columns([
        pl.col("gain")
        .ewm_mean(alpha=alpha, adjust=False)
        .over("instrument_id")
        .alias("avg_gain"),
        pl.col("loss")
        .ewm_mean(alpha=alpha, adjust=False)
        .over("instrument_id")
        .alias("avg_loss"),
    ])

    # Calculate RSI
    df_rsi = df_avg.with_columns([
        pl.when(pl.col("avg_loss") == 0)
          .then(100.0)
          .otherwise(
              100.0 - (100.0 / (1.0 + pl.col("avg_gain") / pl.col("avg_loss")))
          )
          .alias(col_name)
    ])

    # Clean up intermediate columns
    return df_rsi.drop(["change", "gain", "loss", "avg_gain", "avg_loss"])


# =============================================================================
# MACD (Moving Average Convergence Divergence)
# =============================================================================


@register_indicator("MACD")
def macd(
    df: pl.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pl.DataFrame:
    """
    Moving Average Convergence Divergence.

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal_period)
    Histogram = MACD Line - Signal Line

    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal EMA period (default: 9)

    Returns:
        DataFrame with MACD_line, MACD_signal, MACD_hist columns
    """
    # Calculate fast and slow EMAs
    fast_alpha = 2.0 / (1 + fast_period)
    slow_alpha = 2.0 / (1 + slow_period)
    signal_alpha = 2.0 / (1 + signal_period)

    df_sorted = df.sort("date")

    df_ema = df_sorted.with_columns([
        pl.col("close")
        .ewm_mean(alpha=fast_alpha, adjust=False)
        .over("instrument_id")
        .alias(f"EMA_{fast_period}"),
        pl.col("close")
        .ewm_mean(alpha=slow_alpha, adjust=False)
        .over("instrument_id")
        .alias(f"EMA_{slow_period}"),
    ])

    # MACD Line
    df_macd = df_ema.with_columns([
        (pl.col(f"EMA_{fast_period}") - pl.col(f"EMA_{slow_period}"))
        .alias("MACD_line")
    ])

    # Signal Line (EMA of MACD Line)
    df_signal = df_macd.with_columns([
        pl.col("MACD_line")
        .ewm_mean(alpha=signal_alpha, adjust=False)
        .over("instrument_id")
        .alias("MACD_signal")
    ])

    # Histogram
    df_hist = df_signal.with_columns([
        (pl.col("MACD_line") - pl.col("MACD_signal"))
        .alias("MACD_hist")
    ])

    return df_hist


# =============================================================================
# ATR (Average True Range)
# =============================================================================


@register_indicator("ATR")
def atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """
    Average True Range.

    True Range = max(High - Low, |High - Prev Close|, |Low - Prev Close|)
    ATR = EMA of True Range (Wilder's smoothing)

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period (default: 14)

    Returns:
        DataFrame with ATR column added
    """
    col_name = f"ATR_{period}"

    df_sorted = df.sort("date")

    # Previous close
    df_prev = df_sorted.with_columns(
        pl.col("close")
        .shift(1)
        .over("instrument_id")
        .alias("prev_close")
    )

    # Calculate True Range components
    df_tr = df_prev.with_columns([
        # High - Low
        (pl.col("high") - pl.col("low")).alias("hl"),
        # |High - Prev Close|
        (pl.col("high") - pl.col("prev_close")).abs().alias("hpc"),
        # |Low - Prev Close|
        (pl.col("low") - pl.col("prev_close")).abs().alias("lpc"),
    ])

    # True Range = max of the three
    df_tr_val = df_tr.with_columns([
        pl.max_horizontal(["hl", "hpc", "lpc"]).alias("tr")
    ])

    # ATR = EMA of True Range (Wilder's smoothing: alpha = 1/period)
    alpha = 1.0 / period

    df_atr = df_tr_val.with_columns([
        pl.col("tr")
        .ewm_mean(alpha=alpha, adjust=False)
        .over("instrument_id")
        .alias(col_name)
    ])

    return df_atr.drop(["prev_close", "hl", "hpc", "lpc", "tr"])


# =============================================================================
# Bollinger Bands
# =============================================================================


@register_indicator("BB")
def bollinger_bands(
    df: pl.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pl.DataFrame:
    """
    Bollinger Bands.

    Middle Band = SMA(period)
    Upper Band = Middle Band + (std_dev * std(period))
    Lower Band = Middle Band - (std_dev * std(period))

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period (default: 20)
        std_dev: Standard deviation multiplier (default: 2)

    Returns:
        DataFrame with BB_middle, BB_upper, BB_lower columns
    """
    df_sorted = df.sort("date")

    # Calculate SMA and rolling std
    df_bb = df_sorted.with_columns([
        pl.col("close")
        .rolling_mean(window_size=period)
        .over("instrument_id")
        .alias("BB_middle"),
        pl.col("close")
        .rolling_std(window_size=period)
        .over("instrument_id")
        .alias("bb_std"),
    ])

    # Upper and Lower bands
    df_bands = df_bb.with_columns([
        (pl.col("BB_middle") + std_dev * pl.col("bb_std")).alias("BB_upper"),
        (pl.col("BB_middle") - std_dev * pl.col("bb_std")).alias("BB_lower"),
    ])

    # Bandwidth and %B for additional analysis
    df_final = df_bands.with_columns([
        # Bandwidth = (Upper - Lower) / Middle
        ((pl.col("BB_upper") - pl.col("BB_lower")) / pl.col("BB_middle"))
        .alias("BB_width"),
        # %B = (Close - Lower) / (Upper - Lower)
        ((pl.col("close") - pl.col("BB_lower")) / (pl.col("BB_upper") - pl.col("BB_lower")))
        .alias("BB_pct"),
    ])

    return df_final.drop("bb_std")


# =============================================================================
# Momentum Indicators
# =============================================================================


@register_indicator("ROC")
def roc(df: pl.DataFrame, period: int = 10) -> pl.DataFrame:
    """
    Rate of Change (Price Momentum).

    ROC = ((Close - Close_n) / Close_n) * 100

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period

    Returns:
        DataFrame with ROC column added
    """
    col_name = f"ROC_{period}"

    return df.sort("date").with_columns(
        ((pl.col("close") - pl.col("close").shift(period)) / pl.col("close").shift(period) * 100)
        .over("instrument_id")
        .alias(col_name)
    )


@register_indicator("MOM")
def momentum(df: pl.DataFrame, period: int = 10) -> pl.DataFrame:
    """
    Momentum (Price change).

    MOM = Close - Close_n

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period

    Returns:
        DataFrame with MOM column added
    """
    col_name = f"MOM_{period}"

    return df.sort("date").with_columns(
        (pl.col("close") - pl.col("close").shift(period))
        .over("instrument_id")
        .alias(col_name)
    )


# =============================================================================
# Volatility Indicators
# =============================================================================


@register_indicator("STD")
def rolling_std(df: pl.DataFrame, period: int = 20, column: str = "close") -> pl.DataFrame:
    """
    Rolling Standard Deviation (Volatility).

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period
        column: Column to calculate std on

    Returns:
        DataFrame with STD column added
    """
    col_name = f"STD_{period}"

    return df.sort("date").with_columns(
        pl.col(column)
        .rolling_std(window_size=period)
        .over("instrument_id")
        .alias(col_name)
    )


@register_indicator("VOL")
def volatility(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    """
    Historical Volatility (Annualized).

    VOL = std(daily_returns) * sqrt(252)

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period for std calculation

    Returns:
        DataFrame with VOL column added
    """
    col_name = f"VOL_{period}"

    df_sorted = df.sort("date")

    # Calculate daily returns
    df_returns = df_sorted.with_columns(
        (pl.col("close") / pl.col("close").shift(1) - 1)
        .over("instrument_id")
        .alias("daily_return")
    )

    # Rolling std of returns, annualized
    df_vol = df_returns.with_columns(
        (pl.col("daily_return")
         .rolling_std(window_size=period)
         .over("instrument_id")
         * (252 ** 0.5))  # Annualize (252 trading days)
        .alias(col_name)
    )

    return df_vol.drop("daily_return")


# =============================================================================
# Volume Indicators
# =============================================================================


@register_indicator("VMA")
def volume_ma(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    """
    Volume Moving Average.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period

    Returns:
        DataFrame with VMA column added
    """
    col_name = f"VMA_{period}"

    return df.sort("date").with_columns(
        pl.col("volume")
        .rolling_mean(window_size=period)
        .over("instrument_id")
        .alias(col_name)
    )


@register_indicator("OBV")
def obv(df: pl.DataFrame) -> pl.DataFrame:
    """
    On-Balance Volume.

    OBV cumulative sum:
    - If close > prev_close: OBV += volume
    - If close < prev_close: OBV -= volume
    - If close == prev_close: OBV unchanged

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with OBV column added
    """
    df_sorted = df.sort("date")

    # Calculate direction
    df_dir = df_sorted.with_columns([
        pl.col("close")
        .shift(1)
        .over("instrument_id")
        .alias("prev_close")
    ])

    df_obv = df_dir.with_columns([
        pl.when(pl.col("close") > pl.col("prev_close"))
          .then(pl.col("volume"))
          .when(pl.col("close") < pl.col("prev_close"))
          .then(-pl.col("volume"))
          .otherwise(0)
          .alias("volume_change")
    ])

    # Cumulative sum
    df_final = df_obv.with_columns([
        pl.col("volume_change")
        .cum_sum()
        .over("instrument_id")
        .alias("OBV")
    ])

    return df_final.drop(["prev_close", "volume_change"])


# =============================================================================
# Price Level Indicators
# =============================================================================


@register_indicator("MAX")
def rolling_max(df: pl.DataFrame, period: int = 20, column: str = "high") -> pl.DataFrame:
    """
    Rolling Maximum (Highest value in period).

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period
        column: Column to find max (default: high for highest high)

    Returns:
        DataFrame with MAX column added
    """
    col_name = f"MAX_{period}"

    return df.sort("date").with_columns(
        pl.col(column)
        .rolling_max(window_size=period)
        .over("instrument_id")
        .alias(col_name)
    )


@register_indicator("MIN")
def rolling_min(df: pl.DataFrame, period: int = 20, column: str = "low") -> pl.DataFrame:
    """
    Rolling Minimum (Lowest value in period).

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period
        column: Column to find min (default: low for lowest low)

    Returns:
        DataFrame with MIN column added
    """
    col_name = f"MIN_{period}"

    return df.sort("date").with_columns(
        pl.col(column)
        .rolling_min(window_size=period)
        .over("instrument_id")
        .alias(col_name)
    )


@register_indicator("HH")
def highest_high(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    """
    Highest High in period.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period

    Returns:
        DataFrame with HH column added
    """
    return rolling_max(df, period, "high").rename({f"MAX_{period}": f"HH_{period}"})


@register_indicator("LL")
def lowest_low(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    """
    Lowest Low in period.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period

    Returns:
        DataFrame with LL column added
    """
    return rolling_min(df, period, "low").rename({f"MIN_{period}": f"LL_{period}"})


# =============================================================================
# Clenow Momentum Indicators
# =============================================================================


@register_indicator("CLENOW_SCORE")
def clenow_score(df: pl.DataFrame, period: int = 90) -> pl.DataFrame:
    """
    Clenow Momentum Score: annualized exponential regression slope × R².

    For each instrument, fits linear regression on ln(close) vs. day-index
    over a rolling `period`-day window. Annualizes the slope and weights by R²
    so only smooth, consistent uptrends score high.

    Args:
        df: DataFrame with OHLCV data
        period: Regression lookback window in trading days (default: 90)

    Returns:
        DataFrame with CLENOW_SCORE_{period}, CLENOW_SLOPE_{period},
        CLENOW_R2_{period} columns added
    """
    import numpy as np
    from scipy import stats

    slope_col = f"CLENOW_SLOPE_{period}"
    r2_col = f"CLENOW_R2_{period}"
    score_col = f"CLENOW_SCORE_{period}"

    x = np.arange(period, dtype=float)

    all_groups: list[pl.DataFrame] = []
    for (_instrument_id,), group in df.sort(["instrument_id", "date"]).group_by(
        ["instrument_id"], maintain_order=True
    ):
        closes = group["close"].to_numpy()
        n = len(closes)

        slopes: list[float | None] = [None] * n
        r2s: list[float | None] = [None] * n
        scores: list[float | None] = [None] * n

        for i in range(period - 1, n):
            window = closes[i - period + 1 : i + 1]
            if np.any(window <= 0):
                continue
            y = np.log(window)
            fit = stats.linregress(x, y)
            annual_slope = (np.exp(fit.slope * 252) - 1) * 100
            r2 = fit.rvalue ** 2
            slopes[i] = annual_slope
            r2s[i] = r2
            scores[i] = annual_slope * r2

        all_groups.append(
            group.with_columns([
                pl.Series(slope_col, slopes, dtype=pl.Float64),
                pl.Series(r2_col, r2s, dtype=pl.Float64),
                pl.Series(score_col, scores, dtype=pl.Float64),
            ])
        )

    if not all_groups:
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias(slope_col),
            pl.lit(None).cast(pl.Float64).alias(r2_col),
            pl.lit(None).cast(pl.Float64).alias(score_col),
        ])

    return pl.concat(all_groups).sort(["instrument_id", "date"])


@register_indicator("MIN_DAILY_RETURN")
def min_daily_return(df: pl.DataFrame, period: int = 90) -> pl.DataFrame:
    """
    Rolling minimum single-day return over `period` days.

    Used to filter instruments with extreme gap-down events.
    A value below -0.15 means the stock had a >15% single-day drop
    somewhere in the lookback window.

    Args:
        df: DataFrame with OHLCV data
        period: Rolling window in trading days (default: 90)

    Returns:
        DataFrame with MIN_DAILY_RETURN_{period} column added
    """
    col_name = f"MIN_DAILY_RETURN_{period}"

    df_sorted = df.sort("date")

    df_ret = df_sorted.with_columns(
        (pl.col("close") / pl.col("close").shift(1) - 1)
        .over("instrument_id")
        .alias("_daily_ret")
    )

    return df_ret.with_columns(
        pl.col("_daily_ret")
        .rolling_min(window_size=period)
        .over("instrument_id")
        .alias(col_name)
    ).drop("_daily_ret")


@register_indicator("RS_RATING")
def rs_rating(df: pl.DataFrame, period: int = 252) -> pl.DataFrame:
    """
    IBD-style Relative Strength Rating (0-100 cross-sectional rank).

    RS_raw = (2 * close/close[63] + close/close[126]
              + close/close[189] + close/close[252]) / 5
    RS_RATING = percentile rank of RS_raw within each date, scaled 0-100.

    Args:
        df: DataFrame with OHLCV data
        period: Always 252 (kept for indicator_value() column-name compatibility)

    Returns:
        DataFrame with RS_RATING_{period} column added
    """
    col_name = f"RS_RATING_{period}"

    df_sorted = df.sort(["instrument_id", "date"])

    df_raw = df_sorted.with_columns(
        (
            2.0 * pl.col("close") / pl.col("close").shift(63).over("instrument_id")
            + pl.col("close") / pl.col("close").shift(126).over("instrument_id")
            + pl.col("close") / pl.col("close").shift(189).over("instrument_id")
            + pl.col("close") / pl.col("close").shift(252).over("instrument_id")
        ).alias("_rs_raw") / 5.0
    )

    return df_raw.with_columns(
        (
            pl.col("_rs_raw")
            .rank(method="average", descending=False)
            .over("date")
            / pl.col("_rs_raw").count().over("date")
            * 100.0
        ).alias(col_name)
    ).drop("_rs_raw")


@register_indicator("ADR_PCT")
def adr_pct(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    """
    Average Daily Range Percentage.

    Per-bar daily range %  = (high - low) / close.
    ADR_PCT_{period}       = rolling_mean(daily_range_pct, period) per instrument.

    Used by Qullamaggie-style momentum strategies to filter for high-volatility
    stocks suitable for breakout trading (typical threshold: ADR_PCT >= 0.04).

    Args:
        df: DataFrame with OHLCV data
        period: Rolling window in trading days (default: 20)

    Returns:
        DataFrame with ADR_PCT_{period} column added
    """
    col_name = f"ADR_PCT_{period}"

    df_sorted = df.sort("date")

    df_range = df_sorted.with_columns(
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("_daily_range_pct")
    )

    return df_range.with_columns(
        pl.col("_daily_range_pct")
        .rolling_mean(window_size=period)
        .over("instrument_id")
        .alias(col_name)
    ).drop("_daily_range_pct")


# =============================================================================
# Utility Functions
# =============================================================================


def list_indicators() -> list[str]:
    """Get list of registered indicator names."""
    return sorted(_INDICATOR_REGISTRY.keys())


def indicator_info(name: str) -> dict[str, Any]:
    """
    Get information about an indicator.

    Args:
        name: Indicator name

    Returns:
        Dict with indicator info (name, docstring)
    """
    func = get_indicator(name)
    if func is None:
        raise ValueError(f"Unknown indicator: {name}")

    return {
        "name": name,
        "docstring": func.__doc__ or "",
        "signature": str(func.__code__.co_varnames[:func.__code__.co_argcount]),
    }