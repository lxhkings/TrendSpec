# Clenow Momentum Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Clenow quantitative momentum strategy (《Stocks on the Move》) to TrendSpec, with annualized exponential regression slope × R² scoring, ATR-based risk-parity position sizing, and weekly rebalancing.

**Architecture:** Five sequential changes — add scipy dependency, extend Signal with optional `shares` field, teach BacktestEngine to honor that field, register two new indicators (CLENOW_SCORE, MIN_DAILY_RETURN), then implement ClenowMomentumStrategy using cross-sectional ranking inside `next()` with a weekly-rebalance guard. The strategy pattern follows SectorMomentumStrategy: do full cross-sectional work on the first instrument call of a rebalance day, then guard-return for the rest.

**Tech Stack:** Python 3.11, Polars, scipy.stats.linregress, NumPy, pytest

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `pyproject.toml` | Add `scipy>=1.11.0` dependency |
| Modify | `trendspec/strategy/signal.py:44-46` | Add `shares: float \| None` field |
| Modify | `trendspec/engine/backtest_engine.py:330` | Use `signal.shares` when present |
| Modify | `trendspec/strategy/indicators.py` | Add `CLENOW_SCORE` and `MIN_DAILY_RETURN` indicators |
| Create | `trendspec/strategy/examples/clenow_momentum.py` | ClenowMomentumStrategy |
| Modify | `trendspec/strategy/examples/__init__.py` | Export ClenowMomentumStrategy |
| Modify | `tests/test_strategies.py` | Tests for new strategy + signal.shares |

---

## Task 1: Add scipy Dependency

**Files:**
- Modify: `pyproject.toml:20-28`

- [ ] **Step 1: Add scipy to dependencies**

Edit `pyproject.toml` dependencies list to add scipy after polars:

```toml
dependencies = [
    "polars>=1.0.0",
    "scipy>=1.11.0",
    "pydantic-settings>=2.0.0",
    "typer>=0.12.0",
    "rich>=13.0.0",
    "sqlalchemy>=2.0.0",
    "pymysql>=1.1.0",
    "holidays>=0.40.0",
]
```

- [ ] **Step 2: Sync dependencies**

```bash
uv sync
```

Expected: No errors. `scipy` and `numpy` appear in the environment.

- [ ] **Step 3: Verify scipy import**

```bash
uv run python -c "from scipy import stats; print('scipy ok')"
```

Expected: `scipy ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add scipy dependency for Clenow exponential regression"
```

---

## Task 2: Add `shares` Field to Signal

**Files:**
- Modify: `trendspec/strategy/signal.py`
- Test: `tests/test_strategies.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_strategies.py` inside any convenient test class (or as a standalone function at module level):

```python
def test_signal_shares_field() -> None:
    """Signal.shares defaults to None and can be set after creation."""
    sig = Signal(direction="BUY", ticker="AAPL", instrument_id="AAPL", price=150.0)
    assert sig.shares is None

    sig.shares = 42.0
    assert sig.shares == 42.0


def test_signal_shares_not_in_repr() -> None:
    """Signal.shares is excluded from repr (like timestamp)."""
    sig = Signal(direction="BUY", ticker="AAPL", instrument_id="AAPL", price=150.0, shares=10.0)
    assert "shares" not in repr(sig)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_strategies.py::test_signal_shares_field -v
```

Expected: `FAILED` — `Signal.__init__() got an unexpected keyword argument 'shares'`

- [ ] **Step 3: Add shares field to Signal dataclass**

In `trendspec/strategy/signal.py`, add `shares` after the `note` field (line 46):

```python
    direction: Literal["BUY", "SELL"]
    ticker: str
    instrument_id: str
    price: float
    trigger_value: float | None = None
    note: str | None = None
    shares: float | None = field(default=None, repr=False)
    timestamp: float | None = field(default=None, repr=False)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_strategies.py::test_signal_shares_field tests/test_strategies.py::test_signal_shares_not_in_repr -v
```

Expected: Both `PASSED`

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
uv run pytest -v --tb=short
```

Expected: All previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add trendspec/strategy/signal.py tests/test_strategies.py
git commit -m "feat: add optional shares field to Signal for ATR-based position sizing"
```

---

## Task 3: Update BacktestEngine to Honor signal.shares

**Files:**
- Modify: `trendspec/engine/backtest_engine.py:329-330`

- [ ] **Step 1: Write failing test**

Add to `tests/test_strategies.py`:

```python
def test_backtest_engine_uses_signal_shares(populated_us_basic: str) -> None:
    """BacktestEngine uses signal.shares when set, falls back to order_size otherwise."""
    from trendspec.engine.base_engine import EngineConfig
    from trendspec.engine.backtest_engine import BacktestEngine
    from trendspec.data.markets import Market
    from datetime import date

    executed_shares: list[float] = []

    @register_strategy("_test_shares_strategy")
    class SharesTestStrategy(BaseStrategy):
        name = "_test_shares_strategy"

        def init(self, ctx: StrategyContext) -> None:
            pass

        def next(self, ctx: StrategyContext) -> None:
            if not ctx.has_position(ctx.instrument_id):
                sig = ctx.signal("BUY", ctx.instrument_id, ctx.close)
                sig.shares = 7.0  # custom shares

    config = EngineConfig(
        market=Market.US,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 5),
        initial_capital=100000.0,
        data_lake_root=populated_us_basic,
    )
    engine = BacktestEngine(config)
    result = engine.run(SharesTestStrategy)

    # All executed trades should have 7 shares (from signal.shares)
    assert all(t.shares == 7 for t in result.trades), \
        f"Expected 7 shares per trade, got: {[t.shares for t in result.trades]}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_strategies.py::test_backtest_engine_uses_signal_shares -v
```

Expected: `FAILED` — trades have default 100 shares, not 7.

- [ ] **Step 3: Update BacktestEngine to use signal.shares**

In `trendspec/engine/backtest_engine.py`, replace lines 329-330:

```python
            # Submit orders to broker
            for signal in allowed_signals:
                self._broker.submit(signal, shares=ctx.get_param("order_size", 100))
```

With:

```python
            # Submit orders to broker (use signal.shares if set, else order_size)
            for signal in allowed_signals:
                order_shares = (
                    int(signal.shares)
                    if signal.shares is not None
                    else ctx.get_param("order_size", 100)
                )
                self._broker.submit(signal, shares=order_shares)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_strategies.py::test_backtest_engine_uses_signal_shares -v
```

Expected: `PASSED`

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest -v --tb=short
```

Expected: All previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add trendspec/engine/backtest_engine.py tests/test_strategies.py
git commit -m "feat: backtest engine honors signal.shares for per-signal position sizing"
```

---

## Task 4: Add CLENOW_SCORE and MIN_DAILY_RETURN Indicators

**Files:**
- Modify: `trendspec/strategy/indicators.py` (append before the Utility Functions section)
- Test: `tests/test_strategies.py`

**Important:** `indicator_value()` constructs the lookup column as `f"{name}_{params.get('period', '')}"`. Both new indicators must use `period` (not `lookback`) as the parameter name so column lookup works correctly.

- [ ] **Step 1: Write failing tests for indicators**

Add to `tests/test_strategies.py`:

```python
from datetime import date
import polars as pl
from trendspec.strategy.indicators import compute_indicator, list_indicators


def _make_price_df(n_days: int = 150) -> pl.DataFrame:
    """Synthetic OHLCV data for two instruments over n_days."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    for inst in ["AAA", "BBB"]:
        price = 100.0
        for i in range(n_days):
            price *= 1 + rng.normal(0.001, 0.015)
            rows.append({
                "instrument_id": inst,
                "ticker": inst,
                "date": date(2023, 1, 1) + __import__("datetime").timedelta(days=i),
                "open": price * 0.99,
                "high": price * 1.01,
                "low": price * 0.98,
                "close": price,
                "volume": 1_000_000,
            })
    return pl.DataFrame(rows)


class TestClenowScoreIndicator:
    def test_registered(self) -> None:
        assert "CLENOW_SCORE" in list_indicators()

    def test_columns_added(self) -> None:
        df = _make_price_df(150)
        result = compute_indicator(df, "CLENOW_SCORE", period=90)
        assert "CLENOW_SCORE_90" in result.columns
        assert "CLENOW_SLOPE_90" in result.columns
        assert "CLENOW_R2_90" in result.columns

    def test_null_before_lookback(self) -> None:
        df = _make_price_df(150)
        result = compute_indicator(df, "CLENOW_SCORE", period=90)
        # First 89 rows per instrument should be null
        aaa = result.filter(pl.col("instrument_id") == "AAA").sort("date")
        assert aaa["CLENOW_SCORE_90"][:89].is_null().all()

    def test_r2_bounded(self) -> None:
        df = _make_price_df(150)
        result = compute_indicator(df, "CLENOW_SCORE", period=90)
        r2 = result["CLENOW_R2_90"].drop_nulls()
        assert (r2 >= 0).all() and (r2 <= 1).all()

    def test_uptrend_scores_positive(self) -> None:
        """Monotonically increasing prices → positive slope → positive score."""
        rows = [
            {"instrument_id": "UP", "ticker": "UP",
             "date": date(2023, 1, 1) + __import__("datetime").timedelta(days=i),
             "open": 100 + i, "high": 101 + i, "low": 99 + i,
             "close": 100 + i, "volume": 1_000_000}
            for i in range(120)
        ]
        df = pl.DataFrame(rows)
        result = compute_indicator(df, "CLENOW_SCORE", period=90)
        last = result.filter(pl.col("instrument_id") == "UP").sort("date").tail(1)
        assert last["CLENOW_SCORE_90"].item() > 0


class TestMinDailyReturnIndicator:
    def test_registered(self) -> None:
        assert "MIN_DAILY_RETURN" in list_indicators()

    def test_column_added(self) -> None:
        df = _make_price_df(150)
        result = compute_indicator(df, "MIN_DAILY_RETURN", period=90)
        assert "MIN_DAILY_RETURN_90" in result.columns

    def test_gap_detected(self) -> None:
        """A 20% single-day drop must appear as MIN_DAILY_RETURN < -0.15."""
        rows = []
        price = 100.0
        for i in range(150):
            if i == 100:
                price *= 0.80  # 20% gap down
            rows.append({
                "instrument_id": "G", "ticker": "G",
                "date": date(2023, 1, 1) + __import__("datetime").timedelta(days=i),
                "open": price, "high": price * 1.01, "low": price * 0.99,
                "close": price, "volume": 1_000_000,
            })
        df = pl.DataFrame(rows)
        result = compute_indicator(df, "MIN_DAILY_RETURN", period=90)
        g = result.filter(pl.col("instrument_id") == "G").sort("date")
        # After the gap day and within the 90-day window, min should be ≈ -0.20
        post_gap = g.filter(pl.col("date") >= date(2023, 1, 1) + __import__("datetime").timedelta(days=101))
        assert (post_gap["MIN_DAILY_RETURN_90"].drop_nulls() < -0.15).any()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_strategies.py::TestClenowScoreIndicator tests/test_strategies.py::TestMinDailyReturnIndicator -v
```

Expected: `FAILED` — indicators not registered.

- [ ] **Step 3: Implement CLENOW_SCORE indicator**

Append to `trendspec/strategy/indicators.py` before the `# Utility Functions` section:

```python
# =============================================================================
# Clenow Momentum Indicators
# =============================================================================


@register_indicator("CLENOW_SCORE")
def clenow_score(df: pl.DataFrame, period: int = 90) -> pl.DataFrame:
    """
    Clenow Momentum Score: annualized exponential regression slope × R².

    For each instrument over a rolling `period`-day window:
    - Fit linear regression on ln(close) vs. day-index
    - Annualize slope: (exp(slope * 252) - 1) * 100
    - Score = annualized_slope × R²

    High score = strong AND consistent uptrend.

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
    for (instrument_id,), group in df.sort(["instrument_id", "date"]).group_by(
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
            result = stats.linregress(x, y)
            annual_slope = (np.exp(result.slope * 252) - 1) * 100
            r2 = result.rvalue ** 2
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

    Used to filter instruments with extreme gap-down events (e.g., > 15% in one day).

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
```

- [ ] **Step 4: Run indicator tests**

```bash
uv run pytest tests/test_strategies.py::TestClenowScoreIndicator tests/test_strategies.py::TestMinDailyReturnIndicator -v
```

Expected: All `PASSED`

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest -v --tb=short
```

Expected: All passing.

- [ ] **Step 6: Commit**

```bash
git add trendspec/strategy/indicators.py tests/test_strategies.py
git commit -m "feat: add CLENOW_SCORE and MIN_DAILY_RETURN indicators"
```

---

## Task 5: Implement ClenowMomentumStrategy

**Files:**
- Create: `trendspec/strategy/examples/clenow_momentum.py`
- Modify: `trendspec/strategy/examples/__init__.py`

**Design notes for this task:**
- `next()` is called per-instrument per-day by the engine. The cross-sectional rebalance runs only once per rebalance day: the first instrument to trigger `next()` on a Wednesday (default) does all the work; subsequent calls return immediately (`_last_rebalance_date` guard).
- `ctx.signal()` sets `ticker` to `ctx._current_ticker` (the currently-iterated instrument). For cross-sectional signals, fix the ticker after creation using data from `self._data`.
- Position sizing: `shares = int(nav * risk_factor / atr)` where `nav = available_capital + Σ(qty × current_price)` for all positions.
- `indicator_value("CLENOW_SCORE", iid, date, period=90)` returns `CLENOW_SCORE_90` column value. Same for `MA` with `period=200` and `ATR` with `period=20`.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_strategies.py`:

```python
from trendspec.strategy.examples import ClenowMomentumStrategy  # will fail until Task 5 done


class TestClenowMomentumStrategyInit:
    def test_strategy_registration(self) -> None:
        from trendspec.strategy.base import get_strategy
        assert get_strategy("clenow_momentum") is ClenowMomentumStrategy

    def test_default_params(self) -> None:
        strategy = ClenowMomentumStrategy()
        assert strategy.get_param("sma_period", 200) == 200
        assert strategy.get_param("atr_period", 20) == 20
        assert strategy.get_param("score_period", 90) == 90
        assert strategy.get_param("gap_period", 90) == 90
        assert strategy.get_param("risk_factor", 0.001) == 0.001
        assert strategy.get_param("rebalance_weekday", 2) == 2
        assert strategy.get_param("top_pct", 0.8) == 0.8

    def test_invalid_top_pct(self) -> None:
        with pytest.raises(ValueError, match="top_pct"):
            ClenowMomentumStrategy(params={"top_pct": 1.5})

    def test_invalid_risk_factor(self) -> None:
        with pytest.raises(ValueError, match="risk_factor"):
            ClenowMomentumStrategy(params={"risk_factor": -0.001})

    def test_invalid_rebalance_weekday(self) -> None:
        with pytest.raises(ValueError, match="rebalance_weekday"):
            ClenowMomentumStrategy(params={"rebalance_weekday": 7})


class TestClenowMomentumStrategySignals:
    """Integration tests with synthetic data."""

    def _make_trending_df(self, n_days: int = 300) -> pl.DataFrame:
        """Make a DataFrame with a clear uptrend and one stock in downtrend."""
        import numpy as np
        rng = np.random.default_rng(0)
        rows = []
        for inst, trend in [("UP1", 0.002), ("UP2", 0.0015), ("DOWN", -0.003)]:
            price = 100.0
            for i in range(n_days):
                price = max(1.0, price * (1 + trend + rng.normal(0, 0.005)))
                rows.append({
                    "instrument_id": inst, "ticker": inst,
                    "date": date(2022, 1, 1) + __import__("datetime").timedelta(days=i),
                    "open": price * 0.995, "high": price * 1.005,
                    "low": price * 0.990, "close": price,
                    "volume": 1_000_000,
                })
        return pl.DataFrame(rows)

    def test_init_precomputes_indicators(self) -> None:
        """Strategy init() runs without error and caches indicators."""
        from trendspec.strategy.context import StrategyContext
        from trendspec.data.markets import Market

        df = self._make_trending_df(300)
        strategy = ClenowMomentumStrategy(params={
            "sma_period": 50, "score_period": 30, "gap_period": 30, "atr_period": 10,
        })
        ctx = StrategyContext(market=Market.US, strategy=strategy, data=df)
        strategy.init(ctx)

        cache_keys = list(ctx._indicator_cache.keys())
        assert any("CLENOW_SCORE" in k for k in cache_keys)
        assert any("MIN_DAILY_RETURN" in k for k in cache_keys)
        assert any("ATR" in k for k in cache_keys)
        assert any("MA" in k for k in cache_keys)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_strategies.py::TestClenowMomentumStrategyInit tests/test_strategies.py::TestClenowMomentumStrategySignals -v
```

Expected: `FAILED` — `ImportError: cannot import name 'ClenowMomentumStrategy'`

- [ ] **Step 3: Implement ClenowMomentumStrategy**

Create `trendspec/strategy/examples/clenow_momentum.py`:

```python
"""
Clenow Quantitative Momentum Strategy.

Based on Andreas Clenow's "Stocks on the Move".

Strategy logic:
- Score = annualized exponential regression slope × R² over 90-day window
- Filters: price > SMA(200), no single day drop > 15% in 90 days, score > 0
- Rank all qualifying universe stocks by score descending
- Weekly rebalance (default: Wednesday):
    SELL: current positions that dropped below 200 SMA, or rank fell out of top 80%
    BUY:  top-ranked stocks not yet held (ATR-based position sizing)
- Position size: int(total_equity × risk_factor / ATR(20))

Parameters:
    sma_period (int): Trend filter SMA period. Default 200.
    atr_period (int): ATR period for position sizing. Default 20.
    score_period (int): Regression lookback in trading days. Default 90.
    gap_period (int): Window for gap filter (same as score_period). Default 90.
    risk_factor (float): Equity fraction per ATR unit. Default 0.001.
    rebalance_weekday (int): 0=Mon … 4=Fri. Default 2 (Wednesday).
    top_pct (float): Fraction of ranked universe to hold (e.g. 0.8 = top 80%). Default 0.8.
    max_gap (float): Maximum allowed single-day drop (negative). Default -0.15.
"""

from datetime import date as DateType

import polars as pl

from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext


@register_strategy("clenow_momentum")
class ClenowMomentumStrategy(BaseStrategy):
    """Clenow quantitative momentum strategy (Stocks on the Move)."""

    name = "clenow_momentum"
    version = "1.0.0"
    params = {
        "sma_period": 200,
        "atr_period": 20,
        "score_period": 90,
        "gap_period": 90,
        "risk_factor": 0.001,
        "rebalance_weekday": 2,
        "top_pct": 0.8,
        "max_gap": -0.15,
    }

    def _validate_dict_params(self) -> None:
        top_pct = self.get_param("top_pct", 0.8)
        risk_factor = self.get_param("risk_factor", 0.001)
        rebalance_weekday = self.get_param("rebalance_weekday", 2)

        if not (0 < top_pct < 1):
            raise ValueError(f"top_pct ({top_pct}) must be between 0 and 1 exclusive")
        if risk_factor <= 0:
            raise ValueError(f"risk_factor ({risk_factor}) must be > 0")
        if rebalance_weekday not in range(5):
            raise ValueError(f"rebalance_weekday ({rebalance_weekday}) must be 0-4 (Mon-Fri)")

    def init(self, ctx: StrategyContext) -> None:
        """Precompute all indicators once over the full dataset."""
        sma_period = self.get_param("sma_period", 200)
        atr_period = self.get_param("atr_period", 20)
        score_period = self.get_param("score_period", 90)
        gap_period = self.get_param("gap_period", 90)

        ctx.precompute_indicator("MA", period=sma_period)
        ctx.precompute_indicator("ATR", period=atr_period)
        ctx.precompute_indicator("CLENOW_SCORE", period=score_period)
        ctx.precompute_indicator("MIN_DAILY_RETURN", period=gap_period)

        # Cache parameters
        self._sma_period = sma_period
        self._atr_period = atr_period
        self._score_period = score_period
        self._gap_period = gap_period
        self._risk_factor = self.get_param("risk_factor", 0.001)
        self._rebalance_weekday = self.get_param("rebalance_weekday", 2)
        self._top_pct = self.get_param("top_pct", 0.8)
        self._max_gap = self.get_param("max_gap", -0.15)

        # Guard: prevent re-running cross-sectional work on the same day
        self._last_rebalance_date: DateType | None = None

        # Store full data for cross-sectional price lookups
        self._full_data = ctx._data

        ctx.strategy.log(
            f"Initialized: sma={sma_period}, atr={atr_period}, "
            f"score_period={score_period}, rebalance=weekday {self._rebalance_weekday}, "
            f"top_pct={self._top_pct}"
        )

    def next(self, ctx: StrategyContext) -> None:
        """
        Weekly rebalancing via cross-sectional momentum ranking.

        Only runs on the configured weekday. The first instrument call on that day
        does all work; subsequent calls return immediately.
        """
        current_date = ctx.date

        # Guard: skip non-rebalance days
        if current_date.weekday() != self._rebalance_weekday:
            return

        # Guard: skip if already processed this rebalance day
        if current_date == self._last_rebalance_date:
            return

        self._last_rebalance_date = current_date

        # --- 1. Build current-day price lookup ---
        day_data = self._full_data.filter(pl.col("date") == current_date)
        if day_data.is_empty():
            return

        def get_close(instrument_id: str) -> float | None:
            rows = day_data.filter(pl.col("instrument_id") == instrument_id)
            if rows.is_empty():
                return None
            return rows["close"].item()

        def get_ticker(instrument_id: str) -> str:
            rows = day_data.filter(pl.col("instrument_id") == instrument_id)
            if rows.is_empty():
                return instrument_id
            return rows["ticker"].item()

        # --- 2. Score qualifying universe instruments ---
        universe_ids = ctx.pit_universe(current_date)
        scores: dict[str, float] = {}

        for iid in universe_ids:
            sma = ctx.indicator_value("MA", iid, current_date, period=self._sma_period)
            score = ctx.indicator_value("CLENOW_SCORE", iid, current_date, period=self._score_period)
            min_ret = ctx.indicator_value("MIN_DAILY_RETURN", iid, current_date, period=self._gap_period)
            close = get_close(iid)

            if sma is None or score is None or min_ret is None or close is None:
                continue
            if close <= sma:       # Below 200 SMA — trend broken
                continue
            if min_ret < self._max_gap:  # Extreme gap-down in lookback window
                continue
            if score <= 0:         # Negative momentum
                continue

            scores[iid] = score

        # --- 3. Determine top-ranked set ---
        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
        n_keep = max(1, int(len(ranked) * self._top_pct))
        top_set = set(ranked[:n_keep])

        # --- 4. Compute total equity for position sizing ---
        nav = ctx.available_capital
        for iid, qty in ctx.positions.items():
            close = get_close(iid)
            if close is not None:
                nav += qty * close

        # --- 5. Generate SELL signals ---
        for iid in list(ctx.positions.keys()):
            sma = ctx.indicator_value("MA", iid, current_date, period=self._sma_period)
            close = get_close(iid)

            sell_reason = None
            if close is None:
                sell_reason = "no price data"
            elif sma is not None and close <= sma:
                sell_reason = f"price {close:.2f} below SMA{self._sma_period} {sma:.2f}"
            elif iid not in top_set:
                sell_reason = "rank dropped out of top qualified universe"

            if sell_reason:
                sell_price = close or ctx.close
                sig = ctx.signal("SELL", iid, sell_price, note=sell_reason)
                sig.ticker = get_ticker(iid)

        # --- 6. Generate BUY signals for top-ranked, not already held ---
        for iid in ranked[:n_keep]:
            if ctx.has_position(iid):
                continue

            atr = ctx.indicator_value("ATR", iid, current_date, period=self._atr_period)
            close = get_close(iid)

            if atr is None or atr <= 0 or close is None or close <= 0:
                continue

            shares = int(nav * self._risk_factor / atr)
            if shares < 1:
                continue

            sig = ctx.signal(
                "BUY",
                iid,
                close,
                trigger_value=scores[iid],
                note=f"Clenow score={scores[iid]:.2f}, ATR={atr:.2f}, shares={shares}",
            )
            sig.ticker = get_ticker(iid)
            sig.shares = float(shares)
```

- [ ] **Step 4: Add ClenowMomentumStrategy to examples __init__.py**

In `trendspec/strategy/examples/__init__.py`, add the import and export:

```python
from trendspec.strategy.examples.ma_cross import MACrossStrategy
from trendspec.strategy.examples.rsi_reversal import RSIReversalStrategy
from trendspec.strategy.examples.sector_momentum import SectorMomentumStrategy
from trendspec.strategy.examples.clenow_momentum import ClenowMomentumStrategy

__all__ = [
    "MACrossStrategy",
    "RSIReversalStrategy",
    "SectorMomentumStrategy",
    "ClenowMomentumStrategy",
]
```

- [ ] **Step 5: Run strategy tests**

```bash
uv run pytest tests/test_strategies.py::TestClenowMomentumStrategyInit tests/test_strategies.py::TestClenowMomentumStrategySignals -v
```

Expected: All `PASSED`

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest -v --tb=short
```

Expected: All passing.

- [ ] **Step 7: Run ruff linter**

```bash
uv run ruff check trendspec/strategy/examples/clenow_momentum.py trendspec/strategy/indicators.py
```

Expected: No errors. Fix any reported issues before committing.

- [ ] **Step 8: Commit**

```bash
git add trendspec/strategy/examples/clenow_momentum.py trendspec/strategy/examples/__init__.py tests/test_strategies.py
git commit -m "feat: add ClenowMomentumStrategy with ATR-based position sizing and weekly rebalancing"
```

---

## Task 6: End-to-End Smoke Test via CLI

**Files:** No code changes — verify existing CLI integration.

- [ ] **Step 1: Verify strategy appears in list**

```bash
uv run trendspec backtest list
```

Expected: `clenow_momentum` appears in the strategy list.

- [ ] **Step 2: Run screening (no DB needed)**

If `data_lake` has data ingested:

```bash
uv run trendspec screen --strategy clenow_momentum --market us --date 2024-05-15
```

Expected: Command runs without `ImportError` or `AttributeError`. May produce empty results if no qualifying stocks (acceptable for a smoke test).

- [ ] **Step 3: Run linter across all modified files**

```bash
uv run ruff check trendspec/ tests/test_strategies.py
uv run ruff format --check trendspec/ tests/test_strategies.py
```

Expected: No issues.

- [ ] **Step 4: Final full test run**

```bash
uv run pytest -v --tb=short
```

Expected: All tests pass.

---

## Verification Summary

| Check | Command |
|-------|---------|
| All tests pass | `uv run pytest -v --tb=short` |
| scipy available | `uv run python -c "from scipy import stats"` |
| Strategy registered | `uv run trendspec backtest list` |
| Linter clean | `uv run ruff check .` |

---

## Key Invariants to Preserve

1. `Signal.shares = None` → engine uses `order_size` param (backward compatible).
2. `indicator_value("CLENOW_SCORE", iid, date, period=90)` → returns `CLENOW_SCORE_90` column value (not slope or R²).
3. Rebalance runs at most once per calendar day (guarded by `_last_rebalance_date`).
4. SELL signals always generated before BUY signals in the same rebalance cycle.
5. Instruments with insufficient data (None indicator values) are silently skipped.
