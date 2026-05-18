# Clenow Momentum 选股输出增强 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `clenow_momentum` 策略的 BUY 信号输出从 6 列扩展为 10 列决策表（行业 / 排名 / 止损 / R² / 乖离率 / 回撤 / 放量 / 预警），并修复 CSV 文件名（含策略名，适用所有策略）。

**Architecture:** 在 `Signal` 上新增 `extras: dict` 字段作为通用逃生口；clenow 策略在 BUY 信号生成时填充所有展示字段；`ScreeningReport` 按 `strategy_name == "clenow_momentum"` 切换 10 列渲染分支与 13 列 CSV schema。三个新指标（HH、SMA_VOLUME、CLENOW_R2）独立注册到 `indicators.py`。

**Tech Stack:** Python 3.12+ · Polars · scipy · rich · pytest · typer · uv

**Spec:** [`docs/superpowers/specs/2026-05-18-clenow-momentum-screen-output-design.md`](../specs/2026-05-18-clenow-momentum-screen-output-design.md)

---

## 文件改动总览

| 文件 | 改动 |
|------|------|
| `trendspec/strategy/signal.py` | 新增 `Signal.extras: dict` 字段 |
| `trendspec/strategy/indicators.py` | 新增 `HH`、`SMA_VOLUME`、`CLENOW_R2` 三个 indicator |
| `trendspec/strategy/examples/clenow_momentum.py` | 新增 6 个参数 + 校验；`init()` precompute 3 新指标；`next()` BUY 时填 extras |
| `trendspec/screening/report.py` | 按 strategy_name 切渲染分支；CSV 文件名加 `<strategy>` 段；clenow CSV 13 列 schema |
| `tests/test_strategy.py` | 新增 Signal.extras 测试 + 三个新指标测试 |
| `tests/test_strategies.py` | 新增 clenow 参数校验 + extras 完整性 + 阈值规则测试 |
| `tests/test_screening_report.py` | 新建：CSV 文件名 + clenow 10 列渲染 + 13 列 CSV schema |

---

## Task 1: Signal 新增 `extras` 字段

**Files:**
- Modify: `trendspec/strategy/signal.py`
- Test: `tests/test_strategy.py`

- [ ] **Step 1: Write failing test (Signal extras)**

在 `tests/test_strategy.py` 的 `TestSignal` 类末尾追加：

```python
    def test_signal_extras_default_empty(self) -> None:
        """extras defaults to empty dict, not shared across instances."""
        s1 = Signal(direction="BUY", ticker="A", instrument_id="A", price=1.0)
        s2 = Signal(direction="BUY", ticker="B", instrument_id="B", price=2.0)
        assert s1.extras == {}
        assert s2.extras == {}
        s1.extras["foo"] = 1
        assert s2.extras == {}  # 不共享同一 dict 实例

    def test_signal_extras_arbitrary_payload(self) -> None:
        """extras accepts arbitrary keys/values."""
        s = Signal(
            direction="BUY",
            ticker="X",
            instrument_id="X",
            price=10.0,
            extras={"rank": 1, "sector": "Tech", "alerts": ["a", "b"]},
        )
        assert s.extras["rank"] == 1
        assert s.extras["sector"] == "Tech"
        assert s.extras["alerts"] == ["a", "b"]
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_strategy.py::TestSignal::test_signal_extras_default_empty tests/test_strategy.py::TestSignal::test_signal_extras_arbitrary_payload -v
```

Expected: FAIL — `Signal` 不接受 `extras` kwarg 或属性不存在。

- [ ] **Step 3: 实现 — 修改 `trendspec/strategy/signal.py`**

在 `Signal` dataclass 顶部 import 区域追加 `from typing import Any`（如已有则跳过）。在 `timestamp` 字段后追加：

```python
    extras: dict[str, Any] = field(default_factory=dict, repr=False)
```

完整字段顺序：

```python
@dataclass
class Signal:
    direction: Literal["BUY", "SELL"]
    ticker: str
    instrument_id: str
    price: float
    trigger_value: float | None = None
    note: str | None = None
    shares: float | None = field(default=None, repr=False)
    timestamp: float | None = field(default=None, repr=False)
    extras: dict[str, Any] = field(default_factory=dict, repr=False)
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_strategy.py::TestSignal -v
```

Expected: 全部 PASS（含原有 Signal 测试无回归）。

- [ ] **Step 5: Commit**

```
git add trendspec/strategy/signal.py tests/test_strategy.py
git commit -m "feat(signal): add extras dict field for strategy-specific display payload"
```

---

## Task 2: 注册 `HH` indicator（滚动最高收盘价）

**Files:**
- Modify: `trendspec/strategy/indicators.py`
- Test: `tests/test_strategy.py`

- [ ] **Step 1: Write failing test (HH)**

在 `tests/test_strategy.py` 中找到现有 indicator 测试类（如 `TestIndicators` 或类似），追加用例。若类不存在，直接在文件末尾新建：

```python
class TestHHIndicator:
    """Highest High (rolling max of close) indicator."""

    def _sample_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "instrument_id": ["A"] * 5,
            "date": [date(2024, 1, i) for i in range(1, 6)],
            "close": [10.0, 12.0, 11.0, 15.0, 14.0],
            "open": [10.0] * 5, "high": [10.0] * 5,
            "low": [10.0] * 5, "volume": [1000] * 5, "adj_factor": [1.0] * 5,
        })

    def test_hh_period_3(self) -> None:
        from trendspec.strategy.indicators import compute_indicator
        df = self._sample_df()
        out = compute_indicator(df, "HH", period=3)
        vals = out.sort("date")["HH_3"].to_list()
        # 窗口 3 天：第 1/2 日不够 → None；第 3 日 max(10,12,11)=12；
        # 第 4 日 max(12,11,15)=15；第 5 日 max(11,15,14)=15
        assert vals == [None, None, 12.0, 15.0, 15.0]

    def test_hh_per_instrument_isolated(self) -> None:
        """HH computed per instrument_id group, not across instruments."""
        from trendspec.strategy.indicators import compute_indicator
        df = pl.DataFrame({
            "instrument_id": ["A", "A", "A", "B", "B", "B"],
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)] * 2,
            "close": [10.0, 12.0, 11.0, 100.0, 90.0, 95.0],
            "open": [0.0] * 6, "high": [0.0] * 6,
            "low": [0.0] * 6, "volume": [0] * 6, "adj_factor": [1.0] * 6,
        })
        out = compute_indicator(df, "HH", period=2).sort(["instrument_id", "date"])
        a_vals = out.filter(pl.col("instrument_id") == "A")["HH_2"].to_list()
        b_vals = out.filter(pl.col("instrument_id") == "B")["HH_2"].to_list()
        assert a_vals == [None, 12.0, 12.0]
        assert b_vals == [None, 100.0, 95.0]
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_strategy.py::TestHHIndicator -v
```

Expected: FAIL — "Unknown indicator: HH"。

- [ ] **Step 3: 实现 — 修改 `trendspec/strategy/indicators.py`**

在文件末尾追加（紧跟现有 indicator 注册模式，参考 `VMA` 写法）：

```python
@register_indicator("HH")
def highest_high(df: pl.DataFrame, period: int = 63) -> pl.DataFrame:
    """
    Rolling highest close over `period` days, per instrument.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback window

    Returns:
        DataFrame with HH_{period} column added (None for first period-1 rows)
    """
    col_name = f"HH_{period}"
    return df.sort("date").with_columns(
        pl.col("close")
        .rolling_max(window_size=period)
        .over("instrument_id")
        .alias(col_name)
    )
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_strategy.py::TestHHIndicator -v
```

Expected: PASS。

- [ ] **Step 5: Commit**

```
git add trendspec/strategy/indicators.py tests/test_strategy.py
git commit -m "feat(indicators): register HH (highest high) rolling max indicator"
```

---

## Task 3: 注册 `SMA_VOLUME` indicator（成交量均量）

**Files:**
- Modify: `trendspec/strategy/indicators.py`
- Test: `tests/test_strategy.py`

注：现有 `VMA` 已计算 volume 滚动均值，但为保持与 spec 命名一致（避免后续 clenow 策略 indicator name 与 VMA 混淆），新增独立 `SMA_VOLUME` 注册名。

- [ ] **Step 1: Write failing test (SMA_VOLUME)**

```python
class TestSMAVolumeIndicator:
    """SMA of volume column."""

    def test_sma_volume_period_3(self) -> None:
        from trendspec.strategy.indicators import compute_indicator
        df = pl.DataFrame({
            "instrument_id": ["A"] * 5,
            "date": [date(2024, 1, i) for i in range(1, 6)],
            "close": [10.0] * 5, "open": [10.0] * 5,
            "high": [10.0] * 5, "low": [10.0] * 5,
            "volume": [100, 200, 300, 400, 500],
            "adj_factor": [1.0] * 5,
        })
        out = compute_indicator(df, "SMA_VOLUME", period=3).sort("date")
        vals = out["SMA_VOLUME_3"].to_list()
        # 第 3 日 (100+200+300)/3=200；第 4 日 300；第 5 日 400
        assert vals[0] is None
        assert vals[1] is None
        assert vals[2] == pytest.approx(200.0)
        assert vals[3] == pytest.approx(300.0)
        assert vals[4] == pytest.approx(400.0)
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_strategy.py::TestSMAVolumeIndicator -v
```

Expected: FAIL — "Unknown indicator: SMA_VOLUME"。

- [ ] **Step 3: 实现 — 修改 `trendspec/strategy/indicators.py`**

在 `HH` 函数后追加：

```python
@register_indicator("SMA_VOLUME")
def sma_volume(df: pl.DataFrame, period: int = 50) -> pl.DataFrame:
    """
    Simple Moving Average of volume column, per instrument.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback window

    Returns:
        DataFrame with SMA_VOLUME_{period} column added
    """
    col_name = f"SMA_VOLUME_{period}"
    return df.sort("date").with_columns(
        pl.col("volume")
        .rolling_mean(window_size=period)
        .over("instrument_id")
        .alias(col_name)
    )
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_strategy.py::TestSMAVolumeIndicator -v
```

Expected: PASS。

- [ ] **Step 5: Commit**

```
git add trendspec/strategy/indicators.py tests/test_strategy.py
git commit -m "feat(indicators): register SMA_VOLUME rolling volume average"
```

---

## Task 4: 注册 `CLENOW_R2` indicator

**Files:**
- Modify: `trendspec/strategy/indicators.py`
- Test: `tests/test_strategy.py`

- [ ] **Step 1: Write failing test (CLENOW_R2)**

```python
class TestClenowR2Indicator:
    """Standalone R² from log-price linear regression."""

    def test_clenow_r2_range(self) -> None:
        """R² ∈ [0, 1] for any non-degenerate window."""
        import numpy as np
        from trendspec.strategy.indicators import compute_indicator

        rng = np.random.default_rng(42)
        prices = [100.0]
        for _ in range(99):
            prices.append(max(1.0, prices[-1] * (1 + 0.002 + rng.normal(0, 0.01))))

        df = pl.DataFrame({
            "instrument_id": ["A"] * 100,
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(100)],
            "close": prices,
            "open": prices, "high": prices, "low": prices,
            "volume": [1000] * 100, "adj_factor": [1.0] * 100,
        })
        out = compute_indicator(df, "CLENOW_R2", period=60).sort("date")
        col = out["CLENOW_R2_60"].to_list()
        # 前 period-1 行为 None
        assert all(v is None for v in col[:59])
        # 后续行 ∈ [0, 1]
        for v in col[59:]:
            assert v is not None
            assert 0.0 <= v <= 1.0

    def test_clenow_r2_perfect_log_trend(self) -> None:
        """完美对数线性序列 → R² ≈ 1.0"""
        import numpy as np
        from trendspec.strategy.indicators import compute_indicator

        # ln(price) = a + b*i  →  price = exp(a) * exp(b*i)
        prices = [float(np.exp(0.01 * i)) for i in range(60)]
        df = pl.DataFrame({
            "instrument_id": ["A"] * 60,
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(60)],
            "close": prices,
            "open": prices, "high": prices, "low": prices,
            "volume": [1000] * 60, "adj_factor": [1.0] * 60,
        })
        out = compute_indicator(df, "CLENOW_R2", period=30).sort("date")
        last_r2 = out["CLENOW_R2_30"].to_list()[-1]
        assert last_r2 == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_strategy.py::TestClenowR2Indicator -v
```

Expected: FAIL — "Unknown indicator: CLENOW_R2"。

- [ ] **Step 3: 实现 — 修改 `trendspec/strategy/indicators.py`**

在 `SMA_VOLUME` 后追加：

```python
@register_indicator("CLENOW_R2")
def clenow_r2(df: pl.DataFrame, period: int = 90) -> pl.DataFrame:
    """
    Rolling R² of ln(close) vs day-index linear regression.

    Independent from CLENOW_SCORE to keep indicator registry single-output.

    Args:
        df: DataFrame with OHLCV data
        period: Regression lookback window

    Returns:
        DataFrame with CLENOW_R2_{period} column added (None for first period-1 rows
        and for windows containing non-positive prices)
    """
    import numpy as np
    from scipy import stats

    col_name = f"CLENOW_R2_{period}"
    x = np.arange(period, dtype=float)

    all_groups: list[pl.DataFrame] = []
    for (_iid,), group in df.sort(["instrument_id", "date"]).group_by(
        ["instrument_id"], maintain_order=True
    ):
        closes = group["close"].to_numpy()
        n = len(closes)
        r2s: list[float | None] = [None] * n

        for i in range(period - 1, n):
            window = closes[i - period + 1 : i + 1]
            if np.any(window <= 0):
                continue
            y = np.log(window)
            fit = stats.linregress(x, y)
            r2s[i] = float(fit.rvalue ** 2)

        all_groups.append(
            group.with_columns(pl.Series(col_name, r2s, dtype=pl.Float64))
        )

    if not all_groups:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(col_name))

    return pl.concat(all_groups).sort(["instrument_id", "date"])
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_strategy.py::TestClenowR2Indicator -v
```

Expected: PASS。

- [ ] **Step 5: Commit**

```
git add trendspec/strategy/indicators.py tests/test_strategy.py
git commit -m "feat(indicators): register standalone CLENOW_R2 indicator"
```

---

## Task 5: ClenowMomentumStrategy 新增参数与校验

**Files:**
- Modify: `trendspec/strategy/examples/clenow_momentum.py`
- Test: `tests/test_strategies.py`

- [ ] **Step 1: Write failing tests (param defaults + validation)**

在 `tests/test_strategies.py` 的 `TestClenowMomentumStrategyInit` 类末尾追加：

```python
    def test_new_display_param_defaults(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        s = ClenowMomentumStrategy()
        assert s.get_param("atr_stop_k", None) == 3.0
        assert s.get_param("drawdown_period", None) == 63
        assert s.get_param("volume_avg_period", None) == 50
        assert s.get_param("warn_deviation_max", None) == 40.0
        assert s.get_param("warn_vol_mult_low", None) == 1.0
        assert s.get_param("warn_vol_mult_high", None) == 3.0
        assert s.get_param("warn_drawdown_max", None) == -15.0

    def test_invalid_atr_stop_k(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        with pytest.raises(ValueError, match="atr_stop_k"):
            ClenowMomentumStrategy(params={"atr_stop_k": 0})
        with pytest.raises(ValueError, match="atr_stop_k"):
            ClenowMomentumStrategy(params={"atr_stop_k": -1.0})

    def test_invalid_drawdown_period(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        with pytest.raises(ValueError, match="drawdown_period"):
            ClenowMomentumStrategy(params={"drawdown_period": 1})

    def test_invalid_volume_avg_period(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        with pytest.raises(ValueError, match="volume_avg_period"):
            ClenowMomentumStrategy(params={"volume_avg_period": 1})

    def test_invalid_vol_mult_threshold_ordering(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        with pytest.raises(ValueError, match="warn_vol_mult"):
            ClenowMomentumStrategy(params={
                "warn_vol_mult_low": 3.0, "warn_vol_mult_high": 1.0,
            })

    def test_invalid_warn_drawdown_max_nonneg(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        with pytest.raises(ValueError, match="warn_drawdown_max"):
            ClenowMomentumStrategy(params={"warn_drawdown_max": 0.0})

    def test_invalid_warn_deviation_max_nonpos(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        with pytest.raises(ValueError, match="warn_deviation_max"):
            ClenowMomentumStrategy(params={"warn_deviation_max": 0.0})
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_strategies.py::TestClenowMomentumStrategyInit -v
```

Expected: 新加 7 个用例全部 FAIL（默认值不存在；ValueError 未抛）。

- [ ] **Step 3: 实现 — 修改 `trendspec/strategy/examples/clenow_momentum.py`**

3a. 扩展 `params` dict（在 `class ClenowMomentumStrategy` 类属性区，替换现有 `params = {...}`）：

```python
    params = {
        "sma_period": 200,
        "atr_period": 20,
        "score_period": 90,
        "gap_period": 90,
        "risk_factor": 0.001,
        "rebalance_weekday": 2,
        "top_pct": 0.8,
        "max_gap": -0.15,
        # Display-only fields (do not affect entry/exit logic)
        "atr_stop_k": 3.0,
        "drawdown_period": 63,
        "volume_avg_period": 50,
        "warn_deviation_max": 40.0,
        "warn_vol_mult_low": 1.0,
        "warn_vol_mult_high": 3.0,
        "warn_drawdown_max": -15.0,
    }
```

3b. 扩展 `_validate_dict_params()`，在原方法末尾追加：

```python
        atr_stop_k = self.get_param("atr_stop_k", 3.0)
        drawdown_period = self.get_param("drawdown_period", 63)
        volume_avg_period = self.get_param("volume_avg_period", 50)
        warn_deviation_max = self.get_param("warn_deviation_max", 40.0)
        warn_vol_mult_low = self.get_param("warn_vol_mult_low", 1.0)
        warn_vol_mult_high = self.get_param("warn_vol_mult_high", 3.0)
        warn_drawdown_max = self.get_param("warn_drawdown_max", -15.0)

        if atr_stop_k <= 0:
            raise ValueError(f"atr_stop_k ({atr_stop_k}) must be > 0")
        if drawdown_period < 2:
            raise ValueError(f"drawdown_period ({drawdown_period}) must be >= 2")
        if volume_avg_period < 2:
            raise ValueError(f"volume_avg_period ({volume_avg_period}) must be >= 2")
        if warn_deviation_max <= 0:
            raise ValueError(f"warn_deviation_max ({warn_deviation_max}) must be > 0")
        if warn_drawdown_max >= 0:
            raise ValueError(f"warn_drawdown_max ({warn_drawdown_max}) must be < 0")
        if warn_vol_mult_low >= warn_vol_mult_high:
            raise ValueError(
                f"warn_vol_mult_low ({warn_vol_mult_low}) must be < "
                f"warn_vol_mult_high ({warn_vol_mult_high})"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_strategies.py::TestClenowMomentumStrategyInit -v
```

Expected: 全部 PASS（含原有 init 测试无回归）。

- [ ] **Step 5: Commit**

```
git add trendspec/strategy/examples/clenow_momentum.py tests/test_strategies.py
git commit -m "feat(clenow): add display-field params with validation"
```

---

## Task 6: ClenowMomentumStrategy `init()` precompute 新指标

**Files:**
- Modify: `trendspec/strategy/examples/clenow_momentum.py`
- Test: `tests/test_strategies.py`

- [ ] **Step 1: Write failing test**

在 `tests/test_strategies.py::TestClenowMomentumStrategySignals` 类的 `test_init_precomputes_indicators` 方法处（已存在），在其末尾追加新断言。如果不能修改原方法，新增独立用例：

```python
    def test_init_precomputes_display_indicators(self) -> None:
        """init() also precomputes HH, SMA_VOLUME, CLENOW_R2 for display fields."""
        from trendspec.strategy.examples import ClenowMomentumStrategy
        from trendspec.strategy.context import StrategyContext

        df = self._make_trending_df(300)
        strategy = ClenowMomentumStrategy(params={
            "sma_period": 50, "score_period": 30, "gap_period": 30, "atr_period": 10,
            "drawdown_period": 20, "volume_avg_period": 20,
        })
        ctx = StrategyContext(market=Market.US, strategy=strategy, data=df)
        strategy.init(ctx)

        cache_keys = list(ctx._indicator_cache.keys())
        assert any("HH" in k for k in cache_keys), f"HH not precomputed: {cache_keys}"
        assert any("SMA_VOLUME" in k for k in cache_keys), f"SMA_VOLUME not precomputed: {cache_keys}"
        assert any("CLENOW_R2" in k for k in cache_keys), f"CLENOW_R2 not precomputed: {cache_keys}"
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_strategies.py::TestClenowMomentumStrategySignals::test_init_precomputes_display_indicators -v
```

Expected: FAIL — 新指标未在 cache_keys 中。

- [ ] **Step 3: 实现 — 修改 `trendspec/strategy/examples/clenow_momentum.py`**

在 `init()` 方法中，现有 `ctx.precompute_indicator("MIN_DAILY_RETURN", ...)` 后追加 3 行 precompute；并存储新参数到实例：

```python
        ctx.precompute_indicator("HH", period=self.get_param("drawdown_period", 63))
        ctx.precompute_indicator("SMA_VOLUME", period=self.get_param("volume_avg_period", 50))
        ctx.precompute_indicator("CLENOW_R2", period=score_period)

        self._drawdown_period = self.get_param("drawdown_period", 63)
        self._volume_avg_period = self.get_param("volume_avg_period", 50)
        self._atr_stop_k = self.get_param("atr_stop_k", 3.0)
        self._warn_deviation_max = self.get_param("warn_deviation_max", 40.0)
        self._warn_vol_mult_low = self.get_param("warn_vol_mult_low", 1.0)
        self._warn_vol_mult_high = self.get_param("warn_vol_mult_high", 3.0)
        self._warn_drawdown_max = self.get_param("warn_drawdown_max", -15.0)
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_strategies.py::TestClenowMomentumStrategySignals -v
```

Expected: 全部 PASS（含原有 precompute 用例）。

- [ ] **Step 5: Commit**

```
git add trendspec/strategy/examples/clenow_momentum.py tests/test_strategies.py
git commit -m "feat(clenow): precompute HH/SMA_VOLUME/CLENOW_R2 in init()"
```

---

## Task 7: ClenowMomentumStrategy BUY signal 填充 extras

**Files:**
- Modify: `trendspec/strategy/examples/clenow_momentum.py`
- Test: `tests/test_strategies.py`

本任务包含较多测试（extras 完整性 + 各类预警），分多个 step。

- [ ] **Step 1: Write failing tests (extras schema + sector lookup)**

在 `tests/test_strategies.py::TestClenowMomentumStrategySignals` 类追加测试辅助 + 新用例。先在类内增加辅助方法：

```python
    def _run_strategy_and_get_buys(
        self,
        df: pl.DataFrame,
        rebalance_date: date,
        sector_index_mock=None,
        params_override: dict | None = None,
    ) -> list:
        """Helper: init + manually invoke next() for one rebalance day, return BUY signals."""
        from unittest.mock import MagicMock, patch
        from trendspec.strategy.examples import ClenowMomentumStrategy
        from trendspec.strategy.context import StrategyContext

        params = {
            "sma_period": 50, "score_period": 30, "gap_period": 30, "atr_period": 10,
            "drawdown_period": 20, "volume_avg_period": 20,
            "rebalance_weekday": rebalance_date.weekday(),
        }
        if params_override:
            params.update(params_override)

        strategy = ClenowMomentumStrategy(params=params)
        ctx = StrategyContext(market=Market.US, strategy=strategy, data=df)
        ctx._universe = MagicMock()
        ctx._universe.tickers = MagicMock(return_value=list(df["instrument_id"].unique()))
        ctx.pit_universe = lambda d: list(df["instrument_id"].unique())
        ctx._current_date = rebalance_date
        ctx.update_positions({}, 1_000_000.0)
        strategy.init(ctx)

        collected: list = []
        original_signal = ctx.signal

        def capture_signal(*args, **kwargs):
            sig = original_signal(*args, **kwargs)
            collected.append(sig)
            return sig

        ctx.signal = capture_signal

        if sector_index_mock is not None:
            with patch("trendspec.data.sectors.sector", side_effect=sector_index_mock):
                strategy.next(ctx)
        else:
            strategy.next(ctx)

        return [s for s in collected if s.is_buy()]
```

然后添加测试：

```python
    def test_buy_signal_has_full_extras_schema(self) -> None:
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(df, rebalance_date)
        assert len(buys) >= 1
        for sig in buys:
            keys = set(sig.extras.keys())
            assert keys == {
                "sector", "rank", "r2", "deviation_pct",
                "drawdown_pct", "vol_mult", "stop_loss", "alerts",
            }
            assert isinstance(sig.extras["rank"], int)
            assert sig.extras["rank"] >= 1
            assert isinstance(sig.extras["r2"], float)
            assert 0.0 <= sig.extras["r2"] <= 1.0
            assert isinstance(sig.extras["deviation_pct"], float)
            assert isinstance(sig.extras["drawdown_pct"], float)
            assert isinstance(sig.extras["vol_mult"], float)
            assert isinstance(sig.extras["stop_loss"], float)
            assert sig.extras["stop_loss"] > 0
            assert isinstance(sig.extras["alerts"], list)

    def test_buy_rank_monotonic_top_first(self) -> None:
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(df, rebalance_date)
        ranks = [s.extras["rank"] for s in buys]
        assert ranks == sorted(ranks)
        assert ranks[0] == 1

    def test_buy_sector_lookup_returned(self) -> None:
        """sector() mocked → extras['sector'] reflects mock value."""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]

        def mock_sector(market, iid, dt):
            return {"UP1": "Technology", "UP2": "Financials", "DOWN": "Energy"}.get(iid)

        buys = self._run_strategy_and_get_buys(df, rebalance_date, sector_index_mock=mock_sector)
        sectors = {s.instrument_id: s.extras["sector"] for s in buys}
        assert sectors.get("UP1") == "Technology" or sectors.get("UP2") == "Financials"

    def test_buy_sector_missing_returns_none(self) -> None:
        """sector() returns None → extras['sector'] is None, signal still emitted."""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(df, rebalance_date, sector_index_mock=lambda *a: None)
        assert len(buys) >= 1
        assert all(s.extras["sector"] is None for s in buys)
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_strategies.py::TestClenowMomentumStrategySignals -v -k "extras_schema or rank_monotonic or sector"
```

Expected: 全部 FAIL — extras 为空 dict。

- [ ] **Step 3: 实现 — 修改 `trendspec/strategy/examples/clenow_momentum.py`**

修改 `next()` 方法中的 BUY 循环。完整替换 `# BUY: top-ranked stocks not already held` 段落（从 `for iid in ranked[:n_keep]:` 到 `sig.shares = float(shares)`）为：

```python
        from trendspec.data.sectors import sector as sector_lookup

        # BUY: top-ranked stocks not already held
        for rank_pos, iid in enumerate(ranked[:n_keep], start=1):
            if ctx.has_position(iid):
                continue

            atr = ctx.indicator_value("ATR", iid, current_date, period=self._atr_period)
            close = get_close(iid)

            if atr is None or atr <= 0 or close is None or close <= 0:
                continue

            shares = int(nav * self._risk_factor / atr)
            if shares < 1:
                continue

            ma200 = ctx.indicator_value("MA", iid, current_date, period=self._sma_period)
            hh = ctx.indicator_value("HH", iid, current_date, period=self._drawdown_period)
            vol_avg = ctx.indicator_value(
                "SMA_VOLUME", iid, current_date, period=self._volume_avg_period
            )
            r2 = ctx.indicator_value("CLENOW_R2", iid, current_date, period=self._score_period)

            day_rows = day_data.filter(pl.col("instrument_id") == iid)
            today_vol = day_rows["volume"].item() if not day_rows.is_empty() else None

            if ma200 is None or hh is None or vol_avg is None or r2 is None:
                continue
            if vol_avg <= 0 or today_vol is None:
                continue

            deviation_pct = (close - ma200) / ma200 * 100
            drawdown_pct = (close - hh) / hh * 100
            vol_mult = float(today_vol) / float(vol_avg)
            stop_loss = close - self._atr_stop_k * atr

            alerts: list[str] = []
            if deviation_pct > self._warn_deviation_max:
                alerts.append("均线乖离过大")
            if vol_mult < self._warn_vol_mult_low:
                alerts.append("量能萎缩")
            if vol_mult > self._warn_vol_mult_high:
                alerts.append("放量过快")
            if drawdown_pct < self._warn_drawdown_max:
                alerts.append("回撤过深")

            sector_code = sector_lookup(ctx.market, iid, current_date)

            sig = ctx.signal(
                "BUY",
                iid,
                close,
                trigger_value=scores[iid],
                note=f"score={scores[iid]:.2f}, atr={atr:.2f}, shares={shares}",
            )
            sig.ticker = get_ticker(iid)
            sig.shares = float(shares)
            sig.extras = {
                "sector": sector_code,
                "rank": rank_pos,
                "r2": float(r2),
                "deviation_pct": float(deviation_pct),
                "drawdown_pct": float(drawdown_pct),
                "vol_mult": float(vol_mult),
                "stop_loss": float(stop_loss),
                "alerts": alerts,
            }
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_strategies.py::TestClenowMomentumStrategySignals -v -k "extras_schema or rank_monotonic or sector"
```

Expected: 全部 PASS。

- [ ] **Step 5: Write failing tests (stop-loss formula + alert triggers)**

在同一测试类追加：

```python
    def test_stop_loss_formula(self) -> None:
        """stop_loss == close - atr_stop_k * ATR(20)"""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(df, rebalance_date)
        for sig in buys:
            # 反推：close 与 stop_loss 已知；ATR 从 ctx 拿；k=3.0
            # 这里只能验证关系：stop_loss < close 且 (close - stop_loss) / 3 > 0
            assert sig.extras["stop_loss"] < sig.price
            implied_atr = (sig.price - sig.extras["stop_loss"]) / 3.0
            assert implied_atr > 0

    def test_alerts_normal_when_no_threshold_hit(self) -> None:
        """trending smooth df → 大概率无预警。"""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(
            df, rebalance_date,
            params_override={
                # 把阈值放宽到不可触发
                "warn_deviation_max": 9999.0,
                "warn_vol_mult_low": 0.0,
                "warn_vol_mult_high": 9999.0,
                "warn_drawdown_max": -9999.0,
            },
        )
        assert all(s.extras["alerts"] == [] for s in buys)

    def test_alerts_deviation_trigger(self) -> None:
        """warn_deviation_max=0 → 任何正向乖离都触发。"""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(
            df, rebalance_date,
            params_override={
                "warn_deviation_max": 0.01,
                "warn_vol_mult_low": -1.0,    # 不触发
                "warn_vol_mult_high": 9999.0, # 不触发
                "warn_drawdown_max": -9999.0, # 不触发
            },
        )
        assert any("均线乖离过大" in s.extras["alerts"] for s in buys)

    def test_alerts_vol_low_trigger(self) -> None:
        """warn_vol_mult_low 极高 → 一定触发量能萎缩。"""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(
            df, rebalance_date,
            params_override={
                "warn_deviation_max": 9999.0,
                "warn_vol_mult_low": 9999.0,
                "warn_vol_mult_high": 99999.0,
                "warn_drawdown_max": -9999.0,
            },
        )
        assert all("量能萎缩" in s.extras["alerts"] for s in buys)

    def test_alerts_vol_high_trigger(self) -> None:
        """warn_vol_mult_high 极小 → 一定触发放量过快。"""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(
            df, rebalance_date,
            params_override={
                "warn_deviation_max": 9999.0,
                "warn_vol_mult_low": -1.0,
                "warn_vol_mult_high": 0.001,
                "warn_drawdown_max": -9999.0,
            },
        )
        assert all("放量过快" in s.extras["alerts"] for s in buys)

    def test_alerts_drawdown_trigger(self) -> None:
        """warn_drawdown_max 接近 0（如 -0.001）→ 几乎任何回撤都触发。"""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(
            df, rebalance_date,
            params_override={
                "warn_deviation_max": 9999.0,
                "warn_vol_mult_low": -1.0,
                "warn_vol_mult_high": 9999.0,
                "warn_drawdown_max": -0.001,
            },
        )
        # 至少有一只股票 drawdown_pct < -0.001
        has_dd_alert = any("回撤过深" in s.extras["alerts"] for s in buys)
        assert has_dd_alert
```

- [ ] **Step 6: Run tests to verify they pass**

```
uv run pytest tests/test_strategies.py::TestClenowMomentumStrategySignals -v
```

Expected: 全部 PASS（包括之前的用例）。

- [ ] **Step 7: Commit**

```
git add trendspec/strategy/examples/clenow_momentum.py tests/test_strategies.py
git commit -m "feat(clenow): populate extras with rank/sector/r2/stop_loss/alerts on BUY"
```

---

## Task 8: ScreeningReport CSV 文件名包含 strategy（通用修复）

**Files:**
- Modify: `trendspec/screening/report.py`
- Create: `tests/test_screening_report.py`

- [ ] **Step 1: Write failing test**

新建 `tests/test_screening_report.py`：

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_screening_report.py::TestCSVFilename -v
```

Expected: FAIL — 现有文件名 `signals_20260518.csv` 不含 strategy 名。

- [ ] **Step 3: 实现 — 修改 `trendspec/screening/report.py`**

在 `export()` 方法中，找到这两行：

```python
        date_str = self.screening_date.strftime("%Y%m%d")
        signals_path = output_path / f"signals_{date_str}.csv"
```

替换为：

```python
        date_str = self.screening_date.strftime("%Y%m%d")
        signals_path = output_path / f"signals_{self.strategy_name}_{date_str}.csv"
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_screening_report.py::TestCSVFilename -v
```

Expected: PASS。

- [ ] **Step 5: 回归检查**

```
uv run pytest tests/ -v --ignore=tests/test_screening_report.py 2>&1 | tail -20
```

Expected: 无新增失败。若 `test_screening_engine.py` 中有写死文件名的断言，更新断言匹配新格式。

- [ ] **Step 6: Commit**

```
git add trendspec/screening/report.py tests/test_screening_report.py
git commit -m "fix(screening): include strategy name in CSV filename"
```

---

## Task 9: ScreeningReport clenow_momentum 10 列渲染

**Files:**
- Modify: `trendspec/screening/report.py`
- Modify: `tests/test_screening_report.py`

- [ ] **Step 1: Write failing test**

在 `tests/test_screening_report.py` 末尾追加：

```python
class TestClenowBuyTableRendering:
    def _clenow_signal(self, ticker: str, price: float, **extras_override) -> Signal:
        extras = {
            "sector": "Technology",
            "rank": 1,
            "r2": 0.85,
            "deviation_pct": 32.5,
            "drawdown_pct": -2.1,
            "vol_mult": 1.5,
            "stop_loss": price * 0.77,
            "alerts": [],
        }
        extras.update(extras_override)
        return _buy_signal(ticker, price, extras)

    def test_clenow_buy_table_has_10_columns(self) -> None:
        signals = [self._clenow_signal("LITE", 1001.81)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        table = report._create_signals_table(signals, "买入信号")
        assert len(table.columns) == 10
        col_headers = [c.header for c in table.columns]
        assert col_headers == [
            "股票代码", "行业", "选股排名", "建议买入价", "初始止损线",
            "趋势质量 (R²)", "乖离率 (距 MA200)", "回撤 (距 63 日高点)",
            "放量倍数", "备注/预警",
        ]

    def test_non_clenow_buy_table_keeps_6_columns(self) -> None:
        signals = [_buy_signal("AAPL", 100.0)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="ma_cross",
            market="us",
        )
        table = report._create_signals_table(signals, "买入信号")
        assert len(table.columns) == 6

    def test_clenow_sell_table_uses_default_6_columns(self) -> None:
        """SELL signals 仍走原 6 列路径，即使 strategy_name 为 clenow_momentum。"""
        sell = Signal(
            direction="SELL", ticker="LITE", instrument_id="LITE",
            price=900.0, note="below SMA200",
        )
        report = ScreeningReport(
            signals=[sell],
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        table = report._create_signals_table([sell], "卖出信号")
        assert len(table.columns) == 6

    def test_clenow_sector_none_renders_dash(self) -> None:
        signals = [self._clenow_signal("LITE", 1000.0, sector=None)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        # 通过 _row_for_clenow_buy 辅助方法验证（实现时同步添加），或直接渲染检查
        rows = list(report._iter_clenow_buy_rows(signals))
        assert rows[0][1] == "-"  # 行业列

    def test_clenow_alerts_renders_with_prefix(self) -> None:
        signals = [self._clenow_signal(
            "CIEN", 591.57,
            alerts=["均线乖离过大", "量能萎缩"],
        )]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        rows = list(report._iter_clenow_buy_rows(signals))
        assert rows[0][9] == "[警报] 均线乖离过大，量能萎缩"

    def test_clenow_no_alerts_renders_normal(self) -> None:
        signals = [self._clenow_signal("LITE", 1001.81, alerts=[])]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        rows = list(report._iter_clenow_buy_rows(signals))
        assert rows[0][9] == "正常"

    def test_clenow_r2_label_buckets(self) -> None:
        report = ScreeningReport(
            signals=[], screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum", market="us",
        )
        for r2, label in [(0.90, "极平稳"), (0.80, "优秀"), (0.70, "良好"), (0.50, "一般")]:
            sig = self._clenow_signal("X", 100.0, r2=r2)
            rows = list(report._iter_clenow_buy_rows([sig]))
            assert label in rows[0][5]
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_screening_report.py::TestClenowBuyTableRendering -v
```

Expected: 多个用例 FAIL — `_iter_clenow_buy_rows` 不存在；clenow 表仍 6 列。

- [ ] **Step 3: 实现 — 修改 `trendspec/screening/report.py`**

3a. 修改 `_create_signals_table()`：在方法开头加入分流：

```python
    def _create_signals_table(self, signals: list[Any], title: str) -> Table:
        """Create signals table with Chinese column names."""
        if title == "买入信号" and self.strategy_name == "clenow_momentum":
            return self._create_clenow_buy_table(signals)
        # 原 6 列布局
        table = Table(title=title, show_header=True, header_style="bold green")
        table.add_column("股票代码", style="cyan")
        table.add_column("日期", style="cyan")
        table.add_column("方向", style="yellow")
        table.add_column("价格", style="green")
        table.add_column("触发指标值", style="blue")
        table.add_column("备注", style="white")

        for signal in signals:
            table.add_row(
                signal.ticker,
                self.screening_date.isoformat(),
                signal.direction,
                f"{signal.price:.2f}",
                f"{signal.trigger_value:.2f}" if signal.trigger_value else "N/A",
                signal.note or "",
            )

        return table
```

3b. 在类末尾追加新方法：

```python
    @staticmethod
    def _r2_label(r2: float) -> str:
        if r2 >= 0.85:
            return "极平稳"
        if r2 >= 0.75:
            return "优秀"
        if r2 >= 0.65:
            return "良好"
        return "一般"

    def _iter_clenow_buy_rows(self, signals: list[Any]):
        """Yield formatted row tuples (10 items) for clenow BUY signals."""
        for s in signals:
            e = s.extras or {}
            sector = e.get("sector") or "-"
            rank = e.get("rank")
            r2 = e.get("r2", 0.0)
            deviation = e.get("deviation_pct", 0.0)
            drawdown = e.get("drawdown_pct", 0.0)
            vol_mult = e.get("vol_mult", 0.0)
            stop_loss = e.get("stop_loss", 0.0)
            alerts = e.get("alerts") or []
            note = "[警报] " + "，".join(alerts) if alerts else "正常"
            yield (
                s.ticker,
                sector,
                f"#{rank}" if rank is not None else "-",
                f"${s.price:.2f}",
                f"${stop_loss:.2f}",
                f"{r2:.2f} ({self._r2_label(r2)})",
                f"{deviation:+.1f}%",
                f"{drawdown:+.1f}%",
                f"{vol_mult:.1f}x",
                note,
            )

    def _create_clenow_buy_table(self, signals: list[Any]) -> Table:
        table = Table(title="买入信号", show_header=True, header_style="bold green")
        table.add_column("股票代码", style="cyan")
        table.add_column("行业", style="cyan")
        table.add_column("选股排名", style="magenta")
        table.add_column("建议买入价", style="green")
        table.add_column("初始止损线", style="red")
        table.add_column("趋势质量 (R²)", style="blue")
        table.add_column("乖离率 (距 MA200)", style="yellow")
        table.add_column("回撤 (距 63 日高点)", style="yellow")
        table.add_column("放量倍数", style="blue")
        table.add_column("备注/预警", style="white")

        for row, s in zip(self._iter_clenow_buy_rows(signals), signals):
            alerts = (s.extras or {}).get("alerts") or []
            style = "red" if alerts else "white"
            table.add_row(*row, style=style)
        return table
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_screening_report.py::TestClenowBuyTableRendering -v
```

Expected: 全部 PASS。

- [ ] **Step 5: 完整回归**

```
uv run pytest tests/test_screening_report.py tests/test_strategies.py tests/test_strategy.py -v 2>&1 | tail -30
```

Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```
git add trendspec/screening/report.py tests/test_screening_report.py
git commit -m "feat(report): render clenow_momentum BUY as 10-column decision table"
```

---

## Task 10: ScreeningReport clenow_momentum CSV 13 列 schema

**Files:**
- Modify: `trendspec/screening/report.py`
- Modify: `tests/test_screening_report.py`

- [ ] **Step 1: Write failing test**

在 `tests/test_screening_report.py` 末尾追加：

```python
class TestClenowCSVSchema:
    def _clenow_signal(self, ticker: str, price: float, **extras_override) -> Signal:
        extras = {
            "sector": "Technology", "rank": 1, "r2": 0.85,
            "deviation_pct": 32.5, "drawdown_pct": -2.1, "vol_mult": 1.5,
            "stop_loss": price * 0.77, "alerts": [],
        }
        extras.update(extras_override)
        return _buy_signal(ticker, price, extras)

    def test_clenow_csv_has_13_columns(self, tmp_path: Path) -> None:
        signals = [self._clenow_signal("LITE", 1001.81)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        out = report.export(tmp_path)
        df = pl.read_csv(out)
        assert df.columns == [
            "股票代码", "instrument_id", "日期", "方向", "行业",
            "选股排名", "建议买入价", "初始止损线", "趋势质量 (R²)",
            "乖离率 (距 MA200)", "回撤 (距 63 日高点)", "放量倍数", "备注/预警",
        ]

    def test_clenow_csv_buy_row_fully_populated(self, tmp_path: Path) -> None:
        signals = [self._clenow_signal("CIEN", 591.57, alerts=["量能萎缩"])]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        out = report.export(tmp_path)
        df = pl.read_csv(out)
        row = df.row(0, named=True)
        assert row["股票代码"] == "CIEN"
        assert row["方向"] == "BUY"
        assert row["行业"] == "Technology"
        assert row["选股排名"] == 1
        assert row["建议买入价"] == pytest.approx(591.57)
        assert "量能萎缩" in str(row["备注/预警"])

    def test_clenow_csv_sell_row_blanks_display_cols(self, tmp_path: Path) -> None:
        buy = self._clenow_signal("LITE", 1001.81)
        sell = Signal(
            direction="SELL", ticker="OLD", instrument_id="OLD",
            price=50.0, note="below SMA200",
        )
        report = ScreeningReport(
            signals=[buy, sell],
            screening_date=date(2026, 5, 18),
            strategy_name="clenow_momentum",
            market="us",
        )
        out = report.export(tmp_path)
        df = pl.read_csv(out)
        sell_row = df.filter(pl.col("方向") == "SELL").row(0, named=True)
        assert sell_row["股票代码"] == "OLD"
        assert sell_row["建议买入价"] == pytest.approx(50.0)
        assert sell_row["备注/预警"] == "below SMA200"
        for col in ["行业", "选股排名", "初始止损线", "趋势质量 (R²)",
                    "乖离率 (距 MA200)", "回撤 (距 63 日高点)", "放量倍数"]:
            # CSV 空字符串读回会变 None / empty
            v = sell_row[col]
            assert v is None or v == "" or v == 0

    def test_non_clenow_csv_keeps_7_columns(self, tmp_path: Path) -> None:
        signals = [_buy_signal("AAPL", 100.0)]
        report = ScreeningReport(
            signals=signals,
            screening_date=date(2026, 5, 18),
            strategy_name="ma_cross",
            market="us",
        )
        out = report.export(tmp_path)
        df = pl.read_csv(out)
        assert df.columns == [
            "股票代码", "instrument_id", "日期", "方向",
            "价格", "触发指标值", "备注",
        ]
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_screening_report.py::TestClenowCSVSchema -v
```

Expected: FAIL — 当前 CSV schema 不分策略。

- [ ] **Step 3: 实现 — 修改 `trendspec/screening/report.py`**

替换 `_signals_to_dataframe()` 方法：

```python
    def _signals_to_dataframe(self) -> pl.DataFrame:
        """Convert signals to Polars DataFrame, schema varies by strategy."""
        if not self.signals:
            return pl.DataFrame()

        if self.strategy_name == "clenow_momentum":
            return self._signals_to_clenow_dataframe()

        # 原 7 列 schema
        records = []
        for signal in self.signals:
            records.append({
                "股票代码": signal.ticker,
                "instrument_id": signal.instrument_id,
                "日期": self.screening_date.isoformat(),
                "方向": signal.direction,
                "价格": signal.price,
                "触发指标值": signal.trigger_value,
                "备注": signal.note or "",
            })
        return pl.DataFrame(records)

    def _signals_to_clenow_dataframe(self) -> pl.DataFrame:
        """13-column schema: BUY rows fully populated, SELL rows blank display cols."""
        records = []
        for s in self.signals:
            if s.is_buy():
                e = s.extras or {}
                alerts = e.get("alerts") or []
                note = "[警报] " + "，".join(alerts) if alerts else "正常"
                records.append({
                    "股票代码": s.ticker,
                    "instrument_id": s.instrument_id,
                    "日期": self.screening_date.isoformat(),
                    "方向": "BUY",
                    "行业": e.get("sector") or "",
                    "选股排名": e.get("rank"),
                    "建议买入价": s.price,
                    "初始止损线": e.get("stop_loss"),
                    "趋势质量 (R²)": f"{e.get('r2', 0.0):.4f}",
                    "乖离率 (距 MA200)": f"{e.get('deviation_pct', 0.0):.2f}",
                    "回撤 (距 63 日高点)": f"{e.get('drawdown_pct', 0.0):.2f}",
                    "放量倍数": f"{e.get('vol_mult', 0.0):.4f}",
                    "备注/预警": note,
                })
            else:
                records.append({
                    "股票代码": s.ticker,
                    "instrument_id": s.instrument_id,
                    "日期": self.screening_date.isoformat(),
                    "方向": "SELL",
                    "行业": "",
                    "选股排名": None,
                    "建议买入价": s.price,
                    "初始止损线": None,
                    "趋势质量 (R²)": "",
                    "乖离率 (距 MA200)": "",
                    "回撤 (距 63 日高点)": "",
                    "放量倍数": "",
                    "备注/预警": s.note or "",
                })
        return pl.DataFrame(records)
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_screening_report.py::TestClenowCSVSchema -v
```

Expected: 全部 PASS。

- [ ] **Step 5: 完整回归**

```
uv run pytest tests/ -v 2>&1 | tail -30
```

Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```
git add trendspec/screening/report.py tests/test_screening_report.py
git commit -m "feat(report): 13-column CSV schema for clenow_momentum signals"
```

---

## Task 11: 端到端验证

**Files:** 无修改，仅人工验收。

- [ ] **Step 1: 跑全量测试**

```
uv run pytest tests/ 2>&1 | tail -10
```

Expected: 全部 PASS，无新增 warning。

- [ ] **Step 2: 跑 ruff / 类型检查**

```
uv run ruff check trendspec/strategy/signal.py trendspec/strategy/indicators.py trendspec/strategy/examples/clenow_momentum.py trendspec/screening/report.py
```

Expected: 无新增告警。

- [ ] **Step 3: 实跑 clenow_momentum screen 命令**

```
uv run trendspec screen run --strategy clenow_momentum --market us --date 2026-05-15
```

Expected：
- 终端输出 10 列 BUY 表（股票代码、行业、选股排名、建议买入价、初始止损线、趋势质量 (R²)、乖离率、回撤、放量倍数、备注/预警）
- CSV 文件保存为 `signals_clenow_momentum_20260515.csv`

- [ ] **Step 4: 实跑其他策略回归**

```
uv run trendspec screen run --strategy ma_cross --market cn --date 2026-05-15
```

Expected：
- 终端输出原 6 列 BUY 表
- CSV 文件名为 `signals_ma_cross_20260515.csv`（含策略名 = 通用修复生效）
- CSV 内部 schema 为原 7 列

- [ ] **Step 5: 最终 commit（如需更新 CHANGELOG / 文档）**

如本项目维护 CHANGELOG，追加一行：

```
- feat: clenow_momentum screen output expanded to 10 columns (sector/rank/stop/R²/deviation/drawdown/volume/alerts)
- fix: CSV filename now includes strategy name (signals_<strategy>_<date>.csv)
```

否则跳过此步。

---

## Spec 覆盖核对

| Spec 章节 | 实现任务 |
|----------|---------|
| §3 Architecture / 文件改动总览 | Task 1-10 |
| §4 字段计算公式 | Task 7 (核心逻辑) |
| §4.1 R² 独立 indicator | Task 4 |
| §4.2 R² 质量分档 | Task 9 (`_r2_label`) |
| §4.3 sector 数据源 | Task 7 (sector_lookup 调用) |
| §5.1 Signal extras | Task 1 |
| §5.2 extras 8 键结构 | Task 7 |
| §6.1 6 个新参数 | Task 5 |
| §6.2 参数校验 | Task 5 |
| §6.3 4 条预警规则 | Task 7 (Step 5-6 测试) |
| §7.1 渲染分流 | Task 9 |
| §7.2 10 列终端布局 | Task 9 |
| §7.3 CSV 13 列 + 文件名 | Task 8 (文件名) + Task 10 (schema) |
| §8 错误处理 | Task 7 (sector None, indicator None skip) |
| §9 测试矩阵 | Task 1-10 嵌入式 TDD |
| §10 验证清单 | Task 11 |
