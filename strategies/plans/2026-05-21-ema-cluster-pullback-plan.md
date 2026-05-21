# EMA Cluster Pullback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `ema_cluster_pullback` 策略：日 EMA20/60/120 密集缠绕 + 周线股价回踩 EMA20 + 趋势仍向上多头排列，输出 BUY/SELL 信号；同时扩展框架以支持周线数据。

**Architecture:** 群辉 MariaDB `weekly_prices` → 新增 weekly ingestor → `data_lake/<market>/weekly/...` Parquet → `bars()` 支持 `frequency="weekly"` → BaseEngine 同时加载日/周 → `StrategyContext` 新增 weekly indicator API（含日→已完成周映射，防 lookahead） → 新策略 `ema_cluster_pullback`。

**Tech Stack:** Python 3.13 / polars / typer / SQLAlchemy / pydantic-settings / pytest

**Spec:** `strategies/specs/2026-05-21-ema-cluster-pullback-design.md`

---

## File Structure

**新建：**
- `tests/test_weekly_ingestor.py` — 周线 ingestor 测试
- `tests/test_weekly_loader.py` — `bars(frequency="weekly")` 测试
- `tests/test_weekly_context.py` — `StrategyContext` 周线 API 测试
- `tests/strategy/test_ema_cluster_pullback.py` — 策略测试（新建 `tests/strategy/` 目录）
- `tests/strategy/__init__.py` — 空文件
- `trendspec/strategy/examples/ema_cluster_pullback.py` — 策略本体

**修改：**
- `trendspec/ingest/stocks_db_ingestor.py` — 新增 `ingest_us_weekly` + `ingest_cn_weekly`
- `trendspec/cli/ingest_cmd.py` — 新增 `weekly` 子命令
- `trendspec/data/parquet_loader.py` — `bars()` 加 `frequency` 参数
- `trendspec/engine/base_engine.py` — `load_data()` 同时加载周线注入 context
- `trendspec/strategy/context.py` — 新增 weekly indicator API

---

## Task 1: Weekly ingestor — US

**Files:**
- Test: `tests/test_weekly_ingestor.py`
- Modify: `trendspec/ingest/stocks_db_ingestor.py` （文件尾追加）

- [ ] **Step 1.1: 写失败测试**

新建 `tests/test_weekly_ingestor.py`：

```python
"""Tests for weekly ingestor."""
import tempfile
import polars as pl
import pytest
from sqlalchemy import create_engine, text


@pytest.fixture
def stocks_db_with_weekly():
    """SQLite mock of Synology DB with weekly_prices table."""
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE prices (ticker TEXT, date DATE, open REAL, high REAL,
                                  low REAL, close REAL, volume INTEGER)
        """))
        conn.execute(text("""
            CREATE TABLE weekly_prices (ticker TEXT, date DATE, open REAL, high REAL,
                                         low REAL, close REAL, volume INTEGER)
        """))
        conn.execute(text("""
            CREATE TABLE stocks (ticker TEXT PRIMARY KEY, exchange TEXT,
                                  gics_sector TEXT, gics_industry TEXT, is_active INTEGER)
        """))
        conn.execute(text("""
            CREATE TABLE index_constituents (index_id TEXT, snapshot_date DATE, ticker TEXT)
        """))
        conn.execute(text("""
            INSERT INTO index_constituents VALUES
            ('SP500', '2024-01-01', 'AAPL'),
            ('SP500', '2024-01-01', 'MSFT')
        """))
        conn.execute(text("""
            INSERT INTO stocks VALUES
            ('AAPL', 'NYSE', 'Tech', 'Hardware', 1),
            ('MSFT', 'Nasdaq', 'Tech', 'Software', 1),
            ('600000', 'SSE', 'Financials', 'Banks', 1),
            ('000001', 'SZSE', 'Financials', 'Banks', 1)
        """))
        conn.execute(text("""
            INSERT INTO weekly_prices VALUES
            ('AAPL',   '2024-01-05', 180.0, 188.0, 179.0, 187.0, 250000000),
            ('AAPL',   '2024-01-12', 187.0, 192.0, 185.0, 190.0, 260000000),
            ('MSFT',   '2024-01-05', 365.0, 375.0, 364.0, 373.0, 100000000),
            ('600000', '2024-01-05', 7.0,   7.3,   6.9,   7.2,   50000000),
            ('000001', '2024-01-05', 10.0,  10.5,  9.9,   10.3,  80000000)
        """))
        conn.commit()
    yield engine


@pytest.fixture
def temp_root():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_ingest_us_weekly_writes_parquet(stocks_db_with_weekly, temp_root):
    """US weekly Parquet has correct schema and data."""
    from trendspec.data.markets import Market
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.stocks_db_ingestor import ingest_us_weekly

    manifest = Manifest(Market.US, temp_root)
    result = ingest_us_weekly(stocks_db_with_weekly, manifest, temp_root, full_sync=True)

    assert result["row_count"] == 3   # 2 AAPL + 1 MSFT
    assert result["instrument_count"] == 2

    from trendspec.data.parquet_loader import scan_parquet
    lf = scan_parquet(temp_root, Market.US, "weekly")
    df = lf.collect()
    assert set(df.columns) >= {"instrument_id", "date", "open", "high", "low",
                                "close", "volume", "adj_factor"}
    assert df["adj_factor"].unique().to_list() == [1.0]
    assert sorted(df["instrument_id"].unique().to_list()) == ["AAPL", "MSFT"]
```

- [ ] **Step 1.2: 运行测试验证失败**

```bash
uv run pytest tests/test_weekly_ingestor.py::test_ingest_us_weekly_writes_parquet -v
```
Expected: `FAILED` — `ImportError: cannot import name 'ingest_us_weekly'`

- [ ] **Step 1.3: 实现 `ingest_us_weekly`**

在 `trendspec/ingest/stocks_db_ingestor.py` 文件末尾追加：

```python
# =============================================================================
# US Weekly
# =============================================================================


def ingest_us_weekly(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """Ingest US weekly OHLCV from weekly_prices + index_constituents.

    Schema mirrors prices: (ticker, date, open, high, low, close, volume).
    date corresponds to each week's closing day per the Synology table convention.
    """
    last_date = "1970-01-01" if full_sync else _get_last_synced_date(manifest, "weekly")

    sql = text("""
        SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, p.volume
        FROM weekly_prices p
        JOIN (
            SELECT DISTINCT ticker FROM index_constituents
            WHERE index_id IN ('SP500', 'RUSSELL1000')
        ) AS us ON p.ticker = us.ticker
        WHERE p.date > :last_date
        ORDER BY p.date, p.ticker
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, {"last_date": last_date}).fetchall()

    if not rows:
        return {"row_count": 0, "date_range": ("", ""), "instrument_count": 0}

    df = pl.DataFrame(
        rows,
        schema=["ticker", "date", "open", "high", "low", "close", "volume"],
        orient="row",
    )
    df = df.with_columns(pl.col("date").cast(pl.Date))
    df = df.with_columns([
        pl.col("ticker").alias("instrument_id"),
        pl.lit(1.0).alias("adj_factor"),
    ])

    write_parquet(df, Market.US, "weekly", root, overwrite=full_sync)

    dates = df["date"].cast(pl.Utf8)
    date_range = (dates.min(), dates.max())
    instrument_count = df["instrument_id"].n_unique()
    row_count = len(df)

    manifest.update_dataset_state("weekly", row_count, date_range, instrument_count)

    return {"row_count": row_count, "date_range": date_range, "instrument_count": instrument_count}
```

- [ ] **Step 1.4: 运行测试验证通过**

```bash
uv run pytest tests/test_weekly_ingestor.py::test_ingest_us_weekly_writes_parquet -v
```
Expected: `PASSED`

- [ ] **Step 1.5: Commit**

```bash
git add tests/test_weekly_ingestor.py trendspec/ingest/stocks_db_ingestor.py
git commit -m "feat: 新增 ingest_us_weekly 从 weekly_prices 表写入 data_lake/<market>/weekly/"
```

---

## Task 2: Weekly ingestor — CN

**Files:**
- Test: `tests/test_weekly_ingestor.py`（追加）
- Modify: `trendspec/ingest/stocks_db_ingestor.py`（追加）

- [ ] **Step 2.1: 写失败测试**

在 `tests/test_weekly_ingestor.py` 追加：

```python
def test_ingest_cn_weekly_derives_instrument_id(stocks_db_with_weekly, temp_root):
    """CN weekly produces SH/SZ-prefixed instrument_id."""
    from trendspec.data.markets import Market
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_weekly

    manifest = Manifest(Market.CN, temp_root)
    result = ingest_cn_weekly(stocks_db_with_weekly, manifest, temp_root, full_sync=True)

    assert result["row_count"] == 2
    assert result["instrument_count"] == 2

    from trendspec.data.parquet_loader import scan_parquet
    df = scan_parquet(temp_root, Market.CN, "weekly").collect()
    assert sorted(df["instrument_id"].unique().to_list()) == ["SH600000", "SZ000001"]
```

- [ ] **Step 2.2: 运行验证失败**

```bash
uv run pytest tests/test_weekly_ingestor.py::test_ingest_cn_weekly_derives_instrument_id -v
```
Expected: `FAILED` — `ImportError`

- [ ] **Step 2.3: 实现 `ingest_cn_weekly`**

在 `trendspec/ingest/stocks_db_ingestor.py` 末尾追加：

```python
# =============================================================================
# CN Weekly
# =============================================================================


def ingest_cn_weekly(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """Ingest CN weekly OHLCV from weekly_prices joined with stocks.

    instrument_id = SH{ticker} for SSE/SH, SZ{ticker} for SZSE/SZ.
    adj_factor = 1.0 (assume Tushare backward-adjusted, same as daily).
    """
    last_date = "1970-01-01" if full_sync else _get_last_synced_date(manifest, "weekly")

    ex_placeholders = _exchange_placeholder(_CN_EXCHANGES)
    ex_params = _exchange_params(_CN_EXCHANGES)
    ex_params["last_date"] = last_date

    sql = text(f"""
        SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, p.volume,
               s.exchange
        FROM weekly_prices p
        JOIN stocks s ON p.ticker = s.ticker
        WHERE s.exchange IN ({ex_placeholders})
          AND p.date > :last_date
        ORDER BY p.date, p.ticker
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, ex_params).fetchall()

    if not rows:
        return {"row_count": 0, "date_range": ("", ""), "instrument_count": 0}

    df = pl.DataFrame(
        rows,
        schema=["ticker", "date", "open", "high", "low", "close", "volume", "exchange"],
        orient="row",
    )
    df = df.with_columns(pl.col("date").cast(pl.Date))
    df = df.with_columns(
        pl.struct(["ticker", "exchange"])
        .map_elements(
            lambda s: _derive_cn_instrument_id(s["ticker"], s["exchange"]),
            return_dtype=pl.Utf8,
        )
        .alias("instrument_id")
    )
    df = df.with_columns(pl.lit(1.0).alias("adj_factor"))
    df = df.drop("exchange")

    write_parquet(df, Market.CN, "weekly", root, overwrite=full_sync)

    dates = df["date"].cast(pl.Utf8)
    date_range = (dates.min(), dates.max())
    instrument_count = df["instrument_id"].n_unique()
    row_count = len(df)

    manifest.update_dataset_state("weekly", row_count, date_range, instrument_count)

    return {"row_count": row_count, "date_range": date_range, "instrument_count": instrument_count}
```

- [ ] **Step 2.4: 运行验证通过**

```bash
uv run pytest tests/test_weekly_ingestor.py -v
```
Expected: 两个测试均 `PASSED`

- [ ] **Step 2.5: Commit**

```bash
git add tests/test_weekly_ingestor.py trendspec/ingest/stocks_db_ingestor.py
git commit -m "feat: 新增 ingest_cn_weekly 推导 SH/SZ instrument_id 并写入 weekly Parquet"
```

---

## Task 3: `bars()` 支持 `frequency` 参数

**Files:**
- Test: `tests/test_weekly_loader.py`
- Modify: `trendspec/data/parquet_loader.py:156-214`

- [ ] **Step 3.1: 写失败测试**

新建 `tests/test_weekly_loader.py`：

```python
"""Tests for bars() frequency parameter."""
import tempfile
from datetime import date

import polars as pl
import pytest
from sqlalchemy import create_engine, text


@pytest.fixture
def lake_with_weekly():
    """Build a data_lake with weekly Parquet via the weekly ingestor."""
    from trendspec.data.markets import Market
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.stocks_db_ingestor import ingest_us_weekly

    with tempfile.TemporaryDirectory() as d:
        engine = create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE weekly_prices (ticker TEXT, date DATE, open REAL, high REAL, low REAL, close REAL, volume INTEGER)"))
            conn.execute(text("CREATE TABLE index_constituents (index_id TEXT, snapshot_date DATE, ticker TEXT)"))
            conn.execute(text("INSERT INTO index_constituents VALUES ('SP500', '2024-01-01', 'AAPL')"))
            conn.execute(text("""
                INSERT INTO weekly_prices VALUES
                ('AAPL', '2024-01-05', 180.0, 188.0, 179.0, 187.0, 250000000),
                ('AAPL', '2024-01-12', 187.0, 192.0, 185.0, 190.0, 260000000)
            """))
            conn.commit()
        manifest = Manifest(Market.US, d)
        ingest_us_weekly(engine, manifest, d, full_sync=True)
        yield d


def test_bars_loads_weekly_frequency(lake_with_weekly):
    """bars(frequency='weekly') returns weekly Parquet data."""
    from trendspec.data.markets import Market
    from trendspec.data.parquet_loader import bars

    df = bars(market=Market.US, frequency="weekly", root=lake_with_weekly)
    assert len(df) == 2
    assert df["instrument_id"].unique().to_list() == ["AAPL"]
    assert sorted(df["date"].to_list()) == [date(2024, 1, 5), date(2024, 1, 12)]


def test_bars_defaults_to_daily(lake_with_weekly):
    """bars() without frequency arg still loads daily (backward compat)."""
    from trendspec.data.markets import Market
    from trendspec.data.parquet_loader import bars

    # No daily data → empty DataFrame
    df = bars(market=Market.US, root=lake_with_weekly)
    assert df.is_empty()
```

- [ ] **Step 3.2: 运行验证失败**

```bash
uv run pytest tests/test_weekly_loader.py -v
```
Expected: `FAILED` — `bars() got unexpected keyword 'frequency'`

- [ ] **Step 3.3: 修改 `bars()` 签名**

替换 `trendspec/data/parquet_loader.py` 中 `bars` 函数的签名与首行：

```python
def bars(
    market: Market,
    start_date: date | None = None,
    end_date: date | None = None,
    instrument_ids: list[str] | None = None,
    columns: list[str] | None = None,
    adjustment_mode: AdjustmentMode = "forward",
    root: str | None = None,
    frequency: Literal["daily", "weekly"] = "daily",
) -> pl.DataFrame:
    """
    Get OHLCV bars for a market with optional date range and adjustment.

    Args:
        ...（保留原 docstring，追加：）
        frequency: 'daily' (default) or 'weekly'

    Returns:
        Polars DataFrame with OHLCV data
    """
    if adjustment_mode not in ADJUSTMENT_MODES:
        raise ValueError(...)

    lf = scan_parquet(root, market, frequency)   # ← 替换硬编码 "daily"
    ...（其余逻辑不变）
```

并在文件顶部 `from typing import ...` 处加 `Literal`（如未导入）：

```python
from typing import Literal
```

- [ ] **Step 3.4: 运行验证通过**

```bash
uv run pytest tests/test_weekly_loader.py -v
```
Expected: 两个测试 `PASSED`

```bash
uv run pytest tests/test_data_loader.py -v
```
Expected: 现有测试仍 `PASSED`（向后兼容）

- [ ] **Step 3.5: Commit**

```bash
git add tests/test_weekly_loader.py trendspec/data/parquet_loader.py
git commit -m "feat: bars() 新增 frequency 参数支持 weekly 加载, 默认 daily 向后兼容"
```

---

## Task 4: CLI `ingest weekly` 子命令

**Files:**
- Modify: `trendspec/cli/ingest_cmd.py`

- [ ] **Step 4.1: 直接追加 CLI 命令（CLI 层无需单测，run smoke test 即可）**

在 `trendspec/cli/ingest_cmd.py` 文件中、`@app.command("daily")` 之后追加：

```python
@app.command("weekly")
def ingest_weekly(
    market: str = typer.Option("us", "--market", help="市场代码 (cn, us)"),
    incremental: bool = typer.Option(
        True, "--incremental/--full",
        help="增量同步 (默认) 或全量同步",
    ),
) -> None:
    """
    从群辉 stocks DB 导入 OHLCV 周线数据.

    示例:
        trendspec ingest weekly --market us
        trendspec ingest weekly --market cn --full
    """
    from trendspec.data.markets import Market
    from trendspec.config.settings import get_settings
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.mariadb_client import create_engine_from_settings
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_weekly, ingest_us_weekly

    market_enum = Market(market.upper())
    settings = get_settings()
    engine = create_engine_from_settings(settings.db)
    root = settings.data_lake.data_lake_root
    manifest = Manifest(market_enum, root)
    full_sync = not incremental

    console.print(f"[cyan]导入 {market} 周线数据...[/cyan]")
    try:
        if market_enum == Market.US:
            result = ingest_us_weekly(engine, manifest, root, full_sync=full_sync)
        elif market_enum == Market.CN:
            result = ingest_cn_weekly(engine, manifest, root, full_sync=full_sync)
        else:
            console.print(f"[red]不支持的市场: {market}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]完成: {result['row_count']} 行, {result['instrument_count']} 只股票[/green]")
    except Exception as e:
        console.print(f"[red]导入失败: {e}[/red]")
        raise typer.Exit(1)
```

- [ ] **Step 4.2: 验证命令可注册**

```bash
uv run trendspec ingest --help
```
Expected 输出包含 `weekly  从群辉 stocks DB 导入 OHLCV 周线数据.`

```bash
uv run trendspec ingest weekly --help
```
Expected 显示 `--market` / `--incremental/--full` 选项

- [ ] **Step 4.3: Commit**

```bash
git add trendspec/cli/ingest_cmd.py
git commit -m "feat(cli): 新增 trendspec ingest weekly --market <us|cn> [--full] 子命令"
```

---

## Task 5: `StrategyContext` 周线 indicator API

**Files:**
- Test: `tests/test_weekly_context.py`
- Modify: `trendspec/strategy/context.py`

- [ ] **Step 5.1: 写失败测试**

新建 `tests/test_weekly_context.py`：

```python
"""Tests for StrategyContext weekly indicator API."""
from datetime import date

import polars as pl
import pytest


@pytest.fixture
def weekly_df():
    """Hand-crafted weekly DataFrame: AAPL with 5 consecutive weekly bars."""
    return pl.DataFrame({
        "instrument_id": ["AAPL"] * 5,
        "date": [date(2024, 1, 5), date(2024, 1, 12), date(2024, 1, 19),
                  date(2024, 1, 26), date(2024, 2, 2)],
        "open":   [180.0, 187.0, 190.0, 192.0, 195.0],
        "high":   [188.0, 192.0, 194.0, 196.0, 200.0],
        "low":    [179.0, 185.0, 188.0, 191.0, 193.0],
        "close":  [187.0, 190.0, 193.0, 195.0, 199.0],
        "volume": [250_000_000] * 5,
        "adj_factor": [1.0] * 5,
    })


@pytest.fixture
def ctx_with_weekly(weekly_df):
    """StrategyContext with weekly_data injected."""
    from trendspec.data.markets import Market
    from trendspec.strategy.base import BaseStrategy
    from trendspec.strategy.context import StrategyContext

    class _Dummy(BaseStrategy):
        name = "dummy"
        def init(self, ctx): pass
        def next(self, ctx): pass

    strat = _Dummy()
    ctx = StrategyContext(market=Market.US, strategy=strat, data=None,
                          weekly_data=weekly_df)
    return ctx


def test_weekly_indicator_value_returns_completed_week(ctx_with_weekly):
    """as_of_date=周三 (after most recent Friday) should return last Friday's value."""
    ctx_with_weekly.precompute_weekly_indicator("EMA", period=2)
    # 1/22 是周一; 上一已完成周线 bar = 1/19 (周五)
    val = ctx_with_weekly.weekly_indicator_value(
        "EMA", instrument_id="AAPL", as_of_date=date(2024, 1, 22), period=2
    )
    assert val is not None
    # EMA(period=2, smoothing=2) on [187, 190, 193] at 1/19 row:
    #   sf = 2 / (1+2) = 0.6667; iterative EMA result, just sanity-check non-None.


def test_weekly_indicator_value_no_lookahead(ctx_with_weekly):
    """as_of_date 当周尚未结束时不应读未来 bar."""
    ctx_with_weekly.precompute_weekly_indicator("EMA", period=2)
    # 1/8 周一: 已完成的最近周是 1/5, 不能偷看 1/12
    val_mon = ctx_with_weekly.weekly_indicator_value(
        "EMA", "AAPL", date(2024, 1, 8), period=2)
    val_fri = ctx_with_weekly.weekly_indicator_value(
        "EMA", "AAPL", date(2024, 1, 5), period=2)
    assert val_mon == val_fri   # 都指向 1/5 那条 bar


def test_weekly_indicator_value_before_any_data_returns_none(ctx_with_weekly):
    """as_of_date 早于第一周 → None."""
    ctx_with_weekly.precompute_weekly_indicator("EMA", period=2)
    val = ctx_with_weekly.weekly_indicator_value(
        "EMA", "AAPL", date(2023, 12, 1), period=2)
    assert val is None


def test_weekly_indicator_value_missing_weekly_data():
    """ctx without weekly_data → weekly_indicator_value returns None."""
    from trendspec.data.markets import Market
    from trendspec.strategy.base import BaseStrategy
    from trendspec.strategy.context import StrategyContext

    class _Dummy(BaseStrategy):
        name = "dummy"
        def init(self, ctx): pass
        def next(self, ctx): pass

    ctx = StrategyContext(market=Market.US, strategy=_Dummy(), data=None,
                          weekly_data=None)
    val = ctx.weekly_indicator_value("EMA", "AAPL", date(2024, 1, 22), period=20)
    assert val is None
```

- [ ] **Step 5.2: 运行验证失败**

```bash
uv run pytest tests/test_weekly_context.py -v
```
Expected: `FAILED` — `StrategyContext.__init__() got unexpected keyword 'weekly_data'`

- [ ] **Step 5.3: 修改 `StrategyContext.__init__`**

修改 `trendspec/strategy/context.py:55-73`：

```python
    def __init__(
        self,
        market: Market,
        strategy: "BaseStrategy",
        data: pl.DataFrame | None = None,
        root: str | None = None,
        weekly_data: pl.DataFrame | None = None,
    ) -> None:
        """
        Initialize strategy context.
        ...
        Args:
            ...
            weekly_data: Optional weekly OHLCV DataFrame for weekly indicators
        """
        self.market = market
        self.strategy = strategy
        self._data = data
        self._root = root
        self._weekly_data = weekly_data
```

在 `_indicator_fast` 之后增加周线相关字段：

```python
        # Weekly indicator cache (populated by precompute_weekly_indicator)
        self._weekly_indicator_cache: dict[str, pl.DataFrame] = {}
        self._weekly_indicator_fast: dict[str, dict[tuple, float]] = {}
        # Sorted weekly dates per instrument for binary lookup (built lazily)
        self._weekly_dates_by_iid: dict[str, list] = {}
```

- [ ] **Step 5.4: 实现两个新方法**

在 `precompute_indicator` 方法之后追加：

```python
    def precompute_weekly_indicator(
        self,
        name: str,
        **params: Any,
    ) -> pl.DataFrame:
        """
        Precompute indicator on weekly data (mirrors precompute_indicator).

        Args:
            name: Indicator name (MA, EMA, RSI, ...)
            **params: Indicator parameters (period, etc.)

        Returns:
            Weekly DataFrame with indicator column added
        """
        from trendspec.strategy.indicators import compute_indicator

        if self._weekly_data is None:
            raise RuntimeError("No weekly_data; cannot precompute weekly indicator")

        result = compute_indicator(self._weekly_data, name, **params)
        cache_key = f"weekly_{name}_{params}"
        self._weekly_indicator_cache[cache_key] = result

        _col = f"{name}_{params.get('period', '')}" if params else name
        if _col not in result.columns:
            _col = name
        if _col in result.columns:
            self._weekly_indicator_fast[cache_key] = {
                (inst_id, dt): val
                for inst_id, dt, val in result.select(
                    ["instrument_id", "date", _col]
                ).iter_rows()
                if val is not None
            }

        return result

    def weekly_indicator_value(
        self,
        name: str,
        instrument_id: str | None = None,
        as_of_date: DateType | None = None,
        **params: Any,
    ) -> float | None:
        """
        Get weekly indicator value at the most recent completed weekly bar ≤ as_of_date.

        Never reads an incomplete (current) week — strict no-lookahead guarantee.

        Returns None if no weekly bar exists ≤ as_of_date, or weekly data missing.
        """
        if self._weekly_data is None:
            return None

        target_iid = instrument_id or self._current_instrument_id
        target_date = as_of_date or self._current_date
        if target_iid is None or target_date is None:
            return None

        cache_key = f"weekly_{name}_{params}"
        if cache_key not in self._weekly_indicator_fast:
            self.precompute_weekly_indicator(name, **params)

        week_end = self._resolve_week_end(target_iid, target_date)
        if week_end is None:
            return None

        return self._weekly_indicator_fast.get(cache_key, {}).get((target_iid, week_end))

    def _resolve_week_end(self, iid: str, as_of_date: DateType) -> DateType | None:
        """Binary-search largest weekly bar date ≤ as_of_date for `iid`."""
        import bisect

        if iid not in self._weekly_dates_by_iid:
            if self._weekly_data is None:
                return None
            dates = (
                self._weekly_data
                .filter(pl.col("instrument_id") == iid)
                .sort("date")["date"]
                .to_list()
            )
            self._weekly_dates_by_iid[iid] = dates

        dates = self._weekly_dates_by_iid[iid]
        if not dates:
            return None
        # bisect_right gives the insertion point AFTER all equals;
        # idx-1 → largest date ≤ as_of_date
        idx = bisect.bisect_right(dates, as_of_date)
        if idx == 0:
            return None
        return dates[idx - 1]
```

- [ ] **Step 5.5: 运行验证通过**

```bash
uv run pytest tests/test_weekly_context.py -v
```
Expected: 4 个测试均 `PASSED`

```bash
uv run pytest tests/test_strategy.py tests/test_backtest_engine.py -v
```
Expected: 现有测试仍 `PASSED`（向后兼容）

- [ ] **Step 5.6: Commit**

```bash
git add tests/test_weekly_context.py trendspec/strategy/context.py
git commit -m "feat(context): StrategyContext 新增周线 indicator API + 日→已完成周二分映射"
```

---

## Task 6: BaseEngine 加载周线并注入 Context

**Files:**
- Modify: `trendspec/engine/base_engine.py`

- [ ] **Step 6.1: 修改 `load_data()`**

修改 `trendspec/engine/base_engine.py:160-175`（`load_data` 方法），在 `self._data = bars(...)` 之后追加周线加载：

```python
    def load_data(self) -> pl.DataFrame:
        """
        Load OHLCV data for the date range.

        Returns:
            DataFrame with OHLCV data
        """
        if self._data is None:
            self._data = bars(
                market=self.config.market,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                adjustment_mode=self.config.adjustment_mode,
                root=self.root,
            )
            # Best-effort weekly load (may be empty if weekly ingest not run yet)
            try:
                self._weekly_data = bars(
                    market=self.config.market,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    adjustment_mode=self.config.adjustment_mode,
                    root=self.root,
                    frequency="weekly",
                )
                if self._weekly_data.is_empty():
                    self._weekly_data = None
            except Exception:
                self._weekly_data = None
        return self._data
```

在 `__init__` 中初始化字段（如 `self._data = None` 附近）：

```python
        self._weekly_data: pl.DataFrame | None = None
```

- [ ] **Step 6.2: 修改 StrategyContext 构造调用**

查找 `BaseEngine` 中所有 `StrategyContext(...)` 构造，将 `weekly_data=self._weekly_data` 注入。例如：

```bash
grep -n "StrategyContext(" trendspec/engine/base_engine.py
```

逐处补 `weekly_data=self._weekly_data,` 参数。

- [ ] **Step 6.3: 写集成测试**

在 `tests/test_backtest_engine.py` 末尾追加（在已有 fixture 基础上）：

```python
def test_engine_loads_weekly_data_into_context(tmp_path):
    """Engine that finds weekly Parquet injects it into StrategyContext."""
    import polars as pl
    from datetime import date
    from trendspec.data.markets import Market
    from trendspec.ingest.writer import write_parquet

    # Manually craft minimal daily + weekly Parquet
    daily = pl.DataFrame({
        "instrument_id": ["AAPL"] * 3,
        "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        "open": [180.0, 181.0, 182.0],
        "high": [182.0, 183.0, 184.0],
        "low":  [179.0, 180.0, 181.0],
        "close":[181.0, 182.0, 183.0],
        "volume":[1_000_000]*3,
        "adj_factor":[1.0]*3,
    })
    weekly = pl.DataFrame({
        "instrument_id":["AAPL"],
        "date":[date(2024, 1, 5)],
        "open":[180.0], "high":[185.0], "low":[179.0], "close":[183.0],
        "volume":[5_000_000], "adj_factor":[1.0],
    })
    write_parquet(daily, Market.US, "daily", str(tmp_path), overwrite=True)
    write_parquet(weekly, Market.US, "weekly", str(tmp_path), overwrite=True)

    from trendspec.engine.base_engine import EngineConfig
    from trendspec.engine.backtest_engine import BacktestEngine
    # Minimal strategy that just records ctx
    from trendspec.strategy.base import BaseStrategy
    seen = {}
    class _Spy(BaseStrategy):
        name = "spy"
        def init(self, ctx):
            seen["weekly"] = ctx._weekly_data
        def next(self, ctx): pass

    config = EngineConfig(
        market=Market.US,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 4),
        root=str(tmp_path),
    )
    engine = BacktestEngine(config)
    engine.load_data()
    engine.run(_Spy)   # run() takes the CLASS, not an instance

    assert seen["weekly"] is not None
    assert len(seen["weekly"]) == 1
```

- [ ] **Step 6.4: 运行验证**

```bash
uv run pytest tests/test_backtest_engine.py::test_engine_loads_weekly_data_into_context -v
```
Expected: `PASSED`

```bash
uv run pytest tests/test_backtest_engine.py tests/test_screening_engine.py -v
```
Expected: 全部 `PASSED`

- [ ] **Step 6.5: Commit**

```bash
git add tests/test_backtest_engine.py trendspec/engine/base_engine.py
git commit -m "feat(engine): BaseEngine 加载周线并注入 StrategyContext (best-effort, 缺失时 None)"
```

---

## Task 7: `ema_cluster_pullback` 骨架 + 参数注册

**Files:**
- Create: `tests/strategy/__init__.py`（空文件）
- Create: `tests/strategy/test_ema_cluster_pullback.py`
- Create: `trendspec/strategy/examples/ema_cluster_pullback.py`

- [ ] **Step 7.1: 写失败测试（骨架检查）**

```bash
touch tests/strategy/__init__.py
```

新建 `tests/strategy/test_ema_cluster_pullback.py`：

```python
"""Tests for EMACluster Pullback strategy."""
from datetime import date

import polars as pl
import pytest


def test_strategy_registered():
    """Strategy registers under the name 'ema_cluster_pullback'."""
    from trendspec.strategy.base import get_strategy
    import trendspec.strategy.examples.ema_cluster_pullback  # noqa: F401

    cls = get_strategy("ema_cluster_pullback")
    assert cls is not None
    assert cls.name == "ema_cluster_pullback"


def test_strategy_default_params():
    """Strategy ships with the spec's default param values."""
    from trendspec.strategy.examples.ema_cluster_pullback import EMAClusterPullback
    s = EMAClusterPullback()
    assert s.get_param("ema_short") == 20
    assert s.get_param("ema_mid") == 60
    assert s.get_param("ema_long") == 120
    assert s.get_param("daily_cluster_threshold") == 0.04
    assert s.get_param("weekly_proximity_threshold") == 0.025
    assert s.get_param("stop_loss_pct") == 0.08
    assert s.get_param("confirmation_days") == 2
```

- [ ] **Step 7.2: 运行验证失败**

```bash
uv run pytest tests/strategy/test_ema_cluster_pullback.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: trendspec.strategy.examples.ema_cluster_pullback`

- [ ] **Step 7.3: 创建策略骨架**

新建 `trendspec/strategy/examples/ema_cluster_pullback.py`：

```python
"""
EMA Cluster Pullback Strategy.

Signal logic:
  BUY  = 日 EMA20/60/120 密集 (max-min)/min < threshold_daily
       ∧ |daily_close - 周 EMA20| / 周 EMA20 < threshold_weekly
       ∧ 日 EMA120 > 20 交易日前 EMA120
       ∧ 周 EMA20 > 上一已完成周 EMA20
       ∧ 指数 close > 指数 EMA200 (可关)
       ∧ ADV20 ≥ 阈值
       (连续 confirmation_days 日满足)

  SELL = 收盘 < 日 EMA60 连续 confirmation_days 日
       ∨ 收盘 ≤ entry_price * (1 - stop_loss_pct)  (硬止损, 单日触发)
"""

from collections import deque
from datetime import date as DateType
from typing import Any

import polars as pl

from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext


_DEFAULTS = {
    "ema_short": 20,
    "ema_mid": 60,
    "ema_long": 120,
    "daily_cluster_threshold": 0.04,
    "weekly_proximity_threshold": 0.025,
    "ema_long_slope_lookback": 20,
    "weekly_ema_period": 20,
    "adv_lookback": 20,
    "adv_threshold_us": 5_000_000,
    "adv_threshold_cn": 50_000_000,
    "market_index_id_us": "SP500",
    "market_index_id_cn": "CSI800",
    "market_ema_period": 200,
    "market_filter_enabled": True,
    "confirmation_days": 2,
    "stop_loss_pct": 0.08,
    "sell_ma_period": 60,
}


@register_strategy("ema_cluster_pullback")
class EMAClusterPullback(BaseStrategy):
    """日线 EMA 密集缠绕 + 周线 EMA20 回踩 + 多头趋势确认."""

    name = "ema_cluster_pullback"
    version = "1.0.0"
    params: dict[str, Any] = dict(_DEFAULTS)

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Merge user-supplied params over class defaults so get_param() never sees missing keys."""
        merged = dict(_DEFAULTS)
        if params:
            merged.update(params)
        super().__init__(params=merged)

    def init(self, ctx: StrategyContext) -> None:
        """Vectorized precompute of all indicators."""
        s = self.get_param("ema_short")
        m = self.get_param("ema_mid")
        l = self.get_param("ema_long")
        w = self.get_param("weekly_ema_period")

        ctx.precompute_indicator("EMA", period=s)
        ctx.precompute_indicator("EMA", period=m)
        ctx.precompute_indicator("EMA", period=l)
        ctx.precompute_weekly_indicator("EMA", period=w)

        # ADV20 = rolling mean of close*volume
        self._adv20_fast = self._compute_adv20_fast(
            ctx._data, lookback=self.get_param("adv_lookback")
        )

        self._market_ema_cache: dict[tuple, float | None] = {}
        self._entry_price: dict[str, float] = {}
        self._buy_pass_history: dict[str, deque] = {}
        self._sell_break_history: dict[str, deque] = {}

        self._full_data = ctx._data

    @staticmethod
    def _compute_adv20_fast(
        df: pl.DataFrame | None, lookback: int
    ) -> dict[tuple, float]:
        """Build {(iid, date): adv} dict for fast O(1) lookup."""
        if df is None or df.is_empty():
            return {}
        with_adv = df.sort("date").with_columns(
            (pl.col("close") * pl.col("volume"))
            .rolling_mean(window_size=lookback)
            .over("instrument_id")
            .alias("_adv")
        )
        return {
            (iid, dt): val
            for iid, dt, val in with_adv.select(
                ["instrument_id", "date", "_adv"]
            ).iter_rows()
            if val is not None
        }

    def next(self, ctx: StrategyContext) -> None:
        """Implemented in Task 8 (BUY) and Task 9 (SELL)."""
        raise NotImplementedError("next() implemented in subsequent tasks")
```

- [ ] **Step 7.4: 运行验证通过**

```bash
uv run pytest tests/strategy/test_ema_cluster_pullback.py -v
```
Expected: 2 个测试 `PASSED`

- [ ] **Step 7.5: Commit**

```bash
git add tests/strategy/__init__.py tests/strategy/test_ema_cluster_pullback.py trendspec/strategy/examples/ema_cluster_pullback.py
git commit -m "feat(strategy): ema_cluster_pullback 骨架, 注册参数与 init() 预算"
```

---

## Task 8: `next()` BUY 信号实现

**Files:**
- Modify: `tests/strategy/test_ema_cluster_pullback.py`（追加）
- Modify: `trendspec/strategy/examples/ema_cluster_pullback.py:next`

- [ ] **Step 8.1: 写失败测试 — 构造满足全部条件的样本**

在 `tests/strategy/test_ema_cluster_pullback.py` 末尾追加：

```python
def _build_passing_dataset():
    """Build daily + weekly DataFrames where AAPL meets all BUY conditions at end."""
    # 日线: 200 个交易日, 价格平缓上升让 EMA20/60/120 在末日极度贴合, EMA120 持续上行
    from datetime import timedelta
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(250)]
    # 简化: 用线性慢涨 +10% over 250 days, 末段持平让 EMA 收敛
    prices = []
    base = 100.0
    for i in range(250):
        if i < 200:
            base += 0.05   # 缓涨 → 三条 EMA 都接近这个值
        prices.append(base)
    daily = pl.DataFrame({
        "instrument_id": ["AAPL"] * 250,
        "date": dates,
        "open":   prices,
        "high":   [p * 1.005 for p in prices],
        "low":    [p * 0.995 for p in prices],
        "close":  prices,
        "volume": [10_000_000] * 250,
        "adj_factor":[1.0]*250,
    })
    # 周线: 40 周, 收敛到末周收盘 ≈ daily 末日 close, 让 weekly EMA20 也持续上行
    weekly_dates = [date(2024, 1, 5) + timedelta(days=7*i) for i in range(40)]
    w_prices = [100.0 + 0.25 * i for i in range(40)]
    weekly = pl.DataFrame({
        "instrument_id":["AAPL"]*40,
        "date": weekly_dates,
        "open":  w_prices, "high": [p*1.01 for p in w_prices],
        "low":   [p*0.99 for p in w_prices], "close": w_prices,
        "volume":[50_000_000]*40, "adj_factor":[1.0]*40,
    })
    return daily, weekly


def test_buy_signal_emitted_after_confirmation_days():
    """Run the strategy across the dataset; expect at least one BUY signal."""
    from trendspec.data.markets import Market
    from trendspec.strategy.base import get_strategy
    from trendspec.strategy.context import StrategyContext

    daily, weekly = _build_passing_dataset()

    StrategyClass = get_strategy("ema_cluster_pullback")
    # Disable market filter for unit test (no index_close available)
    strat = StrategyClass(params={"market_filter_enabled": False})

    ctx = StrategyContext(market=Market.US, strategy=strat, data=daily,
                          weekly_data=weekly)
    strat.init(ctx)

    # Walk through daily bars
    buy_count = 0
    for dt in daily["date"].to_list():
        ctx._current_date = dt
        ctx._current_instrument_id = "AAPL"
        ctx._current_ticker = "AAPL"
        ctx._pending_signals = []
        try:
            strat.next(ctx)
        except Exception:
            pass
        for sig in ctx._pending_signals:
            if sig.direction == "BUY":
                buy_count += 1
    assert buy_count >= 1, "策略应在密集 + 周回踩 + 多头趋势末段触发至少一次 BUY"
```

- [ ] **Step 8.2: 运行验证失败**

```bash
uv run pytest tests/strategy/test_ema_cluster_pullback.py::test_buy_signal_emitted_after_confirmation_days -v
```
Expected: `FAILED` — `NotImplementedError`

- [ ] **Step 8.3: 实现 `next()` 的 BUY 部分**

替换 `trendspec/strategy/examples/ema_cluster_pullback.py` 中的 `next` 方法：

```python
    def next(self, ctx: StrategyContext) -> None:
        iid = ctx.instrument_id
        t = ctx.date
        close = ctx.close

        s = self.get_param("ema_short")
        m = self.get_param("ema_mid")
        l = self.get_param("ema_long")
        w = self.get_param("weekly_ema_period")
        slope_lb = self.get_param("ema_long_slope_lookback")
        conf = self.get_param("confirmation_days")

        ema_s = ctx.indicator_value("EMA", iid, t, period=s)
        ema_m = ctx.indicator_value("EMA", iid, t, period=m)
        ema_l = ctx.indicator_value("EMA", iid, t, period=l)
        weekly_ema_w = ctx.weekly_indicator_value("EMA", iid, t, period=w)

        # SELL hard stop & EMA60 break — implemented in Task 9
        if self._maybe_sell(ctx, iid, t, close, ema_m, conf):
            return

        # BUY checks
        buy_history = self._buy_pass_history.setdefault(iid, deque(maxlen=conf))
        if any(v is None for v in [ema_s, ema_m, ema_l, weekly_ema_w]):
            buy_history.append(False)
            return

        ema_l_prev = self._lookup_prev_ema(ctx, iid, t, l, slope_lb)
        weekly_ema_w_prev = self._lookup_prev_weekly_ema(ctx, iid, t, w)
        if ema_l_prev is None or weekly_ema_w_prev is None:
            buy_history.append(False)
            return

        c1 = (max(ema_s, ema_m, ema_l) - min(ema_s, ema_m, ema_l)) / min(ema_s, ema_m, ema_l) \
             < self.get_param("daily_cluster_threshold")
        c2 = abs(close - weekly_ema_w) / weekly_ema_w \
             < self.get_param("weekly_proximity_threshold")
        c3 = ema_l > ema_l_prev
        c4 = weekly_ema_w > weekly_ema_w_prev
        c5 = self._market_passes(ctx, t)
        c6 = self._liquid_enough(ctx, iid, t)

        passes = c1 and c2 and c3 and c4 and c5 and c6
        buy_history.append(passes)

        if len(buy_history) == conf and all(buy_history) and not ctx.has_position(iid):
            ctx.signal("BUY", iid, close,
                       note=f"EMA cluster: spread={(max(ema_s,ema_m,ema_l)-min(ema_s,ema_m,ema_l))/min(ema_s,ema_m,ema_l):.3%}")
            self._entry_price[iid] = close

    def _lookup_prev_ema(
        self, ctx: StrategyContext, iid: str, t: DateType,
        period: int, lookback: int,
    ) -> float | None:
        """Return daily EMA at `lookback` trading-day bars before t for iid."""
        if self._full_data is None:
            return None
        dates = (
            self._full_data
            .filter(pl.col("instrument_id") == iid)
            .sort("date")["date"]
        )
        idx = dates.search_sorted(t, side="left")
        if idx < lookback:
            return None
        prev_date = dates[idx - lookback]
        return ctx.indicator_value("EMA", iid, prev_date, period=period)

    def _lookup_prev_weekly_ema(
        self, ctx: StrategyContext, iid: str, t: DateType, period: int,
    ) -> float | None:
        """Return weekly EMA at the week BEFORE the most-recent-completed week ≤ t."""
        from datetime import timedelta
        week_end = ctx._resolve_week_end(iid, t)
        if week_end is None:
            return None
        prev_target = week_end - timedelta(days=1)
        prev_week_end = ctx._resolve_week_end(iid, prev_target)
        if prev_week_end is None:
            return None
        cache_key = f"weekly_EMA_{{'period': {period}}}"
        return ctx._weekly_indicator_fast.get(cache_key, {}).get((iid, prev_week_end))

    def _market_passes(self, ctx: StrategyContext, t: DateType) -> bool:
        """Index close > index EMA200 (if filter enabled)."""
        if not self.get_param("market_filter_enabled"):
            return True
        from trendspec.data.markets import Market
        idx_id = (self.get_param("market_index_id_us") if ctx.market == Market.US
                  else self.get_param("market_index_id_cn"))
        ema_p = self.get_param("market_ema_period")
        idx_close = ctx.index_close(idx_id, t)
        if idx_close is None:
            return True
        cache_key = (idx_id, t, ema_p)
        if cache_key in self._market_ema_cache:
            ema_val = self._market_ema_cache[cache_key]
        else:
            if not hasattr(ctx, "_indices_cache") or ctx._indices_cache is None:
                self._market_ema_cache[cache_key] = None
                return True
            series = (
                ctx._indices_cache
                .filter(pl.col("instrument_id") == idx_id)
                .sort("date")
                .filter(pl.col("date") <= t)["close"]
                .to_list()
            )
            if len(series) < ema_p:
                ema_val = None
            else:
                # Manual EMA computation
                sf = 2.0 / (1 + ema_p)
                ema_val = series[0]
                for x in series[1:]:
                    ema_val = (x - ema_val) * sf + ema_val
            self._market_ema_cache[cache_key] = ema_val
        if ema_val is None:
            return True
        return idx_close > ema_val

    def _liquid_enough(self, ctx: StrategyContext, iid: str, t: DateType) -> bool:
        """ADV20 ≥ market threshold."""
        from trendspec.data.markets import Market
        thr = (self.get_param("adv_threshold_us") if ctx.market == Market.US
               else self.get_param("adv_threshold_cn"))
        adv = self._adv20_fast.get((iid, t))
        if adv is None:
            return False
        return adv >= thr

    def _maybe_sell(self, ctx, iid, t, close, ema_m, conf) -> bool:
        """Placeholder — overridden in Task 9."""
        return False
```

- [ ] **Step 8.4: 运行验证通过**

```bash
uv run pytest tests/strategy/test_ema_cluster_pullback.py -v
```
Expected: 3 个测试 `PASSED`

- [ ] **Step 8.5: Commit**

```bash
git add tests/strategy/test_ema_cluster_pullback.py trendspec/strategy/examples/ema_cluster_pullback.py
git commit -m "feat(strategy): ema_cluster_pullback BUY 信号 6 条件 + 2 日确认"
```

---

## Task 9: SELL 信号 — 跌破 EMA60 + 硬止损

**Files:**
- Modify: `tests/strategy/test_ema_cluster_pullback.py`
- Modify: `trendspec/strategy/examples/ema_cluster_pullback.py`

- [ ] **Step 9.1: 写两个失败测试**

在 `tests/strategy/test_ema_cluster_pullback.py` 末尾追加：

```python
def test_sell_on_stop_loss():
    """Hard stop loss: close ≤ entry_price * (1 - 0.08) emits SELL same day."""
    from datetime import timedelta
    from trendspec.data.markets import Market
    from trendspec.strategy.base import get_strategy
    from trendspec.strategy.context import StrategyContext

    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(5)]
    daily = pl.DataFrame({
        "instrument_id": ["AAPL"] * 5,
        "date": dates,
        "open":   [100.0]*5, "high": [101.0]*5, "low": [90.0]*5,
        "close":  [100.0, 100.0, 100.0, 100.0, 91.0],   # final close = -9%
        "volume": [10_000_000]*5, "adj_factor": [1.0]*5,
    })

    StrategyClass = get_strategy("ema_cluster_pullback")
    strat = StrategyClass(params={"market_filter_enabled": False})
    ctx = StrategyContext(market=Market.US, strategy=strat, data=daily,
                          weekly_data=None)
    strat.init(ctx)

    # Inject position + entry price
    strat._entry_price["AAPL"] = 100.0
    ctx._positions["AAPL"] = 100.0

    ctx._current_date = dates[-1]
    ctx._current_instrument_id = "AAPL"
    ctx._current_ticker = "AAPL"
    ctx._pending_signals = []
    strat.next(ctx)

    sells = [s for s in ctx._pending_signals if s.direction == "SELL"]
    assert len(sells) == 1
    assert "stop_loss" in (sells[0].note or "").lower()


def test_sell_on_break_ema60_two_days():
    """Break EMA60 for 2 consecutive bars while holding → SELL."""
    from datetime import timedelta
    from trendspec.data.markets import Market
    from trendspec.strategy.base import get_strategy
    from trendspec.strategy.context import StrategyContext

    # Build 100 bars so EMA60 is well-defined; final 2 bars below EMA60
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    closes = [100.0] * 98 + [50.0, 49.0]   # crash final 2 days, but only -2% from entry
    daily = pl.DataFrame({
        "instrument_id": ["AAPL"] * 100,
        "date": dates,
        "open":   closes, "high": [c*1.01 for c in closes],
        "low":    [c*0.99 for c in closes], "close": closes,
        "volume": [10_000_000]*100, "adj_factor":[1.0]*100,
    })

    StrategyClass = get_strategy("ema_cluster_pullback")
    strat = StrategyClass(params={
        "market_filter_enabled": False,
        "stop_loss_pct": 0.95,   # very wide → disable hard-stop for this test
    })
    ctx = StrategyContext(market=Market.US, strategy=strat, data=daily,
                          weekly_data=None)
    strat.init(ctx)

    strat._entry_price["AAPL"] = 100.0
    ctx._positions["AAPL"] = 100.0

    # Iterate to populate sell_break_history & trigger 2-day confirmation
    for dt in dates[-3:]:
        ctx._current_date = dt
        ctx._current_instrument_id = "AAPL"
        ctx._current_ticker = "AAPL"
        ctx._pending_signals = []
        strat.next(ctx)

    sells = [s for s in ctx._pending_signals if s.direction == "SELL"]
    assert len(sells) == 1
    assert "ema60" in (sells[0].note or "").lower() or "break" in (sells[0].note or "").lower()
```

- [ ] **Step 9.2: 运行验证失败**

```bash
uv run pytest tests/strategy/test_ema_cluster_pullback.py::test_sell_on_stop_loss \
                  tests/strategy/test_ema_cluster_pullback.py::test_sell_on_break_ema60_two_days -v
```
Expected: 2 个 `FAILED`（`_maybe_sell` 是 placeholder）

- [ ] **Step 9.3: 实现 `_maybe_sell`**

替换 `trendspec/strategy/examples/ema_cluster_pullback.py` 中的 `_maybe_sell` 方法：

```python
    def _maybe_sell(self, ctx: StrategyContext, iid: str, t: DateType,
                    close: float, ema_m: float | None, conf: int) -> bool:
        """
        Return True if a SELL signal was emitted (caller should skip BUY check).
        Two paths:
          - hard stop: close ≤ entry_price * (1 - stop_loss_pct) → emit immediately
          - break EMA60 for `conf` consecutive bars → emit
        """
        if not ctx.has_position(iid):
            return False

        # Hard stop loss
        entry = self._entry_price.get(iid)
        if entry is not None:
            stop_pct = self.get_param("stop_loss_pct")
            if close <= entry * (1.0 - stop_pct):
                ctx.signal("SELL", iid, close, note=f"stop_loss_{stop_pct:.0%}")
                self._cleanup_position(iid)
                return True

        # Break EMA60 (need ema_m and rolling history)
        history = self._sell_break_history.setdefault(iid, deque(maxlen=conf))
        if ema_m is None:
            history.append(False)
            return False
        broken_today = close < ema_m
        history.append(broken_today)
        if len(history) == conf and all(history):
            ctx.signal("SELL", iid, close, note=f"break_ema{self.get_param('sell_ma_period')}_{conf}d")
            self._cleanup_position(iid)
            return True
        return False

    def _cleanup_position(self, iid: str) -> None:
        """Clear per-iid state after a SELL."""
        self._entry_price.pop(iid, None)
        self._sell_break_history.pop(iid, None)
        self._buy_pass_history.pop(iid, None)
```

- [ ] **Step 9.4: 运行验证通过**

```bash
uv run pytest tests/strategy/test_ema_cluster_pullback.py -v
```
Expected: 5 个测试均 `PASSED`

- [ ] **Step 9.5: Commit**

```bash
git add tests/strategy/test_ema_cluster_pullback.py trendspec/strategy/examples/ema_cluster_pullback.py
git commit -m "feat(strategy): ema_cluster_pullback SELL 信号 (跌破 EMA60 连2日 + 8% 硬止损)"
```

---

## Task 10: 端到端 smoke test + 注册到全局

**Files:**
- Modify: `trendspec/strategy/examples/__init__.py`
- Modify: `tests/test_strategies.py`（如有，否则用 inline 测试）

- [ ] **Step 10.1: 在 examples 包暴露新策略**

读取 `trendspec/strategy/examples/__init__.py`，照已有策略导入风格追加一行：

```python
from trendspec.strategy.examples.ema_cluster_pullback import EMAClusterPullback  # noqa: F401
```

- [ ] **Step 10.2: 写 smoke test — backtest run 端到端**

新建 `tests/strategy/test_ema_cluster_pullback_e2e.py`：

```python
"""End-to-end smoke test for ema_cluster_pullback."""
from datetime import date, timedelta

import polars as pl


def test_screen_run_does_not_crash(tmp_path):
    """ScreeningEngine + ema_cluster_pullback completes without error on minimal data."""
    from trendspec.data.markets import Market
    from trendspec.engine.base_engine import EngineConfig
    from trendspec.engine.screening_engine import ScreeningEngine
    from trendspec.ingest.writer import write_parquet
    import trendspec.strategy.examples  # noqa: register strategy

    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(150)]
    closes = [100.0 + 0.01 * i for i in range(150)]
    daily = pl.DataFrame({
        "instrument_id":["AAPL"]*150, "date": dates,
        "open": closes, "high":[c*1.01 for c in closes],
        "low":[c*0.99 for c in closes], "close": closes,
        "volume":[10_000_000]*150, "adj_factor":[1.0]*150,
    })
    weekly_dates = [date(2024, 1, 5) + timedelta(days=7*i) for i in range(25)]
    w_closes = [100.0 + 0.1 * i for i in range(25)]
    weekly = pl.DataFrame({
        "instrument_id":["AAPL"]*25, "date": weekly_dates,
        "open": w_closes, "high":[c*1.01 for c in w_closes],
        "low":[c*0.99 for c in w_closes], "close": w_closes,
        "volume":[50_000_000]*25, "adj_factor":[1.0]*25,
    })
    write_parquet(daily, Market.US, "daily", str(tmp_path), overwrite=True)
    write_parquet(weekly, Market.US, "weekly", str(tmp_path), overwrite=True)

    from trendspec.strategy.base import get_strategy
    StrategyClass = get_strategy("ema_cluster_pullback")

    # ScreeningEngine takes EngineConfig (uses start_date as target)
    config = EngineConfig(
        market=Market.US,
        start_date=dates[-1],
        end_date=dates[-1],
        root=str(tmp_path),
    )
    engine = ScreeningEngine(config)
    # run() takes class + params dict; engine handles load_universe + load_data
    result = engine.run(
        StrategyClass,
        params={"market_filter_enabled": False, "adv_threshold_us": 0},
    )
    # No assertion on signal count — only that no exception
    assert result is not None
```

- [ ] **Step 10.3: 验证全套**

```bash
uv run pytest tests/strategy/ tests/test_weekly_ingestor.py tests/test_weekly_loader.py tests/test_weekly_context.py -v
```
Expected: 全部 `PASSED`

```bash
uv run pytest -x   # 全套回归
```
Expected: 现有测试与新测试均 `PASSED`

- [ ] **Step 10.4: Lint**

```bash
uv run ruff check trendspec/ tests/
uv run ruff format --check trendspec/ tests/
```
若有问题：

```bash
uv run ruff format trendspec/ tests/
uv run ruff check --fix trendspec/ tests/
```

- [ ] **Step 10.5: Commit**

```bash
git add trendspec/strategy/examples/__init__.py tests/strategy/test_ema_cluster_pullback_e2e.py
git commit -m "feat: ema_cluster_pullback 注册到 examples + ScreeningEngine 端到端 smoke"
```

---

## Task 11: 手工冒烟（真实数据，可选）

**Files:** 无（仅运行命令）

- [ ] **Step 11.1: 验证周线 ingest 真实运行**

```bash
uv run trendspec ingest weekly --market us --full
```
观察输出：`完成: N 行, M 只股票`。预期 M ≈ SP500+RUSSELL1000 成分数量；N ≈ M × 历史周数。

- [ ] **Step 11.2: 验证 screen 命令运行**

```bash
uv run trendspec screen run --strategy ema_cluster_pullback --market us --date 2026-05-15
```
Expected: 列出当日满足条件的 ticker 列表（可能为空，但不应崩溃）。

- [ ] **Step 11.3: 简易回测**

```bash
uv run trendspec backtest run \
  --strategy ema_cluster_pullback --market us \
  --start 2020-01-01 --end 2026-05-15
```
观察回测输出（不要求性能曲线特定形态，仅验证不崩）。

- [ ] **Step 11.4: 可选: 放宽阈值测试灵敏度**

```bash
uv run trendspec screen run --strategy ema_cluster_pullback --market us \
  --date 2026-05-15 \
  --param daily_cluster_threshold=0.05 \
  --param weekly_proximity_threshold=0.03
```

无 commit（纯验证步骤）。

---

## 风险与回退

- **回退**：每个 Task 独立 commit；任一 Task 失败可 `git revert <hash>` 单独回滚。
- **数据缺失**：若群辉 `weekly_prices` 没有某只股票，ingest 跳过；strategy 的 `weekly_indicator_value` 返回 None，BUY 不触发。
- **lookahead**：所有周线访问通过 `_resolve_week_end` 二分定位「≤ as_of_date 的最大周日期」，已有专门测试覆盖。
- **市场过滤兜底**：指数数据缺失时通过过滤（同 minervini 范式），日志可后续加。

## Done = 当以下全部满足时

1. `uv run pytest` 全套通过（含新增 4 个测试文件）
2. `uv run ruff check .` 与 `uv run ruff format --check .` 通过
3. `uv run trendspec ingest weekly --market us` 真实跑通
4. `uv run trendspec screen run --strategy ema_cluster_pullback --market us --date <today>` 不崩溃
5. 所有 commit 完成、main 分支干净
