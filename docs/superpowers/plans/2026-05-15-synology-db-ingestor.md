# Synology DB Ingestor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `CN_A` → `CN` throughout, fix the broken CLI ingest commands, and write a custom ingestor that reads the existing Synology NAS `stocks` DB schema into TrendSpec's Parquet data lake.

**Architecture:** New `stocks_db_ingestor.py` reads the existing multi-market DB (prices + stocks + constituent_changes tables) and transforms it to TrendSpec's standard Parquet format. CLI `ingest_cmd.py` is rewritten to call standalone ingestor functions instead of non-existent wrapper classes. US market implemented first, then CN.

**Tech Stack:** Python 3.11+, SQLAlchemy 2.x, Polars, pymysql, uv

---

## File Map

| Action | File |
|--------|------|
| Modify | `trendspec/data/markets.py` — rename `CN_A` enum |
| Rename | `trendspec/ingest/cn_a_ingestor.py` → `trendspec/ingest/cn_ingestor.py` |
| Rename | `trendspec/data/universe/cn_a.py` → `trendspec/data/universe/cn.py` |
| Batch sed | All 32 source + 17 test files — replace `CN_A` references |
| Modify | `trendspec/config/settings.py` — add `ALLOW_ROOT_DB_USER` bypass |
| Modify | `.env` — add `ALLOW_ROOT_DB_USER=true` |
| **Create** | `trendspec/ingest/stocks_db_ingestor.py` — new custom ingestor |
| **Create** | `tests/test_stocks_db_ingestor.py` — tests for new ingestor |
| Modify | `trendspec/cli/ingest_cmd.py` — fix broken CLI + route to new ingestor |

---

## Task 1: Rename Market.CN_A → Market.CN

**Files:**
- Modify: `trendspec/data/markets.py`
- Rename: `trendspec/ingest/cn_a_ingestor.py` → `trendspec/ingest/cn_ingestor.py`
- Rename: `trendspec/data/universe/cn_a.py` → `trendspec/data/universe/cn.py`
- Batch sed: All `*.py` files

- [ ] **Step 1: Edit `trendspec/data/markets.py` — change enum and path**

Change line 60: `CN_A = "CN_A"` → `CN = "CN"`
Change line 168: `Market.CN_A:` → `Market.CN:`
Change line 169: `path="cn_a",` → `path="cn",`

```python
# trendspec/data/markets.py line 60
CN = "CN"  # China A-shares (was CN_A)
```

```python
# trendspec/data/markets.py line 168-169
Market.CN: MarketMetadata(
    path="cn",
```

- [ ] **Step 2: Rename the two source files**

```bash
mv trendspec/ingest/cn_a_ingestor.py trendspec/ingest/cn_ingestor.py
mv trendspec/data/universe/cn_a.py trendspec/data/universe/cn.py
```

- [ ] **Step 3: Batch rename — Market enum and string references**

```bash
# Market.CN_A → Market.CN
find . -name "*.py" -not -path "./.git/*" | xargs sed -i '' 's/Market\.CN_A/Market.CN/g'

# CN_A = "CN_A" string (leftover in tests that compare enum values)
find . -name "*.py" | xargs sed -i '' 's/"CN_A"/"CN"/g'

# path strings "cn_a" → "cn"
find . -name "*.py" | xargs sed -i '' 's/"cn_a"/"cn"/g'

# CLI choice value cn_a → cn (in string literals for CLI options)
find . -name "*.py" | xargs sed -i '' "s/'cn_a'/'cn'/g"

# Variable names CN_A_*MAP → CN_*MAP in schema_map.py
sed -i '' 's/CN_A_DAILY_MAP/CN_DAILY_MAP/g' trendspec/ingest/schema_map.py
sed -i '' 's/CN_A_COMPONENTS_MAP/CN_COMPONENTS_MAP/g' trendspec/ingest/schema_map.py
sed -i '' 's/CN_A_SECTORS_MAP/CN_SECTORS_MAP/g' trendspec/ingest/schema_map.py

# Fix references to the renamed variable names everywhere else
find . -name "*.py" | xargs sed -i '' 's/CN_A_DAILY_MAP/CN_DAILY_MAP/g'
find . -name "*.py" | xargs sed -i '' 's/CN_A_COMPONENTS_MAP/CN_COMPONENTS_MAP/g'
find . -name "*.py" | xargs sed -i '' 's/CN_A_SECTORS_MAP/CN_SECTORS_MAP/g'

# Import paths after file rename
find . -name "*.py" | xargs sed -i '' 's/from trendspec\.ingest\.cn_a_ingestor/from trendspec.ingest.cn_ingestor/g'
find . -name "*.py" | xargs sed -i '' 's/from trendspec\.data\.universe\.cn_a/from trendspec.data.universe.cn/g'

# __init__.py imports
sed -i '' 's/cn_a import CNAUniverse/cn import CNAUniverse/g' trendspec/data/universe/__init__.py
```

- [ ] **Step 4: Rename the function prefix in cn_ingestor.py**

```bash
sed -i '' 's/ingest_cn_a_/ingest_cn_/g' trendspec/ingest/cn_ingestor.py
sed -i '' 's/get_cn_a_/get_cn_/g' trendspec/ingestor/cn_ingestor.py
```

Update the docstring in `trendspec/ingest/cn_ingestor.py` line 1:
```python
"""
A-share (CN) data ingestor.
...
"""
```

- [ ] **Step 5: Run tests to verify no regressions**

```bash
uv run pytest tests/ -x -q 2>&1 | head -40
```

Expected: all tests pass (or existing failures only — not new ones from the rename).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: rename Market.CN_A → Market.CN, path cn_a → cn"
```

---

## Task 2: Fix Settings Root-User Check + .env

**Files:**
- Modify: `trendspec/config/settings.py:33-43`
- Modify: `.env`

- [ ] **Step 1: Write failing test**

Add to `tests/test_settings.py`:

```python
def test_root_user_allowed_with_env_var(monkeypatch, tmp_path):
    """DB_USER=root is accepted when ALLOW_ROOT_DB_USER=true."""
    monkeypatch.setenv("ALLOW_ROOT_DB_USER", "true")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_USER", "root")
    monkeypatch.setenv("DB_PASSWORD", "secret")
    from trendspec.config.settings import DatabaseSettings
    settings = DatabaseSettings()
    assert settings.user == "root"


def test_root_user_rejected_without_env_var(monkeypatch):
    """DB_USER=root raises ValueError when ALLOW_ROOT_DB_USER not set."""
    monkeypatch.delenv("ALLOW_ROOT_DB_USER", raising=False)
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_USER", "root")
    monkeypatch.setenv("DB_PASSWORD", "secret")
    from trendspec.config.settings import DatabaseSettings
    import pytest
    with pytest.raises(ValueError, match="cannot be 'root'"):
        DatabaseSettings()
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
uv run pytest tests/test_settings.py::test_root_user_allowed_with_env_var -v
```

Expected: FAIL — `ValueError` is raised even with the env var.

- [ ] **Step 3: Update `trendspec/config/settings.py` validator**

Replace the `validate_user_not_root` method (lines 33-43) with:

```python
@field_validator("user")
@classmethod
def validate_user_not_root(cls, v: str) -> str:
    """Ensure database user is not root for security."""
    if v.lower() == "root":
        import os
        import warnings
        if os.getenv("ALLOW_ROOT_DB_USER", "").lower() != "true":
            raise ValueError(
                "DB_USER cannot be 'root'. Use a read-only account for security. "
                "Create one with: CREATE USER 'trendspec'@'%' IDENTIFIED BY '<password>'; "
                "GRANT SELECT ON stocks.* TO 'trendspec'@'%'; "
                "Or set ALLOW_ROOT_DB_USER=true for development."
            )
        warnings.warn(
            "DB_USER=root is insecure. Development only.", UserWarning, stacklevel=2
        )
    return v
```

- [ ] **Step 4: Run to confirm PASS**

```bash
uv run pytest tests/test_settings.py -v
```

Expected: all pass including the two new tests.

- [ ] **Step 5: Add `ALLOW_ROOT_DB_USER=true` to `.env`**

Append to `.env`:
```
ALLOW_ROOT_DB_USER=true
```

- [ ] **Step 6: Commit**

```bash
git add trendspec/config/settings.py tests/test_settings.py .env
git commit -m "feat: allow root DB user in dev via ALLOW_ROOT_DB_USER=true env var"
```

---

## Task 3: US Daily Ingestor

**Files:**
- Create: `trendspec/ingest/stocks_db_ingestor.py`
- Create: `tests/test_stocks_db_ingestor.py`

The Synology `stocks` DB has:
- `prices(ticker, date, open, high, low, close, volume)`
- `stocks(ticker, exchange, gics_sector, gics_industry, is_active)`
- `constituent_changes(index_id, ticker, change_type, change_date)`

US stocks: `exchange IN ('NYSE', 'Nasdaq', 'CBOE')`
For US: `instrument_id = ticker` (already uppercase tickers like AAPL, MSFT)

- [ ] **Step 1: Create test file with SQLite fixture**

Create `tests/test_stocks_db_ingestor.py`:

```python
"""Tests for Synology stocks DB custom ingestor."""

import tempfile
from datetime import date

import polars as pl
import pytest
from sqlalchemy import create_engine, text


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def stocks_db():
    """SQLite in-memory mock of the Synology stocks DB."""
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE prices (
                ticker TEXT,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
            )
        """))
        conn.execute(text("""
            CREATE TABLE stocks (
                ticker TEXT PRIMARY KEY,
                exchange TEXT,
                gics_sector TEXT,
                gics_industry TEXT,
                is_active INTEGER
            )
        """))
        conn.execute(text("""
            CREATE TABLE constituent_changes (
                index_id TEXT,
                ticker TEXT,
                change_type TEXT,
                change_date DATE
            )
        """))
        # US stocks metadata
        conn.execute(text("""
            INSERT INTO stocks VALUES
            ('AAPL', 'NYSE', 'Information Technology', 'Technology Hardware', 1),
            ('MSFT', 'Nasdaq', 'Information Technology', 'Systems Software', 1),
            ('JPM', 'NYSE', 'Financials', 'Diversified Banks', 1)
        """))
        # US price data
        conn.execute(text("""
            INSERT INTO prices VALUES
            ('AAPL', '2024-01-02', 185.0, 186.0, 183.0, 185.5, 50000000),
            ('AAPL', '2024-01-03', 185.5, 187.0, 184.0, 186.0, 55000000),
            ('MSFT', '2024-01-02', 370.0, 372.0, 368.0, 371.0, 20000000),
            ('MSFT', '2024-01-03', 371.0, 373.0, 369.0, 372.0, 22000000),
            ('JPM',  '2024-01-02', 150.0, 152.0, 149.0, 151.0, 10000000),
            ('JPM',  '2024-01-03', 151.0, 153.0, 150.0, 152.0, 11000000)
        """))
        conn.commit()
    yield engine


@pytest.fixture
def temp_root():
    with tempfile.TemporaryDirectory() as d:
        yield d


# =============================================================================
# US daily tests
# =============================================================================

def test_ingest_us_daily_schema(stocks_db, temp_root):
    """US daily Parquet has correct columns and types."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_daily
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market
    import polars as pl

    manifest = Manifest(Market.US, temp_root)
    result = ingest_us_daily(stocks_db, manifest, temp_root)

    assert result["row_count"] == 6
    assert result["instrument_count"] == 3

    df = pl.read_parquet(f"{temp_root}/us/daily/")
    assert set(df.columns) >= {"instrument_id", "date", "ticker", "open", "high", "low", "close", "volume", "adj_factor"}


def test_ingest_us_daily_instrument_id_equals_ticker(stocks_db, temp_root):
    """For US stocks, instrument_id == ticker."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_daily
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    ingest_us_daily(stocks_db, manifest, temp_root)

    df = pl.read_parquet(f"{temp_root}/us/daily/")
    mismatched = df.filter(pl.col("instrument_id") != pl.col("ticker"))
    assert len(mismatched) == 0, f"instrument_id != ticker: {mismatched}"


def test_ingest_us_daily_adj_factor_is_one(stocks_db, temp_root):
    """adj_factor must be 1.0 (prices already adjusted via Yahoo API)."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_daily
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    ingest_us_daily(stocks_db, manifest, temp_root)

    df = pl.read_parquet(f"{temp_root}/us/daily/")
    assert df["adj_factor"].unique().to_list() == [1.0]


def test_ingest_us_daily_incremental(stocks_db, temp_root):
    """Second run with same data is a no-op (already synced)."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_daily
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    r1 = ingest_us_daily(stocks_db, manifest, temp_root)
    assert r1["row_count"] == 6

    manifest2 = Manifest(Market.US, temp_root)  # reload manifest from disk
    r2 = ingest_us_daily(stocks_db, manifest2, temp_root)
    assert r2["row_count"] == 0  # no new rows
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
uv run pytest tests/test_stocks_db_ingestor.py::test_ingest_us_daily_schema -v
```

Expected: `ModuleNotFoundError: No module named 'trendspec.ingest.stocks_db_ingestor'`

- [ ] **Step 3: Create `trendspec/ingest/stocks_db_ingestor.py`**

```python
"""
Custom ingestor for the Synology NAS 'stocks' database.

Source schema (existing, do not modify):
  prices(ticker, date, open, high, low, close, volume)
  stocks(ticker, exchange, gics_sector, gics_industry, is_active)
  constituent_changes(index_id, ticker, change_type, change_date)

exchange values:
  US  → NYSE, Nasdaq, CBOE
  CN  → SSE, SH, SZSE, SZ
  HK  → HKEX, HK

instrument_id convention:
  US  → ticker as-is (AAPL, MSFT)
  CN  → SH{ticker} for SSE/SH, SZ{ticker} for SZSE/SZ
  HK  → ticker as-is

Prices are already adjusted:
  US  → Yahoo Finance adjusted close (adj_factor = 1.0)
  CN  → Tushare backward-adjusted (adj_factor = 1.0)
"""

from datetime import date, datetime
from typing import Final

import polars as pl
from sqlalchemy import Engine, text

from trendspec.data.markets import Market
from trendspec.ingest.manifest import Manifest
from trendspec.ingest.writer import write_parquet

# Exchange sets for filtering
_US_EXCHANGES: Final[tuple[str, ...]] = ("NYSE", "Nasdaq", "CBOE")
_CN_EXCHANGES: Final[tuple[str, ...]] = ("SSE", "SH", "SZSE", "SZ")
_HK_EXCHANGES: Final[tuple[str, ...]] = ("HKEX", "HK")


def _exchange_placeholder(exchanges: tuple[str, ...]) -> str:
    """Build SQLAlchemy :param0, :param1, ... placeholder string."""
    return ", ".join(f":ex{i}" for i in range(len(exchanges)))


def _exchange_params(exchanges: tuple[str, ...]) -> dict[str, str]:
    """Build SQLAlchemy parameter dict for exchange IN clause."""
    return {f"ex{i}": ex for i, ex in enumerate(exchanges)}


def _get_last_synced_date(manifest: Manifest, dataset: str) -> str:
    """
    Get the last synced date from manifest for a dataset.

    Returns '1970-01-01' if never synced (pulls all history).
    """
    state = manifest.get_dataset_state(dataset)
    if state is None:
        return "1970-01-01"
    date_range = state.get("date_range", {})
    return date_range.get("end", "1970-01-01")


# =============================================================================
# US Daily
# =============================================================================


def ingest_us_daily(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """
    Ingest US daily OHLCV from prices + stocks tables.

    Args:
        engine: SQLAlchemy engine connected to Synology stocks DB
        manifest: Manifest for tracking sync state
        root: data_lake root directory
        full_sync: If True, pull all history ignoring manifest

    Returns:
        {"row_count": int, "date_range": (str, str), "instrument_count": int}
    """
    last_date = "1970-01-01" if full_sync else _get_last_synced_date(manifest, "daily")

    placeholders = _exchange_placeholder(_US_EXCHANGES)
    params = _exchange_params(_US_EXCHANGES)
    params["last_date"] = last_date

    sql = text(f"""
        SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, p.volume
        FROM prices p
        JOIN stocks s ON p.ticker = s.ticker
        WHERE s.exchange IN ({placeholders})
          AND p.date > :last_date
        ORDER BY p.date, p.ticker
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()

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

    write_parquet(df, Market.US, "daily", root)

    dates = df["date"].cast(pl.Utf8)
    date_range = (dates.min(), dates.max())
    instrument_count = df["instrument_id"].n_unique()
    row_count = len(df)

    manifest.update_dataset_state("daily", row_count, date_range, instrument_count)

    return {"row_count": row_count, "date_range": date_range, "instrument_count": instrument_count}
```

- [ ] **Step 4: Run tests to confirm PASS**

```bash
uv run pytest tests/test_stocks_db_ingestor.py::test_ingest_us_daily_schema \
              tests/test_stocks_db_ingestor.py::test_ingest_us_daily_instrument_id_equals_ticker \
              tests/test_stocks_db_ingestor.py::test_ingest_us_daily_adj_factor_is_one \
              tests/test_stocks_db_ingestor.py::test_ingest_us_daily_incremental -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add trendspec/ingest/stocks_db_ingestor.py tests/test_stocks_db_ingestor.py
git commit -m "feat: add US daily ingestor for Synology stocks DB schema"
```

---

## Task 4: US Components Ingestor

**Files:**
- Modify: `trendspec/ingest/stocks_db_ingestor.py` (append)
- Modify: `tests/test_stocks_db_ingestor.py` (append)

Components schema in this DB: `constituent_changes(index_id, ticker, change_type, change_date)`
- `change_type` = `ADDED` or `REMOVED`
- `index_id` = `SP500`, `CSI800`, `HSI`

TrendSpec components schema: `(instrument_id, date, event, event_details)`
- `event` = `IPO` or `DELIST` (TrendSpec events — we use ADDED→IPO, REMOVED→DELIST as proxies)

We also query `MIN(date)` from `prices` per US ticker as a fallback IPO date for tickers not in SP500.

- [ ] **Step 1: Add test for US components**

Append to `tests/test_stocks_db_ingestor.py`:

```python
@pytest.fixture
def stocks_db_with_changes(stocks_db):
    """Add SP500 constituent changes to the fixture DB."""
    with stocks_db.connect() as conn:
        conn.execute(text("""
            INSERT INTO constituent_changes VALUES
            ('SP500', 'AAPL', 'ADDED', '2020-01-15'),
            ('SP500', 'MSFT', 'ADDED', '2019-06-01'),
            ('SP500', 'JPM', 'REMOVED', '2023-03-10')
        """))
        conn.commit()
    return stocks_db


def test_ingest_us_components_schema(stocks_db_with_changes, temp_root):
    """US components Parquet has correct columns."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_components
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    result = ingest_us_components(stocks_db_with_changes, manifest, temp_root)

    assert result["row_count"] > 0

    df = pl.read_parquet(f"{temp_root}/us/components/")
    assert set(df.columns) >= {"instrument_id", "date", "event", "event_details"}


def test_ingest_us_components_event_mapping(stocks_db_with_changes, temp_root):
    """ADDED maps to IPO, REMOVED maps to DELIST."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_components
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    ingest_us_components(stocks_db_with_changes, manifest, temp_root)

    df = pl.read_parquet(f"{temp_root}/us/components/")
    events = df["event"].unique().to_list()
    assert "IPO" in events
    assert "DELIST" in events
    # No raw ADDED/REMOVED values should survive
    assert "ADDED" not in events
    assert "REMOVED" not in events


def test_ingest_us_components_all_tickers_have_ipo(stocks_db_with_changes, temp_root):
    """Every US ticker in prices should have at least one IPO event."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_components
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    ingest_us_components(stocks_db_with_changes, manifest, temp_root)

    df = pl.read_parquet(f"{temp_root}/us/components/")
    ipos = df.filter(pl.col("event") == "IPO")["instrument_id"].unique().to_list()
    # All 3 tickers in prices must have an IPO event
    assert "AAPL" in ipos
    assert "MSFT" in ipos
    assert "JPM" in ipos
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
uv run pytest tests/test_stocks_db_ingestor.py::test_ingest_us_components_schema -v
```

Expected: `AttributeError: module has no attribute 'ingest_us_components'`

- [ ] **Step 3: Add `ingest_us_components` to `trendspec/ingest/stocks_db_ingestor.py`**

Append after the `ingest_us_daily` function:

```python
# =============================================================================
# US Components
# =============================================================================


def ingest_us_components(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """
    Ingest US component events from constituent_changes + prices.

    SP500 constituent_changes provides ADDED→IPO and REMOVED→DELIST events.
    All US tickers in prices also get an IPO event from their MIN(date) in prices,
    ensuring every ticker has an entry even if not in SP500.

    Args:
        engine: SQLAlchemy engine
        manifest: Manifest for sync tracking
        root: data_lake root directory
        full_sync: Ignored — components always rebuilt from scratch

    Returns:
        {"row_count": int, "date_range": (str, str), "instrument_count": int}
    """
    ex_placeholders = _exchange_placeholder(_US_EXCHANGES)
    ex_params = _exchange_params(_US_EXCHANGES)

    # 1. SP500 constituent changes (ADDED → IPO, REMOVED → DELIST)
    sql_changes = text(f"""
        SELECT c.ticker, c.change_date, c.change_type
        FROM constituent_changes c
        JOIN stocks s ON c.ticker = s.ticker
        WHERE c.index_id = 'SP500'
          AND s.exchange IN ({ex_placeholders})
        ORDER BY c.change_date
    """)

    with engine.connect() as conn:
        changes = conn.execute(sql_changes, ex_params).fetchall()

        # 2. MIN(date) per ticker from prices as IPO fallback
        sql_min = text(f"""
            SELECT p.ticker, MIN(p.date) as first_date
            FROM prices p
            JOIN stocks s ON p.ticker = s.ticker
            WHERE s.exchange IN ({ex_placeholders})
            GROUP BY p.ticker
        """)
        min_dates = conn.execute(sql_min, ex_params).fetchall()

    # Build IPO events from SP500 ADDED changes
    rows = []
    sp500_added: set[str] = set()

    for ticker, change_date, change_type in changes:
        event = "IPO" if change_type == "ADDED" else "DELIST"
        if change_type == "ADDED":
            sp500_added.add(ticker)
        rows.append({
            "instrument_id": ticker,
            "date": change_date,
            "event": event,
            "event_details": f"SP500 {change_type.lower()}",
        })

    # Add IPO events for tickers NOT already covered by SP500 ADDED
    for ticker, first_date in min_dates:
        if ticker not in sp500_added:
            rows.append({
                "instrument_id": ticker,
                "date": first_date,
                "event": "IPO",
                "event_details": "first price record",
            })

    if not rows:
        return {"row_count": 0, "date_range": ("", ""), "instrument_count": 0}

    df = pl.DataFrame(rows)
    df = df.with_columns(pl.col("date").cast(pl.Date))

    write_parquet(df, Market.US, "components", root)

    dates = df["date"].cast(pl.Utf8)
    date_range = (dates.min(), dates.max())
    instrument_count = df["instrument_id"].n_unique()
    row_count = len(df)

    manifest.update_dataset_state("components", row_count, date_range, instrument_count)

    return {"row_count": row_count, "date_range": date_range, "instrument_count": instrument_count}
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_stocks_db_ingestor.py::test_ingest_us_components_schema \
              tests/test_stocks_db_ingestor.py::test_ingest_us_components_event_mapping \
              tests/test_stocks_db_ingestor.py::test_ingest_us_components_all_tickers_have_ipo -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add trendspec/ingest/stocks_db_ingestor.py tests/test_stocks_db_ingestor.py
git commit -m "feat: add US components ingestor for Synology stocks DB"
```

---

## Task 5: US Sectors Ingestor

**Files:**
- Modify: `trendspec/ingest/stocks_db_ingestor.py` (append)
- Modify: `tests/test_stocks_db_ingestor.py` (append)

`stocks.gics_sector` and `gics_industry` are static (no historical date). We use `assign_date = date(2000, 1, 1)` as a constant.

- [ ] **Step 1: Add tests**

Append to `tests/test_stocks_db_ingestor.py`:

```python
def test_ingest_us_sectors_schema(stocks_db, temp_root):
    """US sectors Parquet has correct columns."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_sectors
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    result = ingest_us_sectors(stocks_db, manifest, temp_root)

    assert result["instrument_count"] == 3  # AAPL, MSFT, JPM

    df = pl.read_parquet(f"{temp_root}/us/sectors/")
    assert set(df.columns) >= {"instrument_id", "date", "sector", "sector_name"}


def test_ingest_us_sectors_static_date(stocks_db, temp_root):
    """All sector rows have assign_date = 2000-01-01 (no historical changes in source)."""
    from trendspec.ingest.stocks_db_ingestor import ingest_us_sectors
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.US, temp_root)
    ingest_us_sectors(stocks_db, manifest, temp_root)

    df = pl.read_parquet(f"{temp_root}/us/sectors/")
    dates = df["date"].unique().to_list()
    assert len(dates) == 1
    assert str(dates[0]) == "2000-01-01"
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
uv run pytest tests/test_stocks_db_ingestor.py::test_ingest_us_sectors_schema -v
```

Expected: `AttributeError: module has no attribute 'ingest_us_sectors'`

- [ ] **Step 3: Add `ingest_us_sectors` to `trendspec/ingest/stocks_db_ingestor.py`**

```python
# =============================================================================
# US Sectors
# =============================================================================


def ingest_us_sectors(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """
    Ingest US sector assignments from stocks.gics_sector.

    The source has no historical sector changes (static snapshot).
    All rows get assign_date = 2000-01-01 as a sentinel.

    sector     = gics_sector   (e.g. "Information Technology")
    sector_name = gics_industry (e.g. "Technology Hardware")

    Args:
        engine: SQLAlchemy engine
        manifest: Manifest for sync tracking
        root: data_lake root directory
        full_sync: Ignored — sectors always rebuilt from static source

    Returns:
        {"row_count": int, "date_range": (str, str), "instrument_count": int}
    """
    ex_placeholders = _exchange_placeholder(_US_EXCHANGES)
    ex_params = _exchange_params(_US_EXCHANGES)

    sql = text(f"""
        SELECT ticker, gics_sector, gics_industry
        FROM stocks
        WHERE exchange IN ({ex_placeholders})
          AND gics_sector IS NOT NULL
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, ex_params).fetchall()

    if not rows:
        return {"row_count": 0, "date_range": ("", ""), "instrument_count": 0}

    static_date = date(2000, 1, 1)
    df = pl.DataFrame(
        [
            {
                "instrument_id": ticker,
                "date": static_date,
                "sector": gics_sector or "",
                "sector_name": gics_industry or "",
            }
            for ticker, gics_sector, gics_industry in rows
        ]
    )
    df = df.with_columns(pl.col("date").cast(pl.Date))

    write_parquet(df, Market.US, "sectors", root)

    date_range = ("2000-01-01", "2000-01-01")
    instrument_count = df["instrument_id"].n_unique()
    row_count = len(df)

    manifest.update_dataset_state("sectors", row_count, date_range, instrument_count)

    return {"row_count": row_count, "date_range": date_range, "instrument_count": instrument_count}
```

- [ ] **Step 4: Run all US tests**

```bash
uv run pytest tests/test_stocks_db_ingestor.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add trendspec/ingest/stocks_db_ingestor.py tests/test_stocks_db_ingestor.py
git commit -m "feat: add US sectors ingestor for Synology stocks DB"
```

---

## Task 6: Fix CLI — US Ingest Commands

**Files:**
- Modify: `trendspec/cli/ingest_cmd.py`

The current CLI imports non-existent `CNAIngestor` / `USIngestor` classes. Rewrite to call standalone functions.

- [ ] **Step 1: Rewrite `ingest_daily` command in `trendspec/cli/ingest_cmd.py`**

Replace the `ingest_daily` function body (approximately lines 45-105):

```python
@app.command("daily")
def ingest_daily(
    market: str = typer.Option(
        "us",
        "--market",
        help="市场代码 (cn, us)",
    ),
    since: str = typer.Option(
        "2000-01-01",
        "--since",
        help="起始日期 YYYY-MM-DD",
    ),
    incremental: bool = typer.Option(
        True,
        "--incremental/--full",
        help="增量同步 (默认) 或全量同步",
    ),
) -> None:
    """
    从群辉 stocks DB 导入 OHLCV 日线数据.

    示例:
        trendspec ingest daily --market us --since 2024-01-01
        trendspec ingest daily --market cn --since 2020-01-01
    """
    from datetime import date as date_type
    from trendspec.data.markets import Market
    from trendspec.config.settings import get_settings
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.mariadb_client import create_engine_from_settings
    from trendspec.ingest.stocks_db_ingestor import ingest_us_daily, ingest_cn_daily

    try:
        date_type.fromisoformat(since)
    except ValueError:
        console.print("[red]日期格式错误，请使用 YYYY-MM-DD 格式[/red]")
        raise typer.Exit(1)

    market_enum = Market(market.upper())
    settings = get_settings()
    engine = create_engine_from_settings(settings.db)
    root = settings.data_lake.data_lake_root
    manifest = Manifest(market_enum, root)
    full_sync = not incremental

    console.print(f"[cyan]导入 {market} 日线数据...[/cyan]")

    try:
        if market_enum == Market.US:
            result = ingest_us_daily(engine, manifest, root, full_sync=full_sync)
        elif market_enum == Market.CN:
            result = ingest_cn_daily(engine, manifest, root, full_sync=full_sync)
        else:
            console.print(f"[red]不支持的市场: {market}[/red]")
            raise typer.Exit(1)

        console.print(f"[green]完成: {result['row_count']} 行, {result['instrument_count']} 只股票[/green]")
    except Exception as e:
        console.print(f"[red]导入失败: {e}[/red]")
        raise typer.Exit(1)
```

- [ ] **Step 2: Rewrite `ingest_components` and `ingest_sectors` similarly**

```python
@app.command("components")
def ingest_components(
    market: str = typer.Option("us", "--market", help="市场代码 (cn, us)"),
) -> None:
    """导入成分变动数据."""
    from trendspec.data.markets import Market
    from trendspec.config.settings import get_settings
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.mariadb_client import create_engine_from_settings
    from trendspec.ingest.stocks_db_ingestor import ingest_us_components, ingest_cn_components

    market_enum = Market(market.upper())
    settings = get_settings()
    engine = create_engine_from_settings(settings.db)
    root = settings.data_lake.data_lake_root
    manifest = Manifest(market_enum, root)

    console.print(f"[cyan]导入 {market} 成分数据...[/cyan]")
    try:
        if market_enum == Market.US:
            result = ingest_us_components(engine, manifest, root)
        elif market_enum == Market.CN:
            result = ingest_cn_components(engine, manifest, root)
        else:
            console.print(f"[red]不支持的市场: {market}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]完成: {result['row_count']} 行[/green]")
    except Exception as e:
        console.print(f"[red]导入失败: {e}[/red]")
        raise typer.Exit(1)


@app.command("sectors")
def ingest_sectors(
    market: str = typer.Option("us", "--market", help="市场代码 (cn, us)"),
) -> None:
    """导入行业分类数据."""
    from trendspec.data.markets import Market
    from trendspec.config.settings import get_settings
    from trendspec.ingest.manifest import Manifest
    from trendspec.ingest.mariadb_client import create_engine_from_settings
    from trendspec.ingest.stocks_db_ingestor import ingest_us_sectors, ingest_cn_sectors

    market_enum = Market(market.upper())
    settings = get_settings()
    engine = create_engine_from_settings(settings.db)
    root = settings.data_lake.data_lake_root
    manifest = Manifest(market_enum, root)

    console.print(f"[cyan]导入 {market} 行业数据...[/cyan]")
    try:
        if market_enum == Market.US:
            result = ingest_us_sectors(engine, manifest, root)
        elif market_enum == Market.CN:
            result = ingest_cn_sectors(engine, manifest, root)
        else:
            console.print(f"[red]不支持的市场: {market}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]完成: {result['row_count']} 行[/green]")
    except Exception as e:
        console.print(f"[red]导入失败: {e}[/red]")
        raise typer.Exit(1)
```

- [ ] **Step 3: Check if `mariadb_client.create_engine_from_settings` exists**

```bash
grep -n "create_engine_from_settings\|def create_engine" trendspec/ingest/mariadb_client.py
```

If the function doesn't exist, add it to `trendspec/ingest/mariadb_client.py`:

```python
def create_engine_from_settings(db_settings) -> Engine:
    """Create SQLAlchemy engine from DatabaseSettings."""
    from sqlalchemy import create_engine
    return create_engine(db_settings.connection_url)
```

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/ -x -q 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 5: Test CLI help works**

```bash
uv run trendspec ingest daily --help
uv run trendspec ingest components --help
uv run trendspec ingest sectors --help
```

Expected: each prints help without ImportError.

- [ ] **Step 6: Commit**

```bash
git add trendspec/cli/ingest_cmd.py trendspec/ingest/mariadb_client.py
git commit -m "fix: rewrite ingest CLI to call standalone functions, add US market routing"
```

---

## Task 7: CN Daily Ingestor

**Files:**
- Modify: `trendspec/ingest/stocks_db_ingestor.py` (append)
- Modify: `tests/test_stocks_db_ingestor.py` (append)

CN stocks: `exchange IN ('SSE', 'SH', 'SZSE', 'SZ')`
`instrument_id`: `SH{ticker}` for SSE/SH, `SZ{ticker}` for SZSE/SZ

- [ ] **Step 1: Add CN fixture and tests**

Append to `tests/test_stocks_db_ingestor.py`:

```python
@pytest.fixture
def stocks_db_cn(stocks_db):
    """Add CN stocks and prices to the fixture DB."""
    with stocks_db.connect() as conn:
        conn.execute(text("""
            INSERT INTO stocks VALUES
            ('600000', 'SSE', 'Financials', 'Banks', 1),
            ('000001', 'SZSE', 'Financials', 'Banks', 1),
            ('600036', 'SH', 'Financials', 'Banks', 1)
        """))
        conn.execute(text("""
            INSERT INTO prices VALUES
            ('600000', '2024-01-02', 10.0, 10.5, 9.8, 10.2, 1000000),
            ('600000', '2024-01-03', 10.2, 10.8, 10.0, 10.5, 1100000),
            ('000001', '2024-01-02', 20.0, 20.5, 19.8, 20.2, 500000),
            ('000001', '2024-01-03', 20.2, 20.8, 20.0, 20.5, 550000),
            ('600036', '2024-01-02', 15.0, 15.5, 14.8, 15.2, 800000),
            ('600036', '2024-01-03', 15.2, 15.8, 15.0, 15.5, 850000)
        """))
        conn.commit()
    return stocks_db


def test_ingest_cn_daily_instrument_id_prefix(stocks_db_cn, temp_root):
    """CN instrument_id has SH/SZ prefix based on exchange."""
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_daily
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.CN, temp_root)
    result = ingest_cn_daily(stocks_db_cn, manifest, temp_root)

    assert result["row_count"] == 6
    assert result["instrument_count"] == 3

    df = pl.read_parquet(f"{temp_root}/cn/daily/")
    ids = sorted(df["instrument_id"].unique().to_list())
    assert "SH600000" in ids
    assert "SZ000001" in ids
    assert "SH600036" in ids  # SH exchange → SH prefix


def test_ingest_cn_daily_adj_factor_is_one(stocks_db_cn, temp_root):
    """CN adj_factor = 1.0 (Tushare prices already backward-adjusted)."""
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_daily
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.CN, temp_root)
    ingest_cn_daily(stocks_db_cn, manifest, temp_root)

    df = pl.read_parquet(f"{temp_root}/cn/daily/")
    assert df["adj_factor"].unique().to_list() == [1.0]
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
uv run pytest tests/test_stocks_db_ingestor.py::test_ingest_cn_daily_instrument_id_prefix -v
```

Expected: `AttributeError: module has no attribute 'ingest_cn_daily'`

- [ ] **Step 3: Add `ingest_cn_daily` to `trendspec/ingest/stocks_db_ingestor.py`**

```python
# =============================================================================
# CN Daily
# =============================================================================

_CN_EXCHANGE_TO_PREFIX: Final[dict[str, str]] = {
    "SSE": "SH",
    "SH":  "SH",
    "SZSE": "SZ",
    "SZ":  "SZ",
}


def _derive_cn_instrument_id(ticker: str, exchange: str) -> str:
    """
    Build CN instrument_id from ticker + exchange.

    SSE/SH  → SH{ticker}
    SZSE/SZ → SZ{ticker}
    """
    prefix = _CN_EXCHANGE_TO_PREFIX.get(exchange.upper())
    if prefix is None:
        raise ValueError(f"Unknown CN exchange: {exchange!r} for ticker {ticker!r}")
    return f"{prefix}{ticker}"


def ingest_cn_daily(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """
    Ingest CN (A-share) daily OHLCV from prices + stocks tables.

    instrument_id = SH{ticker} for SSE/SH, SZ{ticker} for SZSE/SZ.
    adj_factor = 1.0 (prices are Tushare backward-adjusted).

    Args:
        engine: SQLAlchemy engine
        manifest: Manifest for sync tracking
        root: data_lake root directory
        full_sync: If True, pull all history

    Returns:
        {"row_count": int, "date_range": (str, str), "instrument_count": int}
    """
    last_date = "1970-01-01" if full_sync else _get_last_synced_date(manifest, "daily")

    ex_placeholders = _exchange_placeholder(_CN_EXCHANGES)
    ex_params = _exchange_params(_CN_EXCHANGES)
    ex_params["last_date"] = last_date

    sql = text(f"""
        SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, p.volume,
               s.exchange
        FROM prices p
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

    # Build instrument_id from ticker + exchange
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

    write_parquet(df, Market.CN, "daily", root)

    dates = df["date"].cast(pl.Utf8)
    date_range = (dates.min(), dates.max())
    instrument_count = df["instrument_id"].n_unique()
    row_count = len(df)

    manifest.update_dataset_state("daily", row_count, date_range, instrument_count)

    return {"row_count": row_count, "date_range": date_range, "instrument_count": instrument_count}
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_stocks_db_ingestor.py::test_ingest_cn_daily_instrument_id_prefix \
              tests/test_stocks_db_ingestor.py::test_ingest_cn_daily_adj_factor_is_one -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add trendspec/ingest/stocks_db_ingestor.py tests/test_stocks_db_ingestor.py
git commit -m "feat: add CN daily ingestor for Synology stocks DB"
```

---

## Task 8: CN Components + CN Sectors

**Files:**
- Modify: `trendspec/ingest/stocks_db_ingestor.py` (append)
- Modify: `tests/test_stocks_db_ingestor.py` (append)

- [ ] **Step 1: Add CN components + sectors tests**

Append to `tests/test_stocks_db_ingestor.py`:

```python
def test_ingest_cn_components_has_ipo_events(stocks_db_cn, temp_root):
    """Every CN ticker gets an IPO event from MIN(date) in prices."""
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_components
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.CN, temp_root)
    result = ingest_cn_components(stocks_db_cn, manifest, temp_root)

    assert result["row_count"] == 3  # one IPO per ticker

    df = pl.read_parquet(f"{temp_root}/cn/components/")
    assert set(df["event"].unique().to_list()) == {"IPO"}
    ids = df["instrument_id"].unique().to_list()
    assert "SH600000" in ids
    assert "SZ000001" in ids


def test_ingest_cn_sectors_maps_to_correct_market(stocks_db_cn, temp_root):
    """CN sectors use CN instrument_id format (SH/SZ prefix)."""
    from trendspec.ingest.stocks_db_ingestor import ingest_cn_sectors
    from trendspec.ingest.manifest import Manifest
    from trendspec.data.markets import Market

    manifest = Manifest(Market.CN, temp_root)
    result = ingest_cn_sectors(stocks_db_cn, manifest, temp_root)

    assert result["instrument_count"] == 3

    df = pl.read_parquet(f"{temp_root}/cn/sectors/")
    ids = df["instrument_id"].unique().to_list()
    assert all(id_.startswith(("SH", "SZ")) for id_ in ids)
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
uv run pytest tests/test_stocks_db_ingestor.py::test_ingest_cn_components_has_ipo_events -v
```

Expected: `AttributeError: module has no attribute 'ingest_cn_components'`

- [ ] **Step 3: Add CN components + sectors functions**

Append to `trendspec/ingest/stocks_db_ingestor.py`:

```python
# =============================================================================
# CN Components
# =============================================================================


def ingest_cn_components(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """
    Ingest CN component events.

    Source has CSI800 constituent changes (ADDED/REMOVED).
    All CN tickers in prices also get an IPO event from MIN(date).

    Args:
        engine: SQLAlchemy engine
        manifest: Manifest for sync tracking
        root: data_lake root directory
        full_sync: Ignored — always rebuilt

    Returns:
        {"row_count": int, "date_range": (str, str), "instrument_count": int}
    """
    ex_placeholders = _exchange_placeholder(_CN_EXCHANGES)
    ex_params = _exchange_params(_CN_EXCHANGES)

    # CSI800 constituent changes
    sql_changes = text(f"""
        SELECT c.ticker, c.change_date, c.change_type, s.exchange
        FROM constituent_changes c
        JOIN stocks s ON c.ticker = s.ticker
        WHERE c.index_id = 'CSI800'
          AND s.exchange IN ({ex_placeholders})
        ORDER BY c.change_date
    """)

    # MIN(date) per ticker from prices
    sql_min = text(f"""
        SELECT p.ticker, MIN(p.date) as first_date, s.exchange
        FROM prices p
        JOIN stocks s ON p.ticker = s.ticker
        WHERE s.exchange IN ({ex_placeholders})
        GROUP BY p.ticker, s.exchange
    """)

    with engine.connect() as conn:
        changes = conn.execute(sql_changes, ex_params).fetchall()
        min_dates = conn.execute(sql_min, ex_params).fetchall()

    rows = []
    csi800_added: set[str] = set()  # tickers already covered

    for ticker, change_date, change_type, exchange in changes:
        instrument_id = _derive_cn_instrument_id(ticker, exchange)
        event = "IPO" if change_type == "ADDED" else "DELIST"
        if change_type == "ADDED":
            csi800_added.add(ticker)
        rows.append({
            "instrument_id": instrument_id,
            "date": change_date,
            "event": event,
            "event_details": f"CSI800 {change_type.lower()}",
        })

    for ticker, first_date, exchange in min_dates:
        if ticker not in csi800_added:
            instrument_id = _derive_cn_instrument_id(ticker, exchange)
            rows.append({
                "instrument_id": instrument_id,
                "date": first_date,
                "event": "IPO",
                "event_details": "first price record",
            })

    if not rows:
        return {"row_count": 0, "date_range": ("", ""), "instrument_count": 0}

    df = pl.DataFrame(rows)
    df = df.with_columns(pl.col("date").cast(pl.Date))

    write_parquet(df, Market.CN, "components", root)

    dates = df["date"].cast(pl.Utf8)
    date_range = (dates.min(), dates.max())
    instrument_count = df["instrument_id"].n_unique()
    row_count = len(df)

    manifest.update_dataset_state("components", row_count, date_range, instrument_count)

    return {"row_count": row_count, "date_range": date_range, "instrument_count": instrument_count}


# =============================================================================
# CN Sectors
# =============================================================================


def ingest_cn_sectors(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """
    Ingest CN sector assignments from stocks.gics_sector.

    Note: The source DB uses GICS classification (not Shenwan/申万).
    assign_date = 2000-01-01 (static, no historical changes in source).

    Args:
        engine: SQLAlchemy engine
        manifest: Manifest for sync tracking
        root: data_lake root directory
        full_sync: Ignored — always rebuilt

    Returns:
        {"row_count": int, "date_range": (str, str), "instrument_count": int}
    """
    ex_placeholders = _exchange_placeholder(_CN_EXCHANGES)
    ex_params = _exchange_params(_CN_EXCHANGES)

    sql = text(f"""
        SELECT ticker, exchange, gics_sector, gics_industry
        FROM stocks
        WHERE exchange IN ({ex_placeholders})
          AND gics_sector IS NOT NULL
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, ex_params).fetchall()

    if not rows:
        return {"row_count": 0, "date_range": ("", ""), "instrument_count": 0}

    static_date = date(2000, 1, 1)
    df = pl.DataFrame(
        [
            {
                "instrument_id": _derive_cn_instrument_id(ticker, exchange),
                "date": static_date,
                "sector": gics_sector or "",
                "sector_name": gics_industry or "",
            }
            for ticker, exchange, gics_sector, gics_industry in rows
        ]
    )
    df = df.with_columns(pl.col("date").cast(pl.Date))

    write_parquet(df, Market.CN, "sectors", root)

    date_range = ("2000-01-01", "2000-01-01")
    instrument_count = df["instrument_id"].n_unique()
    row_count = len(df)

    manifest.update_dataset_state("sectors", row_count, date_range, instrument_count)

    return {"row_count": row_count, "date_range": date_range, "instrument_count": instrument_count}
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/test_stocks_db_ingestor.py -v
uv run pytest tests/ -x -q 2>&1 | tail -10
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add trendspec/ingest/stocks_db_ingestor.py tests/test_stocks_db_ingestor.py
git commit -m "feat: add CN components and sectors ingestors for Synology stocks DB"
```

---

## Task 9: End-to-End Smoke Test Against Real Synology DB

*This task uses the real Synology DB. Requires `.env` configured.*

- [ ] **Step 1: Test US ingest (small date range)**

```bash
uv run trendspec ingest daily --market us --since 2024-01-01
```

Expected output:
```
导入 us 日线数据...
完成: XXXX 行, XXX 只股票
```

- [ ] **Step 2: Verify Parquet written correctly**

```bash
uv run python -c "
import polars as pl
df = pl.read_parquet('./data_lake/us/daily/')
print('shape:', df.shape)
print('schema:', df.schema)
print(df.head(3))
print('date range:', df['date'].min(), '-', df['date'].max())
print('tickers:', df['instrument_id'].n_unique())
"
```

Expected: DataFrame with correct columns, ~500 tickers, dates in 2024.

- [ ] **Step 3: Test US components + sectors**

```bash
uv run trendspec ingest components --market us
uv run trendspec ingest sectors --market us
```

- [ ] **Step 4: Test CN ingest (small date range)**

```bash
uv run trendspec ingest daily --market cn --since 2024-01-01
```

- [ ] **Step 5: Verify CN instrument_id format**

```bash
uv run python -c "
import polars as pl
df = pl.read_parquet('./data_lake/cn/daily/')
ids = df['instrument_id'].unique().head(10).to_list()
print('Sample IDs:', ids)
# All should start with SH or SZ
assert all(id_.startswith(('SH','SZ')) for id_ in ids), 'Bad instrument_id format!'
print('OK — all IDs have SH/SZ prefix')
"
```

- [ ] **Step 6: Final commit**

```bash
git add .
git commit -m "docs: add end-to-end smoke test verification notes"
```

---

## Self-Review

### Spec Coverage

| Requirement | Task |
|-------------|------|
| Rename CN_A → CN | Task 1 |
| Fix broken CLI (CNAIngestor not found) | Task 6 |
| Allow root DB user in dev | Task 2 |
| US daily ingest | Task 3 |
| US components ingest | Task 4 |
| US sectors ingest | Task 5 |
| CN daily ingest | Task 7 |
| CN components ingest | Task 8 |
| CN sectors ingest | Task 8 |
| End-to-end smoke test | Task 9 |

### Known Limitation

CN sectors use GICS classification from the source DB (not Shenwan 申万). TrendSpec's existing code references `"Shenwan_L1"` sector classification in `markets.py`. The ingestor stores GICS values — this is functionally correct for backtesting but the sector names won't match any existing Shenwan-based factor calculations.
