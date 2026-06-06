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

from datetime import date, timedelta
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
    """Build SQLAlchemy :ex0, :ex1, ... placeholder string."""
    return ", ".join(f":ex{i}" for i in range(len(exchanges)))


def _exchange_params(exchanges: tuple[str, ...]) -> dict[str, str]:
    """Build SQLAlchemy parameter dict for exchange IN clause."""
    return {f"ex{i}": ex for i, ex in enumerate(exchanges)}


def _get_last_synced_date(manifest: Manifest, dataset: str) -> str:
    """Return last synced end date from manifest, or '1970-01-01' if never synced."""
    state = manifest.get_dataset_state(dataset)
    if state is None:
        return "1970-01-01"
    date_range = state.get("date_range", {})
    return date_range.get("end", "1970-01-01")


def _resolve_start_exclusive(
    manifest: Manifest, dataset: str, full_sync: bool, since: str | None
) -> str:
    """
    Resolve the exclusive lower-bound date for the `p.date > :last_date` filter.

    Precedence: explicit `since` > full_sync > manifest last synced date.
    `since` is treated as an inclusive start, so it maps to the day before
    (exclusive boundary). Returns 'YYYY-MM-DD'.
    """
    if since is not None:
        return (date.fromisoformat(since) - timedelta(days=1)).isoformat()
    if full_sync:
        return "1970-01-01"
    return _get_last_synced_date(manifest, dataset)


_FETCH_CHUNK: Final[int] = 50_000


def _fetch_df_with_progress(
    engine: Engine,
    data_sql,
    params: dict,
    schema: list[str],
    label: str,
) -> pl.DataFrame:
    """
    Stream a large query into a Polars DataFrame with a Rich progress bar.

    Uses a server-side cursor (stream_results) and pulls rows in chunks, so a
    multi-million-row pull shows live row counts within ~1s of the first chunk
    instead of a silent hang. No COUNT(*) up front (that would block first).
    """
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True).execute(data_sql, params)
        rows: list = []
        progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]{label}[/cyan]"),
            BarColumn(),
            TextColumn("{task.completed:,} 行"),
            TimeElapsedColumn(),
        )
        with progress:
            task = progress.add_task("fetch", total=None)
            for chunk in result.partitions(_FETCH_CHUNK):
                rows.extend(chunk)
                progress.update(task, advance=len(chunk))

    return pl.DataFrame(rows, schema=schema, orient="row")


# =============================================================================
# US Daily
# =============================================================================


def ingest_us_daily(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
    since: str | None = None,
) -> dict:
    """
    Ingest US daily OHLCV from prices + stocks tables.

    Args:
        engine: SQLAlchemy engine connected to Synology stocks DB
        manifest: Manifest for tracking sync state
        root: data_lake root directory
        full_sync: If True, pull all history ignoring manifest
        since: Inclusive start date 'YYYY-MM-DD'; overrides manifest/full_sync

    Returns:
        {"row_count": int, "date_range": (str, str), "instrument_count": int}
    """
    last_date = _resolve_start_exclusive(manifest, "daily", full_sync, since)

    params = {"last_date": last_date}

    sql = text("""
        SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, p.volume
        FROM prices p
        JOIN (
            SELECT DISTINCT ticker FROM index_constituents
            WHERE index_id IN ('SP500', 'RUSSELL1000')
        ) AS us ON p.ticker = us.ticker
        WHERE p.date > :last_date
    """)

    df = _fetch_df_with_progress(
        engine, sql, params,
        schema=["ticker", "date", "open", "high", "low", "close", "volume"],
        label="拉取 us 日线",
    )

    if df.is_empty():
        return {"row_count": 0, "date_range": ("", ""), "instrument_count": 0}

    df = df.with_columns(pl.col("date").cast(pl.Date))
    df = df.with_columns([
        pl.col("ticker").alias("instrument_id"),
        pl.lit(1.0).alias("adj_factor"),
    ])

    write_parquet(df, Market.US, "daily", root, overwrite=full_sync, show_progress=True)

    dates = df["date"].cast(pl.Utf8)
    date_range = (dates.min(), dates.max())
    instrument_count = df["instrument_id"].n_unique()
    row_count = len(df)

    manifest.update_dataset_state("daily", row_count, date_range, instrument_count)

    return {"row_count": row_count, "date_range": date_range, "instrument_count": instrument_count}


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

    Returns:
        {"row_count": int, "date_range": (str, str), "instrument_count": int}
    """
    sql_changes = text("""
        SELECT c.ticker, c.change_date, c.change_type
        FROM constituent_changes c
        WHERE c.index_id = 'SP500'
        ORDER BY c.change_date
    """)

    sql_min = text("""
        SELECT p.ticker, MIN(p.date) as first_date
        FROM prices p
        JOIN (
            SELECT DISTINCT ticker FROM index_constituents
            WHERE index_id IN ('SP500', 'RUSSELL1000')
        ) AS us ON p.ticker = us.ticker
        GROUP BY p.ticker
    """)

    with engine.connect() as conn:
        changes = conn.execute(sql_changes).fetchall()
        min_dates = conn.execute(sql_min).fetchall()

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

    Static snapshot — no historical sector changes in source.
    All rows get assign_date = 2000-01-01 as a sentinel.

    sector      = gics_sector   (e.g. "Information Technology")
    sector_name = gics_industry (e.g. "Technology Hardware")

    Returns:
        {"row_count": int, "date_range": (str, str), "instrument_count": int}
    """
    sql = text("""
        SELECT s.ticker, s.gics_sector, s.gics_industry
        FROM stocks s
        JOIN (
            SELECT DISTINCT ticker FROM index_constituents
            WHERE index_id IN ('SP500', 'RUSSELL1000')
        ) AS us ON s.ticker = us.ticker
        WHERE s.gics_sector IS NOT NULL
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()

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


# =============================================================================
# CN Daily
# =============================================================================

_CN_EXCHANGE_TO_PREFIX: Final[dict[str, str]] = {
    "SSE": "SH",
    "SH": "SH",
    "SZSE": "SZ",
    "SZ": "SZ",
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
    since: str | None = None,
) -> dict:
    """
    Ingest CN (A-share) daily OHLCV from prices + stocks tables.

    instrument_id = SH{ticker} for SSE/SH, SZ{ticker} for SZSE/SZ.
    adj_factor = 1.0 (prices are Tushare backward-adjusted).

    Args:
        since: Inclusive start date 'YYYY-MM-DD'; overrides manifest/full_sync

    Returns:
        {"row_count": int, "date_range": (str, str), "instrument_count": int}
    """
    last_date = _resolve_start_exclusive(manifest, "daily", full_sync, since)

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
    """)

    df = _fetch_df_with_progress(
        engine, sql, ex_params,
        schema=["ticker", "date", "open", "high", "low", "close", "volume", "exchange"],
        label="拉取 cn 日线",
    )

    if df.is_empty():
        return {"row_count": 0, "date_range": ("", ""), "instrument_count": 0}

    df = df.with_columns(pl.col("date").cast(pl.Date))

    df = df.with_columns(
        (
            pl.col("exchange").str.to_uppercase().replace_strict(_CN_EXCHANGE_TO_PREFIX)
            + pl.col("ticker")
        ).alias("instrument_id")
    )
    df = df.with_columns(pl.lit(1.0).alias("adj_factor"))
    df = df.drop("exchange")

    write_parquet(df, Market.CN, "daily", root, overwrite=full_sync, show_progress=True)

    dates = df["date"].cast(pl.Utf8)
    date_range = (dates.min(), dates.max())
    instrument_count = df["instrument_id"].n_unique()
    row_count = len(df)

    manifest.update_dataset_state("daily", row_count, date_range, instrument_count)

    return {"row_count": row_count, "date_range": date_range, "instrument_count": instrument_count}


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

    Returns:
        {"row_count": int, "date_range": (str, str), "instrument_count": int}
    """
    sql_changes = text("""
        SELECT c.ticker, c.change_date, c.change_type, s.exchange
        FROM constituent_changes c
        JOIN stocks s ON c.ticker = s.ticker
        WHERE c.index_id = 'CSI800'
        ORDER BY c.change_date
    """)

    sql_min = text("""
        SELECT p.ticker, MIN(p.date) as first_date, s.exchange
        FROM prices p
        JOIN stocks s ON p.ticker = s.ticker
        JOIN (
            SELECT DISTINCT ticker FROM index_constituents
            WHERE index_id = 'CSI800'
        ) AS cn ON p.ticker = cn.ticker
        GROUP BY p.ticker, s.exchange
    """)

    with engine.connect() as conn:
        changes = conn.execute(sql_changes).fetchall()
        min_dates = conn.execute(sql_min).fetchall()

    rows = []
    csi800_added: set[str] = set()

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

    Note: Source DB uses GICS classification, not Shenwan/申万.
    assign_date = 2000-01-01 (static — no historical changes in source).

    Returns:
        {"row_count": int, "date_range": (str, str), "instrument_count": int}
    """
    sql = text("""
        SELECT s.ticker, s.exchange, s.gics_sector, s.gics_industry
        FROM stocks s
        JOIN (
            SELECT DISTINCT ticker FROM index_constituents
            WHERE index_id = 'CSI800'
        ) AS cn ON s.ticker = cn.ticker
        WHERE s.gics_sector IS NOT NULL
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()

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


# =============================================================================
# US Indices
# =============================================================================


def ingest_us_indices(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """Ingest US index prices (SP500 + sector ETFs) from index_prices table."""
    sql = text("SELECT date, index_id, close FROM index_prices ORDER BY index_id, date")
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    if not rows:
        return {"row_count": 0, "date_range": None, "instrument_count": 0}
    df = pl.DataFrame(
        rows,
        schema={"date": pl.Date, "index_id": pl.String, "close": pl.Float64},
        orient="row",
    ).rename({"index_id": "instrument_id"})
    write_parquet(df, Market.US, "indices", root)
    dates = df["date"]
    index_ids = df["instrument_id"].unique()
    manifest.update_dataset_state(
        "indices",
        row_count=len(df),
        date_range=(dates.min(), dates.max()),
        instrument_count=len(index_ids),
    )
    return {"row_count": len(df), "date_range": (dates.min(), dates.max()), "instrument_count": len(index_ids)}


# =============================================================================
# CN Indices
# =============================================================================


def ingest_cn_indices(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """Ingest CN index prices (CSI800) from index_prices table."""
    sql = text("SELECT date, index_id, close FROM index_prices WHERE index_id = 'CSI800' ORDER BY date")
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    if not rows:
        return {"row_count": 0, "date_range": None, "instrument_count": 0}
    df = pl.DataFrame(
        rows,
        schema={"date": pl.Date, "index_id": pl.String, "close": pl.Float64},
        orient="row",
    ).rename({"index_id": "instrument_id"})
    write_parquet(df, Market.CN, "indices", root)
    dates = df["date"]
    manifest.update_dataset_state(
        "indices",
        row_count=len(df),
        date_range=(dates.min(), dates.max()),
        instrument_count=1,
    )
    return {"row_count": len(df), "date_range": (dates.min(), dates.max()), "instrument_count": 1}


# =============================================================================
# US Weekly
# =============================================================================


def ingest_us_weekly(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """Ingest US weekly OHLCV from prices_weekly + index_constituents.

    Source table: prices_weekly (id, ticker, date, open, high, low, close, volume, created_at)
    date = week ending date as stored by the Synology sync script.
    """
    last_date = "1970-01-01" if full_sync else _get_last_synced_date(manifest, "weekly")

    sql = text("""
        SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, p.volume
        FROM prices_weekly p
        JOIN (
            SELECT DISTINCT ticker COLLATE utf8mb4_unicode_ci AS ticker
            FROM index_constituents
            WHERE index_id IN ('SP500', 'RUSSELL1000')
        ) AS us ON p.ticker COLLATE utf8mb4_unicode_ci = us.ticker
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


# =============================================================================
# US Intraday (1h)
# =============================================================================


def _get_last_synced_datetime(manifest: Manifest, dataset: str) -> str:
    """Return last synced end datetime, or '1970-01-01 00:00:00' if never."""
    state = manifest.get_dataset_state(dataset)
    if state is None:
        return "1970-01-01 00:00:00"
    return state.get("date_range", {}).get("end", "1970-01-01 00:00:00")


def ingest_us_intraday(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """
    Ingest US 1h OHLCV from prices_intraday (raw/unadjusted).

    全部 ticker（无 universe 过滤），instrument_id = ticker。
    PK (instrument_id, datetime)；按 datetime 去重，date 列供按年分区。

    Returns:
        {"row_count": int, "date_range": (str, str), "instrument_count": int}
    """
    last_dt = "1970-01-01 00:00:00" if full_sync else _get_last_synced_datetime(
        manifest, "intraday"
    )
    params = {"last_dt": last_dt}

    sql = text("""
        SELECT ticker, datetime, open, high, low, close, volume
        FROM prices_intraday
        WHERE `interval` = '1h' AND datetime > :last_dt
    """)

    df = _fetch_df_with_progress(
        engine, sql, params,
        schema=["ticker", "datetime", "open", "high", "low", "close", "volume"],
        label="拉取 us 1h",
    )

    if df.is_empty():
        return {"row_count": 0, "date_range": ("", ""), "instrument_count": 0}

    # MariaDB 返回 datetime，SQLite 返回字符串；按 dtype 处理
    if df["datetime"].dtype == pl.String:
        df = df.with_columns(pl.col("datetime").str.to_datetime())
    else:
        df = df.with_columns(pl.col("datetime").cast(pl.Datetime))

    df = df.with_columns([
        pl.col("ticker").alias("instrument_id"),
    ])
    df = df.with_columns(pl.col("datetime").dt.date().alias("date"))

    write_parquet(
        df, Market.US, "intraday", root,
        overwrite=full_sync, show_progress=True,
        dedup_keys=["instrument_id", "datetime"],
    )

    dts = df["datetime"].cast(pl.Utf8)
    date_range = (dts.min(), dts.max())
    instrument_count = df["instrument_id"].n_unique()
    row_count = len(df)

    manifest.update_dataset_state("intraday", row_count, date_range, instrument_count)

    return {"row_count": row_count, "date_range": date_range,
            "instrument_count": instrument_count}


# =============================================================================
# CN Weekly
# =============================================================================


def ingest_cn_weekly(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """Ingest CN weekly OHLCV from prices_weekly joined with stocks.

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
        FROM prices_weekly p
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
        (
            pl.col("exchange").str.to_uppercase().replace_strict(_CN_EXCHANGE_TO_PREFIX)
            + pl.col("ticker")
        ).alias("instrument_id")
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
