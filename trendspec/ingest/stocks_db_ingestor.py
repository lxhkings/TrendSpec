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

from datetime import date
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
    ex_placeholders = _exchange_placeholder(_US_EXCHANGES)
    ex_params = _exchange_params(_US_EXCHANGES)

    sql_changes = text(f"""
        SELECT c.ticker, c.change_date, c.change_type
        FROM constituent_changes c
        JOIN stocks s ON c.ticker = s.ticker
        WHERE c.index_id = 'SP500'
          AND s.exchange IN ({ex_placeholders})
        ORDER BY c.change_date
    """)

    sql_min = text(f"""
        SELECT p.ticker, MIN(p.date) as first_date
        FROM prices p
        JOIN stocks s ON p.ticker = s.ticker
        WHERE s.exchange IN ({ex_placeholders})
        GROUP BY p.ticker
    """)

    with engine.connect() as conn:
        changes = conn.execute(sql_changes, ex_params).fetchall()
        min_dates = conn.execute(sql_min, ex_params).fetchall()

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
) -> dict:
    """
    Ingest CN (A-share) daily OHLCV from prices + stocks tables.

    instrument_id = SH{ticker} for SSE/SH, SZ{ticker} for SZSE/SZ.
    adj_factor = 1.0 (prices are Tushare backward-adjusted).

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
    ex_placeholders = _exchange_placeholder(_CN_EXCHANGES)
    ex_params = _exchange_params(_CN_EXCHANGES)

    sql_changes = text(f"""
        SELECT c.ticker, c.change_date, c.change_type, s.exchange
        FROM constituent_changes c
        JOIN stocks s ON c.ticker = s.ticker
        WHERE c.index_id = 'CSI800'
          AND s.exchange IN ({ex_placeholders})
        ORDER BY c.change_date
    """)

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
