"""Ingest US quarterly fundamentals from the Synology 'stocks' DB.

Source tables (existing, do not modify):
  us_fin_income(ticker, period_end, ann_date, financial_type, period_text, raw_payload)
  us_fin_indicator(ticker, period_end, ann_date, financial_type, period_text, raw_payload)

Only quarterly records (financial_type 1..4) are kept; annual rows (7) carry a
NULL ann_date and cannot be point-in-time aligned.

Output: data_lake/us/fundamentals/instrument_id=<id>/<year>.parquet
PK (instrument_id, date) where date = ann_date (the PIT-visible date).

TTM/YoY are derived on the quarterly series ordered by period_end within ticker.
Fundamentals are low-volume, so the ingest always recomputes the full history.
"""

from datetime import date, timedelta
from typing import Final

import polars as pl
from sqlalchemy import Engine, text

from trendspec.data.markets import Market
from trendspec.ingest.stocks_db_ingestor import _fetch_df_with_progress, _resolve_start_exclusive
from trendspec.ingest.fundamentals_schema import (
    CN_INCOME_FIELDS,
    CN_INDICATOR_FIELDS,
    INCOME_FIELDS,
    INDICATOR_FIELDS,
    parse_flat_payload,
    parse_item_list,
)
from trendspec.ingest.manifest import Manifest
from trendspec.ingest.schema_map import derive_instrument_id_cn
from trendspec.ingest.writer import write_parquet

_ALLOWED_TABLES: frozenset[str] = frozenset({"us_fin_income", "us_fin_indicator"})
_QUARTERLY = ("1", "2", "3", "4")
_CN_ALLOWED_TABLES: frozenset[str] = frozenset({"fin_income", "fin_indicator"})


def _fetch_parsed(engine: Engine, table: str, field_map: dict[int, str]) -> list[dict]:
    """Fetch quarterly rows from `table`, parse item_list into flat columns."""
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Table {table!r} not in allowed list")
    placeholders = ", ".join(f":q{i}" for i in range(len(_QUARTERLY)))
    params = {f"q{i}": q for i, q in enumerate(_QUARTERLY)}
    sql = text(f"""
        SELECT ticker, period_end, ann_date, raw_payload
        FROM {table}
        WHERE financial_type IN ({placeholders})
          AND ann_date IS NOT NULL
    """)
    rows: list[dict] = []
    with engine.connect() as conn:
        for ticker, period_end, ann_date, payload in conn.execute(sql, params):
            parsed = parse_item_list(payload, field_map)
            rows.append({"ticker": ticker, "period_end": period_end,
                         "ann_date": ann_date, **parsed})
    return rows


def ingest_us_fundamentals(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,  # noqa: ARG001 — kept for CLI API compat; always full recompute
) -> dict:
    """Ingest US quarterly fundamentals into the fundamentals dataset.

    Returns {"row_count", "date_range", "instrument_count"}.
    """
    income_rows = _fetch_parsed(engine, "us_fin_income", INCOME_FIELDS)
    indicator_rows = _fetch_parsed(engine, "us_fin_indicator", INDICATOR_FIELDS)

    if not income_rows:
        return {"row_count": 0, "date_range": ("", ""), "instrument_count": 0}

    income = pl.DataFrame(income_rows).with_columns([
        pl.col("period_end").cast(pl.Date),
        pl.col("ann_date").cast(pl.Date),
    ])
    if indicator_rows:
        _IND_COLS = ["roe", "roic", "net_margin", "op_margin"]
        indicator = pl.DataFrame(indicator_rows).with_columns(
            pl.col("period_end").cast(pl.Date)
        )
        available_ind = [c for c in _IND_COLS if c in indicator.columns]
        indicator = indicator.select(["ticker", "period_end"] + available_ind)
        df = income.join(indicator, on=["ticker", "period_end"], how="left")
    else:
        df = income.with_columns([
            pl.lit(None, dtype=pl.Float64).alias(c)
            for c in ("roe", "roic", "net_margin", "op_margin")
        ])

    # Derive TTM / YoY on the quarterly series ordered by period_end per ticker.
    df = df.sort(["ticker", "period_end"]).with_columns([
        pl.col("diluted_eps").rolling_sum(window_size=4).over("ticker").alias("eps_ttm"),
        (pl.col("total_revenue") / pl.col("total_revenue").shift(4).over("ticker") - 1)
            .alias("revenue_yoy"),
        (pl.col("net_income") / pl.col("net_income").shift(4).over("ticker") - 1)
            .alias("net_income_yoy"),
    ])

    # ann_date becomes the PIT 'date' partition key.
    df = df.rename({"ticker": "instrument_id", "ann_date": "date"})

    write_parquet(
        df, Market.US, "fundamentals", root,
        overwrite=True, dedup_keys=["instrument_id", "date"],
    )

    dates = df["date"].cast(pl.Utf8)
    date_range = (dates.min(), dates.max())
    instrument_count = df["instrument_id"].n_unique()
    row_count = len(df)
    manifest.update_dataset_state("fundamentals", row_count, date_range, instrument_count)

    return {"row_count": row_count, "date_range": date_range,
            "instrument_count": instrument_count}


# =============================================================================
# CN Fundamentals (Tushare, via StockPull-populated fin_income / fin_indicator)
# =============================================================================


def _ts_code_to_instrument_id(ts_code: str) -> str | None:
    """Derive CN instrument_id from a Tushare ts_code (e.g. "600519.SH").

    Returns None for exchanges not yet supported by derive_instrument_id_cn
    (e.g. Beijing Stock Exchange ".BJ") rather than raising, so a handful of
    unsupported tickers don't abort the whole ingest.
    """
    exchange = ts_code.rsplit(".", 1)[-1]
    try:
        return derive_instrument_id_cn(ts_code, exchange)
    except ValueError:
        return None


def _fetch_cn_parsed(engine: Engine, table: str, field_map: dict[str, str]) -> list[dict]:
    """Fetch rows from a CN raw_payload-JSON financial table, parsed to flat columns."""
    if table not in _CN_ALLOWED_TABLES:
        raise ValueError(f"Table {table!r} not in allowed list")
    sql = text(f"""
        SELECT ts_code, end_date, ann_date, raw_payload
        FROM {table}
        WHERE ann_date IS NOT NULL
    """)
    rows: list[dict] = []
    with engine.connect() as conn:
        for ts_code, end_date, ann_date, payload in conn.execute(sql):
            parsed = parse_flat_payload(payload, field_map)
            rows.append({"ts_code": ts_code, "end_date": end_date,
                         "ann_date": ann_date, **parsed})
    return rows


def ingest_cn_fundamentals(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,  # noqa: ARG001 — kept for CLI API compat; always full recompute
) -> dict:
    """Ingest CN quarterly fundamentals from StockPull-populated fin_income /
    fin_indicator tables (raw_payload JSON, already flat tushare field names).

    Returns {"row_count", "date_range", "instrument_count"}.
    """
    income_rows = _fetch_cn_parsed(engine, "fin_income", CN_INCOME_FIELDS)
    indicator_rows = _fetch_cn_parsed(engine, "fin_indicator", CN_INDICATOR_FIELDS)

    if not income_rows:
        return {"row_count": 0, "date_range": ("", ""), "instrument_count": 0}

    income = pl.DataFrame(income_rows).with_columns([
        pl.col("end_date").cast(pl.Date),
        pl.col("ann_date").cast(pl.Date),
    ])

    _IND_COLS = list(CN_INDICATOR_FIELDS.values())
    if indicator_rows:
        indicator = pl.DataFrame(indicator_rows).with_columns(
            pl.col("end_date").cast(pl.Date)
        )
        available_ind = [c for c in _IND_COLS if c in indicator.columns]
        indicator = indicator.select(["ts_code", "end_date"] + available_ind)
        df = income.join(indicator, on=["ts_code", "end_date"], how="left")
    else:
        df = income.with_columns([
            pl.lit(None, dtype=pl.Float64).alias(c) for c in _IND_COLS
        ])

    # instrument_id derived from ts_code; drop rows for unsupported exchanges.
    df = df.with_columns(
        pl.col("ts_code")
        .map_elements(_ts_code_to_instrument_id, return_dtype=pl.Utf8)
        .alias("instrument_id")
    ).filter(pl.col("instrument_id").is_not_null())

    # ann_date becomes the PIT 'date' partition key.
    df = df.rename({"ann_date": "date"}).drop("ts_code")

    write_parquet(
        df, Market.CN, "fundamentals", root,
        overwrite=True, dedup_keys=["instrument_id", "date"],
    )

    dates = df["date"].cast(pl.Utf8)
    date_range = (dates.min(), dates.max())
    instrument_count = df["instrument_id"].n_unique()
    row_count = len(df)
    manifest.update_dataset_state("fundamentals", row_count, date_range, instrument_count)

    return {"row_count": row_count, "date_range": date_range,
            "instrument_count": instrument_count}


# =============================================================================
# CN Valuation (Tushare daily_basic, via StockPull-populated cn_valuation_snapshot)
# =============================================================================

_CN_VALUATION_COLS = [
    "pe", "pe_ttm", "pb", "ps", "ps_ttm", "total_mv", "circ_mv", "turnover_rate",
]


_VALUATION_CHUNK_DAYS: Final[int] = 365


def _next_chunk_end(start_exclusive: str, today: date) -> str:
    """365-day window end after `start_exclusive`, capped at `today`."""
    start = date.fromisoformat(start_exclusive)
    end = start + timedelta(days=_VALUATION_CHUNK_DAYS)
    return min(end, today).isoformat()


def ingest_cn_valuation(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
    since: str | None = None,
) -> dict:
    """Ingest CN daily valuation snapshots from StockPull-populated
    cn_valuation_snapshot table (flat columns, no JSON payload).

    Fetches in ~365-day windows via a streaming server-side cursor, writing
    parquet + advancing the manifest watermark after each window. The table
    now holds 10M+ rows after full history backfill; even a single streamed
    (unbuffered) pull of the whole table intermittently gets
    "Connection reset by peer" from the NAS partway through (observed at
    ~3M rows), so a full history sync must be resumable per-window rather
    than all-or-nothing — a crash mid-run only loses the in-flight window,
    and simply re-running (default incremental mode) picks up from the last
    successfully committed window's watermark.

    Output dataset: data_lake/cn/valuation/instrument_id=<id>/<year>.parquet
    PK (instrument_id, date) where date = trade_date (same-day visible, no PIT lag).

    Args:
        full_sync: If True, pull all history ignoring manifest. Only the
            first window of a full_sync run overwrites existing partition
            files; later windows merge (write_parquet overwrite=False),
            otherwise each window's write would wipe out the previous one's
            rows in any partition (instrument_id, year) touched by both.
        since: Inclusive start date 'YYYY-MM-DD'; overrides manifest/full_sync.

    Returns {"row_count", "date_range", "instrument_count"} aggregated across
    all windows fetched in this call.
    """
    cursor = _resolve_start_exclusive(manifest, "valuation", full_sync, since)
    today = date.today()
    cols = ", ".join(["ts_code", "trade_date"] + _CN_VALUATION_COLS)

    total_row_count = 0
    all_instrument_ids: set[str] = set()
    overall_min_date: str | None = None
    overall_max_date: str | None = None
    is_first_window = True

    while True:
        window_end = _next_chunk_end(cursor, today)
        sql = text(
            f"SELECT {cols} FROM cn_valuation_snapshot "
            "WHERE trade_date > :start AND trade_date <= :end"
        )
        df = _fetch_df_with_progress(
            engine, sql, {"start": cursor, "end": window_end},
            schema=["ts_code", "trade_date"] + _CN_VALUATION_COLS,
            label=f"拉取 cn 估值快照 (至 {window_end})",
        )

        if df.is_empty():
            manifest.update_dataset_state("valuation", 0, (cursor, window_end), 0)
        else:
            df = df.with_columns(pl.col("trade_date").cast(pl.Date))
            df = df.with_columns(
                pl.col("ts_code")
                .map_elements(_ts_code_to_instrument_id, return_dtype=pl.Utf8)
                .alias("instrument_id")
            ).filter(pl.col("instrument_id").is_not_null())
            df = df.rename({"trade_date": "date"}).drop("ts_code")

            write_parquet(
                df, Market.CN, "valuation", root,
                overwrite=(full_sync and is_first_window),
                dedup_keys=["instrument_id", "date"],
            )

            dates = df["date"].cast(pl.Utf8)
            window_min, window_max = dates.min(), dates.max()
            manifest.update_dataset_state(
                "valuation", len(df), (window_min, window_max),
                df["instrument_id"].n_unique(),
            )

            total_row_count += len(df)
            all_instrument_ids.update(df["instrument_id"].to_list())
            overall_min_date = window_min if overall_min_date is None else min(overall_min_date, window_min)
            overall_max_date = window_max if overall_max_date is None else max(overall_max_date, window_max)

        is_first_window = False
        if window_end >= today.isoformat():
            break
        cursor = window_end

    if total_row_count == 0:
        return {"row_count": 0, "date_range": ("", ""), "instrument_count": 0}

    return {
        "row_count": total_row_count,
        "date_range": (overall_min_date, overall_max_date),
        "instrument_count": len(all_instrument_ids),
    }
