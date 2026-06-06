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

import polars as pl
from sqlalchemy import Engine, text

from trendspec.data.markets import Market
from trendspec.ingest.fundamentals_schema import (
    INCOME_FIELDS,
    INDICATOR_FIELDS,
    parse_item_list,
)
from trendspec.ingest.manifest import Manifest
from trendspec.ingest.writer import write_parquet

_ALLOWED_TABLES: frozenset[str] = frozenset({"us_fin_income", "us_fin_indicator"})
_QUARTERLY = ("1", "2", "3", "4")


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
