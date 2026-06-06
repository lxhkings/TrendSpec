"""Tests for US fundamentals ingestor."""

import json
import tempfile
from datetime import date

import polars as pl
import pytest
from sqlalchemy import create_engine, text

from trendspec.data.markets import Market
from trendspec.data.parquet_loader import scan_parquet
from trendspec.ingest.fundamentals_schema import (
    INCOME_FIELDS,
    INDICATOR_FIELDS,
    parse_item_list,
)


def test_parse_item_list_income():
    payload = json.dumps({
        "item_list": [
            {"field_id": 8001, "display_name": "总收入", "data": 100.0},
            {"field_id": 8037, "display_name": "净利润", "data": 20.0},
            {"field_id": 8043, "display_name": "归母净利润", "data": 18.0},
            {"field_id": 8048, "display_name": "稀释EPS", "data": 1.5},
            {"field_id": 9999, "display_name": "无关字段", "data": 7.0},
        ]
    })
    out = parse_item_list(payload, INCOME_FIELDS)
    assert out == {
        "total_revenue": 100.0,
        "net_income": 20.0,
        "net_income_attr_p": 18.0,
        "diluted_eps": 1.5,
    }


def test_parse_item_list_skips_header_rows_without_data():
    payload = json.dumps({
        "item_list": [
            {"field_id": 14001, "display_name": "盈利能力TTM"},  # header, no data
            {"field_id": 14029, "display_name": "ROE", "data": 30.0},
            {"field_id": 14005, "display_name": "归母净利率", "data": 10.0},
        ]
    })
    out = parse_item_list(payload, INDICATOR_FIELDS)
    assert out == {"roe": 30.0, "net_margin": 10.0}


# ---------------------------------------------------------------------------
# Task 2: DB → Parquet ingestor tests
# ---------------------------------------------------------------------------

from trendspec.ingest.fundamentals_ingestor import ingest_us_fundamentals  # noqa: E402
from trendspec.ingest.manifest import Manifest  # noqa: E402


def _income_payload(revenue, net_income, eps):
    return json.dumps({"item_list": [
        {"field_id": 8001, "data": revenue},
        {"field_id": 8037, "data": net_income},
        {"field_id": 8043, "data": net_income},
        {"field_id": 8048, "data": eps},
    ]})


def _indicator_payload(roe):
    return json.dumps({"item_list": [
        {"field_id": 14029, "data": roe},
        {"field_id": 14031, "data": roe},
        {"field_id": 14005, "data": 10.0},
        {"field_id": 14003, "data": 12.0},
    ]})


@pytest.fixture
def fundamentals_db():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE us_fin_income (
                ticker TEXT, period_end DATE, ann_date DATE,
                financial_type TEXT, period_text TEXT, raw_payload TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE us_fin_indicator (
                ticker TEXT, period_end DATE, ann_date TEXT,
                financial_type TEXT, period_text TEXT, raw_payload TEXT
            )
        """))
        # 5 consecutive quarters for AAPL so eps_ttm + 1 YoY point exist.
        # revenue grows 100,110,120,130,150 ; eps 1 each quarter
        # YoY uses shift(4): row 5 (index 4) vs row 1 (index 0).
        income = [
            ("AAPL", "2025-03-28", "2025-05-01", "2", "2025/Q2", _income_payload(100.0, 20.0, 1.0)),
            ("AAPL", "2025-06-27", "2025-07-31", "3", "2025/Q3", _income_payload(110.0, 22.0, 1.0)),
            ("AAPL", "2025-09-26", "2025-10-30", "4", "2025/Q4", _income_payload(120.0, 24.0, 1.0)),
            ("AAPL", "2025-12-26", "2026-01-29", "1", "2026/Q1", _income_payload(130.0, 26.0, 1.0)),
            ("AAPL", "2026-03-27", "2026-04-30", "2", "2026/Q2", _income_payload(150.0, 30.0, 1.0)),
            # annual row -> must be dropped (financial_type 7, ann_date NULL)
            ("AAPL", "2025-09-26", None, "7", "2025/FY", _income_payload(450.0, 90.0, 4.0)),
        ]
        for r in income:
            conn.execute(text(
                "INSERT INTO us_fin_income VALUES (:t,:pe,:ad,:ft,:pt,:rp)"
            ), {"t": r[0], "pe": r[1], "ad": r[2], "ft": r[3], "pt": r[4], "rp": r[5]})
        ind = [
            ("AAPL", "2025-03-28", "2025-05-01", "2", "2025/Q2", _indicator_payload(30.0)),
            ("AAPL", "2025-06-27", "2025-07-31", "3", "2025/Q3", _indicator_payload(31.0)),
            ("AAPL", "2025-09-26", "2025-10-30", "4", "2025/Q4", _indicator_payload(32.0)),
            ("AAPL", "2025-12-26", "2026-01-29", "1", "2026/Q1", _indicator_payload(33.0)),
            ("AAPL", "2026-03-27", "2026-04-30", "2", "2026/Q2", _indicator_payload(34.0)),
        ]
        for r in ind:
            conn.execute(text(
                "INSERT INTO us_fin_indicator VALUES (:t,:pe,:ad,:ft,:pt,:rp)"
            ), {"t": r[0], "pe": r[1], "ad": r[2], "ft": r[3], "pt": r[4], "rp": r[5]})
        conn.commit()
    return engine


def test_ingest_us_fundamentals_writes_parquet(fundamentals_db):
    with tempfile.TemporaryDirectory() as root:
        manifest = Manifest(Market.US, root)
        result = ingest_us_fundamentals(fundamentals_db, manifest, root, full_sync=True)
        assert result["row_count"] == 5  # annual row dropped

        df = scan_parquet(root, Market.US, "fundamentals").collect()
        assert df["instrument_id"].unique().to_list() == ["AAPL"]
        # PIT date column == ann_date
        assert df["date"].min() == date(2025, 5, 1)

        latest = df.filter(pl.col("date") == date(2026, 4, 30)).row(0, named=True)
        # eps_ttm = sum of last 4 quarterly diluted_eps = 1+1+1+1 = 4
        assert latest["eps_ttm"] == pytest.approx(4.0)
        # revenue_yoy = 150 / 100 - 1 = 0.5  (Q2'26 vs Q2'25, shift(4))
        assert latest["revenue_yoy"] == pytest.approx(0.5)
        # net_income_yoy = 30 / 20 - 1 = 0.5
        assert latest["net_income_yoy"] == pytest.approx(0.5)
        assert latest["roe"] == pytest.approx(34.0)


def test_ingest_us_fundamentals_drops_annual_rows(fundamentals_db):
    with tempfile.TemporaryDirectory() as root:
        manifest = Manifest(Market.US, root)
        ingest_us_fundamentals(fundamentals_db, manifest, root, full_sync=True)
        df = scan_parquet(root, Market.US, "fundamentals").collect()
        # no row from the FY (annual) record: total_revenue 450 must not appear
        assert 450.0 not in df["total_revenue"].to_list()
