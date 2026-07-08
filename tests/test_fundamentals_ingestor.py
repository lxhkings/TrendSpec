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


# ---------------------------------------------------------------------------
# CN fundamentals / valuation ingestor tests
# ---------------------------------------------------------------------------

from trendspec.ingest.fundamentals_ingestor import (  # noqa: E402
    _ts_code_to_instrument_id,
    ingest_cn_fundamentals,
    ingest_cn_valuation,
)
from trendspec.ingest.fundamentals_schema import parse_flat_payload  # noqa: E402


def test_parse_flat_payload_renames_and_drops_missing():
    payload = json.dumps({
        "roe": 38.4, "netprofit_margin": 52.3, "tr_yoy": 15.7, "unrelated": 1.0,
    })
    out = parse_flat_payload(payload, {
        "roe": "roe", "netprofit_margin": "net_margin", "tr_yoy": "revenue_yoy",
        "op_of_gr": "op_margin",  # absent in payload -> dropped, not KeyError
    })
    assert out == {"roe": 38.4, "net_margin": 52.3, "revenue_yoy": 15.7}


def test_ts_code_to_instrument_id_sh_sz():
    assert _ts_code_to_instrument_id("600519.SH") == "SH600519.SH"
    assert _ts_code_to_instrument_id("000001.SZ") == "SZ000001.SZ"


def test_ts_code_to_instrument_id_unsupported_exchange_returns_none():
    # Beijing Stock Exchange not yet supported by derive_instrument_id_cn.
    assert _ts_code_to_instrument_id("430047.BJ") is None


def _cn_income_payload(revenue, net_income):
    return json.dumps({"total_revenue": revenue, "n_income": net_income, "diluted_eps": 1.0})


def _cn_indicator_payload(roe):
    return json.dumps({
        "roe": roe, "roic": roe, "netprofit_margin": 52.0, "op_of_gr": 68.0,
        "tr_yoy": 0.15, "netprofit_yoy": 0.20,
    })


@pytest.fixture
def cn_fundamentals_db():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE fin_income (
                ts_code TEXT, end_date DATE, ann_date DATE,
                f_ann_date DATE, report_type TEXT, comp_type TEXT, raw_payload TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE fin_indicator (
                ts_code TEXT, end_date DATE, ann_date DATE, raw_payload TEXT
            )
        """))
        conn.execute(
            text("INSERT INTO fin_income VALUES ('600519.SH','2024-09-30','2024-10-26',"
                 "'2024-10-26','1','1',:rp)"),
            {"rp": _cn_income_payload(123122542625.45, 63031462239.55)},
        )
        conn.execute(
            text("INSERT INTO fin_indicator VALUES ('600519.SH','2024-09-30','2024-10-26',:rp)"),
            {"rp": _cn_indicator_payload(38.4)},
        )
        # Beijing exchange row must be silently dropped (unsupported prefix).
        conn.execute(
            text("INSERT INTO fin_income VALUES ('430047.BJ','2024-09-30','2024-10-26',"
                 "'2024-10-26','1','1',:rp)"),
            {"rp": _cn_income_payload(1000.0, 100.0)},
        )
        conn.commit()
    return engine


def test_ingest_cn_fundamentals_writes_parquet(cn_fundamentals_db):
    with tempfile.TemporaryDirectory() as root:
        manifest = Manifest(Market.CN, root)
        result = ingest_cn_fundamentals(cn_fundamentals_db, manifest, root)
        assert result["row_count"] == 1  # BJ row dropped

        df = scan_parquet(root, Market.CN, "fundamentals").collect()
        assert df["instrument_id"].to_list() == ["SH600519.SH"]
        assert df["date"].to_list() == [date(2024, 10, 26)]
        row = df.row(0, named=True)
        assert row["total_revenue"] == pytest.approx(123122542625.45)
        assert row["net_income"] == pytest.approx(63031462239.55)
        assert row["roe"] == pytest.approx(38.4)
        assert row["net_margin"] == pytest.approx(52.0)
        assert row["op_margin"] == pytest.approx(68.0)
        assert row["revenue_yoy"] == pytest.approx(0.15)
        assert row["net_income_yoy"] == pytest.approx(0.20)


@pytest.fixture
def cn_valuation_db():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE cn_valuation_snapshot (
                ts_code TEXT, trade_date DATE, close REAL, turnover_rate REAL,
                volume_ratio REAL, pe REAL, pe_ttm REAL, pb REAL, ps REAL,
                ps_ttm REAL, total_mv REAL, circ_mv REAL
            )
        """))
        conn.execute(text(
            "INSERT INTO cn_valuation_snapshot VALUES "
            "('600519.SH','2024-12-31',1524.0,0.31,1.9,25.6,23.15,9.22,12.96,11.59,191444544.72,191444544.72)"
        ))
        conn.execute(text(
            "INSERT INTO cn_valuation_snapshot VALUES "
            "('430047.BJ','2024-12-31',10.0,1.0,1.0,15.0,14.0,2.0,3.0,3.0,1000.0,1000.0)"
        ))
        conn.commit()
    return engine


def test_ingest_cn_valuation_writes_parquet(cn_valuation_db):
    with tempfile.TemporaryDirectory() as root:
        manifest = Manifest(Market.CN, root)
        result = ingest_cn_valuation(cn_valuation_db, manifest, root)
        assert result["row_count"] == 1  # BJ row dropped

        df = scan_parquet(root, Market.CN, "valuation").collect()
        assert df["instrument_id"].to_list() == ["SH600519.SH"]
        row = df.row(0, named=True)
        assert row["pe_ttm"] == pytest.approx(23.15)
        assert row["pb"] == pytest.approx(9.22)
        assert row["date"] == date(2024, 12, 31)
