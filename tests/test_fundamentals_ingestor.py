"""Tests for US fundamentals ingestor."""

import json

import polars as pl
import pytest
from sqlalchemy import create_engine, text

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
