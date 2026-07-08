"""Field-id maps and JSON line-item parser for US fundamentals.

Source: us_fin_income / us_fin_indicator raw_payload JSON, item_list of
{field_id, display_name, data}. field_id namespace overlaps across statements,
so each statement has its own map.
"""

import json
from typing import Final

# us_fin_income item_list field ids
INCOME_FIELDS: Final[dict[int, str]] = {
    8001: "total_revenue",
    8037: "net_income",
    8043: "net_income_attr_p",
    8048: "diluted_eps",
}

# us_fin_indicator item_list field ids (skip header rows that lack "data")
INDICATOR_FIELDS: Final[dict[int, str]] = {
    14029: "roe",
    14031: "roic",
    14005: "net_margin",
    14003: "op_margin",
}


def parse_item_list(payload_json: str, field_map: dict[int, str]) -> dict[str, float]:
    """Parse a raw_payload JSON string into {canonical_name: value}.

    Only field ids present in field_map and items carrying a "data" key are kept.
    """
    doc = json.loads(payload_json)
    out: dict[str, float] = {}
    for item in doc.get("item_list", []):
        fid = item.get("field_id")
        if fid in field_map and "data" in item:
            out[field_map[fid]] = float(item["data"])
    return out


# CN Tushare raw_payload is already a flat {field_name: value} JSON dict (no
# field_id indirection like the US item_list format) — these maps just rename
# tushare's native field names to the canonical column names shared with US.
CN_INCOME_FIELDS: Final[dict[str, str]] = {
    "total_revenue": "total_revenue",
    "n_income": "net_income",
    "diluted_eps": "diluted_eps",
}

CN_INDICATOR_FIELDS: Final[dict[str, str]] = {
    "roe": "roe",
    "roic": "roic",
    "netprofit_margin": "net_margin",
    "op_of_gr": "op_margin",
    "tr_yoy": "revenue_yoy",
    "netprofit_yoy": "net_income_yoy",
}


def parse_flat_payload(payload_json: str, field_map: dict[str, str]) -> dict[str, float]:
    """Parse a flat {field_name: value} raw_payload JSON string into
    {canonical_name: value}, keeping only fields present in field_map with a
    non-null numeric value.
    """
    doc = json.loads(payload_json)
    out: dict[str, float] = {}
    for src_name, canonical_name in field_map.items():
        val = doc.get(src_name)
        if val is not None:
            out[canonical_name] = float(val)
    return out
