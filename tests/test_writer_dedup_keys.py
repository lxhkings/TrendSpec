from datetime import date, datetime

import polars as pl

from trendspec.data.markets import Market
from trendspec.ingest.writer import write_parquet


def test_dedup_keys_datetime_preserves_intraday_bars(tmp_path):
    """同一天多根 1h bar，按 datetime 去重时全部保留（不塌成 1 根）。"""
    root = str(tmp_path)
    # 先写一根
    df1 = pl.DataFrame({
        "instrument_id": ["A"],
        "datetime": [datetime(2024, 6, 4, 13, 30)],
        "date": [date(2024, 6, 4)],
        "close": [100.0],
    })
    write_parquet(df1, Market.US, "intraday", root,
                  overwrite=False, dedup_keys=["instrument_id", "datetime"])
    # 增量再写同一天另一根（不同 datetime）
    df2 = pl.DataFrame({
        "instrument_id": ["A"],
        "datetime": [datetime(2024, 6, 4, 14, 30)],
        "date": [date(2024, 6, 4)],
        "close": [101.0],
    })
    write_parquet(df2, Market.US, "intraday", root,
                  overwrite=False, dedup_keys=["instrument_id", "datetime"])

    out = pl.read_parquet(tmp_path / "us" / "intraday" / "instrument_id=A" / "2024.parquet")
    assert out.height == 2  # 两根都在，没被 date 去重塌掉
    assert sorted(out["close"].to_list()) == [100.0, 101.0]


def test_dedup_keys_default_unchanged(tmp_path):
    """默认 dedup_keys（date）行为不变：同 date 覆盖。"""
    root = str(tmp_path)
    df1 = pl.DataFrame({
        "instrument_id": ["A"], "date": [date(2024, 6, 4)], "close": [100.0],
    })
    write_parquet(df1, Market.US, "daily", root, overwrite=False)
    df2 = pl.DataFrame({
        "instrument_id": ["A"], "date": [date(2024, 6, 4)], "close": [200.0],
    })
    write_parquet(df2, Market.US, "daily", root, overwrite=False)

    out = pl.read_parquet(tmp_path / "us" / "daily" / "instrument_id=A" / "2024.parquet")
    assert out.height == 1
    assert out["close"].to_list() == [200.0]  # keep="last"