from datetime import datetime

import polars as pl
import pytest
from sqlalchemy import create_engine, text

from trendspec.data.markets import Market
from trendspec.ingest.manifest import Manifest
from trendspec.ingest.stocks_db_ingestor import ingest_us_intraday


@pytest.fixture
def intraday_engine():
    """SQLite 内存库模拟 prices_intraday。"""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE prices_intraday (
                ticker TEXT, `interval` TEXT, datetime TIMESTAMP,
                open REAL, high REAL, low REAL, close REAL, volume INTEGER
            )
        """))
        rows = [
            ("A", "1h", "2024-06-04 13:30:00", 100, 101, 99, 100.5, 1000),
            ("A", "1h", "2024-06-04 14:30:00", 100.5, 102, 100, 101.5, 1200),
            ("AAPL", "1h", "2024-06-04 13:30:00", 200, 201, 199, 200.5, 5000),
            # 噪声：非 1h interval 应被过滤
            ("A", "1d", "2024-06-04 00:00:00", 100, 105, 98, 104, 9999),
        ]
        for r in rows:
            conn.execute(text(
                "INSERT INTO prices_intraday VALUES "
                "(:t,:i,:dt,:o,:h,:l,:c,:v)"
            ), {"t": r[0], "i": r[1], "dt": r[2], "o": r[3],
                "h": r[4], "l": r[5], "c": r[6], "v": r[7]})
    return engine


def test_ingest_us_intraday_writes_parquet(intraday_engine, tmp_path):
    root = str(tmp_path)
    manifest = Manifest(Market.US, root)
    result = ingest_us_intraday(intraday_engine, manifest, root, full_sync=True)

    assert result["row_count"] == 3  # 1d 行被过滤
    assert result["instrument_count"] == 2

    df = pl.read_parquet(tmp_path / "us" / "intraday" / "instrument_id=A" / "2024.parquet")
    assert "datetime" in df.columns
    assert "date" in df.columns
    assert df.height == 2  # 同一天两根 bar 都保留
    assert df["instrument_id"].unique().to_list() == ["A"]


def test_ingest_us_intraday_incremental(intraday_engine, tmp_path):
    root = str(tmp_path)
    manifest = Manifest(Market.US, root)
    ingest_us_intraday(intraday_engine, manifest, root, full_sync=True)

    # 追加更新的 bar
    with intraday_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO prices_intraday VALUES "
            "('A','1h','2024-06-04 15:30:00',101.5,103,101,102.5,1300)"
        ))
    result = ingest_us_intraday(intraday_engine, manifest, root, full_sync=False)
    assert result["row_count"] == 1  # 只拉新的一根

    df = pl.read_parquet(tmp_path / "us" / "intraday" / "instrument_id=A" / "2024.parquet")
    assert df.height == 3  # 旧 2 根 + 新 1 根


def test_read_intraday_roundtrip(intraday_engine, tmp_path):
    from trendspec.data.parquet_loader import read_intraday

    root = str(tmp_path)
    manifest = Manifest(Market.US, root)
    ingest_us_intraday(intraday_engine, manifest, root, full_sync=True)

    df = read_intraday(Market.US, root=root)
    assert "datetime" in df.columns
    assert df.height == 3
    # 升序 + 含两个 ticker
    assert set(df["instrument_id"].unique().to_list()) == {"A", "AAPL"}

    only_a = read_intraday(Market.US, root=root, instrument_ids=["A"])
    assert only_a["instrument_id"].unique().to_list() == ["A"]
    dts = only_a["datetime"].to_list()
    assert dts == sorted(dts)  # 升序