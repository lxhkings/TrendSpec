import datetime as dt
import polars as pl
from trendspec.research.panel_io import write_ipc, read_ipc_mmap


def test_ipc_roundtrip(tmp_path):
    df = pl.DataFrame({"instrument_id": ["A", "B"],
                       "date": [dt.date(2020, 1, 1), dt.date(2020, 1, 2)],
                       "value": [1.5, 2.5]})
    p = tmp_path / "panel.arrow"
    write_ipc(df, p)
    back = read_ipc_mmap(p)
    assert back.equals(df)