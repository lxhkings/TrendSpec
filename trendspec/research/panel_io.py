"""mmap Arrow IPC 零拷贝读写：并行 worker 只读共享面板，走 OS page cache。"""

from pathlib import Path

import polars as pl


def write_ipc(df: pl.DataFrame, path: str | Path) -> None:
    df.write_ipc(str(path))  # Arrow IPC（Feather v2）


def read_ipc_mmap(path: str | Path) -> pl.DataFrame:
    return pl.read_ipc(str(path), memory_map=True)  # 零拷贝 mmap 只读