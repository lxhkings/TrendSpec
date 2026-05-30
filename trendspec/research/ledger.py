"""JSONL ledger（agent 记忆）+ state.json 原子写（面板读）。"""

import json
import os
from pathlib import Path
from typing import Any


def append_ledger(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_ledger(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_state(path: str | Path, state: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, p)  # 原子替换，面板永不读到半截


def read_state(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"phase": "idle"}
    return json.loads(p.read_text(encoding="utf-8"))
