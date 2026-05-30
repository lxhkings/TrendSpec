import json
from pathlib import Path

from trendspec.research.ledger import append_ledger, read_ledger, write_state


def test_append_and_read_ledger(tmp_path: Path):
    p = tmp_path / "ledger.jsonl"
    append_ledger(p, {"round": 1, "hypothesis": {"rationale": "a"}})
    append_ledger(p, {"round": 2, "hypothesis": {"rationale": "b"}})
    rows = read_ledger(p)
    assert [r["round"] for r in rows] == [1, 2]


def test_read_missing_ledger_returns_empty(tmp_path: Path):
    assert read_ledger(tmp_path / "nope.jsonl") == []


def test_write_state_atomic_overwrites(tmp_path: Path):
    p = tmp_path / "state.json"
    write_state(p, {"phase": "running", "round": 1})
    write_state(p, {"phase": "running", "round": 2})
    data = json.loads(p.read_text())
    assert data["round"] == 2
    # 无残留临时文件
    assert list(tmp_path.glob("*.tmp")) == []
