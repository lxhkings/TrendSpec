import json
from pathlib import Path

from starlette.testclient import TestClient

from trendspec.research.dashboard import create_app


def test_api_state_reads_file(tmp_path: Path):
    (tmp_path / "state.json").write_text(json.dumps({"phase": "running", "round": 2}))
    client = TestClient(create_app(str(tmp_path)))
    r = client.get("/api/state")
    assert r.status_code == 200
    assert r.json()["round"] == 2


def test_api_state_idle_when_missing(tmp_path: Path):
    client = TestClient(create_app(str(tmp_path)))
    assert client.get("/api/state").json()["phase"] == "idle"


def test_api_ledger_returns_rows(tmp_path: Path):
    p = tmp_path / "ledger.jsonl"
    p.write_text('{"round": 1}\n{"round": 2}\n')
    client = TestClient(create_app(str(tmp_path)))
    rows = client.get("/api/ledger").json()
    assert [r["round"] for r in rows] == [1, 2]


def test_index_returns_html(tmp_path: Path):
    client = TestClient(create_app(str(tmp_path)))
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<html" in r.text.lower()
