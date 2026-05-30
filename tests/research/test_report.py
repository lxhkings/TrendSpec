from pathlib import Path

from trendspec.research.report import write_advice


def _winner():
    return {
        "spec": {
            "market": "us",
            "factors": [{"name": "momentum", "params": {"period": 60},
                         "direction": "high", "weight": 1.0}],
            "top_k": 20, "rebalance": 5,
            "rationale": "动量优选",
        },
        "oos_sharpe": 1.35,
        "oos_max_drawdown": 0.12,
        "oos_total_return": 0.42,
        "window_sharpes": [1.2, 1.5, 1.35],
    }


def test_write_advice_creates_markdown(tmp_path: Path):
    path = write_advice(tmp_path, _winner(), round_no=3)
    p = Path(path)
    assert p.exists() and p.suffix == ".md"
    text = p.read_text(encoding="utf-8")
    assert "momentum" in text
    assert "1.35" in text          # OOS sharpe
    assert "动量优选" in text       # rationale
    assert "top_k" in text or "20" in text
