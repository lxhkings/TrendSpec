"""Architectural boundaries for trendspec.combo."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
STRAT = ROOT / "trendspec" / "strategy"
COMBO = ROOT / "trendspec" / "combo"

_RESEARCH_IMPORT = re.compile(
    r"^\s*(from\s+trendspec\.research|import\s+trendspec\.research)",
    re.M,
)
_FORBIDDEN_COMBO = re.compile(
    r"^\s*(from|import)\s+trendspec\.(strategy|research|engine|cli)\b",
    re.M,
)


def _py_files(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*.py") if p.is_file())


def test_strategy_does_not_import_research():
    offenders: list[str] = []
    for p in _py_files(STRAT):
        text = p.read_text(encoding="utf-8")
        if _RESEARCH_IMPORT.search(text):
            offenders.append(str(p.relative_to(ROOT)))
    assert offenders == [], f"strategy must not import research: {offenders}"


def test_combo_does_not_import_upper_layers():
    offenders: list[str] = []
    for p in _py_files(COMBO):
        text = p.read_text(encoding="utf-8")
        if _FORBIDDEN_COMBO.search(text):
            offenders.append(str(p.relative_to(ROOT)))
    assert offenders == [], f"combo must not import strategy/research/engine/cli: {offenders}"
