# research_cmd factors/filters 校验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `research ic/quantile` 的 `_load_factor_spec_json` 真正校验 `factors`/`filters`（`FactorTerm`/`FilterTerm`），非法输入以清晰 `Exit(1)` 失败，合法路径返回形状与今兼容的 dict。

**Architecture:** 在 `trendspec/research/spec.py` 增加纯函数 `parse_research_eval_spec(raw: dict) -> dict`（可单测、不依赖 Typer）。CLI `_load_factor_spec_json` 在 `json.loads` 后调用它；`ValidationError` 转红字 + `typer.Exit(1)`。不强制 `top_k`/`rebalance`/`market`。

**Tech Stack:** Python, Pydantic v2, Typer, pytest, uv。

**Spec:** `docs/superpowers/specs/2026-07-15-research-pipeline-structure-cleanup-design.md` §C

## Global Constraints

- 行为冻结：合法 spec 的成功路径与今一致（下游仍 `spec.get("filters")` 等）。
- 错误路径清晰化：非法 op / 未注册名 / 缺 direction → 可读错误，非裸 `KeyError`。
- 不走完整 `FactorSpec`（不要 top_k/rebalance）。
- 不改 `FactorStrategy` 加载路径。
- 测 `parse_research_eval_spec` 纯函数即可；不必起真实行情 CLI e2e。

---

## File map

| 文件 | 职责 |
|------|------|
| `trendspec/research/spec.py` | `parse_research_eval_spec` |
| `trendspec/cli/research_cmd.py` | `_load_factor_spec_json` 接线 |
| `tests/research/test_spec.py` | 纯函数校验测 |

---

### Task 1: `parse_research_eval_spec` 纯函数 + 单测

**Files:**
- Modify: `trendspec/research/spec.py`
- Modify: `tests/research/test_spec.py`

**Interfaces:**
- Produces:
  ```python
  def parse_research_eval_spec(raw: dict[str, Any]) -> dict[str, Any]:
      """Validate ic/quantile subset of a factor spec JSON.

      Requires non-empty factors (each FactorTerm). Optional filters
      (each FilterTerm, default []). Optional group_by, winsorize_pct
      (default 0.01). Does NOT require market/top_k/rebalance.

      Returns a plain dict suitable for compute_rank_ic / compute_quantile_returns:
        {
          "factors": [FactorTerm.model_dump(), ...],
          "filters": [FilterTerm.model_dump(), ...],  # always present, maybe []
          "group_by": ... | omitted if absent in raw,
          "winsorize_pct": float,
        }
      Raises pydantic.ValidationError on bad input.
      """
  ```
- Consumes: `FactorTerm`, `FilterTerm`

- [ ] **Step 1: Write the failing tests**

Append to `tests/research/test_spec.py`（文件顶部已 import pytest / FactorSpec 等；补 import）：

```python
import trendspec.factors  # noqa: F401 — 触发因子注册，name 校验才过
from pydantic import ValidationError

from trendspec.research.spec import (
    FactorSpec,
    FactorTerm,
    FilterTerm,
    parse_research_eval_spec,
)


def test_parse_research_eval_spec_accepts_factors_and_filters():
    raw = {
        "factors": [
            {"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}
        ],
        "filters": [
            {"name": "momentum", "params": {"period": 5}, "op": ">", "value": 0.0}
        ],
        "winsorize_pct": 0.02,
    }
    out = parse_research_eval_spec(raw)
    assert len(out["factors"]) == 1
    assert out["factors"][0]["name"] == "momentum"
    assert out["factors"][0]["direction"] == "high"
    assert len(out["filters"]) == 1
    assert out["filters"][0]["op"] == ">"
    assert out["winsorize_pct"] == 0.02


def test_parse_research_eval_spec_defaults_filters_empty_and_winsorize():
    raw = {
        "factors": [
            {"name": "momentum", "params": {"period": 5}, "direction": "high"}
        ],
    }
    out = parse_research_eval_spec(raw)
    assert out["filters"] == []
    assert out["winsorize_pct"] == 0.01


def test_parse_research_eval_spec_rejects_bad_filter_op():
    raw = {
        "factors": [
            {"name": "momentum", "params": {"period": 5}, "direction": "high"}
        ],
        "filters": [{"name": "momentum", "op": "!=", "value": 0.0}],
    }
    with pytest.raises(ValidationError):
        parse_research_eval_spec(raw)


def test_parse_research_eval_spec_rejects_unknown_factor_name():
    raw = {
        "factors": [
            {"name": "no_such_factor_xyz", "direction": "high"}
        ],
    }
    with pytest.raises(ValidationError):
        parse_research_eval_spec(raw)


def test_parse_research_eval_spec_rejects_missing_direction():
    raw = {
        "factors": [
            {"name": "momentum", "params": {"period": 5}}
        ],
    }
    with pytest.raises(ValidationError):
        parse_research_eval_spec(raw)


def test_parse_research_eval_spec_rejects_empty_factors():
    raw = {"factors": []}
    with pytest.raises(ValidationError):
        parse_research_eval_spec(raw)


def test_parse_research_eval_spec_preserves_group_by():
    raw = {
        "factors": [
            {"name": "momentum", "params": {"period": 5}, "direction": "high"}
        ],
        "group_by": {"金融": ["银行"]},
    }
    out = parse_research_eval_spec(raw)
    assert out["group_by"] == {"金融": ["银行"]}
```

Note: if `tests/research/test_spec.py` already imports `FactorSpec, FactorTerm, FilterTerm` without `trendspec.factors`, add the `import trendspec.factors` so registry is populated (existing tests already call FactorSpec which needs registered names for momentum etc. — check file; if tests already pass without explicit import, factors may be imported elsewhere via conftest — still add `import trendspec.factors  # noqa: F401` at top of new tests' module side if name validation fails).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/research/test_spec.py -k parse_research_eval -xvs`

Expected: FAIL `cannot import name 'parse_research_eval_spec'`

- [ ] **Step 3: Implement `parse_research_eval_spec`**

Add to `trendspec/research/spec.py` (after `FilterTerm`, before or after `FactorSpec` — after both term classes is fine). Need `ValidationError` is from pydantic when constructing models; also use a small helper model or manual construction:

```python
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
```

Actually `ValidationError` is raised automatically by model construction — no need to import it in spec.py unless re-raising.

```python
class _ResearchEvalSpec(BaseModel):
    """ic/quantile 子集：不要求 market/top_k/rebalance。"""

    factors: list[FactorTerm] = Field(min_length=1)
    filters: list[FilterTerm] = Field(default_factory=list)
    group_by: dict[str, list[str]] | None = None
    winsorize_pct: float = 0.01


def parse_research_eval_spec(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate the ic/quantile subset of a factor-spec JSON object.

    Requires non-empty ``factors`` (each a FactorTerm). Optional ``filters``
    (FilterTerm list, default []), optional ``group_by``, optional
    ``winsorize_pct`` (default 0.01). Ignores market/top_k/rebalance if present.

    Returns a plain dict with dumped factors/filters for compute_* callers.
    Raises pydantic.ValidationError on invalid input.
    """
    # Only pick known keys so extra full-FactorSpec fields do not break parsing
    payload = {
        "factors": raw.get("factors"),
        "filters": raw.get("filters", []),
        "winsorize_pct": raw.get("winsorize_pct", 0.01),
    }
    if "group_by" in raw:
        payload["group_by"] = raw.get("group_by")
    parsed = _ResearchEvalSpec.model_validate(payload)
    out: dict[str, Any] = {
        "factors": [t.model_dump() for t in parsed.factors],
        "filters": [t.model_dump() for t in parsed.filters],
        "winsorize_pct": parsed.winsorize_pct,
    }
    if parsed.group_by is not None:
        out["group_by"] = parsed.group_by
    return out
```

If `factors` key is missing entirely, `model_validate` should fail — good.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/research/test_spec.py -xvs`

Expected: PASS（新旧测试全绿）

- [ ] **Step 5: Commit**

```bash
git add trendspec/research/spec.py tests/research/test_spec.py
git commit -m "feat(research): parse_research_eval_spec for ic/quantile JSON"
```

---

### Task 2: 接线 `_load_factor_spec_json`

**Files:**
- Modify: `trendspec/cli/research_cmd.py`

**Interfaces:**
- Consumes: `parse_research_eval_spec(raw) -> dict` (raises `ValidationError`)
- Produces: CLI Exit(1) on validation failure with red message

- [ ] **Step 1: Write a CLI-level unit test (no network)**

Create `tests/research/test_research_cmd_load.py`（新文件）：

```python
from pathlib import Path

import pytest
import typer

import trendspec.factors  # noqa: F401
from trendspec.cli.research_cmd import _load_factor_spec_json


def test_load_factor_spec_json_valid(tmp_path: Path):
    p = tmp_path / "ok.json"
    p.write_text(
        '{"factors":[{"name":"momentum","params":{"period":5},"direction":"high"}]}',
        encoding="utf-8",
    )
    out = _load_factor_spec_json(p)
    assert out["factors"][0]["name"] == "momentum"
    assert out["filters"] == []


def test_load_factor_spec_json_bad_filter_op_exits(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(
        '{"factors":[{"name":"momentum","direction":"high"}],'
        '"filters":[{"name":"momentum","op":"!=","value":0}]}',
        encoding="utf-8",
    )
    with pytest.raises(typer.Exit) as ei:
        _load_factor_spec_json(p)
    assert ei.value.exit_code == 1
```

- [ ] **Step 2: Run test to verify it fails (or passes partially)**

Run: `uv run pytest tests/research/test_research_cmd_load.py -xvs`

Expected: `bad_filter_op` 可能仍 `KeyError` 或通过 raw dict 不 Exit——直到接线完成应 FAIL 在 `exit_code` 断言或 validation 未触发。

- [ ] **Step 3: Wire CLI loader**

Replace `_load_factor_spec_json` in `trendspec/cli/research_cmd.py` with:

```python
def _load_factor_spec_json(spec_file: Path) -> dict:
    """Load and validate a factor spec JSON file for research ic/quantile commands.

    Validates factors/filters/group_by/winsorize_pct via parse_research_eval_spec.
    Does not require top_k/rebalance/market. Exits via typer.Exit(1) on missing
    file, invalid JSON, or validation errors.
    """
    if not spec_file.exists():
        console.print(f"[red]--spec-file 不存在: {spec_file}[/red]")
        raise typer.Exit(1)
    try:
        raw = json.loads(spec_file.read_text())
    except json.JSONDecodeError as e:
        console.print(f"[red]--spec-file 不是合法 JSON: {e}[/red]")
        raise typer.Exit(1) from None

    # 延迟 import：保证调用方已 import trendspec.factors 完成注册
    from pydantic import ValidationError

    from trendspec.research.spec import parse_research_eval_spec

    try:
        return parse_research_eval_spec(raw)
    except ValidationError as e:
        console.print(f"[red]--spec-file 校验失败: {e}[/red]")
        raise typer.Exit(1) from None
```

Update help strings on `research_ic` / `research_quantile` if needed (optional): already mention factors/filters.

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/research/test_research_cmd_load.py tests/research/test_spec.py -xvs
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trendspec/cli/research_cmd.py tests/research/test_research_cmd_load.py
git commit -m "feat(cli): validate research ic/quantile spec factors and filters"
```

---

## Plan complete criteria

- [ ] `parse_research_eval_spec` 覆盖合法 / 坏 op / 未注册 / 缺 direction / 空 factors
- [ ] `_load_factor_spec_json` 校验失败 → `typer.Exit(1)`
- [ ] 合法 JSON 返回含 `filters` 键的 dict，ic/quantile 下游无需改
