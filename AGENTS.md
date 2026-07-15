# infolang-pydantic-ai — agent instructions

InfoLang semantic-memory tools for the **Pydantic AI** agent framework. Package
name: `infolang-pydantic-ai`; import path: `infolang_pydantic_ai`.

## Architecture

- `src/infolang_pydantic_ai/deps.py` — `InfoLangDeps`, the dependency object
  injected via `RunContext.deps` (holds an `infolang.AsyncInfoLang`).
- `src/infolang_pydantic_ai/tools.py` — the `recall` / `investigate` /
  `remember` / `forget` context tools and the `infolang_toolset()` factory.
- `src/infolang_pydantic_ai/models.py` — typed `RecallHit` / `RecallResponse`
  return models (Pydantic AI is schema-first).

## Contract

- Depends only on the **published** `infolang` Python SDK (`>=0.2,<0.3`). Never
  reimplement HTTP, import runtime/engine internals, or reference core-ip.
- Canonical SDK surface: `AsyncInfoLang.from_api_key(...)`, `.recall`,
  `.investigate`, `.remember`, `.forget`. Behavior is defined by the SDK /
  OpenAPI contract, not engine internals.

## Rules

- Verify the Pydantic AI tool/toolset API against the installed package before
  changing tool wiring — do not assume shapes.
- Keep tool docstrings model-facing: they are the descriptions the agent sees.

## Commands

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest
```
