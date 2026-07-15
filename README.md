# infolang-pydantic-ai

[InfoLang](https://infolang.ai) semantic-memory tools for
[Pydantic AI](https://ai.pydantic.dev). Give any Pydantic AI `Agent` durable,
cross-session memory: a typed `InfoLangDeps` dependency plus `recall`,
`investigate`, `remember`, and `forget` tools the model can call.

> Repository: `InfoLang-Inc/infolang-pydantic-ai`. Package:
> `infolang-pydantic-ai` (PyPI). Import path: `infolang_pydantic_ai`.

## Install

```bash
pip install infolang-pydantic-ai
```

This pulls in the `infolang` SDK (`>=0.2,<0.3`) and `pydantic-ai-slim`.

> **Note:** until `infolang` is published to PyPI, install both from source:
>
> ```bash
> pip install "infolang @ git+ssh://git@github.com/InfoLang-Inc/infolang-sdk-python.git"
> pip install "infolang-pydantic-ai @ git+ssh://git@github.com/InfoLang-Inc/infolang-pydantic-ai.git"
> ```

## Quickstart

```python
import asyncio

from pydantic_ai import Agent
from infolang_pydantic_ai import InfoLangDeps, infolang_toolset


agent = Agent(
    "openai:gpt-4o",
    deps_type=InfoLangDeps,
    toolsets=[infolang_toolset()],
    instructions=(
        "You have long-term memory. Use `recall`/`investigate` before answering "
        "questions about the user, and `remember` durable facts they share."
    ),
)


async def main() -> None:
    # INFOLANG_API_KEY is read from the environment (or pass api_key=...).
    async with InfoLangDeps.from_api_key(
        "il_live_...", namespace="user-42"
    ) as deps:
        # First run stores a fact.
        await agent.run("My favorite language is Rust. Remember that.", deps=deps)

        # A later run (even a new process) recalls it.
        result = await agent.run("What's my favorite language?", deps=deps)
        print(result.output)


asyncio.run(main())
```

A runnable version is in [`examples/quickstart.py`](examples/quickstart.py).

## How it maps onto InfoLang

Pydantic AI injects a typed dependency object into every tool call through
`RunContext.deps`. `InfoLangDeps` is that object; it carries an
`infolang.AsyncInfoLang` client plus the `namespace` (bank) and `source` the
tools use.

| Tool | InfoLang SDK call | Returns |
|------|-------------------|---------|
| `recall(query, top_k=5)` | `client.recall(...)` | `RecallResponse` (typed hits) |
| `investigate(query, top_k=5)` | `client.investigate(...)` | `RecallResponse` |
| `remember(text, tags=None)` | `client.remember(...)` | memory id (`str`) |
| `forget(memory_id)` | `client.forget(...)` | `True` |

`recall`/`investigate` return a `RecallResponse`:

```python
class RecallResponse(BaseModel):
    query: str
    namespace: str | None
    hits: list[RecallHit]      # each: id, text, score, tags
    weak: bool                 # True when the top hit is below the 0.85 floor
```

Because Pydantic AI is schema-first, these typed return models are what the
model sees — stable field names instead of the SDK's compact wire aliases.

## Wiring options

**Bundle (recommended):**

```python
agent = Agent(..., deps_type=InfoLangDeps, toolsets=[infolang_toolset()])
```

Select a subset:

```python
infolang_toolset(include=["recall", "remember"])
```

**Individual tools** (e.g. to mix with your own):

```python
from infolang_pydantic_ai import recall, remember

agent = Agent(..., deps_type=InfoLangDeps, tools=[recall, remember])
```

## Scoping (workspace vs namespace)

InfoLang scopes memory by **workspace** (tenant) and **namespace** (bank). Set a
per-user or per-agent namespace so recalls stay isolated:

```python
InfoLangDeps.from_api_key("il_live_...", namespace="user-42", workspace="acme")
```

Managed API-key requests honor `namespace` on both reads and writes. Pass an
existing client instead with `InfoLangDeps.from_client(client, namespace=...)`
— then you own that client's lifecycle.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest
```

Unit tests mock the HTTP layer (`respx`) against a small in-memory fake of the
`il-runtime` memory routes; agent-integration tests drive the tools through
Pydantic AI's `TestModel` so no live LLM or API is needed. An optional live
smoke test (`tests/test_live_smoke.py`) runs against the real InfoLang API when
`INFOLANG_API_KEY` is set and only touches namespaces prefixed
`ittest-pydantic-ai-`:

```bash
INFOLANG_API_KEY=il_live_... pytest tests/test_live_smoke.py -v
```

## Links

- [InfoLang docs](https://docs.infolang.ai)
- [InfoLang Python SDK](https://github.com/InfoLang-Inc/infolang-sdk-python)
- [Pydantic AI](https://ai.pydantic.dev)

## License

Apache-2.0
