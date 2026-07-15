"""InfoLang memory tools for Pydantic AI agents.

Each function is a Pydantic AI *context tool*: it takes ``RunContext[InfoLangDeps]``
as its first argument, so Pydantic AI injects the agent's :class:`InfoLangDeps`
at call time. Register them either individually::

    agent = Agent(..., deps_type=InfoLangDeps, tools=[recall, remember])

or as a bundle via :func:`infolang_toolset`::

    agent = Agent(..., deps_type=InfoLangDeps, toolsets=[infolang_toolset()])

The tool docstrings below are what the model sees as each tool's description, so
they are written for the agent, not just for humans.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from .deps import InfoLangDeps
from .models import RecallResponse


async def recall(
    ctx: RunContext[InfoLangDeps], query: str, top_k: int = 5
) -> RecallResponse:
    """Search long-term memory for information relevant to a query.

    Use this to look up facts, user preferences, decisions, or context saved in
    earlier turns or sessions. Returns the most semantically similar memory
    chunks, ranked best-first, with a ``weak`` flag when the top match is below
    the confidence floor.
    """

    deps = ctx.deps
    result = await deps.client.recall(query, namespace=deps.namespace, top_k=top_k)
    return RecallResponse.from_result(query, result, namespace=deps.namespace)


async def investigate(
    ctx: RunContext[InfoLangDeps], query: str, top_k: int = 5
) -> RecallResponse:
    """Agent-style memory lookup for open-ended "how/why" questions.

    Like ``recall`` but tuned for investigative questions about how something
    works or why a decision was made. Prefer this when you need background
    context rather than a single stored fact.
    """

    deps = ctx.deps
    result = await deps.client.investigate(
        query, namespace_hint=deps.namespace, top_k=top_k
    )
    return RecallResponse.from_result(query, result, namespace=deps.namespace)


async def remember(
    ctx: RunContext[InfoLangDeps], text: str, tags: str | None = None
) -> str:
    """Save a fact to long-term memory so it can be recalled later.

    Store durable information worth keeping across turns or sessions (user
    preferences, decisions, important details). ``tags`` is an optional
    comma-separated list for later filtering. Returns the id of the stored
    memory.
    """

    deps = ctx.deps
    result = await deps.client.remember(
        text, namespace=deps.namespace, source=deps.source, tags=tags
    )
    return result.memory_id or ""


async def forget(ctx: RunContext[InfoLangDeps], memory_id: str) -> bool:
    """Delete a memory by id (as returned by ``recall`` or ``remember``).

    Use this to remove information that is outdated or that the user asked you
    to forget. Returns ``True`` once the delete is issued.
    """

    deps = ctx.deps
    await deps.client.forget(memory_id, namespace=deps.namespace)
    return True


_TOOLS: dict[str, Any] = {
    "recall": recall,
    "investigate": investigate,
    "remember": remember,
    "forget": forget,
}


def infolang_toolset(
    include: Iterable[str] | None = None, **toolset_kwargs: Any
) -> FunctionToolset[InfoLangDeps]:
    """Build a :class:`FunctionToolset` with the InfoLang memory tools.

    Args:
        include: Optional subset of tool names to expose, from
            ``{"recall", "investigate", "remember", "forget"}``. Defaults to all
            four.
        **toolset_kwargs: Forwarded to :class:`FunctionToolset` (e.g.
            ``max_retries``).

    Returns:
        A ``FunctionToolset[InfoLangDeps]`` ready to pass to
        ``Agent(..., toolsets=[...])``.

    Raises:
        ValueError: If ``include`` names an unknown tool or is empty.
    """

    if include is None:
        names = list(_TOOLS)
    else:
        names = list(include)
        unknown = [name for name in names if name not in _TOOLS]
        if unknown:
            raise ValueError(
                f"Unknown InfoLang tool(s): {sorted(unknown)}. "
                f"Available: {sorted(_TOOLS)}."
            )
        if not names:
            raise ValueError("infolang_toolset(include=[]) would expose no tools.")

    tools: list[Any] = [_TOOLS[name] for name in names]
    return FunctionToolset(tools, **toolset_kwargs)
