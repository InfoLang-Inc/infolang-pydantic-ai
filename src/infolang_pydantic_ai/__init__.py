"""InfoLang memory tools for the Pydantic AI agent framework.

Quickstart::

    from pydantic_ai import Agent
    from infolang_pydantic_ai import InfoLangDeps, infolang_toolset

    agent = Agent(
        "openai:gpt-4o",
        deps_type=InfoLangDeps,
        toolsets=[infolang_toolset()],
    )

    async def main() -> None:
        async with InfoLangDeps.from_api_key("il_live_...", namespace="user-42") as deps:
            result = await agent.run("What are my preferences?", deps=deps)
            print(result.output)
"""

from __future__ import annotations

from ._version import __version__
from .deps import DEFAULT_SOURCE, InfoLangDeps
from .models import RecallHit, RecallResponse
from .tools import forget, infolang_toolset, investigate, recall, remember

__all__ = [
    "__version__",
    "InfoLangDeps",
    "DEFAULT_SOURCE",
    "RecallHit",
    "RecallResponse",
    "infolang_toolset",
    "recall",
    "investigate",
    "remember",
    "forget",
]
