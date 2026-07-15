"""Runnable quickstart for infolang-pydantic-ai.

Requires a real InfoLang API key and an LLM provider configured for Pydantic AI
(e.g. ``OPENAI_API_KEY``). Run with::

    INFOLANG_API_KEY=il_live_... OPENAI_API_KEY=sk-... python examples/quickstart.py
"""

from __future__ import annotations

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
    async with InfoLangDeps.from_api_key(
        "il_live_...", namespace="quickstart-user"
    ) as deps:
        first = await agent.run(
            "My favorite programming language is Rust. Please remember that.",
            deps=deps,
        )
        print("stored:", first.output)

        second = await agent.run("What's my favorite programming language?", deps=deps)
        print("recalled:", second.output)


if __name__ == "__main__":
    asyncio.run(main())
