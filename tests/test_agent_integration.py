"""Integration tests that drive the tools through a real Pydantic AI ``Agent``.

These use offline models (``TestModel`` / ``FunctionModel``) so no LLM is
called, but the full framework path is exercised: tool registration, schema
generation, ``RunContext.deps`` injection, and tool execution against the
mocked InfoLang HTTP layer.
"""

from __future__ import annotations

from collections.abc import Sequence

import respx
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from infolang_pydantic_ai import InfoLangDeps, infolang_toolset, recall, remember
from tests.conftest import FakeRuntime, make_deps


@respx.mock
async def test_agent_runs_toolset_with_test_model(runtime: FakeRuntime) -> None:
    runtime.register(respx)
    runtime.seed("test-ns", "seed fact about teal")
    agent = Agent(
        TestModel(call_tools=["recall", "investigate", "remember"]),
        deps_type=InfoLangDeps,
        toolsets=[infolang_toolset(include=["recall", "investigate", "remember"])],
    )
    async with make_deps("test-ns") as deps:
        result = await agent.run("hello", deps=deps)
    assert result.output is not None
    assert runtime.recall_calls >= 1
    assert runtime.remember_calls >= 1


@respx.mock
async def test_agent_runs_individual_tools(runtime: FakeRuntime) -> None:
    runtime.register(respx)
    agent = Agent(
        TestModel(call_tools=["remember"]),
        deps_type=InfoLangDeps,
        tools=[recall, remember],
    )
    async with make_deps("test-ns") as deps:
        await agent.run("hello", deps=deps)
    assert runtime.remember_calls >= 1


def _finish_after_tool(text: str, tool_name: str, args: dict[str, object]):
    def model_fn(messages: Sequence[ModelMessage], info: AgentInfo) -> ModelResponse:
        for message in messages:
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    return ModelResponse(parts=[TextPart(text)])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])

    return model_fn


@respx.mock
async def test_function_model_recall_roundtrip(runtime: FakeRuntime) -> None:
    runtime.register(respx)
    runtime.seed("test-ns", "the answer is 42")
    agent = Agent(
        FunctionModel(_finish_after_tool("done", "recall", {"query": "answer"})),
        deps_type=InfoLangDeps,
        toolsets=[infolang_toolset()],
    )
    async with make_deps("test-ns") as deps:
        result = await agent.run("what is the answer", deps=deps)
    assert result.output == "done"
    assert runtime.recall_calls == 1


@respx.mock
async def test_function_model_forget_roundtrip(runtime: FakeRuntime) -> None:
    runtime.register(respx)
    mid = runtime.seed("test-ns", "obsolete fact")
    agent = Agent(
        FunctionModel(_finish_after_tool("forgotten", "forget", {"memory_id": mid})),
        deps_type=InfoLangDeps,
        toolsets=[infolang_toolset()],
    )
    async with make_deps("test-ns") as deps:
        result = await agent.run("please forget that", deps=deps)
    assert result.output == "forgotten"
    assert runtime.forget_calls == 1
    assert mid not in runtime.store["test-ns"]
