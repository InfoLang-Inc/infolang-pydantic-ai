from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from infolang.errors import NotFoundError

from infolang_pydantic_ai import forget, investigate, recall, remember
from infolang_pydantic_ai.deps import InfoLangDeps
from tests.conftest import BASE_URL, FakeRuntime, make_deps


def ctx(deps: InfoLangDeps) -> Any:
    """A minimal stand-in for RunContext; tools only read ``ctx.deps``."""

    return SimpleNamespace(deps=deps)


@respx.mock
async def test_recall_returns_ranked_hits(runtime: FakeRuntime) -> None:
    runtime.register(respx)
    runtime.seed("test-ns", "the user loves teal")
    runtime.seed("test-ns", "unrelated note")
    async with make_deps("test-ns") as deps:
        response = await recall(ctx(deps), "teal")
    assert response.query == "teal"
    assert response.hits[0].text == "the user loves teal"
    assert response.hits[0].score == 0.95
    assert runtime.recall_calls == 1


@respx.mock
async def test_recall_sends_namespace_and_top_k(runtime: FakeRuntime) -> None:
    runtime.register(respx)
    for i in range(3):
        runtime.seed("test-ns", f"memory {i}")
    async with make_deps("test-ns") as deps:
        response = await recall(ctx(deps), "memory", top_k=1)
    assert len(response.hits) == 1
    assert runtime.last_recall_body is not None
    assert runtime.last_recall_body["namespace"] == "test-ns"
    assert runtime.last_recall_body["top_k"] == 1


@respx.mock
async def test_recall_empty_namespace_returns_no_hits(runtime: FakeRuntime) -> None:
    runtime.register(respx)
    async with make_deps("empty-ns") as deps:
        response = await recall(ctx(deps), "anything")
    assert response.hits == []
    assert response.weak is False


@respx.mock
async def test_recall_weak_when_no_substring_match(runtime: FakeRuntime) -> None:
    runtime.register(respx)
    runtime.seed("test-ns", "completely different content")
    async with make_deps("test-ns") as deps:
        response = await recall(ctx(deps), "zzz-no-match")
    assert response.hits
    assert response.weak is True


@respx.mock
async def test_investigate_uses_namespace_hint_and_default_top_k(
    runtime: FakeRuntime,
) -> None:
    runtime.register(respx)
    runtime.seed("test-ns", "how auth works")
    async with make_deps("test-ns") as deps:
        response = await investigate(ctx(deps), "auth")
    assert response.hits
    assert runtime.last_recall_body is not None
    assert runtime.last_recall_body["namespace"] == "test-ns"
    assert runtime.last_recall_body["top_k"] == 5


@respx.mock
async def test_remember_returns_id_and_sends_metadata(runtime: FakeRuntime) -> None:
    runtime.register(respx)
    async with make_deps("test-ns") as deps:
        memory_id = await remember(ctx(deps), "remember this", tags="fact,demo")
    assert memory_id.startswith("mem-")
    assert runtime.remember_calls == 1
    assert runtime.last_remember_body is not None
    assert runtime.last_remember_body["text"] == "remember this"
    assert runtime.last_remember_body["source"] == "pydantic-ai"
    assert runtime.last_remember_body["tags"] == "fact,demo"
    assert runtime.last_remember_body["namespace"] == "test-ns"


@respx.mock
async def test_remember_without_tags_omits_tags(runtime: FakeRuntime) -> None:
    runtime.register(respx)
    async with make_deps("test-ns") as deps:
        await remember(ctx(deps), "no tags here")
    assert runtime.last_remember_body is not None
    assert "tags" not in runtime.last_remember_body


@respx.mock
async def test_remember_empty_id_falls_back_to_empty_string() -> None:
    respx.post(f"{BASE_URL}/v1/remember").mock(
        return_value=httpx.Response(200, json={})
    )
    async with make_deps("test-ns") as deps:
        memory_id = await remember(ctx(deps), "x")
    assert memory_id == ""


@respx.mock
async def test_forget_deletes_and_returns_true(runtime: FakeRuntime) -> None:
    runtime.register(respx)
    mid = runtime.seed("test-ns", "delete me")
    async with make_deps("test-ns") as deps:
        ok = await forget(ctx(deps), mid)
    assert ok is True
    assert runtime.forget_calls == 1
    assert mid not in runtime.store["test-ns"]


@respx.mock
async def test_forget_missing_raises_not_found(runtime: FakeRuntime) -> None:
    runtime.register(respx)
    async with make_deps("test-ns") as deps:
        with pytest.raises(NotFoundError):
            await forget(ctx(deps), "does-not-exist")


@respx.mock
async def test_recall_with_none_namespace_omits_namespace(runtime: FakeRuntime) -> None:
    runtime.register(respx)
    async with make_deps(None) as deps:
        await recall(ctx(deps), "anything")
    assert runtime.last_recall_body is not None
    assert "namespace" not in runtime.last_recall_body


@respx.mock
async def test_investigate_weak_when_no_match(runtime: FakeRuntime) -> None:
    runtime.register(respx)
    runtime.seed("test-ns", "totally different")
    async with make_deps("test-ns") as deps:
        response = await investigate(ctx(deps), "zzz-nope")
    assert response.weak is True


@respx.mock
async def test_remember_then_recall_roundtrip(runtime: FakeRuntime) -> None:
    runtime.register(respx)
    async with make_deps("test-ns") as deps:
        await remember(ctx(deps), "the sky is teal today")
        response = await recall(ctx(deps), "teal")
    assert any("teal" in hit.text for hit in response.hits)
