from __future__ import annotations

from infolang import AsyncInfoLang

from infolang_pydantic_ai import DEFAULT_SOURCE, InfoLangDeps


def test_default_source_value() -> None:
    assert DEFAULT_SOURCE == "pydantic-ai"


async def test_from_api_key_builds_async_client() -> None:
    async with InfoLangDeps.from_api_key("il_test", namespace="ns") as deps:
        assert isinstance(deps.client, AsyncInfoLang)
        assert deps.namespace == "ns"
        assert deps.source == DEFAULT_SOURCE
        assert deps.client.namespace == "ns"


async def test_from_api_key_custom_source() -> None:
    async with InfoLangDeps.from_api_key("il_test", source="my-app") as deps:
        assert deps.source == "my-app"


async def test_from_client_inherits_client_namespace() -> None:
    client = AsyncInfoLang.from_api_key("il_test", namespace="from-client")
    try:
        deps = InfoLangDeps.from_client(client)
        assert deps.namespace == "from-client"
        assert deps.client is client
    finally:
        await client.aclose()


async def test_from_client_explicit_namespace_overrides() -> None:
    client = AsyncInfoLang.from_api_key("il_test", namespace="from-client")
    try:
        deps = InfoLangDeps.from_client(client, namespace="override")
        assert deps.namespace == "override"
    finally:
        await client.aclose()


async def test_aclose_is_callable() -> None:
    deps = InfoLangDeps.from_api_key("il_test")
    await deps.aclose()


async def test_context_manager_returns_self() -> None:
    async with InfoLangDeps.from_api_key("il_test") as deps:
        assert isinstance(deps, InfoLangDeps)


async def test_from_api_key_sets_workspace() -> None:
    async with InfoLangDeps.from_api_key("il_test", workspace="acme") as deps:
        assert deps.client.workspace == "acme"


async def test_from_client_default_source() -> None:
    client = AsyncInfoLang.from_api_key("il_test", namespace="ns")
    try:
        deps = InfoLangDeps.from_client(client)
        assert deps.source == DEFAULT_SOURCE
    finally:
        await client.aclose()
