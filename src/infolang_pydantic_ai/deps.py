"""``InfoLangDeps`` -- the dependency object injected into a Pydantic AI agent.

Pydantic AI passes a typed dependencies object to every tool call through
``RunContext.deps``. :class:`InfoLangDeps` carries an ``AsyncInfoLang`` client
plus the namespace/source the InfoLang tools should use, so an ``Agent`` wires
up as::

    agent = Agent("openai:gpt-4o", deps_type=InfoLangDeps, toolsets=[infolang_toolset()])
    deps = InfoLangDeps.from_api_key("il_live_...", namespace="user-42")
    await agent.run("What are my preferences?", deps=deps)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from infolang import AsyncInfoLang

DEFAULT_SOURCE = "pydantic-ai"


@dataclass
class InfoLangDeps:
    """InfoLang dependencies for a Pydantic AI agent.

    Attributes:
        client: The ``AsyncInfoLang`` client the tools call.
        namespace: InfoLang namespace (bank) reads/writes target. When ``None``
            the client's own default namespace (constructor arg or
            ``INFOLANG_NAMESPACE``) applies.
        source: ``source`` tag written on every ``remember`` for provenance.
    """

    client: AsyncInfoLang
    namespace: str | None = None
    source: str | None = DEFAULT_SOURCE

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        *,
        namespace: str | None = None,
        workspace: str | None = None,
        source: str | None = DEFAULT_SOURCE,
        **client_kwargs: Any,
    ) -> InfoLangDeps:
        """Construct deps that own a freshly built managed-cloud client.

        The returned deps own the client; release it with :meth:`aclose` (or use
        ``async with``).
        """

        client = AsyncInfoLang.from_api_key(
            api_key, namespace=namespace, workspace=workspace, **client_kwargs
        )
        return cls(client=client, namespace=namespace, source=source)

    @classmethod
    def from_client(
        cls,
        client: AsyncInfoLang,
        *,
        namespace: str | None = None,
        source: str | None = DEFAULT_SOURCE,
    ) -> InfoLangDeps:
        """Wrap an existing client. The caller keeps ownership of ``client``."""

        return cls(client=client, namespace=namespace or client.namespace, source=source)

    async def aclose(self) -> None:
        """Close the underlying InfoLang client."""

        await self.client.aclose()

    async def __aenter__(self) -> InfoLangDeps:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
