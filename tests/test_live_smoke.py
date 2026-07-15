"""Optional live smoke test against the real InfoLang API.

Skipped unless ``INFOLANG_API_KEY`` is set -- NOT part of the default ``pytest``
run and excluded from the coverage gate. Only touches namespaces prefixed
``ittest-pydantic-ai-`` and cleans up in a ``finally`` block, so it is safe to
run against a shared account.

Run it with::

    INFOLANG_API_KEY=il_live_... pytest tests/test_live_smoke.py -v
"""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from infolang_pydantic_ai import InfoLangDeps, forget, recall, remember

pytestmark = pytest.mark.skipif(
    not os.environ.get("INFOLANG_API_KEY"),
    reason="live smoke test requires INFOLANG_API_KEY",
)


def _ctx(deps: InfoLangDeps) -> Any:
    return SimpleNamespace(deps=deps)


async def test_live_round_trip() -> None:
    namespace = f"ittest-pydantic-ai-{uuid.uuid4().hex[:8]}"
    memory_id = ""
    async with InfoLangDeps.from_api_key(
        os.environ["INFOLANG_API_KEY"], namespace=namespace
    ) as deps:
        try:
            memory_id = await remember(
                _ctx(deps), "InfoLang pydantic-ai live smoke fact", tags="smoke"
            )
            assert memory_id

            response = await recall(_ctx(deps), "smoke fact")
            assert any("smoke" in hit.text for hit in response.hits)
        finally:
            if memory_id:
                await forget(_ctx(deps), memory_id)
