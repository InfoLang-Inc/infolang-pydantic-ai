"""Shared test fixtures.

``FakeRuntime`` is a small in-memory stand-in for the ``il-runtime`` memory
routes the ``infolang`` SDK calls (``POST /v1/recall``, ``POST /v1/remember``,
``DELETE /v1/memories/{id}``, ``GET /v1/memories``, ``POST /v1/execute``). Tests
build a real ``AsyncInfoLang`` pointed at ``BASE_URL`` and let ``respx``
intercept, so they exercise end-to-end request/response shaping rather than
mocking the SDK's methods directly.
"""

from __future__ import annotations

import itertools
import json
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import pytest

from infolang_pydantic_ai import InfoLangDeps

BASE_URL = "https://api.test.infolang.ai"


class FakeRuntime:
    """In-memory fake of the InfoLang memory routes."""

    def __init__(self) -> None:
        self._next_id = itertools.count(1)
        # namespace -> {id: {"id","text","tags"}}
        self.store: dict[str, dict[str, dict[str, Any]]] = {}
        self.recall_calls = 0
        self.remember_calls = 0
        self.forget_calls = 0
        self.execute_calls = 0
        self.list_calls = 0
        self.last_recall_body: dict[str, Any] | None = None
        self.last_remember_body: dict[str, Any] | None = None
        self._poisoned: set[str] = set()

    # -- helpers -----------------------------------------------------------

    def seed(self, namespace: str, text: str, *, tags: str | None = None) -> str:
        mid = f"mem-{next(self._next_id)}"
        self.store.setdefault(namespace, {})[mid] = {
            "id": mid,
            "text": text,
            "tags": tags,
        }
        return mid

    def poison_forget(self, memory_id: str) -> None:
        """Make the next ``forget`` for this id 404 (simulate a delete race)."""

        self._poisoned.add(memory_id)

    # -- route handlers ----------------------------------------------------

    def recall(self, request: httpx.Request) -> httpx.Response:
        self.recall_calls += 1
        body = json.loads(request.content)
        self.last_recall_body = body
        namespace = body.get("namespace") or "default"
        top_k = body.get("top_k")
        query = str(body.get("query", ""))
        records = list(self.store.get(namespace, {}).values())
        scored = [
            {
                "id": rec["id"],
                "text": rec["text"],
                "tags": rec.get("tags"),
                "similarity": 0.95 if query.lower() in rec["text"].lower() else 0.6,
            }
            for rec in records
        ]
        scored.sort(key=lambda hit: hit["similarity"], reverse=True)
        if isinstance(top_k, int):
            scored = scored[:top_k]
        return httpx.Response(200, json={"hits": scored})

    def remember(self, request: httpx.Request) -> httpx.Response:
        self.remember_calls += 1
        body = json.loads(request.content)
        self.last_remember_body = body
        namespace = body.get("namespace") or "default"
        mid = self.seed(namespace, body["text"], tags=body.get("tags"))
        return httpx.Response(200, json={"id": mid})

    def forget(self, request: httpx.Request) -> httpx.Response:
        self.forget_calls += 1
        memory_id = unquote(request.url.path.rsplit("/", 1)[-1])
        if memory_id in self._poisoned:
            self._poisoned.discard(memory_id)
            return httpx.Response(404, json={"error": "no such memory"})
        for namespace_store in self.store.values():
            if memory_id in namespace_store:
                del namespace_store[memory_id]
                return httpx.Response(200, json={})
        return httpx.Response(404, json={"error": "no such memory"})

    def list_recent(self, request: httpx.Request) -> httpx.Response:
        self.list_calls += 1
        qs = parse_qs(urlparse(str(request.url)).query)
        namespace = qs.get("namespace", ["default"])[0]
        records = list(self.store.get(namespace, {}).values())
        return httpx.Response(200, json={"memories": records})

    def execute(self, request: httpx.Request) -> httpx.Response:
        self.execute_calls += 1
        body = json.loads(request.content)
        op = body["operations"][0]
        args = op["args"]
        namespace = args.get("namespace") or "default"
        results = []
        for item in args["items"]:
            mid = self.seed(namespace, item["text"], tags=item.get("tags"))
            results.append({"id": mid})
        return httpx.Response(
            200,
            json={
                "results": [
                    {"op": "remember_batch", "ok": True, "payload": {"results": results}}
                ]
            },
        )

    def register(self, mock: Any) -> None:
        mock.post(f"{BASE_URL}/v1/recall").mock(side_effect=self.recall)
        mock.post(f"{BASE_URL}/v1/remember").mock(side_effect=self.remember)
        mock.post(f"{BASE_URL}/v1/execute").mock(side_effect=self.execute)
        mock.get(url__regex=rf"{BASE_URL}/v1/memories(\?.*)?$").mock(
            side_effect=self.list_recent
        )
        mock.delete(url__regex=rf"{BASE_URL}/v1/memories/.+").mock(
            side_effect=self.forget
        )


def make_deps(namespace: str | None = "test-ns", **kwargs: Any) -> InfoLangDeps:
    """Build ``InfoLangDeps`` whose client targets the fake runtime."""

    return InfoLangDeps.from_api_key(
        "il_test_key", base_url=BASE_URL, namespace=namespace, **kwargs
    )


@pytest.fixture
def runtime() -> FakeRuntime:
    return FakeRuntime()
