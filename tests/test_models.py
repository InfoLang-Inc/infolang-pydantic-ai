from __future__ import annotations

from infolang import Chunk, RecallResult

from infolang_pydantic_ai import RecallHit, RecallResponse


def test_recall_hit_defaults() -> None:
    hit = RecallHit(id="m1", text="hello")
    assert hit.score is None
    assert hit.tags is None


def test_recall_hit_full() -> None:
    hit = RecallHit(id="m1", text="hello", score=0.9, tags="a,b")
    assert hit.score == 0.9
    assert hit.tags == "a,b"


def test_from_result_maps_all_fields() -> None:
    result = RecallResult(
        chunks=[
            Chunk(id="m1", text="first", score=0.9, tags="a,b"),
            Chunk(id="m2", text="second", score=0.88, tags=None),
        ],
        namespace="user-1",
    )
    response = RecallResponse.from_result("q", result)
    assert response.query == "q"
    assert response.namespace == "user-1"
    assert [h.id for h in response.hits] == ["m1", "m2"]
    assert response.hits[0].tags == "a,b"
    assert response.hits[0].score == 0.9


def test_from_result_weak_true_when_top_below_floor() -> None:
    result = RecallResult(chunks=[Chunk(id="m1", text="x", score=0.5)])
    response = RecallResponse.from_result("q", result)
    assert response.weak is True


def test_from_result_weak_false_when_top_above_floor() -> None:
    result = RecallResult(chunks=[Chunk(id="m1", text="x", score=0.9)])
    response = RecallResponse.from_result("q", result)
    assert response.weak is False


def test_from_result_empty_is_not_weak() -> None:
    response = RecallResponse.from_result("q", RecallResult(chunks=[]))
    assert response.hits == []
    assert response.weak is False


def test_from_result_namespace_fallback_when_result_has_none() -> None:
    result = RecallResult(chunks=[])
    response = RecallResponse.from_result("q", result, namespace="fallback-ns")
    assert response.namespace == "fallback-ns"


def test_from_result_prefers_result_namespace_over_fallback() -> None:
    result = RecallResult(chunks=[], namespace="real-ns")
    response = RecallResponse.from_result("q", result, namespace="fallback-ns")
    assert response.namespace == "real-ns"
