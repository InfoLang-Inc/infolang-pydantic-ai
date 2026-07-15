"""Typed result models for the InfoLang Pydantic AI tools.

Pydantic AI is schema-first: tool return types become part of the model-facing
contract. Rather than hand back the raw ``infolang`` SDK objects (whose fields
carry compact wire aliases), the recall/investigate tools return these small,
self-documenting models so the schema an agent sees is stable and readable.
"""

from __future__ import annotations

from infolang import RecallResult
from pydantic import BaseModel, Field


class RecallHit(BaseModel):
    """A single memory chunk recalled from InfoLang."""

    id: str
    """Stable id of the stored memory (use with ``forget``)."""

    text: str
    """The remembered text."""

    score: float | None = None
    """Similarity score in ``[0, 1]`` when the runtime reports one."""

    tags: str | None = None
    """Comma-separated tags stored alongside the memory, if any."""


class RecallResponse(BaseModel):
    """Structured result of a ``recall`` / ``investigate`` tool call."""

    query: str
    """The query that produced these hits."""

    namespace: str | None = None
    """InfoLang namespace (bank) the recall ran against."""

    hits: list[RecallHit] = Field(default_factory=list)
    """Matching memory chunks, most relevant first."""

    weak: bool = False
    """True when the top hit scores below InfoLang's 0.85 confidence floor."""

    @classmethod
    def from_result(
        cls, query: str, result: RecallResult, *, namespace: str | None = None
    ) -> RecallResponse:
        """Build a :class:`RecallResponse` from an SDK :class:`RecallResult`."""

        hits = [
            RecallHit(id=chunk.id, text=chunk.text, score=chunk.score, tags=chunk.tags)
            for chunk in result.chunks
        ]
        return cls(
            query=query,
            namespace=result.namespace or namespace,
            hits=hits,
            weak=result.weak,
        )
