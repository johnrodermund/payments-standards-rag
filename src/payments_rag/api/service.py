"""Wires the real dependencies (local embeddings, pgvector, Claude) behind /ask.

Kept apart from the endpoint so tests can swap in fakes and so retrieval changes
touch exactly one place (`build_retriever`).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from payments_rag.audit.log import AuditLog
from payments_rag.llm.cache import ResponseCache
from payments_rag.llm.generation import Generator
from payments_rag.retrieval.store import database_url

# Provisional. Calibrate against on-topic vs off-topic scores from the retrieval evaluation
# (cosine similarity of the top chunk) before relying on it.
DEFAULT_MIN_SCORE = 0.6
DEFAULT_K = 5


@dataclass
class Service:
    retriever: Callable[[str], list[dict[str, Any]]]
    generator: Generator
    min_score: float
    audit: Any  # anything with .record(entry) -> hash; AuditLog in production
    audit_fail_closed: bool = True


def build_retriever(k: int = DEFAULT_K) -> Callable[[str], list[dict[str, Any]]]:
    from payments_rag.ingestion.embedding import FastEmbedEmbedder
    from payments_rag.retrieval import store

    embedder = FastEmbedEmbedder()
    conn = store.connect()

    def retrieve(question: str) -> list[dict[str, Any]]:
        return store.hybrid_search(conn, question, embedder.embed_query(question), k=k)

    return retrieve


def build_service() -> Service:
    return Service(
        retriever=build_retriever(int(os.environ.get("RETRIEVAL_K", DEFAULT_K))),
        generator=Generator(cache=ResponseCache()),
        min_score=float(os.environ.get("ESCALATION_MIN_SCORE", DEFAULT_MIN_SCORE)),
        audit=AuditLog(database_url()),
        # Fail closed by default: no answer leaves the system unless it was recorded.
        audit_fail_closed=os.environ.get("AUDIT_FAIL_OPEN", "0") != "1",
    )
