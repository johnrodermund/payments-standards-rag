"""Wires the real dependencies (local embeddings, pgvector, Claude) behind /ask.

Kept apart from the endpoint so tests can swap in fakes and so retrieval changes
touch exactly one place (`build_retriever`).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
    # Refuses retrieved chunks whose text no longer matches the committed extract (see
    # guardrails.provenance). None disables the check.
    provenance: Any = None


def build_retriever(k: int = DEFAULT_K, embedder: Any = None) -> Callable[[str], list[dict[str, Any]]]:
    """RETRIEVAL_BACKEND=postgres (default: pgvector) or memory (the index file at INDEX_PATH, for Lambda)."""
    if embedder is None:
        from payments_rag.ingestion.embedding import FastEmbedEmbedder

        embedder = FastEmbedEmbedder()

    if os.environ.get("RETRIEVAL_BACKEND", "postgres") == "memory":
        from payments_rag.retrieval.memory_store import DEFAULT_INDEX, MemoryIndex

        index = MemoryIndex.load(os.environ.get("INDEX_PATH", DEFAULT_INDEX), expect_model=embedder.name, expect_dim=embedder.dim)

        def retrieve_memory(question: str) -> list[dict[str, Any]]:
            return index.hybrid_search(question, embedder.embed_query(question), k=k)

        return retrieve_memory

    from payments_rag.retrieval import store

    conn = store.connect()

    def retrieve(question: str) -> list[dict[str, Any]]:
        return store.hybrid_search(conn, question, embedder.embed_query(question), k=k)

    return retrieve


def build_provenance() -> Any:
    """On by default when the committed extract is present; PROVENANCE_CHECK=0 turns it off."""
    from payments_rag.guardrails.provenance import DEFAULT_EXTRACT, ProvenanceCheck

    if os.environ.get("PROVENANCE_CHECK", "1") == "0":
        return None
    path = Path(os.environ.get("PROVENANCE_EXTRACT", DEFAULT_EXTRACT))
    return ProvenanceCheck.from_extract(path) if path.exists() else None


def build_audit() -> Any:
    """DynamoDB when AUDIT_TABLE and AUDIT_HEAD_TABLE are set (serverless), else the Postgres audit log."""
    if os.environ.get("AUDIT_TABLE") and os.environ.get("AUDIT_HEAD_TABLE"):
        from payments_rag.audit.dynamo import DynamoAuditLog

        return DynamoAuditLog(os.environ["AUDIT_TABLE"], os.environ["AUDIT_HEAD_TABLE"])
    return AuditLog(database_url())


def build_service() -> Service:
    return Service(
        retriever=build_retriever(int(os.environ.get("RETRIEVAL_K", DEFAULT_K))),
        generator=Generator.from_env(cache=ResponseCache()),
        min_score=float(os.environ.get("ESCALATION_MIN_SCORE", DEFAULT_MIN_SCORE)),
        audit=build_audit(),
        # Fail closed by default: no answer leaves the system unless it was recorded.
        audit_fail_closed=os.environ.get("AUDIT_FAIL_OPEN", "0") != "1",
        provenance=build_provenance(),
    )
