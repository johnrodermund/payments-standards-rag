"""Ingest chunks: validate, embed only what changed, store in pgvector.

Usage:
    python -m payments_rag.ingestion.pipeline data/processed/chunks.jsonl \
        --expect pacs.008 --expect pacs.002
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from payments_rag.ingestion.embedding import Embedder
from payments_rag.ingestion.validation import validate_chunks
from payments_rag.retrieval import store

BATCH = 64


class IngestError(RuntimeError):
    pass


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest(
    chunks: list[dict[str, Any]],
    embedder: Embedder,
    conn: Any,
    expected: list[str] | None = None,
) -> dict[str, Any]:
    """Validate then store. Unchanged chunks are not re-embedded, so reruns are free."""
    report = validate_chunks(chunks, expected)
    if not report["passed"]:
        first = [i for i in report["issues"] if i["severity"] == "error"][:5]
        raise IngestError(f"data-quality validation failed ({report['errors']} errors): {first}")

    store.init_schema(conn, embedder.dim, embedder.name)
    # "flow" is always a group: flows whose messages were not all ingested this time are deleted,
    # so the index never holds a flow that points at messages it lacks.
    message_ids = sorted({c["metadata"]["message_id"] for c in chunks} | {"flow"})
    known = store.existing_hashes(conn, message_ids)

    todo = [c for c in chunks if known.get(c["id"]) != _hash(c["text"])]
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        vectors = embedder.embed([c["text"] for c in batch])
        store.upsert(
            conn,
            [
                {
                    "id": c["id"],
                    "kind": c["kind"],
                    "message_id": c["metadata"]["message_id"],
                    "text": c["text"],
                    "metadata": c["metadata"],
                    "content_hash": _hash(c["text"]),
                    "vector": vector,
                }
                for c, vector in zip(batch, vectors, strict=True)
            ],
        )

    deleted = store.delete_missing(conn, message_ids, [c["id"] for c in chunks])
    return {
        "total": len(chunks),
        "embedded": len(todo),
        "skipped": len(chunks) - len(todo),
        "deleted": deleted,
        "warnings": report["warnings"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate, embed and store chunks.")
    parser.add_argument("chunks_path")
    parser.add_argument("--expect", action="append", default=[])
    args = parser.parse_args(argv)

    from payments_rag.ingestion.embedding import FastEmbedEmbedder

    lines = Path(args.chunks_path).read_text(encoding="utf-8").splitlines()
    chunks = [json.loads(line) for line in lines if line.strip()]
    conn = store.connect()
    try:
        stats = ingest(chunks, FastEmbedEmbedder(), conn, args.expect)
    except IngestError as exc:
        print(f"REFUSED: {exc}")
        return 1
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
