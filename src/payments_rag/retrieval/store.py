"""pgvector storage and similarity search for chunks."""

from __future__ import annotations

import json
import os
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

DEFAULT_URL = "postgresql://payments:payments@localhost:5432/payments"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


def connect(url: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(url or database_url(), autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection, dim: int, model: str) -> None:
    """Create tables if needed; refuse to mix embedding models or dimensions."""
    conn.execute("CREATE TABLE IF NOT EXISTS index_meta (key text PRIMARY KEY, value text NOT NULL)")
    row = conn.execute("SELECT value FROM index_meta WHERE key = 'embedding'").fetchone()
    wanted = json.dumps({"model": model, "dim": dim})
    if row and row[0] != wanted:
        raise RuntimeError(
            f"index was built with {row[0]} but this run uses {wanted}; re-ingest into a fresh database"
        )
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS chunks (
            id text PRIMARY KEY,
            kind text NOT NULL,
            message_id text NOT NULL,
            text text NOT NULL,
            metadata jsonb NOT NULL,
            content_hash text NOT NULL,
            embedding vector({int(dim)}) NOT NULL,
            ingested_at timestamptz NOT NULL DEFAULT now()
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    conn.execute("INSERT INTO index_meta (key, value) VALUES ('embedding', %s) ON CONFLICT DO NOTHING", (wanted,))


def existing_hashes(conn: psycopg.Connection, message_ids: list[str]) -> dict[str, str]:
    rows = conn.execute(
        "SELECT id, content_hash FROM chunks WHERE message_id = ANY(%s)", (message_ids,)
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def upsert(conn: psycopg.Connection, rows: list[dict[str, Any]]) -> None:
    """rows: dicts with id, kind, message_id, text, metadata, content_hash, vector."""
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks (id, kind, message_id, text, metadata, content_hash, embedding)
               VALUES (%(id)s, %(kind)s, %(message_id)s, %(text)s, %(metadata)s, %(content_hash)s, %(vector)s)
               ON CONFLICT (id) DO UPDATE SET
                 kind = EXCLUDED.kind, message_id = EXCLUDED.message_id, text = EXCLUDED.text,
                 metadata = EXCLUDED.metadata, content_hash = EXCLUDED.content_hash,
                 embedding = EXCLUDED.embedding, ingested_at = now()""",
            [{**r, "metadata": json.dumps(r["metadata"])} for r in rows],
        )


def delete_missing(conn: psycopg.Connection, message_ids: list[str], keep_ids: list[str]) -> int:
    """Remove rows of these messages whose ids are no longer produced by the chunker."""
    cur = conn.execute(
        "DELETE FROM chunks WHERE message_id = ANY(%s) AND NOT (id = ANY(%s))", (message_ids, keep_ids)
    )
    return cur.rowcount


def count(conn: psycopg.Connection) -> int:
    return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


def search(
    conn: psycopg.Connection, vector: list[float], k: int = 5, kinds: list[str] | None = None
) -> list[dict[str, Any]]:
    """Nearest chunks by cosine similarity (score = 1 - cosine distance)."""
    params: list[Any] = [vector]
    where = ""
    if kinds:
        where = "WHERE kind = ANY(%s)"
        params.append(kinds)
    params += [vector, k]
    rows = conn.execute(
        f"""SELECT id, kind, text, metadata, 1 - (embedding <=> %s::vector) AS score
            FROM chunks {where}
            ORDER BY embedding <=> %s::vector LIMIT %s""",
        params,
    ).fetchall()
    return [
        {"id": r[0], "kind": r[1], "text": r[2], "metadata": r[3], "score": float(r[4])} for r in rows
    ]
