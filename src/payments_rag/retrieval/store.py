"""pgvector storage and similarity search for chunks."""

from __future__ import annotations

import json
import os
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from payments_rag.retrieval import ranking

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
    # No approximate (HNSW) index: with ~1,200 chunks an exact scan takes a few milliseconds,
    # and HNSW recall was poor on this corpus (many near-identical vectors), silently
    # dropping true nearest neighbours: hit@5 0.38 with the index vs 0.54 exact on the eval set.
    # Revisit (e.g. raise hnsw.ef_search) only if the corpus grows past ~100k chunks.
    conn.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
    # Keyword ranking is BM25 in Python now (ranking.LexicalIndex); drop the old tsvector column.
    conn.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS fts")
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


def message_prefix_from_query(query: str) -> str | None:
    """'... in pacs.008?' -> 'pacs.008'. Only when exactly one message is named."""
    found = {m.group(1).lower() for m in ranking.MESSAGE_MENTION.finditer(query)}
    return found.pop() if len(found) == 1 else None


def _where(kinds: list[str] | None, message_prefix: str | None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if kinds:
        clauses.append("kind = ANY(%s)")
        params.append(kinds)
    if message_prefix:
        # Flow chunks span several messages; keep the ones that use the named message.
        clauses.append("(message_id LIKE %s OR (kind = 'flow' AND metadata->'messages' ? %s))")
        params += [message_prefix + ".%", message_prefix]
    return ("WHERE " + " AND ".join(clauses)) if clauses else "", params


def doc_allowed(doc: dict[str, Any], kinds: list[str] | None, prefix: str | None) -> bool:
    """Kind filter plus message filter; a flow chunk passes the message filter when it uses that message."""
    if kinds and doc["kind"] not in kinds:
        return False
    if not prefix or doc["message_id"].startswith(prefix + "."):
        return True
    return doc["kind"] == "flow" and prefix in (doc.get("metadata") or {}).get("messages", [])


def _row(r: tuple[Any, ...], score: float) -> dict[str, Any]:
    return {"id": r[0], "kind": r[1], "text": r[2], "metadata": r[3], "score": score}


def search(
    conn: psycopg.Connection,
    vector: list[float],
    k: int = 5,
    kinds: list[str] | None = None,
    message_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Nearest chunks by cosine similarity (score = 1 - cosine distance), exact."""
    where, params = _where(kinds, message_prefix)
    rows = conn.execute(
        f"""SELECT id, kind, text, metadata, 1 - (embedding <=> %s::vector) AS score
            FROM chunks {where}
            ORDER BY embedding <=> %s::vector LIMIT %s""",
        [vector, *params, vector, k],
    ).fetchall()
    return [_row(r, float(r[4])) for r in rows]


class _Corpus:
    """All chunks plus their BM25 index, loaded once per connection and reloaded when the
    table changes (row count or latest ingest time differ)."""

    def __init__(self, version: tuple[Any, ...], docs: list[dict[str, Any]]):
        self.version = version
        self.docs = docs
        self.position = {d["id"]: i for i, d in enumerate(docs)}
        self.lexical = ranking.LexicalIndex(d["text"] for d in docs)


_corpora: dict[int, _Corpus] = {}


def _corpus(conn: psycopg.Connection) -> _Corpus:
    version = tuple(conn.execute("SELECT count(*), max(ingested_at) FROM chunks").fetchone())
    cached = _corpora.get(id(conn))
    if cached is None or cached.version != version:
        rows = conn.execute("SELECT id, kind, text, metadata, message_id FROM chunks ORDER BY id").fetchall()
        docs = [{"id": r[0], "kind": r[1], "text": r[2], "metadata": r[3], "message_id": r[4]} for r in rows]
        cached = _corpora[id(conn)] = _Corpus(version, docs)
    return cached


def hybrid_search(
    conn: psycopg.Connection,
    query: str,
    vector: list[float],
    k: int = 5,
    kinds: list[str] | None = None,
    message_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Vector, BM25 and element-name rankings fused (see payments_rag.retrieval.ranking).

    If the question names one message (e.g. "pacs.008") results are limited to it.
    `score` is the cosine similarity of the chunk to the question: comparable across
    queries, so it is the field for a low-confidence cutoff. `rrf` is the fused score
    used for ordering; it only means something relative to other results of the same query.
    """
    prefix = message_prefix or message_prefix_from_query(query)
    corpus = _corpus(conn)
    allowed = [doc_allowed(d, kinds, prefix) for d in corpus.docs]
    by_vector = search(conn, vector, k=ranking.POOL, kinds=kinds, message_prefix=prefix)
    scores = {h["id"]: h["score"] for h in by_vector}
    vector_order = [corpus.position[h["id"]] for h in by_vector if h["id"] in corpus.position]
    prior = None if prefix else [ranking.message_prior(d) for d in corpus.docs]
    fused = ranking.fuse(query, corpus.docs, vector_order, corpus.lexical, allowed, prior)[:k]

    missing = [corpus.docs[i]["id"] for i, _ in fused if corpus.docs[i]["id"] not in scores]
    if missing:
        rows = conn.execute(
            "SELECT id, 1 - (embedding <=> %s::vector) FROM chunks WHERE id = ANY(%s)", (vector, missing)
        ).fetchall()
        scores.update({cid: float(score) for cid, score in rows})
    results = []
    for i, rrf in fused:
        d = corpus.docs[i]
        results.append(
            {"id": d["id"], "kind": d["kind"], "text": d["text"], "metadata": d["metadata"],
             "score": scores[d["id"]], "rrf": rrf}
        )
    return results
