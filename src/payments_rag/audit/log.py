"""Append-only, tamper-evident audit log of every question, answer and source.

* Stored in Postgres; triggers block UPDATE, DELETE and TRUNCATE.
* Each row stores sha256(previous row hash + canonical JSON of the entry), so an edited
  row or a deleted row in the middle breaks the chain. `verify` re-checks all of it.
* Limits (be honest about them): a database superuser can disable triggers, and dropping
  rows from the END of the log leaves a valid chain. The chain detects edits and
  mid-log deletions; to also catch tail truncation, record the head hash somewhere the
  database admin cannot write (a periodic export to separate storage).

Usage:
    python -m payments_rag.audit.log verify
    python -m payments_rag.audit.log tail --n 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from typing import Any

import psycopg

GENESIS = "0" * 64
LOCK_KEY = 7261950001  # arbitrary constant for pg_advisory_xact_lock


def _canonical(entry: dict[str, Any]) -> str:
    return json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _row_hash(prev_hash: str, canonical: str) -> str:
    return hashlib.sha256((prev_hash + canonical).encode("utf-8")).hexdigest()


def build_entry(
    request_id: str,
    question: str,
    result: dict[str, Any] | None,
    min_score: float,
    latency_ms: int,
    error: str | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Shape one request's record. Sources are stored as ids and scores (the text lives in the index)."""
    result = result or {}
    if error:
        outcome = "error"
    else:
        outcome = "escalated" if result.get("escalated") else "answered"
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "question": question,
        "outcome": outcome,
        "answer": result.get("answer"),
        "escalated": bool(result.get("escalated")) if not error else None,
        "reason": result.get("reason"),
        "sources": [{"id": s["id"], "score": s.get("score")} for s in result.get("sources", [])],
        "retrieval": list(result.get("retrieval", [])),
        # Security evidence: passages screened out and any answer blocked by the output policy.
        "quarantined": list(result.get("quarantined", [])),
        "violations": list(result.get("violations", [])),
        "blocked_answer": result.get("_blocked_answer"),
        "model": result.get("model"),
        "cached": result.get("cached"),
        "usage": result.get("usage"),
        "min_score": min_score,
        "latency_ms": latency_ms,
        "error": error,
    }
    if trace_id:  # links this row to its trace in Phoenix; absent when tracing is off, so old rows keep their shape
        entry["trace_id"] = trace_id
    return entry


class AuditLog:
    def __init__(self, url: str):
        self.url = url
        self._ready = False

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.url, autocommit=True)

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS audit_log (
                    seq bigserial PRIMARY KEY,
                    ts timestamptz NOT NULL DEFAULT now(),
                    request_id text NOT NULL,
                    outcome text NOT NULL,
                    canonical text NOT NULL,
                    prev_hash text NOT NULL,
                    row_hash text NOT NULL
                )"""
            )
            conn.execute(
                """CREATE OR REPLACE FUNCTION audit_log_block() RETURNS trigger AS $$
                   BEGIN RAISE EXCEPTION 'audit_log is append-only'; END; $$ LANGUAGE plpgsql"""
            )
            conn.execute("DROP TRIGGER IF EXISTS audit_log_no_change ON audit_log")
            conn.execute(
                "CREATE TRIGGER audit_log_no_change BEFORE UPDATE OR DELETE ON audit_log "
                "FOR EACH ROW EXECUTE FUNCTION audit_log_block()"
            )
            conn.execute("DROP TRIGGER IF EXISTS audit_log_no_truncate ON audit_log")
            conn.execute(
                "CREATE TRIGGER audit_log_no_truncate BEFORE TRUNCATE ON audit_log "
                "FOR EACH STATEMENT EXECUTE FUNCTION audit_log_block()"
            )
        self._ready = True

    def record(self, entry: dict[str, Any]) -> str:
        """Append one entry and return its row hash. Raises on any failure (callers decide to fail closed)."""
        if not self._ready:
            self.init_schema()
        canonical = _canonical(entry)
        with self._connect() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_KEY,))  # serialise the chain
            row = conn.execute("SELECT row_hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else GENESIS
            digest = _row_hash(prev, canonical)
            conn.execute(
                "INSERT INTO audit_log (request_id, outcome, canonical, prev_hash, row_hash) VALUES (%s, %s, %s, %s, %s)",
                (entry["request_id"], entry["outcome"], canonical, prev, digest),
            )
        return digest

    def verify(self) -> dict[str, Any]:
        if not self._ready:
            self.init_schema()
        errors: list[str] = []
        prev = GENESIS
        rows = 0
        with self._connect() as conn:
            for seq, canonical, prev_hash, row_hash in conn.execute(
                "SELECT seq, canonical, prev_hash, row_hash FROM audit_log ORDER BY seq"
            ):
                rows += 1
                if prev_hash != prev:
                    errors.append(f"seq {seq}: chain broken (previous row missing or changed)")
                if _row_hash(prev_hash, canonical) != row_hash:
                    errors.append(f"seq {seq}: hash mismatch (row content was modified)")
                prev = row_hash
        return {"ok": not errors, "rows": rows, "head_hash": prev, "errors": errors}

    def head(self, n: int = 20) -> list[dict[str, Any]]:
        """The first n entries, oldest first (used to take a baseline from early real traffic)."""
        if not self._ready:
            self.init_schema()
        with self._connect() as conn:
            rows = conn.execute("SELECT canonical FROM audit_log ORDER BY seq LIMIT %s", (n,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        if not self._ready:
            self.init_schema()
        with self._connect() as conn:
            rows = conn.execute("SELECT canonical FROM audit_log ORDER BY seq DESC LIMIT %s", (n,)).fetchall()
        return [json.loads(r[0]) for r in reversed(rows)]


def main(argv: list[str] | None = None) -> int:
    from payments_rag.retrieval.store import database_url

    parser = argparse.ArgumentParser(description="Inspect the audit log.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    tail = sub.add_parser("tail")
    tail.add_argument("--n", type=int, default=20)
    args = parser.parse_args(argv)

    import os

    if os.environ.get("AUDIT_TABLE") and os.environ.get("AUDIT_HEAD_TABLE"):  # the serverless deployment
        from payments_rag.audit.dynamo import DynamoAuditLog

        log: Any = DynamoAuditLog(os.environ["AUDIT_TABLE"], os.environ["AUDIT_HEAD_TABLE"])
        if args.command == "tail":
            raise SystemExit("tail is not implemented for the DynamoDB audit log; use verify")
    else:
        log = AuditLog(database_url())
    if args.command == "verify":
        report = log.verify()
        print(f"{'OK' if report['ok'] else 'TAMPERED'}: {report['rows']} rows, head {report['head_hash']}")
        for e in report["errors"]:
            print(f"  {e}")
        return 0 if report["ok"] else 1
    for entry in log.tail(args.n):
        print(json.dumps(entry, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
