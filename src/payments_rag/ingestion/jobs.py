"""Background ingest jobs: run the ingestion pipeline off the request thread and report on it.

Design:
* One job at a time. `create_if_idle` takes a Postgres advisory lock, so this also holds across several
  app replicas: a second request while a job is queued or running is refused (the API returns 409).
* Job state lives in the `ingest_jobs` table (queued -> running -> succeeded | failed), with the current stage.
  The work runs in a daemon thread in the process that accepted the request. If that process dies, the job is
  left `running`; after `STALE_SECONDS` without progress the next request marks it failed ("interrupted")
  and a new job may start. A job is never silently resumed.
* The caller chooses WHAT to ingest only from a fixed allowlist; the source file is never taken from the
  request. `source="extract"` re-chunks the committed extract (data/processed/iso20022_pacs.json); `source=
  "repository"` re-parses the ISO e-Repository zip configured by the operator (INGEST_SOURCE_ZIP) and refuses
  it unless its sha256 equals the one pinned in the committed extract.
* Re-ingesting is cheap and safe: unchanged chunks are not re-embedded, removed chunks are deleted, and a
  validation failure (including suspected injection) writes nothing.

Limits: a job that is killed mid-embedding leaves the index partly updated (chunks are upserted in
batches, not in one transaction); rerunning it completes the update. Progress is by stage, not by chunk.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

log = logging.getLogger("payments_rag.ingest")

ALLOWED_MESSAGES = ("pacs.008", "pacs.002")
SOURCES = ("extract", "repository")
DEFAULT_EXTRACT = Path("data/processed/iso20022_pacs.json")
DEFAULT_ZIP = Path("data/raw/iso20022/20260904_ISO20022_2013_eRepository.zip")
STALE_SECONDS = 1800
LOCK_KEY = 7261950002  # arbitrary constant for pg_advisory_xact_lock (the audit log uses ...001)
ACTIVE = ("queued", "running")
MAX_ERROR_CHARS = 2000
FIELDS = ("status", "stage", "result", "error", "started_at", "finished_at")

Runner = Callable[[dict[str, Any], Callable[[str], None]], dict[str, Any]]


class ActiveJobError(RuntimeError):
    def __init__(self, active: dict[str, Any]):
        super().__init__(f"ingest job {active['id']} is already {active['status']}")
        self.active = active


def _row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    out["id"] = str(out["id"])
    for key in ("created_at", "started_at", "finished_at", "updated_at"):
        if out.get(key) is not None:
            out[key] = out[key].isoformat()
    return out


class JobStore:
    def __init__(self, url: str, stale_seconds: int | None = None):
        self.url = url
        self.stale_seconds = stale_seconds if stale_seconds is not None else int(os.environ.get("INGEST_STALE_SECONDS", STALE_SECONDS))
        self._ready = False

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.url, autocommit=True, row_factory=dict_row)

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS ingest_jobs (
                    id uuid PRIMARY KEY,
                    status text NOT NULL CHECK (status IN ('queued','running','succeeded','failed')),
                    stage text NOT NULL DEFAULT 'queued',
                    params jsonb NOT NULL,
                    result jsonb,
                    error text,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    started_at timestamptz,
                    finished_at timestamptz,
                    updated_at timestamptz NOT NULL DEFAULT now()
                )"""
            )
        self._ready = True

    def _ensure(self) -> None:
        if not self._ready:
            self.init_schema()

    def create_if_idle(self, params: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Insert a queued job unless one is active. Returns (new_job, None) or (None, active_job)."""
        self._ensure()
        with self._connect() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_KEY,))
            conn.execute(
                """UPDATE ingest_jobs SET status = 'failed', stage = 'interrupted', finished_at = now(), updated_at = now(), error = %s
                   WHERE status IN ('queued', 'running') AND updated_at < now() - make_interval(secs => %s)""",
                (
                    f"interrupted: no progress reported for {self.stale_seconds} seconds (the worker process probably stopped)",
                    self.stale_seconds,
                ),
            )
            active = conn.execute(
                "SELECT * FROM ingest_jobs WHERE status IN ('queued', 'running') ORDER BY created_at LIMIT 1"
            ).fetchone()
            if active:
                return None, _row(active)
            job_id = uuid.uuid4()
            created = conn.execute(
                "INSERT INTO ingest_jobs (id, status, stage, params) VALUES (%s, 'queued', 'queued', %s) RETURNING *",
                (job_id, Jsonb(params)),
            ).fetchone()
            return _row(created), None

    def update(self, job_id: str, **fields: Any) -> None:
        unknown = set(fields) - set(FIELDS)
        if unknown:
            raise ValueError(f"unknown job fields: {sorted(unknown)}")
        values = {k: (Jsonb(v) if k == "result" and v is not None else v) for k, v in fields.items()}
        assignments = ", ".join([f"{k} = %s" for k in values] + ["updated_at = now()"])
        with self._connect() as conn:
            conn.execute(f"UPDATE ingest_jobs SET {assignments} WHERE id = %s", (*values.values(), job_id))

    def get(self, job_id: str) -> dict[str, Any] | None:
        self._ensure()
        try:
            uuid.UUID(job_id)
        except ValueError:
            return None
        with self._connect() as conn:
            return _row(conn.execute("SELECT * FROM ingest_jobs WHERE id = %s", (job_id,)).fetchone())

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        self._ensure()
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM ingest_jobs ORDER BY created_at DESC LIMIT %s", (n,)).fetchall()
        return [_row(r) for r in rows]  # type: ignore[misc]


def default_runner(
    database_url: str,
    embedder_factory: Callable[[], Any] | None = None,
    extract_path: Path | None = None,
    zip_path: Path | None = None,
) -> Runner:
    """The real pipeline: load -> chunk -> validate, embed what changed, store."""

    def run(params: dict[str, Any], progress: Callable[[str], None]) -> dict[str, Any]:
        from payments_rag.ingestion.chunking import chunk_payload
        from payments_rag.ingestion.pipeline import ingest
        from payments_rag.retrieval import store

        keys = list(params["messages"])
        committed = Path(extract_path or os.environ.get("INGEST_EXTRACT", DEFAULT_EXTRACT))
        extract = json.loads(committed.read_text(encoding="utf-8"))

        progress("loading")
        if params["source"] == "repository":
            from payments_rag.ingestion.iso20022_repository import _sha256, extract_messages

            zip_file = Path(zip_path or os.environ.get("INGEST_SOURCE_ZIP", DEFAULT_ZIP))
            if not zip_file.exists():
                raise FileNotFoundError(f"source zip is not on the server: {zip_file.name}")
            pinned, actual = extract["source"]["sha256"], _sha256(zip_file)
            if actual != pinned:
                raise ValueError(f"source zip sha256 {actual} does not match the pinned {pinned}; refusing to ingest it")
            payload = {"source": extract["source"], "messages": extract_messages(zip_file, keys)}
        else:
            wanted = tuple(f"{k}." for k in keys)
            selected = [m for m in extract["messages"] if m["id"].startswith(wanted)]
            missing = [k for k in keys if not any(m["id"].startswith(f"{k}.") for m in selected)]
            if missing:
                raise ValueError(f"message(s) not in the committed extract: {', '.join(missing)}")
            payload = {"source": extract["source"], "messages": selected}

        progress("chunking")
        chunks = chunk_payload(payload)

        progress("storing")  # validates, embeds only what changed, upserts, deletes what disappeared
        embedder = (embedder_factory or _fastembed)()
        conn = store.connect(database_url)
        try:
            stats = ingest(chunks, embedder, conn, keys)
        finally:
            conn.close()
        return {**stats, "messages": [m["id"] for m in payload["messages"]], "source": params["source"]}

    return run


def _fastembed() -> Any:
    from payments_rag.ingestion.embedding import FastEmbedEmbedder

    return FastEmbedEmbedder()


class IngestService:
    def __init__(self, store: JobStore, runner: Runner):
        self.store = store
        self.runner = runner

    def submit(self, params: dict[str, Any]) -> dict[str, Any]:
        """Queue a job and start it in the background. Raises ActiveJobError if one is already active."""
        job, active = self.store.create_if_idle(params)
        if active:
            raise ActiveJobError(active)
        assert job is not None
        thread = threading.Thread(target=self._run, args=(job["id"], params), name=f"ingest-{job['id'][:8]}", daemon=True)
        thread.start()
        return job

    def _run(self, job_id: str, params: dict[str, Any]) -> None:
        def progress(stage: str) -> None:
            self.store.update(job_id, stage=stage)

        try:
            self.store.update(job_id, status="running", stage="starting", started_at=datetime.now(UTC))
            result = self.runner(params, progress)
        except Exception as exc:
            log.exception("ingest job %s failed", job_id)
            self._finish(job_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS])
        else:
            log.info("ingest job %s succeeded: %s", job_id, result)
            self._finish(job_id, status="succeeded", result=result)

    def _finish(self, job_id: str, **fields: Any) -> None:
        try:
            self.store.update(job_id, stage="done" if fields["status"] == "succeeded" else "failed", finished_at=datetime.now(UTC), **fields)
        except Exception:
            log.exception("could not record the outcome of ingest job %s", job_id)
