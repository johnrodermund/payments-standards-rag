"""POST /ingest, GET /ingest/{id}, GET /ingest: start and follow a background re-index.

Protection (this endpoint writes to the index and burns CPU, so it is closed by default):
* Disabled (503) unless INGEST_API_TOKEN is set to at least 16 characters.
* Every call needs `Authorization: Bearer <token>`; compared in constant time.
* The request chooses only from an allowlist (which messages, and the committed extract or the pinned
  e-Repository zip). It can never name a path or URL.
* One job at a time (409 while one is queued or running).
"""

from __future__ import annotations

import hmac
import os
import threading
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from payments_rag.ingestion.jobs import (
    ALLOWED_MESSAGES,
    ActiveJobError,
    IngestService,
    JobStore,
    default_runner,
)

router = APIRouter()
MIN_TOKEN_CHARS = 16

_service: IngestService | None = None
_lock = threading.Lock()


def get_ingest_service() -> IngestService:
    global _service
    with _lock:
        if _service is None:
            from payments_rag.retrieval.store import database_url

            url = database_url()
            _service = IngestService(JobStore(url), default_runner(url))
        return _service


def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
    expected = os.environ.get("INGEST_API_TOKEN", "")
    if len(expected) < MIN_TOKEN_CHARS:
        raise HTTPException(status_code=503, detail=f"ingest is disabled: set INGEST_API_TOKEN (at least {MIN_TOKEN_CHARS} characters)")
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token", headers={"WWW-Authenticate": "Bearer"})


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a "path" or "url" field is an error, not silently ignored

    source: Literal["extract", "repository"] = "extract"
    messages: list[Literal["pacs.008", "pacs.002"]] = Field(default_factory=lambda: sorted(ALLOWED_MESSAGES), min_length=1)

    @field_validator("messages")
    @classmethod
    def unique_sorted(cls, value: list[str]) -> list[str]:
        return sorted(set(value))


@router.post("/ingest", status_code=202, dependencies=[Depends(require_token)])
def start_ingest(request: IngestRequest, service: Annotated[IngestService, Depends(get_ingest_service)]) -> JSONResponse:
    try:
        job = service.submit(request.model_dump())
    except ActiveJobError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "an ingest job is already active", "active_job_id": exc.active["id"], "status": exc.active["status"]},
        ) from exc
    url = f"/ingest/{job['id']}"
    return JSONResponse({"job_id": job["id"], "status": job["status"], "status_url": url}, status_code=202, headers={"Location": url})


@router.get("/ingest/{job_id}", dependencies=[Depends(require_token)])
def get_ingest_job(job_id: str, service: Annotated[IngestService, Depends(get_ingest_service)]) -> dict[str, Any]:
    job = service.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such ingest job")
    return job


@router.get("/ingest", dependencies=[Depends(require_token)])
def list_ingest_jobs(service: Annotated[IngestService, Depends(get_ingest_service)]) -> dict[str, Any]:
    return {"jobs": service.store.recent(20)}
