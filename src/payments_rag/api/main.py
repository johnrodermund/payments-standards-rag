import logging
import threading
import time
import uuid
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from payments_rag.api.rag import answer_question
from payments_rag.api.service import Service, build_service
from payments_rag.audit.log import build_entry
from payments_rag.llm.budget import BudgetExceededError
from payments_rag.llm.generation import ReplayMissError

app = FastAPI(title="Payments standards assistant")
log = logging.getLogger("payments_rag")

_service: Service | None = None
_lock = threading.Lock()


def get_service() -> Service:
    """Build the real dependencies on first use (loads the embedding model, opens the DB)."""
    global _service
    with _lock:
        if _service is None:
            _service = build_service()
        return _service


class AskRequest(BaseModel):
    question: str = Field(max_length=1000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _record(service: Service, entry: dict[str, Any]) -> None:
    """Write the audit entry. Fail closed unless the operator opted out."""
    try:
        service.audit.record(entry)
    except Exception as exc:
        log.exception("audit log write failed for request %s", entry["request_id"])
        if service.audit_fail_closed:
            raise HTTPException(status_code=503, detail="audit log unavailable; answer withheld") from exc


@app.post("/ask")
def ask(request: AskRequest, service: Annotated[Service, Depends(get_service)]) -> dict[str, Any]:
    request_id = str(uuid.uuid4())
    started = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    try:
        result = answer_question(request.question, service.retriever, service.generator, service.min_score)
    except (ReplayMissError, BudgetExceededError) as exc:
        _record(service, build_entry(request_id, request.question, None, service.min_score, elapsed_ms(), error=f"{type(exc).__name__}: {exc}"))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        _record(service, build_entry(request_id, request.question, None, service.min_score, elapsed_ms(), error=f"{type(exc).__name__}: {exc}"))
        raise

    _record(service, build_entry(request_id, request.question, result, service.min_score, elapsed_ms()))
    public = {k: v for k, v in result.items() if not k.startswith("_")}  # audit-only fields stay internal
    return {"request_id": request_id, **public}
