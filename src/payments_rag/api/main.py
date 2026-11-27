import logging
import threading
import time
import uuid
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from payments_rag.api import ratelimit
from payments_rag.api.flows import router as flows_router
from payments_rag.api.ingest import router as ingest_router
from payments_rag.api.page import router as page_router
from payments_rag.api.rag import answer_question
from payments_rag.api.service import Service, build_service
from payments_rag.audit.log import build_entry
from payments_rag.llm.backends import LLMUnavailableError
from payments_rag.llm.budget import BudgetExceededError
from payments_rag.llm.generation import ReplayMissError
from payments_rag.observability.tracing import current_trace_id, setup_tracing, span

app = FastAPI(title="Payments standards assistant")
app.include_router(ingest_router)
app.include_router(flows_router)
app.include_router(page_router)
log = logging.getLogger("payments_rag")
setup_tracing()  # no-op unless PHOENIX_COLLECTOR_ENDPOINT (or OTEL_EXPORTER_OTLP_ENDPOINT) is set

_service: Service | None = None
_lock = threading.Lock()
_UNSET = object()
_limiter: Any = _UNSET  # built from RATE_LIMIT_TABLE on first use; None means no limit


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


def rate_limit(http_request: Request) -> None:
    global _limiter
    if _limiter is _UNSET:
        _limiter = ratelimit.limiter_from_env()
    retry_after = ratelimit.check(_limiter, http_request)
    if retry_after is not None:
        raise HTTPException(status_code=429, detail="too many questions; please wait and try again",
                            headers={"Retry-After": str(retry_after)})


@app.post("/ask", dependencies=[Depends(rate_limit)])
def ask(request: AskRequest, service: Annotated[Service, Depends(get_service)]) -> dict[str, Any]:
    request_id = str(uuid.uuid4())
    with span("POST /ask", "CHAIN", rag__request_id=request_id, langsmith__is_root=True) as root:
        root.content("input.value", request.question)
        return _ask(request, service, request_id, root)


def _ask(request: AskRequest, service: Service, request_id: str, root: Any) -> dict[str, Any]:
    started = time.perf_counter()
    trace_id = current_trace_id()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    def entry_for(result: dict[str, Any] | None, error: str | None = None) -> dict[str, Any]:
        return build_entry(request_id, request.question, result, service.min_score, elapsed_ms(), error=error, trace_id=trace_id)

    try:
        result = answer_question(
            request.question, service.retriever, service.generator, service.min_score,
            provenance=getattr(service, "provenance", None),
        )
    except (ReplayMissError, BudgetExceededError, LLMUnavailableError) as exc:
        root.update(rag__outcome="error", rag__error=type(exc).__name__)
        _record(service, entry_for(None, f"{type(exc).__name__}: {exc}"))
        # Provider errors can name accounts or resources: the caller gets a generic message, the audit log the detail.
        if isinstance(exc, LLMUnavailableError):
            detail = "the language model is temporarily unavailable"
        elif isinstance(exc, BudgetExceededError):
            detail = "monthly question budget reached; please try again next month"  # no figures or paths
        else:
            detail = str(exc)
        raise HTTPException(status_code=503, detail=detail) from exc
    except Exception as exc:
        root.update(rag__outcome="error", rag__error=type(exc).__name__)
        _record(service, entry_for(None, f"{type(exc).__name__}: {exc}"))
        raise

    scores = [h["score"] for h in result.get("retrieval", []) if h.get("score") is not None]
    root.update(
        rag__outcome="escalated" if result.get("escalated") else "answered",
        rag__reason=result.get("reason"),
        rag__top_score=max(scores) if scores else None,
        rag__min_score=service.min_score,
    )
    root.content("output.value", result.get("answer") or f"ESCALATED: {result.get('reason')}")
    _record(service, entry_for(result))
    public = {k: v for k, v in result.items() if not k.startswith("_")}  # audit-only fields stay internal
    return {"request_id": request_id, **public}
