"""Optional OpenTelemetry tracing to LangSmith or Arize Phoenix (OpenInference attribute names, plus the
langsmith.* and gen_ai.* names LangSmith maps to run types, models and token usage).

Off unless LangSmith or a collector is configured (LANGSMITH_TRACING=true with LANGSMITH_API_KEY, or
PHOENIX_COLLECTOR_ENDPOINT / OTEL_EXPORTER_OTLP_ENDPOINT): otherwise every
helper here is a cheap no-op. Tracing can never break a request: every OpenTelemetry call is guarded, and a
failure is logged and ignored. A private TracerProvider is used (never the global one), so this does not
interfere with other OpenTelemetry users in the process, such as DeepEval.

Spans for one /ask request (the root span is opened by the endpoint):

    POST /ask            CHAIN       question, outcome, reason, top score, request id
      retrieve           RETRIEVER   retrieved passages: id, score, content
      screen             GUARDRAIL   provenance and injection screening: what was removed, and how
      generate           LLM         model, token counts, cached or not, stop reason, answer
      output_check       GUARDRAIL   output-policy violations, if any

Privacy: question, passage and answer TEXT is attached to spans so a trace is readable in Phoenix. Set
TRACE_CONTENT=0 to send ids, scores, counts and timings only. The audit log already stores questions and
answers verbatim; a collector is a second place that text lives. LangSmith is a third party: with it on, the
text of every question, the passages found and the answer are copied there (the owner chose this on
2026-11-27; TRACE_CONTENT=0 turns it off). The Phoenix container in docker-compose.yml runs with authentication off and is bound to localhost only: do not publish port 6006.

Configuration (LangSmith wins if both are set):
    LANGSMITH_TRACING            "true" to send traces to LangSmith (needs LANGSMITH_API_KEY)
    LANGSMITH_API_KEY            LangSmith API key (on Lambda, read from SSM at cold start)
    LANGSMITH_OTEL_ENDPOINT      default https://api.smith.langchain.com/otel (EU: https://eu.api.smith.langchain.com/otel)
    LANGSMITH_PROJECT            project the traces land in (default: payments-assistant)
    LANGSMITH_AGENT_ID, LANGSMITH_AGENT_ENVIRONMENT
                                 route to a registered LangSmith agent instead of a project (both or neither)
    PHOENIX_COLLECTOR_ENDPOINT   base URL of the collector, e.g. http://localhost:6006 (spans go to /v1/traces)
    PHOENIX_PROJECT_NAME         project shown in Phoenix (default: payments-assistant)
    PHOENIX_API_KEY              bearer token, needed when Phoenix has authentication on (the default in Phoenix 21+)
    TRACE_CONTENT                "0" to omit text content
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from typing import Any

log = logging.getLogger("payments_rag.tracing")

SERVICE_NAME = "payments-rag"
DEFAULT_PROJECT = "payments-assistant"
MAX_CONTENT_CHARS = 2000

LANGSMITH_OTEL_ENDPOINT = "https://api.smith.langchain.com/otel"
# OpenInference span kind -> LangSmith run type (LangSmith has no guardrail type; a check is a chain step).
LANGSMITH_RUN_TYPES = {"CHAIN": "chain", "RETRIEVER": "retriever", "LLM": "llm", "GUARDRAIL": "chain",
                       "TOOL": "tool", "EMBEDDING": "embedding"}

_provider: Any = None
_tracer: Any = None


def _content_enabled() -> bool:
    return os.environ.get("TRACE_CONTENT", "1") != "0"


def _clip(value: Any) -> str:
    text = str(value)
    return text if len(text) <= MAX_CONTENT_CHARS else text[: MAX_CONTENT_CHARS - 3] + "..."


def _clean(attributes: dict[str, Any]) -> dict[str, Any]:
    """Keep only values OpenTelemetry accepts (str, bool, int, float and lists of them)."""
    out: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, str | bool | int | float):
            out[key] = value
        elif isinstance(value, list | tuple) and all(isinstance(v, str | bool | int | float) for v in value):
            out[key] = list(value)
        else:
            out[key] = _clip(value)
    return out


class _Handle:
    """What the code inside `with span(...)` talks to. Every method is safe to call and never raises."""

    def __init__(self, otel_span: Any | None):
        self._span = otel_span

    def set(self, key: str, value: Any) -> None:
        self.update(**{key: value})

    def update(self, **attributes: Any) -> None:
        if self._span is None:
            return
        try:
            for key, value in _clean(attributes).items():
                self._span.set_attribute(key.replace("__", "."), value)
        except Exception:
            log.debug("tracing: set_attribute failed", exc_info=True)

    def content(self, key: str, value: Any) -> None:
        """Attach text (question, passage, answer) unless TRACE_CONTENT=0."""
        if value is not None and _content_enabled():
            self.set(key, _clip(value))

    def documents(self, hits: list[dict[str, Any]]) -> None:
        """OpenInference retrieval attributes: id and score always, passage text unless TRACE_CONTENT=0."""
        for i, hit in enumerate(hits):
            prefix = f"retrieval.documents.{i}.document"
            self.set(f"{prefix}.id", hit.get("id"))
            if hit.get("score") is not None:
                self.set(f"{prefix}.score", float(hit["score"]))
            self.content(f"{prefix}.content", hit.get("text"))


_NOOP = _Handle(None)


def enabled() -> bool:
    return _tracer is not None


def langsmith_settings() -> tuple[str, dict[str, str]] | None:
    """(OTLP base endpoint, headers) when LangSmith tracing is configured, else None."""
    key = os.environ.get("LANGSMITH_API_KEY")
    if os.environ.get("LANGSMITH_TRACING", "").lower() not in ("1", "true") or not key:
        return None
    headers = {"x-api-key": key}
    agent, environment = os.environ.get("LANGSMITH_AGENT_ID"), os.environ.get("LANGSMITH_AGENT_ENVIRONMENT")
    if agent and environment:  # LangSmith rejects the agent headers alongside a project header
        headers |= {"Langsmith-Agent-Id": agent, "Langsmith-Agent-Environment": environment}
    else:
        headers["Langsmith-Project"] = os.environ.get("LANGSMITH_PROJECT", DEFAULT_PROJECT)
    return os.environ.get("LANGSMITH_OTEL_ENDPOINT", LANGSMITH_OTEL_ENDPOINT), headers


def setup_tracing(endpoint: str | None = None, exporter: Any | None = None, project: str | None = None) -> bool:
    """Turn tracing on. Returns True if spans will be exported. Safe to call repeatedly and when OTel is missing."""
    global _provider, _tracer
    if _tracer is not None:
        return True
    langsmith = None if (endpoint or exporter) else langsmith_settings()
    headers: dict[str, str] | None = None
    if langsmith:
        endpoint, headers = langsmith
    else:
        api_key = os.environ.get("PHOENIX_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    endpoint = endpoint or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if exporter is None and not endpoint:
        return False
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

        resource = Resource.create(
            {
                "service.name": SERVICE_NAME,
                "openinference.project.name": project or os.environ.get("PHOENIX_PROJECT_NAME", DEFAULT_PROJECT),
            }
        )
        provider = TracerProvider(resource=resource)
        if exporter is not None:
            provider.add_span_processor(SimpleSpanProcessor(exporter))  # synchronous: used by tests
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces", headers=headers, timeout=5))
            )
        _provider, _tracer = provider, provider.get_tracer("payments_rag")
    except Exception:
        log.warning("tracing requested but could not be set up; continuing without it", exc_info=True)
        return False
    log.info("tracing enabled (%s)", "LangSmith" if langsmith else "OTLP collector")
    return True


def flush(timeout_millis: int = 2000) -> None:
    """Send buffered spans now. On Lambda the process is frozen as soon as a response is returned, so the
    handler calls this after every invocation; otherwise spans would wait (or be lost) until the next one."""
    if _provider is not None:
        with contextlib.suppress(Exception):
            _provider.force_flush(timeout_millis)


def shutdown_tracing() -> None:
    """Flush and turn tracing off (tests, and a clean exit)."""
    global _provider, _tracer
    provider, _provider, _tracer = _provider, None, None
    if provider is not None:
        with contextlib.suppress(Exception):
            provider.shutdown()


def current_trace_id() -> str | None:
    """Hex trace id of the active span, for linking an audit-log row to its trace."""
    if _tracer is None:
        return None
    try:
        from opentelemetry import trace

        ctx = trace.get_current_span().get_span_context()
        return format(ctx.trace_id, "032x") if ctx.is_valid else None
    except Exception:  # noqa: BLE001 - never fail a request over tracing
        return None


@contextlib.contextmanager
def span(name: str, kind: str = "CHAIN", **attributes: Any) -> Iterator[_Handle]:
    """Open a span named `name`. `kind` is an OpenInference span kind (CHAIN, RETRIEVER, LLM, GUARDRAIL...).

    The body always runs, traced or not. An exception in the body is recorded on the span and re-raised.
    """
    if _tracer is None:
        yield _NOOP
        return
    try:
        base = {"openinference.span.kind": kind, "langsmith.span.kind": LANGSMITH_RUN_TYPES.get(kind, "chain")}
        manager = _tracer.start_as_current_span(name, attributes=_clean({**base, **{k.replace("__", "."): v for k, v in attributes.items()}}))
        otel_span = manager.__enter__()
    except Exception:
        log.debug("tracing: could not start span", exc_info=True)
        yield _NOOP
        return
    try:
        yield _Handle(otel_span)
    except BaseException as exc:
        with contextlib.suppress(Exception):
            manager.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        with contextlib.suppress(Exception):
            manager.__exit__(None, None, None)
