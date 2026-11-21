"""Orchestration for one question: retrieve, screen, gate on confidence, generate, verify.

Anything the system cannot stand behind is escalated to a human instead of answered.
Keys starting with "_" in the result are for the audit log only and never sent to callers.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol

from payments_rag.guardrails.injection import check_output, screen_passages

CITATION = re.compile(r"\[(\d+)\]")
SNIPPET_CHARS = 400


class SupportsGenerate(Protocol):
    def generate(self, question: str, hits: list[dict[str, Any]]) -> Any: ...


def _source(hit: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(hit["text"].split())
    return {
        "id": hit["id"],
        "kind": hit.get("kind"),
        "score": hit.get("score"),
        "snippet": text if len(text) <= SNIPPET_CHARS else text[: SNIPPET_CHARS - 3] + "...",
    }


def _retrieval(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Everything retrieved, ids and scores only: lets a reviewer (and the audit log) see what was considered."""
    return [{"id": h["id"], "score": h.get("score")} for h in hits]


def answer_question(
    question: str,
    retriever: Callable[[str], list[dict[str, Any]]],
    generator: SupportsGenerate,
    min_score: float,
    guard: bool = True,
) -> dict[str, Any]:
    retrieved = retriever(question)
    kept, quarantined = screen_passages(retrieved) if guard else (retrieved, [])
    base = {"retrieval": _retrieval(retrieved), "retrieved": len(retrieved), "quarantined": quarantined}

    def escalate(reason: str, **extra: Any) -> dict[str, Any]:
        return {
            "answer": None,
            "escalated": True,
            "reason": reason,
            "sources": [_source(h) for h in kept],
            **base,
            **extra,
        }

    if not kept and quarantined:
        return escalate("all_passages_quarantined")
    if not kept or (kept[0].get("score") or 0.0) < min_score:
        return escalate("low_retrieval_confidence")

    gen = generator.generate(question, kept)
    meta = {"model": gen.model, "cached": gen.cached, "usage": gen.usage}

    if gen.stop_reason == "refusal":
        return escalate("model_declined", **meta)
    if gen.stop_reason == "max_tokens":
        return escalate("truncated", **meta)
    if not gen.text or gen.text.strip() == "INSUFFICIENT_CONTEXT":
        return escalate("insufficient_context", **meta)

    if guard and (violations := check_output(gen.text)):
        return escalate("output_policy_violation", violations=violations, _blocked_answer=gen.text, **meta)

    cited = sorted({int(n) for n in CITATION.findall(gen.text) if 1 <= int(n) <= len(kept)})
    if not cited:
        return escalate("no_valid_citation", **meta)

    return {
        "answer": gen.text,
        "escalated": False,
        "reason": None,
        "sources": [_source(kept[n - 1]) for n in cited],
        "_contexts": [h["text"] for h in kept],
        **base,
        **meta,
    }
