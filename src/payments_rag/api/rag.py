"""Orchestration for one question: retrieve, verify, screen, gate on confidence, generate, verify.

Anything the system cannot stand behind is escalated to a human instead of answered.
Keys starting with "_" in the result are for the audit log only and never sent to callers.

`quarantined` lists every passage a control intervened on, as
{"id", "reasons", "action", "removed"} where action is "neutralised" (offending sentences cut,
the rest still shown to the model), "dropped" (nothing usable left) or "rejected_provenance"
(text does not match the committed source). A neutralised passage does not by itself cause an
escalation: the question is escalated only when no usable passage remains.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol

from payments_rag.guardrails.injection import Classifier, check_output, screen_passages
from payments_rag.observability.tracing import span

CITATION = re.compile(r"\[(\d+)\]")
SNIPPET_CHARS = 400

Hits = list[dict[str, Any]]


class SupportsGenerate(Protocol):
    def generate(self, question: str, hits: list[dict[str, Any]]) -> Any: ...


def _source(hit: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(hit["text"].split())
    return {
        "id": hit["id"],
        "kind": hit.get("kind"),
        "score": hit.get("score"),
        "source": (hit.get("metadata") or {}).get("source"),
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
    provenance: Callable[[Hits], tuple[Hits, Hits]] | None = None,
    classifier: Classifier | None = None,
) -> dict[str, Any]:
    with span("retrieve", "RETRIEVER") as step:
        step.content("input.value", question)
        retrieved = retriever(question)
        step.documents(retrieved)
        step.set("rag.retrieved", len(retrieved))
    with span("screen", "GUARDRAIL") as step:
        verified, rejected = provenance(retrieved) if provenance else (retrieved, [])
        kept, screened = screen_passages(verified, classifier=classifier) if guard else (verified, [])
        quarantined = rejected + screened
        step.update(
            rag__guard=guard,
            rag__kept=len(kept),
            rag__quarantined=len(quarantined),
            rag__quarantine_actions=sorted({q["action"] for q in quarantined}),
        )
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

    with span("generate", "LLM") as step:
        step.content("input.value", question)
        gen = generator.generate(question, kept)
        usage = gen.usage or {}
        step.update(
            llm__model_name=gen.model,
            llm__token_count__prompt=usage.get("input_tokens"),
            llm__token_count__completion=usage.get("output_tokens"),
            # The same facts in the gen_ai names LangSmith uses for the model and cost columns.
            gen_ai__system=getattr(getattr(generator, "backend", None), "name", None),
            gen_ai__request__model=gen.model,
            gen_ai__usage__input_tokens=usage.get("input_tokens"),
            gen_ai__usage__output_tokens=usage.get("output_tokens"),
            rag__cached=gen.cached,
            rag__stop_reason=gen.stop_reason,
        )
        step.content("output.value", gen.text)
    meta = {"model": gen.model, "cached": gen.cached, "usage": gen.usage}

    if gen.stop_reason == "refusal":
        return escalate("model_declined", **meta)
    if gen.stop_reason == "max_tokens":
        return escalate("truncated", **meta)
    if not gen.text or gen.text.strip() == "INSUFFICIENT_CONTEXT":
        return escalate("insufficient_context", **meta)

    violations: list[str] = []
    if guard:
        with span("output_check", "GUARDRAIL") as step:
            violations = check_output(gen.text, [h["text"] for h in kept], question)
            step.update(rag__violations=list(violations))
    if violations:
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
