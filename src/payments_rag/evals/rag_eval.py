"""Score the whole RAG pipeline on the golden set.

Three families of metrics:

* retrieval   hit@1/3/5 and MRR: does the evidence chunk appear in what was retrieved? (no LLM)
* behaviour   answered / escalated rates: answerable questions should be answered, unanswerable
              ones handed to a human. (no LLM beyond the generation itself)
* judged      DeepEval faithfulness and answer relevancy on the answers that were produced.
              Judge is "jev" (System One) or "claude"; see judge.py.

Usage:
    uv run python -m payments_rag.evals.rag_eval --judge jev --out evals/results/latest.json
    uv run python -m payments_rag.evals.rag_eval --judge none --limit 10     # retrieval + behaviour only

Generation is cached, so rerunning an unchanged setup costs nothing and `--cache-dir` can point at a
committed recording for CI replay.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from payments_rag.api.rag import answer_question


def load_golden(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def run_golden(
    golden: list[dict[str, Any]],
    retriever: Callable[[str], list[dict[str, Any]]],
    generator: Any,
    min_score: float,
    guard: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    for item in golden:
        result = answer_question(item["question"], retriever, generator, min_score, guard=guard)
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "answerable": item["answerable"],
                "cohort": item.get("cohort", "original"),
                "reference_ids": item["reference_ids"],
                "reference_answer": item["reference_answer"],
                "retrieved_ids": [r["id"] for r in result["retrieval"]],
                "answer": result["answer"],
                "escalated": result["escalated"],
                "reason": result["reason"],
                "contexts": result.get("_contexts", []),
                "model": result.get("model"),
                "cached": result.get("cached"),
                "usage": result.get("usage"),
            }
        )
    return rows


def run_retrieval_only(
    golden: list[dict[str, Any]], retriever: Callable[[str], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Retrieval without generation: needs no API key, so it can gate every CI run."""
    rows = []
    for item in golden:
        hits = retriever(item["question"])
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "answerable": item["answerable"],
                "cohort": item.get("cohort", "original"),
                "reference_ids": item["reference_ids"],
                "retrieved_ids": [h["id"] for h in hits],
                "top_score": hits[0].get("score") if hits else None,
            }
        )
    return rows


def confidence_metrics(rows: list[dict[str, Any]], min_score: float) -> dict[str, float | int]:
    """How the escalation cutoff splits the golden set (the cutoff depends only on retrieval scores)."""
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    def passes(r: dict[str, Any]) -> bool:
        return (r.get("top_score") or 0.0) >= min_score

    def escalation_rate(subset: list[dict[str, Any]]) -> float:
        return sum(not passes(r) for r in subset) / len(subset) if subset else 0.0

    off_topic = [r for r in unanswerable if r["category"] == "unanswerable_off_topic"]
    not_in_corpus = [r for r in unanswerable if r["category"] == "unanswerable_not_in_corpus"]
    return {
        "min_score": min_score,
        "answerable_pass_rate": sum(passes(r) for r in answerable) / len(answerable) if answerable else 0.0,
        # A score cutoff separates off-topic questions. In-domain questions about material that was never
        # ingested score close to answerable ones, so catching those is left to the LLM (behaviour metrics).
        "off_topic_escalation_rate": escalation_rate(off_topic),
        "not_in_corpus_escalation_rate": escalation_rate(not_in_corpus),
        "unanswerable_escalation_rate": escalation_rate(unanswerable),
    }


def retrieval_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    answerable = [r for r in rows if r["answerable"]]
    ranks = []
    for r in answerable:
        rank = next((i + 1 for i, cid in enumerate(r["retrieved_ids"]) if cid in r["reference_ids"]), None)
        ranks.append(rank)
    n = len(answerable)
    out: dict[str, float | int] = {f"hit@{k}": sum(1 for x in ranks if x is not None and x <= k) / n for k in (1, 3, 5)} if n else {}
    out["mrr"] = sum(1 / x for x in ranks if x) / n if n else 0.0
    out["questions"] = n
    return out


def behaviour_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    return {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "answerable_coverage": sum(not r["escalated"] for r in answerable) / len(answerable) if answerable else 0.0,
        "unanswerable_escalation_rate": sum(r["escalated"] for r in unanswerable) / len(unanswerable) if unanswerable else 0.0,
    }


def judge_rows(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    """Score every produced answer to an answerable question. Adds `scores` to those rows."""
    from deepeval.test_case import LLMTestCase

    scored: dict[str, list[float]] = {name: [] for name in metrics}
    errors = 0
    for row in rows:
        if not row["answerable"] or row["escalated"] or not row["answer"]:
            continue
        case = LLMTestCase(input=row["question"], actual_output=row["answer"], retrieval_context=row["contexts"])
        row["scores"] = {}
        for name, metric in metrics.items():
            try:
                metric.measure(case)
                row["scores"][name] = float(metric.score)
                scored[name].append(float(metric.score))
            except Exception as exc:  # noqa: BLE001 - a judge failure must not abort the whole run
                row["scores"][name] = None
                row.setdefault("judge_errors", []).append(f"{name}: {type(exc).__name__}: {exc}")
                errors += 1
    out: dict[str, Any] = {f"{name}_mean": (statistics.fmean(v) if v else None) for name, v in scored.items()}
    out["scored_answers"] = max((len(v) for v in scored.values()), default=0)
    out["judge_errors"] = errors
    return out


def cohort_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Retrieval metrics for the original pacs.008/pacs.002 questions alone, so a drop there cannot hide
    behind gains on the questions added with the flows (cohort "flows-v1")."""
    return {"retrieval_original": retrieval_metrics([r for r in rows if r.get("cohort", "original") == "original"])}


def summarize(rows: list[dict[str, Any]], judged: dict[str, Any] | None) -> dict[str, Any]:
    return {"retrieval": retrieval_metrics(rows), **cohort_summary(rows), "behaviour": behaviour_metrics(rows), "judged": judged}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the golden set.")
    parser.add_argument("--golden", default="evals/golden_set.jsonl")
    parser.add_argument("--judge", choices=["jev", "claude", "none"], default="none")
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, help="only the first N questions (cheap smoke run)")
    parser.add_argument("--cache-dir", help="generation cache (default .cache/llm)")
    parser.add_argument("--no-guard", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true", help="no generation, no API key: retrieval + confidence metrics")
    args = parser.parse_args(argv)

    from payments_rag.api.service import DEFAULT_MIN_SCORE, build_retriever

    golden = load_golden(args.golden)[: args.limit]

    if args.retrieval_only:
        rows = run_retrieval_only(golden, build_retriever())
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "mode": "retrieval_only",
            "questions": len(golden),
            "summary": {
                "retrieval": retrieval_metrics(rows),
                **cohort_summary(rows),
                "confidence": confidence_metrics(rows, DEFAULT_MIN_SCORE),
            },
            "cases": rows,
        }
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report["summary"], indent=2))
        return 0

    from payments_rag.llm.cache import ResponseCache
    from payments_rag.llm.generation import Generator

    generator = Generator.from_env(cache=ResponseCache(args.cache_dir))
    rows = run_golden(golden, build_retriever(), generator, DEFAULT_MIN_SCORE, guard=not args.no_guard)

    judged = None
    if args.judge != "none":
        from payments_rag.evals.judge import build_metrics

        judged = judge_rows(rows, build_metrics(args.judge))

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generator_model": next((r["model"] for r in rows if r["model"]), generator.model),
        "judge": args.judge,
        "questions": len(golden),
        "summary": summarize(rows, judged),
        "cases": rows,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
