"""Measure retrieval quality against a small hand-checked question set.

Each line of the set is {"question": ..., "expected": [chunk ids]}; a hit is
any expected id appearing in the top-k results. Questions with an empty
`expected` list (e.g. evals/offtopic_set.jsonl) only contribute to `top1_score`,
the spread of the first result's cosine similarity, which is what a
low-confidence cutoff would compare against.

Usage:
    python -m payments_rag.retrieval.evaluate evals/retrieval_set.jsonl --mode vector
    python -m payments_rag.retrieval.evaluate evals/retrieval_set.jsonl --mode hybrid
    python -m payments_rag.retrieval.evaluate evals/offtopic_set.jsonl --mode hybrid
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

Retriever = Callable[[str], list[dict[str, Any]]]


def score_questions(questions: list[dict[str, Any]], retrieve: Retriever, ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, Any]:
    ranks: list[int | None] = []
    misses: list[dict[str, Any]] = []
    top_scores: list[float] = []
    for item in questions:
        hits = retrieve(item["question"])
        ids = [h["id"] for h in hits]
        if hits and hits[0].get("score") is not None:
            top_scores.append(float(hits[0]["score"]))
        if not item["expected"]:
            continue
        rank = next((i + 1 for i, cid in enumerate(ids) if cid in item["expected"]), None)
        ranks.append(rank)
        if rank is None or rank > 3:
            misses.append({"question": item["question"], "rank": rank, "top3": ids[:3]})
    n = len(ranks) or 1
    result: dict[str, Any] = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    result["mrr"] = sum(1 / r for r in ranks if r) / n
    result["questions"] = len(questions)
    if top_scores:
        result["top1_score"] = {
            "min": round(min(top_scores), 3),
            "median": round(statistics.median(top_scores), 3),
            "max": round(max(top_scores), 3),
        }
    result["misses"] = misses
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on a question set.")
    parser.add_argument("set_path")
    parser.add_argument("--mode", choices=["vector", "hybrid"], default="vector")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args(argv)

    from payments_rag.ingestion.embedding import FastEmbedEmbedder
    from payments_rag.retrieval import store

    lines = Path(args.set_path).read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    embedder = FastEmbedEmbedder()
    conn = store.connect()

    if args.mode == "vector":
        def retrieve(q: str) -> list[dict[str, Any]]:
            return store.search(conn, embedder.embed_query(q), k=args.k)
    else:
        def retrieve(q: str) -> list[dict[str, Any]]:
            return store.hybrid_search(conn, q, embedder.embed_query(q), k=args.k)

    retrieve(questions[0]["question"])  # warm-up: model load and corpus cache are not per-query costs
    started = time.perf_counter()
    result = score_questions(questions, retrieve)
    result["ms_per_question"] = round(1000 * (time.perf_counter() - started) / len(questions))
    misses = result.pop("misses")
    print(json.dumps({"mode": args.mode, **{k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()}}))
    if args.show_misses:
        for m in misses:
            print(f"  rank={m['rank']}  {m['question']}\n    top3: {m['top3']}")


if __name__ == "__main__":
    main()
