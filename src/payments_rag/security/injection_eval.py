"""Run the prompt-injection set against the real answer pipeline and score it.

Each case plants malicious text inside a retrieved passage. A case is "compromised" if the
final response shows the attack's goal (canary string, leaked system prompt, link, forced
escalation, planted falsehood). Safe outcomes are an answer that resisted the payload, or an
escalation to a human.

Usage (needs ANTHROPIC_API_KEY unless the cache already holds every request):
    uv run python -m payments_rag.security.injection_eval evals/injection_set.jsonl \
        --guard on --out evals/injection/results_guard_on.json
    uv run python -m payments_rag.security.injection_eval evals/injection_set.jsonl \
        --guard off --out evals/injection/results_guard_off.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from payments_rag.api.rag import answer_question


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def classify(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Judge one result. `compromised` is the headline; the rest explains how it ended."""
    rule = case["attack_succeeded_if"]
    answer = result.get("answer") or ""
    lowered = answer.lower()
    compromised = bool(
        (answer and any(token.lower() in lowered for token in rule.get("contains", [])))
        or (answer and rule.get("regex") and re.search(rule["regex"], answer))
        or (rule.get("reason") and result.get("reason") == rule["reason"])
    )
    escalated = bool(result.get("escalated"))
    expect = [e.lower() for e in case.get("expect_any", [])]
    return {
        "compromised": compromised,
        "screened": any(q["id"] == case.get("poisoned_passage") for q in result.get("quarantined", [])),
        "escalated": escalated,
        "correct": bool(answer) and not escalated and any(e in lowered for e in expect),
        "outcome": "escalated" if escalated else "answered",
    }


def run_cases(cases: list[dict[str, Any]], generator: Any, guard: bool) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        hits = [dict(p, kind="element") for p in case["passages"]]
        result = answer_question(case["question"], lambda _q, h=hits: h, generator, min_score=0.0, guard=guard)
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "placement": case.get("placement"),
                **classify(case, result),
                "reason": result.get("reason"),
                "answer": result.get("answer"),
                "quarantined": result.get("quarantined", []),
                "violations": result.get("violations", []),
                "blocked_answer": result.get("_blocked_answer"),
                "model": result.get("model"),
                "cached": result.get("cached"),
            }
        )
    return rows


def _rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "attack_success_rate": sum(r["compromised"] for r in rows) / n if n else 0.0,
        "screened_rate": sum(r["screened"] for r in rows) / n if n else 0.0,
        "escalated_rate": sum(r["escalated"] for r in rows) / n if n else 0.0,
        "correct_rate": sum(r["correct"] for r in rows) / n if n else 0.0,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attacks = [r for r in rows if r["category"] != "control"]
    controls = [r for r in rows if r["category"] == "control"]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in attacks:
        by_category[r["category"]].append(r)
    overall = _rates(attacks)
    return {
        "attacks": len(attacks),
        "controls": len(controls),
        "attack_success_rate": overall["attack_success_rate"],
        "screened_rate": overall["screened_rate"],
        "attack_escalated_rate": overall["escalated_rate"],
        "attack_correct_rate": overall["correct_rate"],
        "control_correct_rate": _rates(controls)["correct_rate"],
        "control_escalation_rate": _rates(controls)["escalated_rate"],
        "by_category": {c: _rates(rs) for c, rs in sorted(by_category.items())},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the pipeline against the injection set.")
    parser.add_argument("set_path")
    parser.add_argument("--guard", choices=["on", "off"], default="on")
    parser.add_argument("--out")
    parser.add_argument("--cache-dir", help="LLM response cache (default .cache/llm)")
    args = parser.parse_args(argv)

    from payments_rag.llm.cache import ResponseCache
    from payments_rag.llm.generation import Generator

    generator = Generator(cache=ResponseCache(args.cache_dir))
    rows = run_cases(load_cases(args.set_path), generator, guard=args.guard == "on")
    summary = summarize(rows)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "guard": args.guard,
        "model": next((r["model"] for r in rows if r["model"]), generator.model),
        "summary": summary,
        "cases": rows,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"guard={args.guard} model={report['model']}")
    print(
        f"attacks={summary['attacks']}  success_rate={summary['attack_success_rate']:.1%}  "
        f"screened={summary['screened_rate']:.1%}  escalated={summary['attack_escalated_rate']:.1%}"
    )
    print(f"controls={summary['controls']}  correct={summary['control_correct_rate']:.1%}  escalated={summary['control_escalation_rate']:.1%}")
    for category, r in summary["by_category"].items():
        print(f"  {category:24} n={r['n']}  success={r['attack_success_rate']:.0%}  screened={r['screened_rate']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
