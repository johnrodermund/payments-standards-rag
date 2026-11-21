"""CI gate: exit 1 if any metric in a results file is below its threshold (or missing).

Thresholds live in evals/thresholds.json as {"retrieval.hit@5": {"min": 0.6}, ...}. A missing metric
fails the gate: a run that silently skipped the judge must not pass.

Usage:
    python -m payments_rag.evals.gate evals/results/latest.json evals/thresholds.json
    python -m payments_rag.evals.gate results.json thresholds.json --only retrieval,behaviour
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def lookup(summary: dict[str, Any], dotted: str) -> Any:
    node: Any = summary
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def check(summary: dict[str, Any], thresholds: dict[str, Any], only: list[str] | None = None) -> list[dict[str, Any]]:
    """One entry per checked metric: {metric, value, min, ok, reason}."""
    results = []
    for metric, rule in thresholds.items():
        if metric.startswith("_"):
            continue
        if only and metric.split(".")[0] not in only:
            continue
        value = lookup(summary, metric)
        minimum = rule["min"]
        if value is None:
            results.append({"metric": metric, "value": None, "min": minimum, "ok": False, "reason": "missing from results"})
        else:
            ok = value >= minimum
            results.append({"metric": metric, "value": value, "min": minimum, "ok": ok, "reason": None if ok else "below threshold"})
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail if eval metrics are below thresholds.")
    parser.add_argument("results_path")
    parser.add_argument("thresholds_path")
    parser.add_argument("--only", help="comma-separated metric groups to check, e.g. retrieval,behaviour")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.results_path).read_text(encoding="utf-8"))
    thresholds = json.loads(Path(args.thresholds_path).read_text(encoding="utf-8"))
    outcomes = check(report["summary"], thresholds, args.only.split(",") if args.only else None)

    for o in outcomes:
        shown = "n/a" if o["value"] is None else f"{o['value']:.3f}"
        print(f"{'PASS' if o['ok'] else 'FAIL'}  {o['metric']:42} {shown:>7}  (min {o['min']})" + (f"  [{o['reason']}]" if o["reason"] else ""))
    if thresholds.get("_provisional"):
        print("NOTE: thresholds are provisional (not yet calibrated from a measured run).")
    if not outcomes:
        print("FAIL: no metrics were checked")
        return 1
    failed = [o for o in outcomes if not o["ok"]]
    print(f"{'GATE FAILED' if failed else 'GATE PASSED'}: {len(outcomes) - len(failed)}/{len(outcomes)} metrics ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
