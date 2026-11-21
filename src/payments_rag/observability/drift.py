"""Drift check: has live behaviour moved away from a baseline?

Three signals are read from the audit log (one record per /ask call):

* top retrieval score        (does the corpus still match what people ask?)
* answer length in words     (answered rows only; a changed model or prompt shows up here)
* escalation rate            (a rising share of hand-offs means retrieval or the cutoff has shifted)

The recent window is compared with a stored baseline: a two-sample Kolmogorov-Smirnov test for the two
continuous signals and a two-proportion z-test for the escalation rate. An alarm needs BOTH a small
p-value and a minimum effect size, so a large window cannot raise an alarm over a trivial shift. With fewer
than `min_n` observations on either side the check reports `insufficient_data` and exits non-zero; it never
reports "ok" on too little evidence.

Honest limits:
* The committed baseline is built from the golden-set run, which is NOT production traffic (it contains 15%
  unanswerable questions and templated phrasing). Until a baseline is taken from real traffic
  (`baseline --from-audit`), a drift alarm may reflect that difference and not real drift.
* The thresholds (alpha 0.01, minimum effects below) are defaults, checked by simulation in the tests, not
  tuned on production data. Three signals at alpha 0.01 give roughly a 3% chance of a false alarm per check
  under no drift.
* Power is limited at small windows. Simulated (60 baseline vs 100 recent requests): a 0.10 drop in mean top
  score or a doubled answer length is caught at least 95% of the time; escalations rising from 15% to 45% are
  caught about 91% of the time, and smaller shifts less often. Collect more requests to catch subtler drift.
* It detects distribution shifts, not wrong answers. Quality is the eval's job.

Usage:
    python -m payments_rag.observability.drift check [--baseline evals/drift_baseline.json] [--recent 200]
    python -m payments_rag.observability.drift baseline --from-results RETRIEVAL_ONLY.json LIVE.json --out evals/drift_baseline.json
    python -m payments_rag.observability.drift baseline --from-audit [--first 200] --out evals/drift_baseline.json
Exit codes: 0 ok, 1 drift detected, 2 insufficient data.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALPHA = 0.01
MIN_N = 30
MIN_KS_EFFECT = 0.25  # KS statistic D: largest gap between the two cumulative distributions
MIN_RATE_EFFECT = 0.10  # absolute change in escalation rate
DEFAULT_RECENT = 200
DEFAULT_BASELINE = Path("evals/drift_baseline.json")


@dataclass(frozen=True)
class Observation:
    top_score: float | None
    answer_words: int | None
    escalated: bool | None


def observation_from_entry(entry: dict[str, Any]) -> Observation | None:
    """One audit entry -> one observation. Error rows carry no behaviour signal and are skipped."""
    if entry.get("outcome") == "error":
        return None
    scores = [h.get("score") for h in entry.get("retrieval", []) if h.get("score") is not None]
    answer = entry.get("answer")
    return Observation(
        top_score=max(scores) if scores else None,
        answer_words=len(answer.split()) if answer else None,
        escalated=bool(entry["escalated"]) if entry.get("escalated") is not None else None,
    )


def observations_from_entries(entries: list[dict[str, Any]]) -> list[Observation]:
    return [o for o in (observation_from_entry(e) for e in entries) if o is not None]


# ---- statistics (pure Python; no SciPy dependency) ----


def ks_2samp(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sample KS statistic and asymptotic two-sided p-value."""
    if not a or not b:
        raise ValueError("both samples must be non-empty")
    xs, ys = sorted(a), sorted(b)
    n, m = len(xs), len(ys)
    i = j = 0
    d = 0.0
    while i < n and j < m:
        value = min(xs[i], ys[j])
        while i < n and xs[i] <= value:
            i += 1
        while j < m and ys[j] <= value:
            j += 1
        d = max(d, abs(i / n - j / m))
    ne = n * m / (n + m)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    if lam < 0.27:
        return d, 1.0
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, min(1.0, max(0.0, p))


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z statistic and two-sided p-value."""
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (x2 / n2 - x1 / n1) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


# ---- baseline ----


def baseline_from_observations(obs: list[Observation], source: str) -> dict[str, Any]:
    flagged = [o.escalated for o in obs if o.escalated is not None]
    return {
        "created": datetime.now(UTC).isoformat(),
        "source": source,
        "top_score": [round(o.top_score, 6) for o in obs if o.top_score is not None],
        "answer_words": [o.answer_words for o in obs if o.answer_words is not None],
        "escalated": sum(flagged),
        "escalation_n": len(flagged),
    }


def baseline_from_results(retrieval_only: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    """Join a golden-set retrieval-only run (top scores) with a live run (answers and escalations) by case id."""
    scores = {c["id"]: c.get("top_score") for c in retrieval_only["cases"]}
    obs = []
    for case in live["cases"]:
        answer = case.get("answer")
        obs.append(
            Observation(
                top_score=scores.get(case["id"]),
                answer_words=len(answer.split()) if answer else None,
                escalated=bool(case["escalated"]),
            )
        )
    return baseline_from_observations(obs, "golden set (not production traffic)")


# ---- comparison ----


def _continuous(name: str, base: list[float], recent: list[float], min_n: int, alpha: float) -> dict[str, Any]:
    result: dict[str, Any] = {"metric": name, "n_baseline": len(base), "n_recent": len(recent)}
    if len(base) < min_n or len(recent) < min_n:
        return {**result, "status": "insufficient_data"}
    d, p = ks_2samp(base, recent)
    drifted = p < alpha and d >= MIN_KS_EFFECT
    return {
        **result,
        "status": "drift" if drifted else "ok",
        "ks_statistic": round(d, 4),
        "p_value": round(p, 6),
        "median_baseline": round(statistics.median(base), 4),
        "median_recent": round(statistics.median(recent), 4),
    }


def compare(
    baseline: dict[str, Any], recent: list[Observation], min_n: int = MIN_N, alpha: float = ALPHA
) -> dict[str, Any]:
    checks = [
        _continuous("top_score", baseline["top_score"], [o.top_score for o in recent if o.top_score is not None], min_n, alpha),
        _continuous(
            "answer_words", baseline["answer_words"], [o.answer_words for o in recent if o.answer_words is not None], min_n, alpha
        ),
    ]
    flagged = [o.escalated for o in recent if o.escalated is not None]
    rate: dict[str, Any] = {"metric": "escalation_rate", "n_baseline": baseline["escalation_n"], "n_recent": len(flagged)}
    if baseline["escalation_n"] < min_n or len(flagged) < min_n:
        rate["status"] = "insufficient_data"
    else:
        z, p = two_proportion_z(baseline["escalated"], baseline["escalation_n"], sum(flagged), len(flagged))
        base_rate = baseline["escalated"] / baseline["escalation_n"]
        recent_rate = sum(flagged) / len(flagged)
        rate.update(
            status="drift" if p < alpha and abs(recent_rate - base_rate) >= MIN_RATE_EFFECT else "ok",
            z=round(z, 3),
            p_value=round(p, 6),
            rate_baseline=round(base_rate, 4),
            rate_recent=round(recent_rate, 4),
        )
    checks.append(rate)
    statuses = {c["status"] for c in checks}
    overall = "drift" if "drift" in statuses else "insufficient_data" if "insufficient_data" in statuses else "ok"
    return {
        "status": overall,
        "alpha": alpha,
        "min_n": min_n,
        "baseline_source": baseline.get("source"),
        "baseline_created": baseline.get("created"),
        "checks": checks,
    }


# ---- CLI ----


def _load_entries(args: argparse.Namespace, first: bool = False) -> list[dict[str, Any]]:
    if getattr(args, "entries", None):
        lines = Path(args.entries).read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    from payments_rag.audit.log import AuditLog
    from payments_rag.retrieval.store import database_url

    log = AuditLog(database_url())
    n = args.first if first else args.recent
    return log.head(n) if first else log.tail(n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drift check over the audit log.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="compare the most recent requests with the baseline")
    check.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    check.add_argument("--recent", type=int, default=DEFAULT_RECENT)
    check.add_argument("--min-n", type=int, default=MIN_N)
    check.add_argument("--entries", help="read audit entries from a JSONL file instead of the database")
    check.add_argument("--out")

    base = sub.add_parser("baseline", help="write a baseline file")
    group = base.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-results", nargs=2, metavar=("RETRIEVAL_ONLY", "LIVE"))
    group.add_argument("--from-audit", action="store_true")
    base.add_argument("--first", type=int, default=DEFAULT_RECENT, help="with --from-audit: use the first N rows")
    base.add_argument("--entries", help="with --from-audit: read entries from a JSONL file")
    base.add_argument("--out", required=True)

    args = parser.parse_args(argv)

    if args.command == "baseline":
        if args.from_results:
            retrieval_only = json.loads(Path(args.from_results[0]).read_text(encoding="utf-8"))
            live = json.loads(Path(args.from_results[1]).read_text(encoding="utf-8"))
            baseline = baseline_from_results(retrieval_only, live)
        else:
            obs = observations_from_entries(_load_entries(args, first=True))
            baseline = baseline_from_observations(obs, f"first {len(obs)} audited requests")
        Path(args.out).write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        print(f"wrote {args.out}: {len(baseline['top_score'])} scores, {len(baseline['answer_words'])} answer lengths, source: {baseline['source']}")
        return 0

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    report = compare(baseline, observations_from_entries(_load_entries(args)), min_n=args.min_n)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"drift check: {report['status'].upper()} (baseline: {report['baseline_source']})")
    for c in report["checks"]:
        detail = {k: v for k, v in c.items() if k not in ("metric", "status")}
        print(f"  {c['metric']:16s} {c['status']:18s} {detail}")
    return {"ok": 0, "drift": 1, "insufficient_data": 2}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
