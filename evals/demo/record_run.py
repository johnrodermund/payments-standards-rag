"""Record a run of the golden set and the off-topic set against a deployed service.

    python evals/demo/record_run.py http://<host> evals/demo/deployed_run.json

Sends every question to POST /ask and saves exactly what came back (answer, escalation, sources, status).
No model call is made by this script; whether the service calls a model depends on how it was deployed
(in replay mode it only replays answers recorded earlier). Standard library only.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def ask(base: str, question: str) -> dict:
    request = urllib.request.Request(
        f"{base}/ask", data=json.dumps({"question": question}).encode(), headers={"content-type": "application/json"}
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body, status = json.load(response), response.status
    except urllib.error.HTTPError as exc:
        body, status = {"detail": exc.read().decode()[:300]}, exc.code
    return {"status": status, "seconds": round(time.perf_counter() - started, 2), **body}


def main(base: str, out: str) -> None:
    golden = [json.loads(line) for line in (ROOT / "evals/golden_set.jsonl").read_text(encoding="utf-8").splitlines()]
    offtopic = [json.loads(line) for line in (ROOT / "evals/offtopic_set.jsonl").read_text(encoding="utf-8").splitlines()]
    results = []
    for row in golden:
        got = ask(base, row["question"])
        retrieved = [r["id"] for r in got.get("retrieval", [])]
        results.append({"set": "golden", "id": row["id"], "category": row["category"], "question": row["question"],
                        "answerable": row["answerable"], "reference_ids": row["reference_ids"],
                        "hit_at_5": bool(set(row["reference_ids"]) & set(retrieved[:5])) if row["answerable"] else None, **got})
    for i, row in enumerate(offtopic, 1):
        results.append({"set": "offtopic", "id": f"off-{i:02d}", "question": row["question"], "answerable": False, **ask(base, row["question"])})
    answerable = [r for r in results if r["set"] == "golden" and r["answerable"]]
    rest = [r for r in results if not (r["set"] == "golden" and r["answerable"])]
    summary = {
        "service": base,
        "answerable_questions": len(answerable),
        "answered": sum(1 for r in answerable if r["status"] == 200 and not r.get("escalated")),
        "answerable_escalated": sum(1 for r in answerable if r.get("escalated")),
        "answerable_http_errors": sum(1 for r in answerable if r["status"] != 200),
        "hit_at_5": sum(1 for r in answerable if r["hit_at_5"]),
        "should_escalate": len(rest),
        "escalated_correctly": sum(1 for r in rest if r["status"] == 200 and r.get("escalated")),
        "answered_but_should_escalate": sum(1 for r in rest if r["status"] == 200 and not r.get("escalated")),
        "http_errors_total": sum(1 for r in results if r["status"] != 200),
    }
    Path(out).write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
