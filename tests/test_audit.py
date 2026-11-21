import os
import threading

import psycopg
import pytest

from payments_rag.audit.log import GENESIS, AuditLog, build_entry

BASE_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://payments:payments@localhost:5432/postgres")
TEST_DB = "payments_audit_test"


@pytest.fixture
def audit():
    try:
        admin = psycopg.connect(BASE_URL, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable (docker compose up -d db)")
    admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
    admin.execute(f"CREATE DATABASE {TEST_DB}")
    admin.close()
    url = BASE_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"
    yield AuditLog(url)


def entry(n=1, **over):
    base = {
        "request_id": f"req-{n}",
        "question": f"question {n}?",
        "outcome": "answered",
        "answer": f"answer {n} [1]",
        "escalated": False,
        "reason": None,
        "sources": [{"id": "pacs.008.001.14#Root/A", "score": 0.8}],
        "retrieval": [{"id": "pacs.008.001.14#Root/A", "score": 0.8}, {"id": "x", "score": 0.5}],
        "model": "claude-haiku-5-5",
        "cached": False,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "min_score": 0.6,
        "latency_ms": 12,
    }
    base.update(over)
    return base


def test_build_entry_captures_question_answer_sources_and_decision():
    result = {
        "answer": "It is the creditor's bank [1].",
        "escalated": False,
        "reason": None,
        "sources": [{"id": "a", "kind": "element", "score": 0.8, "snippet": "long text..."}],
        "retrieval": [{"id": "a", "score": 0.8}],
        "retrieved": 1,
        "model": "claude-haiku-5-5",
        "cached": True,
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    e = build_entry("rid", "What is it?", result, min_score=0.6, latency_ms=5)
    assert e["request_id"] == "rid" and e["question"] == "What is it?"
    assert e["outcome"] == "answered" and e["answer"].startswith("It is")
    assert e["sources"] == [{"id": "a", "score": 0.8}]  # ids and scores, not snippets
    assert e["retrieval"] == [{"id": "a", "score": 0.8}]
    assert e["cached"] is True and e["min_score"] == 0.6 and e["latency_ms"] == 5 and e["ts"]


def test_build_entry_marks_escalations_and_errors():
    esc = build_entry("r", "q", {"answer": None, "escalated": True, "reason": "low_retrieval_confidence", "sources": [], "retrieved": 0}, 0.6, 1)
    assert esc["outcome"] == "escalated" and esc["reason"] == "low_retrieval_confidence"
    err = build_entry("r", "q", None, 0.6, 1, error="ReplayMissError: nope")
    assert err["outcome"] == "error" and err["error"] == "ReplayMissError: nope" and err["answer"] is None


def test_rows_are_chained_and_verify_passes(audit):
    h1 = audit.record(entry(1))
    h2 = audit.record(entry(2))
    assert h1 != h2
    report = audit.verify()
    assert report["ok"] is True and report["rows"] == 2 and report["head_hash"] == h2


def test_empty_log_verifies(audit):
    audit.init_schema()
    report = audit.verify()
    assert report["ok"] is True and report["rows"] == 0 and report["head_hash"] == GENESIS


def test_update_and_delete_and_truncate_are_blocked(audit):
    audit.record(entry(1))
    with psycopg.connect(audit.url, autocommit=True) as c:
        for sql in ("UPDATE audit_log SET outcome = 'x'", "DELETE FROM audit_log", "TRUNCATE audit_log"):
            with pytest.raises(psycopg.Error, match="append-only"):
                c.execute(sql)
    assert audit.verify()["rows"] == 1


def test_tampering_with_a_row_is_detected(audit):
    audit.record(entry(1))
    audit.record(entry(2))
    audit.record(entry(3))
    with psycopg.connect(audit.url, autocommit=True) as c:  # a superuser can bypass the triggers...
        c.execute("ALTER TABLE audit_log DISABLE TRIGGER USER")
        c.execute("UPDATE audit_log SET canonical = replace(canonical, 'answer 2', 'answer TWO') WHERE seq = (SELECT min(seq) + 1 FROM audit_log)")
    report = audit.verify()  # ...but the hash chain still shows it
    assert report["ok"] is False
    assert any("hash mismatch" in e for e in report["errors"])


def test_deleting_a_middle_row_is_detected(audit):
    for n in (1, 2, 3):
        audit.record(entry(n))
    with psycopg.connect(audit.url, autocommit=True) as c:
        c.execute("ALTER TABLE audit_log DISABLE TRIGGER USER")
        c.execute("DELETE FROM audit_log WHERE seq = (SELECT min(seq) + 1 FROM audit_log)")
    report = audit.verify()
    assert report["ok"] is False and any("chain broken" in e for e in report["errors"])


def test_concurrent_writers_keep_the_chain_intact(audit):
    audit.init_schema()
    errors = []

    def write(n):
        try:
            audit.record(entry(n))
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(15)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    report = audit.verify()
    assert report["ok"] is True and report["rows"] == 15


def test_recorded_content_is_retrievable(audit):
    audit.record(entry(7, question="What is the UETR?"))
    rows = audit.tail(1)
    assert rows[0]["question"] == "What is the UETR?" and rows[0]["outcome"] == "answered"
    assert rows[0]["sources"][0]["id"] == "pacs.008.001.14#Root/A"


def test_head_returns_the_first_entries_oldest_first(audit):
    for n in range(1, 6):
        audit.record(entry(n))
    assert [e["request_id"] for e in audit.head(3)] == ["req-1", "req-2", "req-3"]
    assert [e["request_id"] for e in audit.tail(2)] == ["req-4", "req-5"]
