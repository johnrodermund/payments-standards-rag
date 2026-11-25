import json
from collections import Counter
from pathlib import Path

import pytest

from payments_rag.evals.gate import check
from payments_rag.evals.gate import main as gate_main
from payments_rag.evals.rag_eval import (
    behaviour_metrics,
    load_golden,
    retrieval_metrics,
    run_golden,
    summarize,
)
from payments_rag.llm.generation import Generation

GOLDEN = Path("evals/golden_set.jsonl")
SOURCE = Path("data/processed/iso20022_messages.json")
CHUNKS = Path("data/processed/chunks.jsonl")


def test_golden_set_size_and_mix():
    cases = load_golden(GOLDEN)
    assert 40 <= len(cases) <= 60
    answerable = [c for c in cases if c["answerable"]]
    assert len(answerable) >= 35 and len(cases) - len(answerable) >= 5
    cats = Counter(c["category"] for c in answerable)
    assert {"max_length", "requirement", "definition", "codes", "children", "message", "rule"} <= set(cats)
    assert {c["message"] for c in answerable} == {"pacs.008.001.14", "pacs.002.001.16"}


def test_ids_and_questions_are_unique():
    cases = load_golden(GOLDEN)
    assert len({c["id"] for c in cases}) == len(cases)
    assert len({c["question"] for c in cases}) == len(cases)


def test_answerable_cases_have_references_and_unanswerable_do_not():
    for c in load_golden(GOLDEN):
        if c["answerable"]:
            assert c["reference_answer"] and c["reference_ids"]
        else:
            assert c["reference_answer"] is None and c["reference_ids"] == []


def test_reference_ids_exist_in_the_source_data():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    valid = set()
    for m in data["messages"]:
        mid = m["id"]
        valid |= {f"{mid}#{e['xml_path']}" for e in m["elements"]}
        valid |= {f"{mid}#{e['xml_path']}#children" for e in m["elements"]}
        valid |= {f"{mid}#rule:{c['name']}" for c in m["constraints"]}
        valid.add(f"{mid}#message:scope")
    for c in load_golden(GOLDEN):
        for ref in c["reference_ids"]:
            assert ref in valid, f"{c['id']}: {ref}"


def test_reference_ids_exist_in_the_current_chunks_if_generated():
    if not CHUNKS.exists():
        pytest.skip("chunks.jsonl not generated")
    ids = {json.loads(line)["id"] for line in CHUNKS.read_text(encoding="utf-8").splitlines()}
    missing = [(c["id"], r) for c in load_golden(GOLDEN) for r in c["reference_ids"] if r not in ids]
    assert missing == [], f"golden references missing from the chunk ids (did the id scheme change?): {missing[:3]}"


def test_max_length_answers_match_the_data():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    by_id = {f"{m['id']}#{e['xml_path']}": e for m in data["messages"] for e in m["elements"]}
    checked = 0
    for c in load_golden(GOLDEN):
        if c["category"] == "max_length":
            el = by_id[c["reference_ids"][0]]
            assert el["type_info"]["max_length"] in c["reference_answer"]
            checked += 1
    assert checked >= 6


def test_unanswerable_questions_include_off_topic_and_not_in_corpus():
    cats = {c["category"] for c in load_golden(GOLDEN) if not c["answerable"]}
    assert cats == {"unanswerable_off_topic", "unanswerable_not_in_corpus"}


# ---- runner and metrics (no network: stub retriever and generator) ----


class StubGenerator:
    def generate(self, question, hits):
        return Generation(text="Answer [1].", model="stub", cached=False, usage={"input_tokens": 1, "output_tokens": 1}, stop_reason="end_turn")


def make_retriever(ranking):
    return lambda q: [{"id": cid, "kind": "element", "text": f"text of {cid}", "score": score} for cid, score in ranking.get(q, [])]


def test_run_golden_records_retrieval_answer_and_contexts():
    golden = [
        {"id": "g1", "category": "definition", "question": "q1", "answerable": True, "reference_ids": ["b"], "reference_answer": "ref", "message": "m"},
        {"id": "g2", "category": "unanswerable_off_topic", "question": "q2", "answerable": False, "reference_ids": [], "reference_answer": None, "message": None},
    ]
    retriever = make_retriever({"q1": [("a", 0.9), ("b", 0.8)], "q2": [("a", 0.3)]})
    rows = run_golden(golden, retriever, StubGenerator(), min_score=0.6)
    assert rows[0]["retrieved_ids"] == ["a", "b"] and rows[0]["answer"] == "Answer [1]." and rows[0]["contexts"] == ["text of a", "text of b"]
    assert rows[1]["escalated"] is True and rows[1]["answer"] is None
    s = summarize(rows, None)
    assert s["retrieval"]["hit@3"] == 1.0 and s["retrieval"]["hit@1"] == 0.0 and s["retrieval"]["mrr"] == 0.5
    assert s["behaviour"] == {"answerable": 1, "unanswerable": 1, "answerable_coverage": 1.0, "unanswerable_escalation_rate": 1.0}


def test_retrieval_metrics_ignore_unanswerable_questions():
    rows = [
        {"answerable": True, "retrieved_ids": ["x"], "reference_ids": ["x"]},
        {"answerable": False, "retrieved_ids": ["y"], "reference_ids": []},
    ]
    assert retrieval_metrics(rows)["questions"] == 1 and retrieval_metrics(rows)["hit@1"] == 1.0
    assert behaviour_metrics([{"answerable": False, "escalated": False}])["unanswerable_escalation_rate"] == 0.0


def test_retrieval_only_mode_needs_no_generator_and_reports_confidence_split():
    from payments_rag.evals.rag_eval import confidence_metrics, run_retrieval_only

    golden = [
        {"id": "g1", "category": "definition", "question": "q1", "answerable": True, "reference_ids": ["b"], "reference_answer": "r", "message": "m"},
        {"id": "g2", "category": "definition", "question": "q2", "answerable": True, "reference_ids": ["z"], "reference_answer": "r", "message": "m"},
        {"id": "g3", "category": "unanswerable_off_topic", "question": "q3", "answerable": False, "reference_ids": [], "reference_answer": None, "message": None},
        {"id": "g4", "category": "unanswerable_off_topic", "question": "q4", "answerable": False, "reference_ids": [], "reference_answer": None, "message": None},
    ]
    retriever = make_retriever({"q1": [("a", 0.8), ("b", 0.7)], "q2": [("a", 0.5)], "q3": [("a", 0.4)], "q4": [("a", 0.7)]})
    rows = run_retrieval_only(golden, retriever)
    assert [r["top_score"] for r in rows] == [0.8, 0.5, 0.4, 0.7]
    assert retrieval_metrics(rows)["hit@5"] == 0.5
    conf = confidence_metrics(rows, min_score=0.6)
    assert conf["answerable_pass_rate"] == 0.5  # q2 would be wrongly escalated
    assert conf["unanswerable_escalation_rate"] == 0.5  # q4 would wrongly pass
    assert conf["off_topic_escalation_rate"] == 0.5 and conf["not_in_corpus_escalation_rate"] == 0.0


def test_not_in_corpus_and_off_topic_are_reported_separately():
    from payments_rag.evals.rag_eval import confidence_metrics

    rows = [
        {"answerable": False, "category": "unanswerable_off_topic", "top_score": 0.4},
        {"answerable": False, "category": "unanswerable_not_in_corpus", "top_score": 0.65},
        {"answerable": False, "category": "unanswerable_not_in_corpus", "top_score": None},
    ]
    conf = confidence_metrics(rows, min_score=0.6)
    assert conf["off_topic_escalation_rate"] == 1.0
    assert conf["not_in_corpus_escalation_rate"] == 0.5  # no result at all counts as an escalation


# ---- gate ----

SUMMARY = {"retrieval": {"hit@5": 0.7, "mrr": 0.4}, "behaviour": {"answerable_coverage": 0.9}, "judged": {"faithfulness_mean": 0.8}}


def test_gate_passes_when_every_metric_meets_its_minimum():
    out = check(SUMMARY, {"retrieval.hit@5": {"min": 0.7}, "judged.faithfulness_mean": {"min": 0.75}})
    assert all(o["ok"] for o in out)


def test_gate_fails_below_threshold_and_on_missing_metrics():
    out = {o["metric"]: o for o in check(SUMMARY, {"retrieval.mrr": {"min": 0.5}, "judged.answer_relevancy_mean": {"min": 0.5}})}
    assert out["retrieval.mrr"]["ok"] is False and out["retrieval.mrr"]["reason"] == "below threshold"
    assert out["judged.answer_relevancy_mean"]["ok"] is False and out["judged.answer_relevancy_mean"]["reason"] == "missing from results"


def test_gate_only_checks_the_requested_groups():
    out = check(SUMMARY, {"retrieval.hit@5": {"min": 0.7}, "judged.faithfulness_mean": {"min": 0.99}}, only=["retrieval"])
    assert [o["metric"] for o in out] == ["retrieval.hit@5"]


def test_gate_cli_exit_codes(tmp_path):
    results = tmp_path / "r.json"
    results.write_text(json.dumps({"summary": SUMMARY}), encoding="utf-8")
    ok, bad = tmp_path / "ok.json", tmp_path / "bad.json"
    ok.write_text(json.dumps({"retrieval.hit@5": {"min": 0.6}}), encoding="utf-8")
    bad.write_text(json.dumps({"retrieval.hit@5": {"min": 0.9}}), encoding="utf-8")
    assert gate_main([str(results), str(ok)]) == 0
    assert gate_main([str(results), str(bad)]) == 1


def test_gate_with_nothing_to_check_fails():
    assert gate_main.__name__ == "main"  # sanity
    assert check(SUMMARY, {"_comment": "x"}) == []


def test_committed_thresholds_reference_metrics_the_runner_produces():
    thresholds = json.loads(Path("evals/thresholds.json").read_text(encoding="utf-8"))
    summary = {
        "retrieval": {"hit@1": 1, "hit@5": 1, "mrr": 1},
        "confidence": {"answerable_pass_rate": 1, "off_topic_escalation_rate": 1},
        "behaviour": {"answerable_coverage": 1, "unanswerable_escalation_rate": 1},
        "judged": {"faithfulness_mean": 1, "answer_relevancy_mean": 1},
    }
    out = check(summary, thresholds)
    assert out and all(o["ok"] for o in out)
