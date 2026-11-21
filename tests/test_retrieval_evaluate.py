import pytest

from payments_rag.retrieval.evaluate import score_questions

QUESTIONS = [
    {"question": "q1", "expected": ["a"]},
    {"question": "q2", "expected": ["b", "c"]},
    {"question": "q3", "expected": ["z"]},
    {"question": "q4", "expected": ["d"]},
]

RESULTS = {
    "q1": ["a", "x", "y"],  # rank 1
    "q2": ["x", "c", "y"],  # rank 2 (any expected id counts)
    "q3": ["x", "y", "w"],  # never found
    "q4": ["x", "y", "w", "v", "d"],  # rank 5
}


def retrieve(question):
    return [{"id": i} for i in RESULTS[question]]


def test_hit_rates_and_mrr():
    result = score_questions(QUESTIONS, retrieve)
    assert result["hit@1"] == pytest.approx(1 / 4)
    assert result["hit@3"] == pytest.approx(2 / 4)
    assert result["hit@5"] == pytest.approx(3 / 4)
    assert result["mrr"] == pytest.approx((1 + 1 / 2 + 0 + 1 / 5) / 4)


def test_misses_list_questions_outside_top_three():
    result = score_questions(QUESTIONS, retrieve)
    missed = {m["question"]: m["rank"] for m in result["misses"]}
    assert missed == {"q3": None, "q4": 5}


def test_top_score_summary_uses_the_score_of_the_first_result():
    scores = {"q1": 0.9, "q2": 0.7, "q3": 0.5, "q4": 0.8}

    def scored(question):
        return [{"id": i, "score": scores[question]} for i in RESULTS[question]]

    result = score_questions(QUESTIONS, scored)
    assert result["top1_score"] == {"min": 0.5, "median": 0.75, "max": 0.9}


def test_questions_without_expected_ids_only_contribute_scores():
    result = score_questions([{"question": "q1", "expected": []}], lambda q: [{"id": "a", "score": 0.4}])
    assert result["top1_score"]["median"] == 0.4
    assert result["misses"] == []
