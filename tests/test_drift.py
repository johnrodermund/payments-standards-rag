import json
import random

import pytest

from payments_rag.observability.drift import (
    Observation,
    baseline_from_observations,
    baseline_from_results,
    compare,
    ks_2samp,
    main,
    observation_from_entry,
    observations_from_entries,
    two_proportion_z,
)


def sample(rng, n, score_mean=0.78, words_mean=60.0, esc_rate=0.15):
    return [
        Observation(
            top_score=rng.gauss(score_mean, 0.06),
            answer_words=None if (esc := rng.random() < esc_rate) else max(3, int(rng.gauss(words_mean, 15))),
            escalated=esc,
        )
        for _ in range(n)
    ]


def baseline(rng):
    return baseline_from_observations(sample(rng, 60), "test")


# ---- statistics ----


def test_ks_identical_samples_have_no_difference():
    d, p = ks_2samp([1, 2, 3, 4], [1, 2, 3, 4])
    assert d == 0 and p == 1.0


def test_ks_fully_separated_samples():
    d, p = ks_2samp(list(range(50)), list(range(100, 150)))
    assert d == 1.0 and p < 1e-10


def test_ks_handles_ties():
    d, _ = ks_2samp([1, 1, 1, 2], [1, 2, 2, 2])
    assert d == pytest.approx(0.5)


def test_two_proportion_z():
    z, p = two_proportion_z(10, 100, 10, 100)
    assert z == 0 and p == 1.0
    _, p = two_proportion_z(10, 100, 40, 100)
    assert p < 1e-4


# ---- detection power and false alarms (simulation) ----


def run(trials, **shift):
    alarms = 0
    for seed in range(trials):
        rng = random.Random(seed)
        report = compare(baseline(rng), sample(rng, 100, **shift))
        alarms += report["status"] == "drift"
    return alarms / trials


def test_no_drift_rarely_alarms():
    assert run(300) <= 0.06  # three signals at alpha 0.01: about 3% expected


def test_a_drop_in_retrieval_scores_is_detected():
    assert run(100, score_mean=0.68) >= 0.95


def test_a_change_in_answer_length_is_detected():
    assert run(100, words_mean=120.0) >= 0.95


def test_a_rise_in_escalations_is_detected():
    # Measured power for 15% -> 45% with 60 baseline / 100 recent requests at alpha 0.01 is about 0.91.
    assert run(100, esc_rate=0.45) >= 0.85
    assert run(100, esc_rate=0.65) >= 0.99


def test_a_trivial_shift_in_a_huge_window_does_not_alarm():
    rng = random.Random(1)
    base = baseline_from_observations(sample(rng, 5000), "big")
    report = compare(base, sample(rng, 5000, score_mean=0.785))
    top = next(c for c in report["checks"] if c["metric"] == "top_score")
    assert top["p_value"] < 0.01 and top["ks_statistic"] < 0.25 and top["status"] == "ok"


# ---- insufficient data is never reported as ok ----


def test_too_few_observations_is_insufficient_data():
    rng = random.Random(0)
    report = compare(baseline(rng), sample(rng, 5))
    assert report["status"] == "insufficient_data"
    assert all(c["status"] == "insufficient_data" for c in report["checks"])


# ---- reading entries and baselines ----


def entry(outcome="answered", score=0.8, answer="one two three", escalated=False):
    return {
        "outcome": outcome,
        "answer": answer,
        "escalated": escalated,
        "retrieval": [{"id": "a", "score": score}, {"id": "b", "score": score - 0.2}],
    }


def test_observation_uses_top_score_and_word_count_and_skips_errors():
    assert observation_from_entry(entry()) == Observation(0.8, 3, False)
    assert observation_from_entry(entry("escalated", answer=None, escalated=True)) == Observation(0.8, None, True)
    assert observation_from_entry({"outcome": "error", "retrieval": []}) is None
    assert len(observations_from_entries([entry(), {"outcome": "error"}])) == 1


def test_baseline_from_results_joins_scores_and_answers_by_case_id():
    retrieval_only = {"cases": [{"id": "g1", "top_score": 0.9}, {"id": "g2", "top_score": 0.4}]}
    live = {"cases": [{"id": "g1", "answer": "a b c d", "escalated": False}, {"id": "g2", "answer": None, "escalated": True}]}
    b = baseline_from_results(retrieval_only, live)
    assert b["top_score"] == [0.9, 0.4] and b["answer_words"] == [4]
    assert (b["escalated"], b["escalation_n"]) == (1, 2)
    assert "not production" in b["source"]


# ---- CLI ----


def write_entries(path, observations):
    lines = []
    for o in observations:
        lines.append(
            json.dumps(
                {
                    "outcome": "escalated" if o.escalated else "answered",
                    "answer": None if o.escalated else " ".join(["w"] * (o.answer_words or 1)),
                    "escalated": o.escalated,
                    "retrieval": [{"id": "a", "score": o.top_score}],
                }
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def test_cli_exit_codes(tmp_path):
    rng = random.Random(3)
    base_file = tmp_path / "base.json"
    base_file.write_text(json.dumps(baseline(rng)), encoding="utf-8")
    same, shifted, tiny = tmp_path / "same.jsonl", tmp_path / "shifted.jsonl", tmp_path / "tiny.jsonl"
    write_entries(same, sample(rng, 100))
    write_entries(shifted, sample(rng, 100, score_mean=0.6))
    write_entries(tiny, sample(rng, 4))
    common = ["check", "--baseline", str(base_file)]
    assert main([*common, "--entries", str(same)]) == 0
    assert main([*common, "--entries", str(shifted)]) == 1
    assert main([*common, "--entries", str(tiny)]) == 2


def test_cli_baseline_from_entries(tmp_path):
    rng = random.Random(4)
    entries = tmp_path / "e.jsonl"
    write_entries(entries, sample(rng, 80))
    out = tmp_path / "b.json"
    assert main(["baseline", "--from-audit", "--entries", str(entries), "--out", str(out)]) == 0
    assert len(json.loads(out.read_text())["top_score"]) == 80
