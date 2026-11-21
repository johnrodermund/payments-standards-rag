import json
from types import SimpleNamespace

import pytest

pytest.importorskip("deepeval")  # optional `eval` dependency group

from deepeval.test_case import LLMTestCase

from payments_rag.evals.judge import ClaudeJudge, build_metrics
from payments_rag.llm.cache import ResponseCache


class ScriptedClient:
    """Stands in for anthropic.Anthropic. Reads the JSON schema the judge appended to the prompt
    and answers with a minimal valid object, so DeepEval's real metric code runs unmodified."""

    def __init__(self, verdicts=("yes", "no"), garbage_first=False, fenced=False):
        self.calls = 0
        self.verdicts, self.garbage_first, self.fenced = verdicts, garbage_first, fenced
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        prompt = kwargs["messages"][0]["content"]
        if self.garbage_first and self.calls == 1:
            text = "not json at all"
        elif '"truths"' in prompt:
            text = json.dumps({"truths": ["Max35Text allows 1 to 35 characters."]})
        elif '"claims"' in prompt:
            text = json.dumps({"claims": ["It allows 35 characters.", "It is optional."]})
        elif '"statements"' in prompt:
            text = json.dumps({"statements": ["It allows 35 characters.", "It is a text type."]})
        elif '"verdicts"' in prompt:
            text = json.dumps({"verdicts": [{"verdict": v, "reason": "r"} for v in self.verdicts]})
        elif '"reason"' in prompt:
            text = json.dumps({"reason": "fine"})
        else:
            text = "plain text reply"
        if self.fenced:
            text = f"Here you go:\n```json\n{text}\n```"
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=50, output_tokens=10),
        )


def case():
    return LLMTestCase(
        input="What is the maximum length of the message identification?",
        actual_output="It allows 35 characters [1].",
        retrieval_context=["Message Identification: Type: Max35Text (text, 1 to 35 characters); occurrence 1..1 (mandatory)"],
    )


def test_faithfulness_runs_end_to_end_with_a_scripted_judge():
    judge = ClaudeJudge(client=ScriptedClient(verdicts=("yes", "no")), model="judge-test")
    metric = build_metrics("claude", claude_judge=judge)["faithfulness"]
    metric.measure(case())
    assert metric.score == pytest.approx(0.5)  # one of two claims supported
    assert judge.usage["calls"] >= 3 and judge.usage["input_tokens"] > 0


def test_answer_relevancy_runs_end_to_end_with_a_scripted_judge():
    judge = ClaudeJudge(client=ScriptedClient(verdicts=("yes", "yes")), model="judge-test")
    metric = build_metrics("claude", claude_judge=judge)["answer_relevancy"]
    metric.measure(case())
    assert metric.score == pytest.approx(1.0)


def test_judge_retries_once_on_invalid_json_then_succeeds():
    client = ScriptedClient(garbage_first=True)
    judge = ClaudeJudge(client=client, model="judge-test")
    from deepeval.metrics.faithfulness.schema import Truths

    assert judge.generate("prompt", Truths).truths
    assert client.calls == 2


def test_judge_accepts_json_inside_code_fences():
    judge = ClaudeJudge(client=ScriptedClient(fenced=True), model="judge-test")
    from deepeval.metrics.faithfulness.schema import Claims

    assert len(judge.generate("prompt", Claims).claims) == 2


def test_judge_gives_up_after_two_invalid_replies():
    class Bad(ScriptedClient):
        def _create(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="nope")], usage=SimpleNamespace(input_tokens=1, output_tokens=1))

    bad = Bad()
    bad.messages = SimpleNamespace(create=bad._create)
    judge = ClaudeJudge(client=bad, model="judge-test")
    from deepeval.metrics.faithfulness.schema import Truths

    with pytest.raises(ValueError, match="valid JSON"):
        judge.generate("prompt", Truths)
    assert bad.calls == 2


def test_identical_judge_requests_are_served_from_cache(tmp_path):
    client = ScriptedClient()
    judge = ClaudeJudge(client=client, model="judge-test", cache=ResponseCache(tmp_path))
    judge.generate("same prompt")
    judge.generate("same prompt")
    assert client.calls == 1 and judge.usage["cache_hits"] == 1


def test_unknown_judge_is_rejected():
    with pytest.raises(ValueError, match="unknown judge"):
        build_metrics("gpt")


def test_telemetry_is_opted_out():
    import os

    assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"].upper() in {"YES", "1", "TRUE"}
