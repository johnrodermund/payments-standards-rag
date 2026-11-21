from payments_rag.llm import bedrock_check
from payments_rag.llm.backends import BedrockBackend, LLMUnavailableError


def test_not_configured_exits_2_without_calling_anything(monkeypatch, capsys):
    monkeypatch.delenv("BEDROCK_GENERATION_MODEL_ID", raising=False)
    assert bedrock_check.main() == 2
    assert "NOT CONFIGURED" in capsys.readouterr().out


def test_failure_prints_the_cause_and_a_hint(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("BEDROCK_GENERATION_MODEL_ID", "us.test.profile")
    monkeypatch.setenv("LLM_LEDGER", str(tmp_path / "spend.json"))

    def refuse(self, request):
        raise LLMUnavailableError("Bedrock ThrottlingException (after retries)")

    monkeypatch.setattr(BedrockBackend, "send", refuse)
    assert bedrock_check.main() == 1
    out = capsys.readouterr().out
    assert "ThrottlingException" in out and "quota" in out
    assert not (tmp_path / "spend.json").exists()  # a failed call is not recorded as spend


def test_success_is_recorded_in_the_ledger(monkeypatch, capsys, tmp_path):
    from payments_rag.llm.backends import BackendResponse

    monkeypatch.setenv("BEDROCK_GENERATION_MODEL_ID", "us.test.profile")
    monkeypatch.setenv("LLM_LEDGER", str(tmp_path / "spend.json"))
    monkeypatch.setenv("LLM_PRICES", '{"us.test.profile": [1.0, 5.0]}')
    reply = BackendResponse(text="OK", model="us.test.profile", usage={"input_tokens": 30, "output_tokens": 2}, stop_reason="end_turn", request_id="r1")
    monkeypatch.setattr(BedrockBackend, "send", lambda self, request: reply)
    assert bedrock_check.main() == 0
    assert "reply: 'OK'" in capsys.readouterr().out
    assert (tmp_path / "spend.json").exists()
