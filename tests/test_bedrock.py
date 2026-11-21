import boto3
import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.stub import Stubber
from fastapi.testclient import TestClient

from payments_rag.api.main import app, get_service
from payments_rag.llm.backends import (
    AnthropicBackend,
    BedrockBackend,
    LLMUnavailableError,
    make_backend,
)
from payments_rag.llm.budget import Budget, cost_usd, price_for
from payments_rag.llm.cache import ResponseCache
from payments_rag.llm.generation import (
    MAX_OUTPUT_TOKENS,
    SYSTEM_PROMPT,
    Generator,
    ReplayMissError,
    build_prompt,
)
from tests.test_ask import HITS, StubAudit, make_service

MODEL = "us.anthropic.claude-test-profile"
QUESTION = "What is the creditor agent?"


def real_client():
    """A real boto3 client with fake credentials: Stubber then checks every request and response against AWS's API model."""
    return boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test")


def converse_response(text="Answer [1].", stop="end_turn", tokens=(120, 12)):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": stop,
        "usage": {"inputTokens": tokens[0], "outputTokens": tokens[1], "totalTokens": sum(tokens)},
        "metrics": {"latencyMs": 40},
    }


def expected_params():
    system, messages = build_prompt(QUESTION, HITS)
    return {
        "modelId": MODEL,
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": messages[0]["content"]}]}],
        "inferenceConfig": {"maxTokens": MAX_OUTPUT_TOKENS},
    }


@pytest.fixture
def stubbed():
    client = real_client()
    with Stubber(client) as stub:
        yield client, stub
        stub.assert_no_pending_responses()


# ---- request and response mapping ----


def test_request_is_valid_against_the_converse_api_and_sets_max_tokens(stubbed):
    client, stub = stubbed
    stub.add_response("converse", converse_response(), expected_params())  # fails if the shape is not valid Converse input
    gen = Generator(backend=BedrockBackend(client=client, model_id=MODEL))
    result = gen.generate(QUESTION, HITS)
    assert result.text == "Answer [1]." and result.model == MODEL and result.cached is False
    assert result.usage == {"input_tokens": 120, "output_tokens": 12} and result.stop_reason == "end_turn"


def test_system_prompt_and_passages_are_the_same_as_for_claude_api():
    system, messages = build_prompt(QUESTION, HITS)
    bedrock = BedrockBackend(client=object(), model_id=MODEL).build_request(system, messages, 700)
    claude = AnthropicBackend(client=object(), model="m").build_request(system, messages, 700)
    assert bedrock["system"][0]["text"] == claude["system"] == SYSTEM_PROMPT
    assert bedrock["messages"][0]["content"][0]["text"] == claude["messages"][0]["content"]
    assert bedrock["inferenceConfig"] == {"maxTokens": 700}


def test_the_claude_api_request_is_unchanged_so_recorded_answers_still_replay():
    system, messages = build_prompt(QUESTION, HITS)
    assert AnthropicBackend(client=object(), model="claude-haiku-5-5").build_request(system, messages, 700) == {
        "model": "claude-haiku-5-5",
        "max_tokens": 700,
        "system": system,
        "messages": messages,
        "output_config": {"effort": "low"},
    }


@pytest.mark.parametrize(
    ("bedrock_stop", "ours"),
    [("end_turn", "end_turn"), ("max_tokens", "max_tokens"), ("content_filtered", "refusal"), ("guardrail_intervened", "refusal")],
)
def test_stop_reasons_are_translated_so_escalation_rules_still_apply(stubbed, bedrock_stop, ours):
    client, stub = stubbed
    stub.add_response("converse", converse_response(stop=bedrock_stop), expected_params())
    assert Generator(backend=BedrockBackend(client=client, model_id=MODEL)).generate(QUESTION, HITS).stop_reason == ours


def test_reasoning_blocks_are_ignored_and_text_blocks_joined(stubbed):
    client, stub = stubbed
    body = converse_response()
    body["output"]["message"]["content"] = [{"reasoningContent": {"reasoningText": {"text": "thinking"}}}, {"text": "Part one. "}, {"text": "Part two [1]."}]
    stub.add_response("converse", body, expected_params())
    assert Generator(backend=BedrockBackend(client=client, model_id=MODEL)).generate(QUESTION, HITS).text == "Part one. Part two [1]."


# ---- shared safeguards: cache, replay, spend cap ----


def test_second_identical_question_is_served_from_the_cache_without_calling_aws(stubbed, tmp_path):
    client, stub = stubbed
    stub.add_response("converse", converse_response(), expected_params())  # only one response is queued
    gen = Generator(backend=BedrockBackend(client=client, model_id=MODEL), cache=ResponseCache(tmp_path))
    assert gen.generate(QUESTION, HITS).cached is False
    assert gen.generate(QUESTION, HITS).cached is True


def test_replay_only_never_calls_aws(stubbed, tmp_path):
    client, _ = stubbed
    gen = Generator(backend=BedrockBackend(client=client, model_id=MODEL), cache=ResponseCache(tmp_path), replay_only=True)
    with pytest.raises(ReplayMissError):
        gen.generate(QUESTION, HITS)


def test_bedrock_and_claude_requests_never_share_a_cache_entry():
    system, messages = build_prompt(QUESTION, HITS)
    a = ResponseCache.key(AnthropicBackend(client=object(), model=MODEL).build_request(system, messages, 700))
    b = ResponseCache.key(BedrockBackend(client=object(), model_id=MODEL).build_request(system, messages, 700))
    assert a != b


def test_spend_is_recorded_at_the_configured_price(stubbed, tmp_path, monkeypatch):
    client, stub = stubbed
    monkeypatch.setenv("LLM_PRICES", f'{{"{MODEL}": [1.0, 5.0]}}')
    stub.add_response("converse", converse_response(tokens=(1000, 100)), expected_params())
    budget = Budget(0.50, tmp_path / "spend.json")
    Generator(backend=BedrockBackend(client=client, model_id=MODEL), budget=budget).generate(QUESTION, HITS)
    assert budget.spent() == pytest.approx(1000 * 1.0 / 1e6 + 100 * 5.0 / 1e6)


def test_a_call_that_could_pass_the_cap_is_refused_before_it_is_sent(stubbed, tmp_path):
    client, _ = stubbed  # no response queued: a call would raise from Stubber
    from payments_rag.llm.budget import BudgetExceededError

    gen = Generator(backend=BedrockBackend(client=client, model_id=MODEL), budget=Budget(0.001, tmp_path / "spend.json"))
    with pytest.raises(BudgetExceededError):  # the unknown model is priced at the high fallback, so the cap trips early
        gen.generate(QUESTION, HITS)


def test_price_overrides_and_the_safe_fallback(monkeypatch):
    monkeypatch.delenv("LLM_PRICES", raising=False)
    assert price_for("some-unlisted-model") == (5.0, 25.0)
    monkeypatch.setenv("LLM_PRICES", '{"some-unlisted-model": [0.2, 0.9]}')
    assert price_for("some-unlisted-model") == (0.2, 0.9)
    assert cost_usd("some-unlisted-model", 1_000_000, 1_000_000) == pytest.approx(1.1)
    monkeypatch.setenv("LLM_PRICES", "")  # an empty line in .env counts as unset, quietly
    assert price_for("some-unlisted-model") == (5.0, 25.0)
    monkeypatch.setenv("LLM_PRICES", "not json")
    assert price_for("some-unlisted-model") == (5.0, 25.0)
    assert price_for("claude-haiku-5-5") == (0.10, 0.50)  # the built-in table wins


# ---- configuration ----


def test_the_model_id_is_never_guessed(monkeypatch):
    monkeypatch.delenv("BEDROCK_GENERATION_MODEL_ID", raising=False)
    with pytest.raises(ValueError, match="BEDROCK_GENERATION_MODEL_ID"):
        BedrockBackend()


def test_provider_selection(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert make_backend(client=object()).name == "anthropic"
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("BEDROCK_GENERATION_MODEL_ID", MODEL)
    backend = make_backend()
    assert backend.name == "bedrock" and backend.model == MODEL
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        make_backend()


def test_the_real_client_uses_the_runtime_endpoint_adaptive_retries_and_the_configured_region(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    client = BedrockBackend(model_id=MODEL).client  # created lazily; nothing is sent
    assert client.meta.service_model.service_name == "bedrock-runtime"
    assert client.meta.region_name == "us-east-2"
    # botocore normalises max_attempts=5 (retries) to total_max_attempts=6 (one try plus five retries)
    assert client.meta.config.retries == {"mode": "adaptive", "total_max_attempts": 6}


# ---- errors ----


def error_client(code, message="boom"):
    client = real_client()
    stub = Stubber(client)
    stub.add_client_error("converse", service_error_code=code, service_message=message)
    stub.activate()
    return client


@pytest.mark.parametrize("code", ["ThrottlingException", "ServiceUnavailableException", "ModelTimeoutException", "InternalServerException"])
def test_transient_provider_errors_become_unavailable(code):
    gen = Generator(backend=BedrockBackend(client=error_client(code), model_id=MODEL))
    with pytest.raises(LLMUnavailableError, match=code):
        gen.generate(QUESTION, HITS)


@pytest.mark.parametrize("code", ["AccessDeniedException", "ResourceNotFoundException", "ValidationException"])
def test_configuration_errors_become_unavailable_with_the_cause(code):
    gen = Generator(backend=BedrockBackend(client=error_client(code, "no access to arn:aws:bedrock:us-east-1:123456789012:x"), model_id=MODEL))
    with pytest.raises(LLMUnavailableError, match=code):
        gen.generate(QUESTION, HITS)


def test_an_unrecognised_error_is_not_swallowed():
    gen = Generator(backend=BedrockBackend(client=error_client("SomethingNew"), model_id=MODEL))
    with pytest.raises(ClientError):
        gen.generate(QUESTION, HITS)


def test_missing_credentials_become_unavailable():
    class NoCreds:
        def converse(self, **_):
            raise NoCredentialsError()

    gen = Generator(backend=BedrockBackend(client=NoCreds(), model_id=MODEL))
    with pytest.raises(LLMUnavailableError, match="NoCredentialsError"):
        gen.generate(QUESTION, HITS)


def test_the_api_returns_a_generic_503_and_audits_the_detail():
    class Down:
        def generate(self, question, hits):
            raise LLMUnavailableError("Bedrock AccessDeniedException: no access to arn:aws:bedrock:us-east-1:123456789012:x")

    service = make_service(generator=Down(), audit=StubAudit())
    app.dependency_overrides[get_service] = lambda: service
    try:
        r = TestClient(app).post("/ask", json={"question": QUESTION})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 503 and r.json()["detail"] == "the language model is temporarily unavailable"
    assert "123456789012" not in r.text
    entry = service.audit.entries[0]
    assert entry["outcome"] == "error" and "AccessDeniedException" in entry["error"]
