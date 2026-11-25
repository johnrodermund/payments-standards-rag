import json

import httpx
import openai
import pytest

from payments_rag.llm.backends import LLMUnavailableError, OpenAIBackend, make_backend
from payments_rag.llm.budget import price_for
from payments_rag.llm.generation import SYSTEM_PROMPT, Generator, build_prompt
from tests.test_ask import HITS


def response_body(text="Creditor Agent is the creditor's bank [1].", status="completed", incomplete=None, usage=(120, 30)):
    return {
        "id": "resp_123",
        "object": "response",
        "created_at": 1764000000,
        "model": "gpt-6-luna",
        "status": status,
        "incomplete_details": incomplete,
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": text, "annotations": []}]},
        ],
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1], "total_tokens": sum(usage),
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 10}},
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def backend_with(handler, model="gpt-6-luna"):
    seen = []

    def record(request):
        seen.append(request)
        return handler(request)

    client = openai.OpenAI(api_key="test-key", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(record)))
    return OpenAIBackend(client=client, model=model), seen


def ok(body=None, status=200, headers=None):
    return lambda request: httpx.Response(status, json=body if body is not None else response_body(), headers=headers or {"x-request-id": "req_abc"})


def test_request_uses_the_responses_api_with_low_effort_and_a_token_cap():
    backend, seen = backend_with(ok())
    system, messages = build_prompt("What is the creditor agent?", HITS)
    backend.send(backend.build_request(system, messages, 700))
    sent = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/responses"
    assert sent["model"] == "gpt-6-luna"
    assert sent["instructions"] == SYSTEM_PROMPT
    assert sent["input"] == messages
    assert sent["max_output_tokens"] == 700
    assert sent["reasoning"] == {"effort": "low"}
    assert sent["store"] is False


def test_text_usage_model_and_request_id_are_read_from_the_response():
    backend, _ = backend_with(ok())
    out = backend.send(backend.build_request("s", [{"role": "user", "content": "q"}], 700))
    assert out.text == "Creditor Agent is the creditor's bank [1]."
    assert out.usage == {"input_tokens": 120, "output_tokens": 30}
    assert out.model == "gpt-6-luna"
    assert out.stop_reason == "end_turn"
    assert out.request_id == "req_abc"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("max_output_tokens", "max_tokens"), ("content_filter", "refusal")],
)
def test_incomplete_responses_map_to_the_stop_reasons_the_escalation_rules_use(reason, expected):
    backend, _ = backend_with(ok(response_body(status="incomplete", incomplete={"reason": reason})))
    assert backend.send(backend.build_request("s", [{"role": "user", "content": "q"}], 700)).stop_reason == expected


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_errors_become_llm_unavailable(status):
    backend, _ = backend_with(ok({"error": {"message": "busy", "type": "server_error"}}, status=status))
    with pytest.raises(LLMUnavailableError, match="OpenAI"):
        backend.send(backend.build_request("s", [{"role": "user", "content": "q"}], 700))


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_configuration_errors_become_llm_unavailable_without_the_key(status):
    backend, _ = backend_with(ok({"error": {"message": "bad model gpt-x", "type": "invalid_request_error"}}, status=status))
    with pytest.raises(LLMUnavailableError) as exc:
        backend.send(backend.build_request("s", [{"role": "user", "content": "q"}], 700))
    assert "test-key" not in str(exc.value)


def test_connection_failure_becomes_llm_unavailable():
    def boom(request):
        raise httpx.ConnectError("no route")

    backend, _ = backend_with(boom)
    with pytest.raises(LLMUnavailableError):
        backend.send(backend.build_request("s", [{"role": "user", "content": "q"}], 700))


def test_model_defaults_to_luna_and_can_be_set_by_env(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert OpenAIBackend(client=object()).model == "gpt-6-luna"
    monkeypatch.setenv("OPENAI_MODEL", "gpt-6.1-terra")
    assert OpenAIBackend(client=object()).model == "gpt-6.1-terra"


def test_llm_provider_openai_selects_the_backend(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert isinstance(make_backend(client=object()), OpenAIBackend)


def test_prices_are_recorded_for_luna_and_terra():
    assert price_for("gpt-6-luna") == (0.10, 0.50)
    assert price_for("gpt-6.1-terra") == (1.0, 5.0)


def test_openai_and_claude_requests_never_share_a_cache_key():
    from payments_rag.llm.backends import AnthropicBackend

    system, messages = build_prompt("q", HITS)
    a = AnthropicBackend(client=object()).build_request(system, messages, 700)
    o = OpenAIBackend(client=object()).build_request(system, messages, 700)
    assert a != o


def test_generator_answers_through_openai_with_citations(tmp_path):
    from payments_rag.llm.cache import ResponseCache

    backend, seen = backend_with(ok())
    gen = Generator(backend=backend, cache=ResponseCache(tmp_path / "cache.jsonl"))
    result = gen.generate("What is the creditor agent?", HITS)
    assert "[1]" in result.text
    assert len(seen) == 1
    gen.generate("What is the creditor agent?", HITS)
    assert len(seen) == 1  # second call replayed from the cache
