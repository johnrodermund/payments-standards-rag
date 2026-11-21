import pytest

from payments_rag.llm.budget import Budget, BudgetExceededError, cost_usd
from payments_rag.llm.cache import ResponseCache
from payments_rag.llm.generation import Generator

HITS = [{"id": "m#a", "text": "A passage."}]


class Usage:
    input_tokens = 1000
    output_tokens = 100


class Block:
    type = "text"
    text = "Answer [1]."


class Response:
    content = (Block(),)
    usage = Usage()
    stop_reason = "end_turn"
    model = "claude-haiku-5-5"


class Messages:
    def __init__(self):
        self.calls = 0

    def create(self, **_):
        self.calls += 1
        return Response()


class Client:
    def __init__(self):
        self.messages = Messages()


def test_cost_uses_the_price_table_and_a_pessimistic_fallback():
    assert cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert cost_usd("some-new-model", 1_000_000, 0) == pytest.approx(5.0)


def test_calls_are_recorded_and_cached_calls_are_free(tmp_path):
    budget = Budget(0.50, tmp_path / "spend.json")
    client = Client()
    gen = Generator(client=client, cache=ResponseCache(tmp_path / "c"), budget=budget)
    gen.generate("q", HITS)
    spent = budget.spent()
    assert spent == pytest.approx(cost_usd("claude-haiku-5-5", 1000, 100))
    assert gen.generate("q", HITS).cached is True
    assert budget.spent() == spent and client.messages.calls == 1


def test_call_is_refused_before_it_is_sent_when_it_would_pass_the_limit(tmp_path):
    budget = Budget(0.0001, tmp_path / "spend.json")
    client = Client()
    gen = Generator(client=client, budget=budget)
    with pytest.raises(BudgetExceededError):
        gen.generate("q", HITS)
    assert client.messages.calls == 0


def test_limit_trips_across_calls_and_survives_a_new_budget_object(tmp_path):
    ledger = tmp_path / "spend.json"
    client = Client()
    gen = Generator(client=client, budget=Budget(0.0005, ledger))
    sent = 0
    with pytest.raises(BudgetExceededError):
        for i in range(50):
            gen.generate(f"question {i}", HITS)
            sent += 1
    assert 0 < sent < 50
    with pytest.raises(BudgetExceededError):
        Generator(client=client, budget=Budget(0.0005, ledger)).generate("another", HITS)


def test_default_limit_is_fifty_cents(monkeypatch):
    monkeypatch.delenv("LLM_BUDGET_USD", raising=False)
    assert Budget.from_env().limit_usd == 0.50
    monkeypatch.setenv("LLM_BUDGET_USD", "1.25")
    assert Budget.from_env().limit_usd == 1.25
