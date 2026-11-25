import threading
from datetime import UTC, datetime

import boto3
import pytest
from moto import mock_aws

from payments_rag.llm.backends import BackendResponse, LLMUnavailableError
from payments_rag.llm.budget import BudgetExceededError, cost_usd
from payments_rag.llm.generation import Generator
from payments_rag.llm.spend_cap import DynamoSpendCap, create_table
from tests.test_ask import HITS

REGION = "us-east-2"
MODEL = "claude-haiku-5-5"
REQUEST = {"model": MODEL, "messages": [{"role": "user", "content": "x" * 400}]}


@pytest.fixture
def ddb(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    with mock_aws():
        client = boto3.client("dynamodb", region_name=REGION)
        create_table(client, "spend")
        yield client


def cap(ddb, limit=0.01, now=None):
    return DynamoSpendCap("spend", limit, client=ddb, clock=(lambda: now) if now else None)


def committed(ddb, month):
    item = ddb.get_item(TableName="spend", Key={"pk": {"S": month}}).get("Item", {})
    return float(item.get("committed", {}).get("N", "0")), float(item.get("spent", {}).get("N", "0"))


def test_reservation_is_settled_to_the_actual_cost(ddb):
    c = cap(ddb)
    reserved = c.check(MODEL, REQUEST, max_tokens=700)
    assert reserved > 0
    cost = c.record(MODEL, 500, 50, reserved=reserved)
    month = datetime.now(UTC).strftime("%Y-%m")
    total, spent = committed(ddb, month)
    assert spent == pytest.approx(cost) == pytest.approx(cost_usd(MODEL, 500, 50))
    assert total == pytest.approx(cost)


def test_call_is_refused_when_the_worst_case_would_pass_the_limit(ddb):
    c = cap(ddb, limit=0.0003)
    with pytest.raises(BudgetExceededError) as exc:
        c.check(MODEL, REQUEST, max_tokens=700)
    assert "$" not in str(exc.value) and "spend" not in str(exc.value).lower().replace("monthly", "")


def test_concurrent_reservations_never_pass_the_limit(ddb):
    c = cap(ddb, limit=0.002)
    worst = cost_usd(MODEL, len(str(REQUEST)) // 4 + 1, 700)
    granted, refused = [], []

    def go():
        try:
            granted.append(c.check(MODEL, REQUEST, max_tokens=700))
        except BudgetExceededError:
            refused.append(1)

    threads = [threading.Thread(target=go) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(granted) * worst <= 0.002 + 1e-9
    assert refused


def test_release_returns_the_reservation(ddb):
    c = cap(ddb)
    reserved = c.check(MODEL, REQUEST, max_tokens=700)
    c.release(reserved)
    total, spent = committed(ddb, datetime.now(UTC).strftime("%Y-%m"))
    assert total == pytest.approx(0) and spent == 0


def test_a_new_month_starts_from_zero(ddb):
    nov = cap(ddb, limit=0.0006, now=datetime(2026, 11, 30, 23, 59, tzinfo=UTC))
    nov.record(MODEL, 0, 1100, reserved=nov.check(MODEL, REQUEST, max_tokens=700))
    with pytest.raises(BudgetExceededError):
        nov.check(MODEL, REQUEST, max_tokens=700)
    dec = cap(ddb, limit=0.0006, now=datetime(2026, 12, 1, 0, 0, tzinfo=UTC))
    assert dec.check(MODEL, REQUEST, max_tokens=700) > 0


def test_spend_is_emitted_as_a_cloudwatch_metric(ddb, capsys):
    c = cap(ddb)
    c.record(MODEL, 500, 50, reserved=c.check(MODEL, REQUEST, max_tokens=700))
    out = capsys.readouterr().out
    assert '"CloudWatchMetrics"' in out and '"MonthlySpendUSD"' in out


class FailingBackend:
    name, model = "fake", MODEL

    def build_request(self, system, messages, max_tokens):
        return {"model": MODEL, "messages": messages, "max_tokens": max_tokens}

    def send(self, request):
        raise LLMUnavailableError("down")


class OkBackend(FailingBackend):
    def send(self, request):
        return BackendResponse(text="Answer [1].", model=MODEL, usage={"input_tokens": 400, "output_tokens": 40}, stop_reason="end_turn")


def test_generator_releases_the_reservation_when_the_provider_call_fails(ddb):
    c = cap(ddb)
    with pytest.raises(LLMUnavailableError):
        Generator(backend=FailingBackend(), budget=c).generate("q", HITS)
    total, _ = committed(ddb, datetime.now(UTC).strftime("%Y-%m"))
    assert total == pytest.approx(0)


def test_generator_settles_the_actual_cost(ddb):
    c = cap(ddb)
    Generator(backend=OkBackend(), budget=c).generate("q", HITS)
    total, spent = committed(ddb, datetime.now(UTC).strftime("%Y-%m"))
    assert spent == pytest.approx(cost_usd(MODEL, 400, 40)) == pytest.approx(total)


def test_budget_from_env_picks_the_dynamo_cap(ddb, monkeypatch):
    from payments_rag.llm.spend_cap import budget_from_env

    monkeypatch.setenv("SPEND_TABLE", "spend")
    monkeypatch.setenv("LLM_BUDGET_USD", "5")
    b = budget_from_env()
    assert isinstance(b, DynamoSpendCap) and b.limit_usd == 5.0


def test_budget_refusal_is_generic_to_the_caller():
    from fastapi.testclient import TestClient

    from payments_rag.api.main import app, get_service
    from tests.test_ask import StubAudit, make_service

    class Broke:
        model = MODEL

        def generate(self, question, hits):
            raise BudgetExceededError("API budget $5.00 would be exceeded: spent $4.9999 ... delete .cache/llm/spend.json")

    app.dependency_overrides[get_service] = lambda: make_service(audit=StubAudit(), generator=Broke())
    try:
        r = TestClient(app).post("/ask", json={"question": "What is the creditor agent?"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 503
    assert "$" not in r.json()["detail"] and "spend.json" not in r.json()["detail"]
    assert "budget" in r.json()["detail"]


def test_a_call_spanning_month_end_settles_into_the_month_it_was_reserved_in(ddb):
    now = [datetime(2026, 11, 30, 23, 59, 59, tzinfo=UTC)]
    c = DynamoSpendCap("spend", 0.01, client=ddb, clock=lambda: now[0])
    reserved = c.check(MODEL, REQUEST, max_tokens=700)
    now[0] = datetime(2026, 12, 1, 0, 0, 1, tzinfo=UTC)
    cost = c.record(MODEL, 500, 50, reserved=reserved)
    assert committed(ddb, "2026-11")[0] == pytest.approx(cost)
    assert committed(ddb, "2026-12") == (0, 0)


def test_release_after_month_end_goes_back_to_the_reserving_month(ddb):
    now = [datetime(2026, 11, 30, 23, 59, 59, tzinfo=UTC)]
    c = DynamoSpendCap("spend", 0.01, client=ddb, clock=lambda: now[0])
    reserved = c.check(MODEL, REQUEST, max_tokens=700)
    now[0] = datetime(2026, 12, 1, 0, 0, 1, tzinfo=UTC)
    c.release(reserved)
    assert committed(ddb, "2026-11")[0] == pytest.approx(0)
    assert committed(ddb, "2026-12") == (0, 0)


def test_a_failed_release_does_not_hide_the_provider_error(ddb, monkeypatch):
    c = cap(ddb)

    def broken_release(reserved):
        raise RuntimeError("dynamodb throttled")

    monkeypatch.setattr(c, "release", broken_release)
    with pytest.raises(LLMUnavailableError):
        Generator(backend=FailingBackend(), budget=c).generate("q", HITS)
