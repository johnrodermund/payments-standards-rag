from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from payments_rag.api.ratelimit import DynamoRateLimiter, client_key, create_table

REGION = "us-east-2"


@pytest.fixture
def ddb(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    with mock_aws():
        client = boto3.client("dynamodb", region_name=REGION)
        create_table(client, "limits")
        yield client


def req(headers=None, host="10.0.0.1"):
    return SimpleNamespace(headers={k.lower(): v for k, v in (headers or {}).items()}, client=SimpleNamespace(host=host))


def test_key_comes_from_the_cloudfront_viewer_address_without_the_port():
    a = client_key(req({"CloudFront-Viewer-Address": "203.0.113.7:51234"}))
    b = client_key(req({"CloudFront-Viewer-Address": "203.0.113.7:40000"}))
    assert a == b and "203.0.113.7" not in a  # hashed, raw IP not stored


def test_ipv6_clients_are_counted_per_64_so_rotating_addresses_does_not_help():
    a = client_key(req({"CloudFront-Viewer-Address": "2001:db8:0:1::1:443"}))
    b = client_key(req({"CloudFront-Viewer-Address": "2001:db8:0:1:ffff:1234:5678:9abc:443"}))
    other = client_key(req({"CloudFront-Viewer-Address": "2001:db8:0:2::1:443"}))
    assert a == b and a != other


def test_forged_x_forwarded_for_changes_nothing():
    base = req({"CloudFront-Viewer-Address": "203.0.113.7:1"})
    forged = req({"CloudFront-Viewer-Address": "203.0.113.7:1", "X-Forwarded-For": "1.2.3.4"})
    assert client_key(base) == client_key(forged)


def test_without_cloudfront_the_socket_address_is_used():
    assert client_key(req(host="10.0.0.1")) != client_key(req(host="10.0.0.2"))


def limiter(ddb, per_minute=3, per_day=5, now=None):
    return DynamoRateLimiter("limits", per_minute, per_day, client=ddb, clock=(lambda: now) if now else None)


def test_minute_limit_then_retry_after(ddb):
    t = datetime(2026, 11, 25, 12, 0, 10, tzinfo=UTC)
    rl = limiter(ddb, now=t)
    assert [rl.hit("k") for _ in range(3)] == [None, None, None]
    assert rl.hit("k") == 50  # seconds left in this minute


def test_other_clients_are_not_affected(ddb):
    rl = limiter(ddb, per_minute=1)
    assert rl.hit("a") is None
    assert rl.hit("b") is None


def test_day_limit_spans_minutes(ddb):
    t = datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)
    for minute in range(5):
        assert limiter(ddb, now=t + timedelta(minutes=minute)).hit("k") is None
    retry = limiter(ddb, now=t + timedelta(minutes=6)).hit("k")
    assert retry is not None and retry > 3600


def test_counters_expire_with_ttl(ddb):
    rl = limiter(ddb)
    rl.hit("k")
    items = ddb.scan(TableName="limits")["Items"]
    assert items and all("expires_at" in i for i in items)


def test_ask_returns_429_with_retry_after(ddb, monkeypatch):
    from payments_rag.api import main
    from tests.test_ask import make_service

    monkeypatch.setattr(main, "_limiter", limiter(ddb, per_minute=1))
    main.app.dependency_overrides[main.get_service] = lambda: make_service()
    try:
        http = TestClient(main.app)
        assert http.post("/ask", json={"question": "What is the creditor agent?"}).status_code == 200
        r = http.post("/ask", json={"question": "What is the creditor agent?"})
    finally:
        main.app.dependency_overrides.clear()
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0


def test_no_table_means_no_limit(monkeypatch):
    from payments_rag.api.ratelimit import limiter_from_env

    monkeypatch.delenv("RATE_LIMIT_TABLE", raising=False)
    assert limiter_from_env() is None


def test_an_unreachable_table_allows_the_request():
    from payments_rag.api.ratelimit import check

    class Down:
        def hit(self, key):
            raise RuntimeError("dynamodb unreachable")

    assert check(Down(), req()) is None
