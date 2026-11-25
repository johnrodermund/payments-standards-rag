import pytest
from fastapi.testclient import TestClient

from payments_rag.api.ingest import get_ingest_service
from payments_rag.api.main import app
from payments_rag.ingestion.jobs import ALLOWED_MESSAGES, ActiveJobError

TOKEN = "a-sufficiently-long-test-token"
JOB = {"id": "11111111-1111-1111-1111-111111111111", "status": "queued", "stage": "queued", "params": {}}
OTHER_ID = "22222222-2222-2222-2222-222222222222"


class FakeStore:
    def get(self, job_id):
        return JOB if job_id == JOB["id"] else None

    def recent(self, n):
        return [JOB]


class FakeService:
    def __init__(self, active=None):
        self.submitted, self.active, self.store = [], active, FakeStore()

    def submit(self, params):
        if self.active:
            raise ActiveJobError(self.active)
        self.submitted.append(params)
        return JOB


@pytest.fixture
def service():
    fake = FakeService()
    app.dependency_overrides[get_ingest_service] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture
def client(service, monkeypatch):
    monkeypatch.setenv("INGEST_API_TOKEN", TOKEN)
    return TestClient(app)


def auth(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


def test_ingest_is_disabled_when_no_token_is_configured(service, monkeypatch):
    monkeypatch.delenv("INGEST_API_TOKEN", raising=False)
    http = TestClient(app)
    for response in (http.post("/ingest", json={}, headers=auth()), http.get("/ingest", headers=auth()), http.get(f"/ingest/{JOB['id']}")):
        assert response.status_code == 503 and "INGEST_API_TOKEN" in response.json()["detail"]
    assert service.submitted == []


def test_a_short_token_counts_as_not_configured(service, monkeypatch):
    monkeypatch.setenv("INGEST_API_TOKEN", "short")
    assert TestClient(app).post("/ingest", json={}, headers=auth("short")).status_code == 503


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong-token-value-here"}, {"Authorization": TOKEN}, {"Authorization": f"Basic {TOKEN}"}, {"Authorization": "Bearer "}],
)
def test_missing_or_wrong_credentials_are_rejected_and_start_nothing(client, service, headers):
    for response in (client.post("/ingest", json={}, headers=headers), client.get("/ingest", headers=headers), client.get(f"/ingest/{JOB['id']}", headers=headers)):
        assert response.status_code == 401 and response.headers["www-authenticate"] == "Bearer"
    assert service.submitted == []


def test_accepted_request_returns_202_with_a_status_url(client, service):
    r = client.post("/ingest", json={}, headers=auth())
    assert r.status_code == 202
    assert r.json() == {"job_id": JOB["id"], "status": "queued", "status_url": f"/ingest/{JOB['id']}"}
    assert r.headers["location"] == f"/ingest/{JOB['id']}"
    assert service.submitted == [{"source": "extract", "messages": sorted(ALLOWED_MESSAGES)}]


def test_messages_are_deduplicated_and_sorted(client, service):
    client.post("/ingest", json={"messages": ["pacs.008", "pacs.008"], "source": "repository"}, headers=auth())
    assert service.submitted == [{"source": "repository", "messages": ["pacs.008"]}]


@pytest.mark.parametrize(
    "body",
    [
        {"messages": ["camt.056"]},
        {"messages": []},
        {"source": "https://example.com/evil.zip"},
        {"path": "/etc/passwd"},
        {"url": "http://169.254.169.254/latest/meta-data"},
        {"zip_path": "../../secrets.zip"},
    ],
)
def test_callers_cannot_choose_paths_urls_or_unknown_messages(client, service, body):
    assert client.post("/ingest", json=body, headers=auth()).status_code == 422
    assert service.submitted == []


def test_a_second_request_while_a_job_is_active_gets_409(monkeypatch):
    busy = FakeService(active={"id": OTHER_ID, "status": "running"})
    app.dependency_overrides[get_ingest_service] = lambda: busy
    monkeypatch.setenv("INGEST_API_TOKEN", TOKEN)
    try:
        r = TestClient(app).post("/ingest", json={}, headers=auth())
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 409
    assert r.json()["detail"] == {"error": "an ingest job is already active", "active_job_id": OTHER_ID, "status": "running"}


def test_status_and_listing(client):
    assert client.get(f"/ingest/{JOB['id']}", headers=auth()).json()["status"] == "queued"
    assert client.get("/ingest/33333333-3333-3333-3333-333333333333", headers=auth()).status_code == 404
    assert [j["id"] for j in client.get("/ingest", headers=auth()).json()["jobs"]] == [JOB["id"]]


def test_the_token_is_never_echoed_back(client):
    r = client.post("/ingest", json={"messages": ["pain.001"]}, headers=auth())
    assert TOKEN not in r.text


def test_default_request_asks_for_all_six_messages(client, service):
    from typing import get_args

    from payments_rag.api.ingest import IngestRequest
    from payments_rag.ingestion.jobs import ALLOWED_MESSAGES

    assert client.post("/ingest", json={}, headers=auth()).status_code == 202
    assert service.submitted[-1]["messages"] == sorted(ALLOWED_MESSAGES)
    assert len(ALLOWED_MESSAGES) == 6
    literal = get_args(IngestRequest.model_fields["messages"].annotation)[0]
    assert set(get_args(literal)) == set(ALLOWED_MESSAGES)


def test_unknown_message_is_rejected(client, service):
    assert client.post("/ingest", json={"messages": ["camt.056"]}, headers=auth()).status_code == 422
    assert service.submitted == []
