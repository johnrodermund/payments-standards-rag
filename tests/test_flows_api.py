from fastapi.testclient import TestClient

from payments_rag.api.main import app

client = TestClient(app)


def test_list_flows_returns_four_flows_with_message_names():
    r = client.get("/flows")
    assert r.status_code == 200
    body = r.json()
    assert "Not an ISO publication" in body["source_note"]
    assert {f["id"] for f in body["flows"]} == {"serial-credit-transfer", "cover-method", "fi-credit-transfer", "return"}
    step = next(f for f in body["flows"] if f["id"] == "serial-credit-transfer")["steps"][0]
    assert step["message"] == "pain.001"
    assert step["message_id"] == "pain.001.001.13"
    assert step["message_name"] == "Customer Credit Transfer Initiation V13"
    assert {"n", "from", "to", "purpose"} <= set(step)
    assert "cites" not in step


def test_get_flow_by_id():
    r = client.get("/flows/return")
    assert r.status_code == 200
    assert r.json()["title"] == "Payment return"
    assert r.json()["source_note"]


def test_unknown_flow_is_404():
    assert client.get("/flows/direct-debit").status_code == 404


def test_flows_work_without_a_database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:nothing@127.0.0.1:1/none")
    assert client.get("/flows").status_code == 200


def test_flows_response_is_cacheable():
    assert client.get("/flows").headers["cache-control"] == "public, max-age=300"
    assert client.get("/flows/return").headers["cache-control"] == "public, max-age=300"
