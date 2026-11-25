import boto3
import pytest
from moto import mock_aws

from payments_rag.audit.dynamo import DynamoAuditLog, create_tables

REGION = "us-east-2"


@pytest.fixture
def ddb(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    with mock_aws():
        client = boto3.client("dynamodb", region_name=REGION)
        create_tables(client, "entries", "head")
        yield client


def entry(n):
    return {"request_id": f"r{n}", "question": f"q{n}", "outcome": "answered", "ts": f"2026-11-25T00:00:0{n}Z"}


def log(ddb):
    return DynamoAuditLog("entries", "head", client=ddb)


def test_records_chain_and_verify_passes(ddb):
    audit = log(ddb)
    hashes = [audit.record(entry(i)) for i in range(3)]
    assert len(set(hashes)) == 3
    report = audit.verify()
    assert report == {"ok": True, "rows": 3, "head_hash": hashes[-1], "errors": []}


def test_empty_log_verifies(ddb):
    assert log(ddb).verify()["ok"] is True


def test_concurrent_writer_conflict_is_retried_and_the_chain_stays_linear(ddb, monkeypatch):
    a, b = log(ddb), log(ddb)
    real = a._read_head
    stale = real()
    calls = {"n": 0}

    def head_once_stale():
        calls["n"] += 1
        return stale if calls["n"] == 1 else real()

    monkeypatch.setattr(a, "_read_head", head_once_stale)
    b.record(entry(1))  # moves the head after A read it
    a.record(entry(2))
    assert calls["n"] == 2  # A's first transaction was refused and retried
    report = a.verify()
    assert report["ok"] and report["rows"] == 2


def test_an_edited_entry_is_detected(ddb):
    audit = log(ddb)
    for i in range(3):
        audit.record(entry(i))
    item = ddb.get_item(TableName="entries", Key={"pk": {"S": "chain"}, "seq": {"N": "2"}})["Item"]
    item["canonical"] = {"S": item["canonical"]["S"].replace("q1", "edited")}
    ddb.put_item(TableName="entries", Item=item)
    report = audit.verify()
    assert not report["ok"] and any("seq 2" in e for e in report["errors"])


def test_a_deleted_entry_in_the_middle_is_detected(ddb):
    audit = log(ddb)
    for i in range(3):
        audit.record(entry(i))
    ddb.delete_item(TableName="entries", Key={"pk": {"S": "chain"}, "seq": {"N": "2"}})
    assert not audit.verify()["ok"]


def test_entries_deleted_from_the_end_are_detected_through_the_head(ddb):
    audit = log(ddb)
    for i in range(3):
        audit.record(entry(i))
    ddb.delete_item(TableName="entries", Key={"pk": {"S": "chain"}, "seq": {"N": "3"}})
    report = audit.verify()
    assert not report["ok"] and any("head" in e for e in report["errors"])


def test_build_service_uses_dynamo_audit_when_tables_are_configured(ddb, monkeypatch):
    from payments_rag.api import service

    monkeypatch.setenv("AUDIT_TABLE", "entries")
    monkeypatch.setenv("AUDIT_HEAD_TABLE", "head")
    assert isinstance(service.build_audit(), DynamoAuditLog)


def test_verify_cli_reads_dynamo_when_tables_are_set(ddb, monkeypatch, capsys):
    from payments_rag.audit.log import main

    log(ddb).record(entry(1))
    monkeypatch.setenv("AUDIT_TABLE", "entries")
    monkeypatch.setenv("AUDIT_HEAD_TABLE", "head")
    monkeypatch.setattr("payments_rag.audit.dynamo.DynamoAuditLog.client", property(lambda self: ddb))
    assert main(["verify"]) == 0
    assert "OK: 1 rows" in capsys.readouterr().out


def test_entries_added_during_verify_do_not_raise_a_false_alarm(ddb, monkeypatch):
    audit = log(ddb)
    audit.record(entry(1))
    real_entries = audit._entries

    def entries_then_new_question(up_to):
        rows = real_entries(up_to)
        audit.record(entry(2))  # a question answered between reading the head and the entries
        return rows

    monkeypatch.setattr(audit, "_entries", entries_then_new_question)
    assert audit.verify()["ok"] is True


def _rewrite_chain_from(ddb, audit, start):
    """What the API role could do: replace entries with unconditional puts and recompute the chain and head."""
    from payments_rag.audit.log import GENESIS, _canonical, _row_hash

    items = ddb.query(TableName="entries", KeyConditionExpression="pk = :c",
                      ExpressionAttributeValues={":c": {"S": "chain"}})["Items"]
    prev = GENESIS if start == 1 else items[start - 2]["row_hash"]["S"]
    for item in items[start - 1:]:
        canonical = _canonical({"request_id": "forged", "outcome": "answered", "question": "rewritten"})
        digest = _row_hash(prev, canonical)
        ddb.put_item(TableName="entries", Item={**item, "canonical": {"S": canonical}, "prev_hash": {"S": prev}, "row_hash": {"S": digest}})
        prev = digest
    ddb.put_item(TableName="head", Item={"pk": {"S": "head"}, "seq": {"N": str(len(items))}, "row_hash": {"S": prev}})


def test_a_consistent_rewrite_before_the_last_anchor_is_caught(ddb):
    from payments_rag.audit.dynamo import create_anchor_table

    create_anchor_table(ddb, "anchor")
    audit = DynamoAuditLog("entries", "head", client=ddb, anchor_table="anchor")
    for i in range(3):
        audit.record(entry(i))
    assert audit.verify(update_anchor=True)["ok"] is True  # the daily job anchors seq 3
    audit.record(entry(4))
    _rewrite_chain_from(ddb, audit, 2)
    assert log(ddb).verify()["ok"] is True  # without the anchor, the rewrite is invisible
    report = audit.verify(update_anchor=True)
    assert report["ok"] is False and "anchor at seq 3" in report["errors"][0]
    anchor = ddb.get_item(TableName="anchor", Key={"pk": {"S": "anchor"}})["Item"]
    assert anchor["seq"]["N"] == "3"  # a failed verify never moves the anchor


def test_verify_without_update_leaves_the_anchor_alone(ddb):
    from payments_rag.audit.dynamo import create_anchor_table

    create_anchor_table(ddb, "anchor")
    audit = DynamoAuditLog("entries", "head", client=ddb, anchor_table="anchor")
    audit.record(entry(1))
    assert audit.verify()["ok"] is True
    assert "Item" not in ddb.get_item(TableName="anchor", Key={"pk": {"S": "anchor"}})


def test_record_gives_up_after_its_deadline(ddb, monkeypatch):
    import payments_rag.audit.dynamo as mod

    audit = log(ddb)
    monkeypatch.setattr(mod, "WRITE_DEADLINE_SECONDS", 0.05)

    def always_cancelled(**kwargs):
        raise ddb.exceptions.TransactionCanceledException({"Error": {"Code": "TransactionCanceledException"}}, "TransactWriteItems")

    monkeypatch.setattr(ddb, "transact_write_items", always_cancelled)
    with pytest.raises(RuntimeError, match="contention"):
        audit.record(entry(1))
