import copy

from payments_rag.ingestion.chunking import MAX_CHARS, chunk_payload
from payments_rag.ingestion.flows import FLOW_SOURCE, check_flows, flow_chunks
from payments_rag.ingestion.validation import validate_chunks


def message(msg_id, name):
    return {
        "id": msg_id,
        "name": name,
        "definition": "Scope\nSent between agents.\nUsage\nUsed in credit transfers.",
        "registration_status": "Registered",
        "constraints": [],
        "elements": [
            {
                "xml_path": "Root/GrpHdr",
                "path": f"{name} > GroupHeader",
                "name": "GroupHeader",
                "definition": "Set of characteristics shared by all transactions.",
                "min_occurs": "1",
                "max_occurs": "1",
                "type": "GroupHeader1",
                "constraints": [],
            }
        ],
    }


def payload(*ids):
    known = {
        "pacs.008": message("pacs.008.001.14", "FIToFICustomerCreditTransferV14"),
        "pacs.002": message("pacs.002.001.16", "FIToFIPaymentStatusReportV16"),
    }
    return {"source": {"file": "x.zip", "sha256": "0" * 64}, "messages": [known[i] for i in ids or known]}


def flows(**overrides):
    flow = {
        "id": "demo",
        "title": "Demo transfer",
        "summary": "A credit transfer between two agents and its status report.",
        "parties": ["Debtor agent", "Creditor agent"],
        "steps": [
            {"n": 1, "from": "Debtor agent", "to": "Creditor agent", "message": "pacs.008",
             "purpose": "Moves the funds.", "cites": ["pacs.008.001.14#message:scope"]},
            {"n": 2, "from": "Creditor agent", "to": "Debtor agent", "message": "pacs.002",
             "purpose": "Reports the status.", "cites": ["pacs.002.001.16#message:scope"]},
        ],
        "notes": ["Status may be positive or negative."],
        "cites": ["pacs.008.001.14#message:usage"],
    }
    flow.update(overrides)
    return {"source_note": "Written for this project from the cited ISO text.", "flows": [flow]}


def only(chunks, chunk_id):
    return next(c for c in chunks if c["id"] == chunk_id)


def test_flow_chunk_has_id_kind_and_metadata():
    c = only(flow_chunks(flows(), payload()), "flow:demo")
    assert c["kind"] == "flow"
    assert c["metadata"]["message_id"] == "flow"
    assert c["metadata"]["message_name"] == "Demo transfer"
    assert c["metadata"]["source"] == FLOW_SOURCE
    assert c["metadata"]["messages"] == ["pacs.008", "pacs.002"]
    assert "pacs.008.001.14#message:scope" in c["metadata"]["cites"]


def test_flow_text_lists_numbered_steps_with_parties_and_message_names():
    text = only(flow_chunks(flows(), payload()), "flow:demo")["text"]
    assert "not ISO text" in text
    assert "1. Debtor agent → Creditor agent: pacs.008" in text
    assert "FI To FI Customer Credit Transfer" in text
    assert "Parties: Debtor agent, Creditor agent" in text


def test_flow_skipped_when_a_message_is_missing():
    assert flow_chunks(flows(), payload("pacs.008")) == []


def test_index_chunk_lists_included_flows():
    index = only(flow_chunks(flows(), payload()), "flow:index")
    assert index["kind"] == "flow"
    assert "Demo transfer" in index["text"]


def test_long_flow_is_split_below_max_chars():
    many = [
        {"n": n, "from": "Debtor agent", "to": "Creditor agent", "message": "pacs.008",
         "purpose": "Forwards the payment to the next agent in the chain. " * 3, "cites": ["pacs.008.001.14#message:scope"]}
        for n in range(1, 16)
    ]
    chunks = [c for c in flow_chunks(flows(steps=many), payload()) if c["id"] != "flow:index"]
    assert len(chunks) > 1
    assert all(len(c["text"]) <= MAX_CHARS for c in chunks)
    assert {"flow:demo", "flow:demo:2"} <= {c["id"] for c in chunks}


def test_check_flows_reports_unknown_cite_unknown_message_duplicate_id_and_missing_note():
    bad = flows()
    bad["flows"][0]["steps"][0]["cites"] = ["nope#x"]
    bad["flows"][0]["steps"][1]["message"] = "camt.056"
    bad["flows"].append(copy.deepcopy(bad["flows"][0]))
    bad["source_note"] = ""
    errors = " | ".join(check_flows(bad, payload()))
    assert "nope#x" in errors
    assert "camt.056" in errors
    assert "duplicate" in errors
    assert "source_note" in errors


def test_check_flows_rejects_a_bad_id():
    assert any("id" in e for e in check_flows(flows(id="Bad Id"), payload()))


def test_check_flows_accepts_valid_file():
    assert check_flows(flows(), payload()) == []


def test_chunk_payload_without_flows_is_unchanged():
    plain = [c["id"] for c in chunk_payload(payload())]
    with_flows = [c["id"] for c in chunk_payload(payload(), flows())]
    assert with_flows[: len(plain)] == plain
    assert set(with_flows) - set(plain) == {"flow:demo", "flow:index"}


def test_validation_accepts_flow_kind_and_requires_messages_and_cites():
    chunks = chunk_payload(payload(), flows())
    assert validate_chunks(chunks)["passed"]
    broken = copy.deepcopy(chunks)
    del only(broken, "flow:demo")["metadata"]["cites"]
    report = validate_chunks(broken)
    assert any(i["check"] == "missing_metadata" and i["chunk_id"] == "flow:demo" for i in report["issues"])


def test_cli_refuses_a_flows_file_that_fails_checks(tmp_path):
    import json

    import pytest

    from payments_rag.ingestion.chunking import main

    extract, bad, out = tmp_path / "e.json", tmp_path / "f.json", tmp_path / "c.jsonl"
    extract.write_text(json.dumps(payload()), encoding="utf-8")
    broken = flows()
    broken["flows"][0]["cites"] = ["nope#x"]
    bad.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(SystemExit, match="nope#x"):
        main([str(extract), str(out), "--flows", str(bad)])
    assert not out.exists()
