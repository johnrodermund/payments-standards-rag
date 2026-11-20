import json

import pytest

from payments_rag.ingestion.chunking import MAX_CHARS, chunk_payload, main


def element(xml_path, name, definition, typ="", lo="1", hi="1", constraints=None):
    return {
        "xml_path": xml_path,
        "path": " > ".join(["MsgV1"] + [p for p in name.split("/")]),
        "name": name.split("/")[-1],
        "definition": definition,
        "min_occurs": lo,
        "max_occurs": hi,
        "type": typ,
        "constraints": constraints or [],
    }


def payload(messages=None, elements=None):
    msg = {
        "id": "pacs.008.001.14",
        "name": "MsgV1",
        "definition": "Scope\nSent by the debtor agent.\nUsage\nExchanged between agents.",
        "registration_status": "Registered",
        "constraints": [{"name": "MsgRule", "definition": "A message level rule."}],
        "elements": elements
        if elements is not None
        else [
            element("Root/Dbtr", "Debtor", "Party that owes money.", "Party1"),
            element("Root/Dbtr/Nm", "Debtor/Name", "Name of the party.", "Max140Text", "0", "1"),
            element("Root/Cdtr", "Creditor", "Party to which money is due.", "Party1"),
            element("Root/Cdtr/Nm", "Creditor/Name", "Name of the party.", "Max140Text", "0", "1"),
        ],
    }
    return {"source": {"file": "x.zip", "sha256": "0" * 64}, "messages": messages or [msg]}


def by_kind(chunks, kind):
    return [c for c in chunks if c["kind"] == kind]


def test_ids_are_unique_and_stable():
    first = chunk_payload(payload())
    second = chunk_payload(payload())
    ids = [c["id"] for c in first]
    assert len(ids) == len(set(ids))
    assert ids == [c["id"] for c in second]


def test_identical_definitions_stay_distinguishable_by_path():
    chunks = by_kind(chunk_payload(payload()), "element")
    debtor_name = next(c for c in chunks if c["metadata"]["xml_path"] == "Root/Dbtr/Nm")
    creditor_name = next(c for c in chunks if c["metadata"]["xml_path"] == "Root/Cdtr/Nm")
    assert debtor_name["text"] != creditor_name["text"]
    assert "Debtor" in debtor_name["text"] and "Party that owes money." in debtor_name["text"]
    assert "Creditor" in creditor_name["text"]


def test_element_chunk_has_context_fields():
    chunk = next(
        c for c in by_kind(chunk_payload(payload()), "element") if c["metadata"]["xml_path"] == "Root/Dbtr/Nm"
    )
    for expected in ("pacs.008.001.14", "Debtor > Name", "Max140Text", "0..1", "Name of the party."):
        assert expected in chunk["text"]
    assert chunk["metadata"]["message_id"] == "pacs.008.001.14"
    assert chunk["metadata"]["type"] == "Max140Text"


def test_parent_elements_get_a_children_chunk():
    chunks = by_kind(chunk_payload(payload()), "children")
    debtor = next(c for c in chunks if c["metadata"]["xml_path"] == "Root/Dbtr")
    assert "Name" in debtor["text"] and "Dbtr/Nm" in debtor["text"] or "Nm" in debtor["text"]
    assert not any(c["metadata"]["xml_path"] == "Root/Dbtr/Nm" for c in chunks)  # leaves have none


def test_message_definition_split_into_labelled_sections():
    chunks = by_kind(chunk_payload(payload()), "message")
    texts = " ".join(c["text"] for c in chunks)
    assert "Scope" in texts and "Usage" in texts
    assert "Sent by the debtor agent." in texts


def test_rules_become_their_own_chunks_and_are_deduplicated():
    rule = {"name": "SharedRule", "definition": "Must be consistent."}
    elems = [
        element("Root/A", "A", "First.", "T", constraints=[rule]),
        element("Root/B", "B", "Second.", "T", constraints=[rule]),
    ]
    rules = by_kind(chunk_payload(payload(elements=elems)), "rule")
    names = [c["metadata"]["name"] for c in rules]
    assert names.count("SharedRule") == 1
    shared = next(c for c in rules if c["metadata"]["name"] == "SharedRule")
    assert shared["metadata"]["xml_paths"] == ["Root/A", "Root/B"]
    assert "MsgRule" in names


def test_long_text_is_split_below_max_chars():
    many = [element(f"Root/P/C{i}", f"P/Child{i}", "x" * 80, "T") for i in range(80)]
    many.insert(0, element("Root/P", "P", "Parent.", "Comp"))
    chunks = chunk_payload(payload(elements=many))
    assert all(len(c["text"]) <= MAX_CHARS for c in chunks)
    parts = [c for c in by_kind(chunks, "children") if c["metadata"]["xml_path"] == "Root/P"]
    assert len(parts) > 1
    assert len({c["id"] for c in parts}) == len(parts)


def test_empty_definition_is_still_chunked_but_flagged():
    elems = [element("Root/Z", "Z", "", "T")]
    chunk = next(c for c in by_kind(chunk_payload(payload(elements=elems)), "element"))
    assert chunk["metadata"]["has_definition"] is False


def test_cli_writes_jsonl(tmp_path):
    src = tmp_path / "in.json"
    out = tmp_path / "chunks.jsonl"
    src.write_text(json.dumps(payload()), encoding="utf-8")
    main([str(src), str(out)])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows and all({"id", "kind", "text", "metadata"} <= set(r) for r in rows)


@pytest.mark.parametrize("bad", [{"messages": []}, {}])
def test_payload_without_messages_is_an_error(bad):
    with pytest.raises(ValueError):
        chunk_payload(bad)
