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


def rich_chunk(name, definition, type_info, lo="1", hi="1", typ="T"):
    el = element(f"Root/{name}", name, definition, typ, lo, hi)
    el["type_info"] = type_info
    chunks = chunk_payload(payload(elements=[el]))
    return next(c for c in chunks if c["kind"] == "element")["text"]


def test_camel_case_names_are_split_into_words_and_lead_the_text():
    text = rich_chunk("EndToEndIdentification", "Unique id from the initiating party.", None)
    assert text.splitlines()[0].startswith("End To End Identification: Unique id")
    assert text.index("End To End Identification") < text.index("ISO 20022 pacs.008.001.14")


def test_text_type_states_its_length_limit():
    info = {"kind": "Text", "min_length": "1", "max_length": "35", "definition": "d", "codes": [], "code_count": 0}
    assert "text, 1 to 35 characters" in rich_chunk("MessageIdentification", "Message id.", info, typ="Max35Text")


def test_code_set_type_lists_codes_and_counts_the_rest():
    info = {"kind": "CodeSet", "definition": "Who pays charges.", "codes": ["BorneByDebtor", "Shared"], "code_count": 5}
    text = rich_chunk("ChargeBearer", "Which party pays.", info, typ="ChargeBearerType1Code")
    assert "Borne By Debtor, Shared and 3 more" in text


def test_mandatory_optional_and_repeatable_are_spelled_out():
    assert "mandatory" in rich_chunk("A", "d", None, lo="1", hi="1")
    assert "optional" in rich_chunk("B", "d", None, lo="0", hi="1")
    assert "repeatable" in rich_chunk("C", "d", None, lo="1", hi="unbounded")


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


def shared_component_payload():
    """Debtor and Creditor share Party1, which contains a PostalAddress (Addr1) with a TownName."""
    elems = [
        element("Root/Dbtr", "Debtor", "Party that owes money.", "Party1"),
        element("Root/Dbtr/Nm", "Debtor/Name", "Name of the party.", "Max140Text", "0"),
        element("Root/Dbtr/PstlAdr", "Debtor/PostalAddress", "Postal address.", "Addr1", "0"),
        element("Root/Dbtr/PstlAdr/TwnNm", "Debtor/PostalAddress/TownName", "Town.", "Max35Text", "0"),
        element("Root/Cdtr", "Creditor", "Party to which money is due.", "Party1"),
        element("Root/Cdtr/Nm", "Creditor/Name", "Name of the party.", "Max140Text", "0"),
        element("Root/Cdtr/PstlAdr", "Creditor/PostalAddress", "Postal address.", "Addr1", "0"),
        element("Root/Cdtr/PstlAdr/TwnNm", "Creditor/PostalAddress/TownName", "Town.", "Max35Text", "0"),
    ]
    return chunk_payload(payload(elements=elems))


def test_repeated_component_subtree_is_indexed_once():
    paths = {c["metadata"]["xml_path"] for c in by_kind(shared_component_payload(), "element")}
    assert "Root/Dbtr/PstlAdr/TwnNm" in paths  # first use of the component: indexed in full
    assert "Root/Cdtr/PstlAdr/TwnNm" not in paths  # identical copy deeper inside a repeat: not
    # Direct children of every use stay, so "is the creditor name mandatory?" has its own chunk.
    assert {"Root/Cdtr", "Root/Cdtr/Nm", "Root/Cdtr/PstlAdr"} <= paths


def test_first_use_of_a_shared_component_lists_the_other_uses():
    chunks = shared_component_payload()
    debtor = next(c for c in by_kind(chunks, "element") if c["metadata"]["xml_path"] == "Root/Dbtr")
    assert debtor["metadata"]["also_at"] == ["Root/Cdtr"]
    assert "Creditor" in debtor["text"].split("Also used at:", 1)[1]
    children = {c["metadata"]["xml_path"] for c in by_kind(chunks, "children")}
    assert {"Root/Dbtr", "Root/Cdtr", "Root/Dbtr/PstlAdr", "Root/Cdtr/PstlAdr"} <= children


def test_deep_copy_inside_a_shared_component_says_where_else_it_applies():
    chunks = shared_component_payload()
    town = next(c for c in by_kind(chunks, "element") if c["metadata"]["xml_path"] == "Root/Dbtr/PstlAdr/TwnNm")
    assert "Creditor > Postal Address" in town["text"]


def test_rule_text_spells_out_camel_case_names():
    rule = {"name": "TotalAmountRule", "definition": "If GroupHeader/TotalAmount is present, it must match."}
    elems = [element("Root/A", "A", "First.", "T", constraints=[rule])]
    text = next(c for c in by_kind(chunk_payload(payload(elements=elems)), "rule") if "TotalAmountRule" in c["id"])["text"]
    assert "Total Amount Rule" in text and "Group Header / Total Amount" in text


def test_long_message_section_is_split_into_balanced_parts():
    usage = "\n".join(f"Paragraph {i} " + "word " * 60 for i in range(6))
    msg = payload()["messages"][0] | {"definition": "Scope\nSent by the debtor agent.\nUsage\n" + usage}
    parts = [c for c in chunk_payload(payload(messages=[msg])) if c["id"].startswith("pacs.008.001.14#message:usage")]
    sizes = [len(c["text"]) for c in parts]
    assert len(parts) == 2 and max(sizes) - min(sizes) < 400


def test_message_section_label_explains_what_it_holds():
    scope = next(c for c in by_kind(chunk_payload(payload()), "message") if c["id"].endswith("message:scope"))
    assert "purpose" in scope["text"].lower()


@pytest.mark.parametrize("bad", [{"messages": []}, {}])
def test_payload_without_messages_is_an_error(bad):
    with pytest.raises(ValueError):
        chunk_payload(bad)
