import json

from payments_rag.ingestion.chunking import MAX_CHARS
from payments_rag.ingestion.validation import main, validate_chunks


def chunk(cid="pacs.008.001.14#Root/A", kind="element", text=None, **meta):
    base = {
        "message_id": "pacs.008.001.14",
        "message_name": "MsgV1",
        "source": "ISO 20022 e-Repository",
        "xml_path": "Root/A",
        "name": "A",
        "has_definition": True,
    }
    base.update(meta)
    return {
        "id": cid,
        "kind": kind,
        "text": text or "ISO 20022 pacs.008.001.14 (MsgV1)\nElement: A\nDefinition: A well formed definition.",
        "metadata": base,
    }


def checks(report, severity=None):
    return {i["check"] for i in report["issues"] if severity in (None, i["severity"])}


def good_set():
    return [
        chunk(),
        chunk("pacs.008.001.14#Root/B", xml_path="Root/B", name="B", text="B " * 40),
        chunk("pacs.008.001.14#Root#children", kind="children", xml_path="Root", text="children " * 10),
    ]


def test_clean_set_passes():
    report = validate_chunks(good_set(), expected=["pacs.008"])
    assert report["passed"] is True
    assert report["errors"] == 0
    assert report["stats"]["chunks"] == 3


def test_duplicate_ids_and_empty_text_are_errors():
    items = good_set() + [chunk(), chunk("pacs.008.001.14#Root/C", xml_path="Root/C", text="   ")]
    report = validate_chunks(items)
    assert {"duplicate_id", "empty_text"} <= checks(report, "error")
    assert report["passed"] is False


def test_too_short_and_too_long_text():
    items = good_set() + [
        chunk("x#short", xml_path="Root/S", text="hi"),
        chunk("x#long", xml_path="Root/L", text="y" * (MAX_CHARS + 1)),
    ]
    assert {"text_too_short", "text_too_long"} <= checks(validate_chunks(items), "error")


def test_missing_metadata_is_an_error():
    bad = chunk("x#nometa", xml_path="Root/M")
    del bad["metadata"]["message_id"]
    assert "missing_metadata" in checks(validate_chunks(good_set() + [bad]), "error")
    no_path = chunk("x#nopath", xml_path="Root/N")
    del no_path["metadata"]["xml_path"]
    assert "missing_metadata" in checks(validate_chunks(good_set() + [no_path]), "error")


def test_invalid_kind_is_an_error():
    assert "invalid_kind" in checks(validate_chunks(good_set() + [chunk("x#k", kind="weird")]), "error")


def test_bad_encoding_is_detected():
    replacement = chunk("x#enc1", xml_path="Root/E1", text="Name of the party � " * 5)
    mojibake = chunk("x#enc2", xml_path="Root/E2", text="Name of the partyâ€™s account " * 3)
    control = chunk("x#enc3", xml_path="Root/E3", text="Name\x00 of the party " * 5)
    report = validate_chunks(good_set() + [replacement, mojibake, control])
    flagged = {i["chunk_id"] for i in report["issues"] if i["check"] == "bad_encoding"}
    assert flagged == {"x#enc1", "x#enc2", "x#enc3"}


def test_orphan_children_chunk():
    orphan = chunk("x#orphan", kind="children", xml_path="Root/Missing/Deeper", text="children " * 10)
    assert "orphan_children" in checks(validate_chunks(good_set() + [orphan]), "error")


def test_expected_message_missing_or_empty():
    report = validate_chunks(good_set(), expected=["pacs.008", "pacs.002"])
    assert "missing_expected_message" in checks(report, "error")


def test_duplicate_text_is_a_warning_only():
    items = good_set() + [chunk("x#dup", xml_path="Root/D", text=good_set()[1]["text"])]
    report = validate_chunks(items)
    assert "duplicate_text" in checks(report, "warning")
    assert report["passed"] is True


def test_missing_definitions_warn_then_fail_above_ratio():
    few = [chunk(f"x#{i}", xml_path=f"Root/{i}", has_definition=(i != 0), text=f"definition {i} " * 8) for i in range(20)]
    report = validate_chunks(few, max_no_definition_ratio=0.2)
    assert "no_definition" in checks(report, "warning")
    assert report["passed"] is True
    many = [chunk(f"y#{i}", xml_path=f"Root/{i}", has_definition=False, text=f"d {i} " * 20) for i in range(10)]
    report = validate_chunks(many, max_no_definition_ratio=0.2)
    assert "no_definition_ratio" in checks(report, "error")


def test_placeholder_definitions_warn():
    ph = chunk("x#ph", xml_path="Root/P", text="ISO 20022 pacs.008.001.14\nDefinition: TBD placeholder text TBD TBD")
    assert "placeholder_text" in checks(validate_chunks(good_set() + [ph]), "warning")


def test_cli_exit_codes_and_report(tmp_path):
    ok = tmp_path / "ok.jsonl"
    ok.write_text("\n".join(json.dumps(c) for c in good_set()), encoding="utf-8")
    out = tmp_path / "report.json"
    assert main([str(ok), "--expect", "pacs.008", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is True

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(chunk(text="   ")) + "\n", encoding="utf-8")
    assert main([str(bad)]) == 1
