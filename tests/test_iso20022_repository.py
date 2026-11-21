import json
import zipfile

import pytest

from payments_rag.ingestion.iso20022_repository import extract_messages, main

XMI = "http://www.omg.org/XMI"
XSI = "http://www.w3.org/2001/XMLSchema-instance"

FIXTURE = f"""<?xml version="1.0" encoding="utf-8"?>
<iso20022:Repository xmlns:iso20022="urn:iso:std:iso:20022:2013:ecore" xmlns:xmi="{XMI}" xmlns:xsi="{XSI}">
  <dataDictionary>
    <topLevelDictionaryEntry xsi:type="iso20022:Text" xmi:id="T1" name="Max35Text" definition="Up to 35 chars."
        minLength="1" maxLength="35"/>
    <topLevelDictionaryEntry xsi:type="iso20022:MessageComponent" xmi:id="C_HDR" name="GroupHeader1"
        definition="Group header.">
      <constraint xmi:id="K1" name="HdrRule" definition="Must not be empty." expression="&lt;x/&gt;"/>
      <messageElement xsi:type="iso20022:MessageAttribute" xmi:id="E1" name="MessageIdentification"
          definition="Unique id." minOccurs="1" maxOccurs="1" xmlTag="MsgId" simpleType="T1"/>
      <messageElement xsi:type="iso20022:MessageAssociationEnd" xmi:id="E2" name="Loop"
          definition="Points back." minOccurs="0" maxOccurs="1" xmlTag="Loop" complexType="C_HDR"/>
    </topLevelDictionaryEntry>
  </dataDictionary>
  <businessProcessCatalogue><topLevelCatalogueEntry xsi:type="iso20022:BusinessArea" xmi:id="BA" name="Payments">
    <messageDefinition xmi:id="M_OLD" name="FIToFIOldV13"
        definition="Old version." registrationStatus="Registered" xmlTag="FIToFIOld">
      <messageDefinitionIdentifier businessArea="pacs" messageFunctionality="008" flavour="001" version="13"/>
    </messageDefinition>
    <messageDefinition xmi:id="M_NEW" name="FIToFINewV14"
        definition="Scope&#xA;New version." registrationStatus="Registered" xmlTag="FIToFICstmrCdtTrf">
      <messageDefinitionIdentifier businessArea="pacs" messageFunctionality="008" flavour="001" version="14"/>
      <constraint xmi:id="K2" name="MsgRule" definition="Message level rule." expression="&lt;x/&gt;"/>
      <messageBuildingBlock xmi:id="B1" name="GroupHeader" definition="Block."
          minOccurs="1" maxOccurs="1" xmlTag="GrpHdr" complexType="C_HDR"/>
    </messageDefinition>
    <messageDefinition xmi:id="M_OTHER" name="OtherV01"
        definition="Not requested." xmlTag="Other">
      <messageDefinitionIdentifier businessArea="camt" messageFunctionality="053" flavour="001" version="01"/>
    </messageDefinition>
  </topLevelCatalogueEntry></businessProcessCatalogue>
</iso20022:Repository>
"""


@pytest.fixture
def repo_zip(tmp_path):
    path = tmp_path / "repo.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("repo.iso20022", FIXTURE)
    return path


def test_keeps_only_latest_version_of_requested_messages(repo_zip):
    messages = extract_messages(repo_zip, ["pacs.008"])
    assert [m["id"] for m in messages] == ["pacs.008.001.14"]


def test_message_level_fields(repo_zip):
    msg = extract_messages(repo_zip, ["pacs.008"])[0]
    assert msg["name"] == "FIToFINewV14"
    assert msg["definition"].startswith("Scope")
    assert msg["registration_status"] == "Registered"
    assert msg["constraints"] == [{"name": "MsgRule", "definition": "Message level rule."}]


def test_elements_have_paths_types_and_cardinality(repo_zip):
    msg = extract_messages(repo_zip, ["pacs.008"])[0]
    by_path = {e["xml_path"]: e for e in msg["elements"]}
    msg_id = by_path["FIToFICstmrCdtTrf/GrpHdr/MsgId"]
    assert msg_id["name"] == "MessageIdentification"
    assert msg_id["definition"] == "Unique id."
    assert msg_id["type"] == "Max35Text"
    assert (msg_id["min_occurs"], msg_id["max_occurs"]) == ("1", "1")
    assert "GrpHdr" in {e["xml_path"].split("/")[-1] for e in msg["elements"]}


def test_type_info_carries_kind_and_length_limits(repo_zip):
    msg = extract_messages(repo_zip, ["pacs.008"])[0]
    msg_id = next(e for e in msg["elements"] if e["xml_path"].endswith("/MsgId"))
    assert msg_id["type_info"]["kind"] == "Text"
    assert (msg_id["type_info"]["min_length"], msg_id["type_info"]["max_length"]) == ("1", "35")
    header = next(e for e in msg["elements"] if e["xml_path"] == "FIToFICstmrCdtTrf/GrpHdr")
    assert header["type_info"] is None  # components are described by their children


def test_component_rules_are_attached_and_cycles_do_not_loop(repo_zip):
    msg = extract_messages(repo_zip, ["pacs.008"])[0]
    header = next(e for e in msg["elements"] if e["xml_path"] == "FIToFICstmrCdtTrf/GrpHdr")
    assert header["constraints"] == [{"name": "HdrRule", "definition": "Must not be empty."}]
    loop = [e for e in msg["elements"] if e["xml_path"].endswith("/GrpHdr/Loop")]
    assert len(loop) == 1  # recursion into the same component stops


def test_missing_max_occurs_means_unbounded(repo_zip):
    messages = extract_messages(repo_zip, ["pacs.008"])
    loop = next(e for e in messages[0]["elements"] if e["xml_path"].endswith("/GrpHdr/Loop"))
    assert loop["max_occurs"] == "1"
    # Strip maxOccurs from the fixture's attribute element and re-check.
    import zipfile as zf

    with zf.ZipFile(repo_zip, "w") as z:
        z.writestr("repo.iso20022", FIXTURE.replace(' maxOccurs="1" xmlTag="MsgId"', ' xmlTag="MsgId"'))
    messages = extract_messages(repo_zip, ["pacs.008"])
    msg_id = next(e for e in messages[0]["elements"] if e["xml_path"].endswith("/MsgId"))
    assert msg_id["max_occurs"] == "unbounded"


def test_unknown_message_is_an_error(repo_zip):
    with pytest.raises(ValueError, match="pacs.999"):
        extract_messages(repo_zip, ["pacs.999"])


def test_cli_writes_pinned_source_and_messages(repo_zip, tmp_path):
    out = tmp_path / "out.json"
    main([str(repo_zip), str(out), "--message", "pacs.008"])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["source"]["file"] == "repo.zip"
    assert len(data["source"]["sha256"]) == 64
    assert data["messages"][0]["id"] == "pacs.008.001.14"
