import pytest

from payments_rag.retrieval.ranking import (
    LexicalIndex,
    fuse,
    intent,
    kind_factor,
    name_match,
    tokenize,
)


def doc(xml_path, path, kind="element", text=None, message_id="pacs.008.001.14"):
    return {
        "id": f"{message_id}#{xml_path}" + ("#children" if kind == "children" else ""),
        "kind": kind,
        "text": text or path,
        "metadata": {"message_id": message_id, "xml_path": xml_path, "path": path},
    }


def ranked_ids(query, docs):
    lexical = LexicalIndex(d["text"] for d in docs)
    return [docs[i]["id"] for i, _ in fuse(query, docs, list(range(len(docs))), lexical)]


def test_tokenize_splits_camel_case_but_keeps_the_tag():
    assert tokenize("CdtrAgt holds Transactions") == ["cdtragt", "cdtr", "agt", "hold", "transaction"]


def test_intent_from_question_wording():
    assert intent("What fields does the debtor contain?") == "children"
    assert intent("What is the rule on the total amount?") == "rule"
    assert intent("What is the creditor agent?") == "definition"


def test_name_match_counts_trailing_path_components_found_in_the_question():
    header = doc("Root/GrpHdr/InstgAgt", "Group Header > Instructing Agent")
    tx = doc("Root/CdtTrfTxInf/InstgAgt", "Credit Transfer Transaction Information > Instructing Agent")
    q = "What is the instructing agent in the group header?"
    assert name_match(q, header).components == 2
    assert name_match(q, tx).components == 1
    assert name_match("What is the debtor?", tx) is None


def test_name_match_accepts_the_xml_tag():
    uetr = doc("Root/PmtId/UETR", "Payment Identification > UETR")
    assert name_match("what is the uetr", uetr) is not None


def test_element_named_in_question_beats_lookalikes_that_vectors_prefer():
    docs = [
        doc("Root/Tx/CdtrAgtAcct", "Transaction > Creditor Agent Account", text="Creditor Agent Account: account of the creditor agent."),
        doc("Root/Tx", "Transaction", kind="children", text="Transaction contains Creditor Agent, Creditor Agent Account."),
        doc("Root/Tx/CdtrAgt", "Transaction > Creditor Agent", text="Creditor Agent: institution servicing the creditor."),
    ]
    assert ranked_ids("What is the creditor agent?", docs)[0] == "pacs.008.001.14#Root/Tx/CdtrAgt"


def test_structure_question_prefers_the_children_list():
    docs = [
        doc("Root/Tx/Dbtr", "Transaction > Debtor", text="Debtor: party that owes money."),
        doc("Root/Tx/Dbtr", "Transaction > Debtor", kind="children", text="Debtor contains Name, Postal Address."),
    ]
    assert ranked_ids("What fields does the debtor contain?", docs)[0] == "pacs.008.001.14#Root/Tx/Dbtr#children"
    assert ranked_ids("What is the debtor?", docs)[0] == "pacs.008.001.14#Root/Tx/Dbtr"


def test_shallow_element_wins_a_tie_with_a_deeper_copy():
    deep = doc("Root/Tx/Tax/Dbtr", "Transaction > Tax > Debtor", text="Debtor: party that owes money.")
    shallow = doc("Root/Tx/Dbtr", "Transaction > Debtor", text="Debtor: party that owes money.")
    assert ranked_ids("What is the debtor?", [deep, shallow])[0] == shallow["id"]


def test_disallowed_docs_never_come_back():
    docs = [doc("Root/A", "Amount"), doc("Root/A", "Amount", message_id="pacs.002.001.16")]
    lexical = LexicalIndex(d["text"] for d in docs)
    out = fuse("what is the amount", docs, [0], lexical, allowed=[True, False])
    assert [i for i, _ in out] == [0]



@pytest.mark.parametrize(
    "question",
    [
        "What is sent after a pacs.008?",
        "Which message does the creditor agent use to send funds back?",
        "Walk me through the cover method",
        "Who sends a pain.002 and to whom?",
        "What is the difference between a reject and a return?",
        "What flows do you know?",
        "What comes next after the debtor agent receives a pain.001?",
    ],
)
def test_flow_questions_get_flow_intent(question):
    assert intent(question) == "flow"


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What is the return reason in pacs.004?", "definition"),
        ("What is the maximum length of the returned interbank settlement amount in pacs.004?", "definition"),
        ("What does the group header contain in pacs.009?", "children"),
        ("What is the rule on settlement method?", "rule"),
        ("What is the creditor agent in pacs.008?", "definition"),
    ],
)
def test_field_questions_keep_their_intent(question, expected):
    assert intent(question) == expected


def test_flow_intent_prefers_flow_chunks():
    assert kind_factor("flow", "flow") == 1.0
    assert kind_factor("element", "flow") < 1.0
    assert kind_factor("flow", "definition") < 1.0


def test_element_named_end_to_end_is_not_a_flow_question():
    assert intent("Which pacs.002 element holds the original end to end identification?") == "definition"


def test_message_prior_prefers_the_interbank_core_messages():
    from payments_rag.retrieval.ranking import message_prior

    core = {"kind": "element", "message_id": "pacs.008.001.14"}
    status = {"kind": "element", "message_id": "pacs.002.001.16"}
    flow = {"kind": "flow", "message_id": "flow"}
    other = {"kind": "element", "message_id": "pacs.009.001.13"}
    assert message_prior(core) == message_prior(status) == message_prior(flow) == 1.0
    assert 0.5 < message_prior(other) < 1.0


def test_fuse_applies_a_prior_to_break_ties_between_identical_elements():
    docs = [
        {"id": "pacs.009.001.13#R/UETR", "kind": "element", "text": "UETR: unique end to end reference.",
         "metadata": {"xml_path": "R/UETR", "path": "R > UETR", "name": "UETR"}},
        {"id": "pacs.008.001.14#R/UETR", "kind": "element", "text": "UETR: unique end to end reference.",
         "metadata": {"xml_path": "R/UETR", "path": "R > UETR", "name": "UETR"}},
    ]
    lexical = LexicalIndex(d["text"] for d in docs)
    plain = fuse("What is the UETR?", docs, [0, 1], lexical)
    weighted = fuse("What is the UETR?", docs, [0, 1], lexical, prior=[0.8, 1.0])
    assert plain[0][0] == 0
    assert weighted[0][0] == 1


def test_flow_questions_keep_message_scope_and_usage_chunks():
    assert kind_factor("message", "flow") == 1.0
    assert kind_factor("children", "flow") < 1.0
