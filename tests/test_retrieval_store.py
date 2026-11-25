from payments_rag.retrieval.store import _where, doc_allowed


def doc(kind, message_id, messages=None):
    meta = {"messages": messages} if messages is not None else {}
    return {"kind": kind, "message_id": message_id, "metadata": meta}


def test_doc_allowed_keeps_flows_that_use_the_named_message():
    assert doc_allowed(doc("flow", "flow", ["pain.001", "pacs.008"]), None, "pacs.008")
    assert doc_allowed(doc("element", "pacs.008.001.14"), None, "pacs.008")


def test_doc_allowed_drops_flows_that_do_not():
    assert not doc_allowed(doc("flow", "flow", ["pacs.009"]), None, "pacs.008")
    assert not doc_allowed(doc("element", "pacs.002.001.16"), None, "pacs.008")


def test_doc_allowed_without_prefix_and_with_kinds():
    assert doc_allowed(doc("flow", "flow", ["pacs.009"]), None, None)
    assert not doc_allowed(doc("flow", "flow", ["pacs.008"]), ["element"], "pacs.008")


def test_where_lets_flow_chunks_through_the_message_filter():
    sql, params = _where(None, "pacs.008")
    assert "kind = 'flow'" in sql and "metadata->'messages' ?" in sql
    assert params == ["pacs.008.%", "pacs.008"]


def test_core_message_prior_only_when_no_message_is_named():
    from payments_rag.retrieval.store import use_core_prior

    assert use_core_prior("What is the UETR in a credit transfer?", None)
    assert not use_core_prior("What is the UETR in pacs.009?", None)
    assert not use_core_prior("What is the difference between pacs.002 and pacs.004?", None)
    assert not use_core_prior("What is the UETR?", "pacs.009")
