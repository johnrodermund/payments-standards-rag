import pytest

from payments_rag.ingestion.embedding import HashEmbedder
from payments_rag.retrieval.memory_store import MemoryIndex, build_index


def chunk(cid, text, kind="element", message_id="pacs.008.001.14", **meta):
    return {"id": cid, "kind": kind, "text": text, "metadata": {"message_id": message_id, **meta}}


CHUNKS = [
    chunk("pacs.008.001.14#Root/CdtrAgt", "Creditor Agent: financial institution servicing an account for the creditor.",
          xml_path="Root/CdtrAgt", name="CreditorAgent", path="Credit Transfer > Creditor Agent"),
    chunk("pacs.008.001.14#Root/DbtrAgt", "Debtor Agent: financial institution servicing an account for the debtor.",
          xml_path="Root/DbtrAgt", name="DebtorAgent", path="Credit Transfer > Debtor Agent"),
    chunk("pacs.002.001.16#Root/TxSts", "Transaction Status: specifies the status of a transaction.",
          message_id="pacs.002.001.16", xml_path="Root/TxSts", name="TransactionStatus", path="Transaction Status"),
    chunk("flow:return", "Interbank flow: Payment return. pacs.004 returns funds after settlement.",
          kind="flow", message_id="flow", messages=["pacs.004", "pacs.002"], cites=["x"]),
]


@pytest.fixture
def index(tmp_path):
    embedder = HashEmbedder(dim=32)
    path = tmp_path / "index.npz"
    build_index(CHUNKS, embedder, path)
    return MemoryIndex.load(path, expect_model=embedder.name, expect_dim=embedder.dim), embedder


def search(index_and_embedder, query, **kw):
    index, embedder = index_and_embedder
    return index.hybrid_search(query, embedder.embed_query(query), **kw)


def test_round_trip_keeps_ids_text_kind_and_metadata(index):
    idx, _ = index
    assert [d["id"] for d in idx.docs] == [c["id"] for c in CHUNKS]
    assert idx.docs[0]["metadata"]["xml_path"] == "Root/CdtrAgt"
    assert idx.docs[3]["message_id"] == "flow"


def test_name_match_ranks_the_named_element_first_with_a_cosine_score(index):
    hits = search(index, "What is the creditor agent in pacs.008?")
    assert hits[0]["id"] == "pacs.008.001.14#Root/CdtrAgt"
    assert -1.0 <= hits[0]["score"] <= 1.0
    assert {"id", "kind", "text", "metadata", "score", "rrf"} <= set(hits[0])


def test_naming_one_message_filters_to_it(index):
    hits = search(index, "What is the transaction status in pacs.002?", k=10)
    assert all(h["metadata"]["message_id"].startswith("pacs.002") or h["kind"] == "flow" for h in hits)


def test_message_filter_keeps_flows_that_use_the_message(index):
    hits = search(index, "What is sent after a pacs.002 in the return flow?", k=10)
    assert "flow:return" in [h["id"] for h in hits]


def test_kinds_filter(index):
    hits = search(index, "creditor agent", kinds=["flow"], k=10)
    assert [h["kind"] for h in hits] == ["flow"]


def test_model_mismatch_is_refused(tmp_path):
    path = tmp_path / "i.npz"
    build_index(CHUNKS, HashEmbedder(dim=32), path)
    with pytest.raises(ValueError, match="embedding"):
        MemoryIndex.load(path, expect_model="BAAI/bge-small-en-v1.5", expect_dim=384)


def test_build_retriever_uses_memory_backend_from_env(tmp_path, monkeypatch):
    from payments_rag.api import service

    path = tmp_path / "i.npz"
    embedder = HashEmbedder(dim=32)
    build_index(CHUNKS, embedder, path)
    monkeypatch.setenv("RETRIEVAL_BACKEND", "memory")
    monkeypatch.setenv("INDEX_PATH", str(path))
    retrieve = service.build_retriever(k=2, embedder=embedder)
    assert len(retrieve("creditor agent")) == 2
