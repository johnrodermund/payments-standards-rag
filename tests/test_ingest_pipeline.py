import os

import psycopg
import pytest

from payments_rag.ingestion.embedding import HashEmbedder
from payments_rag.ingestion.pipeline import IngestError, ingest
from payments_rag.retrieval import store

BASE_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://payments:payments@localhost:5432/postgres")
TEST_DB = "payments_test"


def make_chunk(cid, text, message_id="pacs.008.001.14", kind="element"):
    return {
        "id": cid,
        "kind": kind,
        "text": text,
        "metadata": {
            "message_id": message_id,
            "message_name": "MsgV1",
            "source": "ISO 20022 e-Repository",
            "xml_path": cid.split("#", 1)[1],
            "name": "N",
            "has_definition": True,
        },
    }


def chunks_v1():
    return [
        make_chunk("pacs.008.001.14#Root/A", "ISO 20022 pacs.008\nElement: A\nDefinition: the debtor party of the payment."),
        make_chunk("pacs.008.001.14#Root/B", "ISO 20022 pacs.008\nElement: B\nDefinition: the creditor agent institution."),
        make_chunk("pacs.008.001.14#Root/C", "ISO 20022 pacs.008\nElement: C\nDefinition: the settlement amount and currency."),
    ]


class CountingEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.embedded = 0

    def embed(self, texts):
        self.embedded += len(texts)
        return super().embed(texts)


@pytest.fixture
def conn():
    try:
        admin = psycopg.connect(BASE_URL, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("Postgres with pgvector not reachable (docker compose up -d db)")
    admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
    admin.execute(f"CREATE DATABASE {TEST_DB}")
    admin.close()
    test_url = BASE_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"
    connection = store.connect(test_url)
    yield connection
    connection.close()


def test_invalid_chunks_are_refused_before_touching_the_database():
    bad = chunks_v1() + [make_chunk("pacs.008.001.14#Root/D", "   ")]
    with pytest.raises(IngestError, match="validation failed"):
        ingest(bad, HashEmbedder(), conn=None)


def test_ingest_stores_everything(conn):
    stats = ingest(chunks_v1(), CountingEmbedder(), conn)
    assert stats["embedded"] == 3 and stats["skipped"] == 0
    assert store.count(conn) == 3


def test_rerun_embeds_nothing(conn):
    embedder = CountingEmbedder()
    ingest(chunks_v1(), embedder, conn)
    before = embedder.embedded
    stats = ingest(chunks_v1(), embedder, conn)
    assert embedder.embedded == before
    assert stats["embedded"] == 0 and stats["skipped"] == 3


def test_changed_chunk_is_reembedded_and_removed_chunk_deleted(conn):
    embedder = CountingEmbedder()
    ingest(chunks_v1(), embedder, conn)
    changed = chunks_v1()[:2]  # C removed
    changed[0] = make_chunk("pacs.008.001.14#Root/A", "ISO 20022 pacs.008\nElement: A\nDefinition: a changed definition text.")
    stats = ingest(changed, embedder, conn)
    assert stats["embedded"] == 1 and stats["deleted"] == 1
    assert store.count(conn) == 2


def test_search_returns_most_similar_first_with_scores(conn):
    embedder = HashEmbedder()
    items = chunks_v1()
    ingest(items, embedder, conn)
    hits = store.search(conn, embedder.embed_query(items[1]["text"]), k=3)
    assert hits[0]["id"] == items[1]["id"]
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]
    assert hits[0]["metadata"]["message_id"] == "pacs.008.001.14"


def test_vector_search_is_exact_not_approximate(conn):
    ingest(chunks_v1(), HashEmbedder(), conn)
    indexes = [r[0] for r in conn.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'chunks'")]
    assert not any("hnsw" in i for i in indexes)


def test_hybrid_search_reloads_its_corpus_after_new_chunks_arrive(conn):
    embedder = HashEmbedder()
    ingest(chunks_v1(), embedder, conn)
    store.hybrid_search(conn, "settlement", embedder.embed_query("settlement"), k=3)
    extra = make_chunk("pacs.008.001.14#Root/U", "ISO 20022 pacs.008 UETR: Universally unique reference.")
    ingest(chunks_v1() + [extra], embedder, conn)
    hits = store.hybrid_search(conn, "what is the uetr", embedder.embed_query("what is the uetr"), k=3)
    assert hits[0]["id"] == extra["id"]


def test_message_prefix_only_when_exactly_one_message_is_named():
    assert store.message_prefix_from_query("creditor agent in pacs.008?") == "pacs.008"
    assert store.message_prefix_from_query("compare pacs.008 with pacs.002") is None
    assert store.message_prefix_from_query("creditor agent") is None


def test_hybrid_search_finds_exact_word_that_vectors_cannot_rank(conn):
    embedder = HashEmbedder()  # random-ish vectors: only the keyword side can find this
    items = chunks_v1() + [
        make_chunk("pacs.008.001.14#Root/U", "ISO 20022 pacs.008\nUETR: Universally unique end-to-end transaction reference.")
    ]
    ingest(items, embedder, conn)
    hits = store.hybrid_search(conn, "what is the uetr", embedder.embed_query("what is the uetr"), k=3)
    assert hits[0]["id"] == "pacs.008.001.14#Root/U"
    assert all(h["score"] is not None for h in hits)
    assert hits[0]["rrf"] >= hits[-1]["rrf"]


def test_hybrid_search_limits_to_the_message_named_in_the_question(conn):
    embedder = HashEmbedder()
    other = make_chunk("pacs.002.001.16#Root/A", "ISO 20022 pacs.002\nElement: A\nDefinition: the debtor party status.", "pacs.002.001.16")
    ingest(chunks_v1() + [other], embedder, conn)
    hits = store.hybrid_search(conn, "debtor party in pacs.002", embedder.embed_query("debtor party in pacs.002"), k=10)
    assert {h["metadata"]["message_id"] for h in hits} == {"pacs.002.001.16"}


def test_mixing_embedding_models_is_refused(conn):
    ingest(chunks_v1(), HashEmbedder(name="model-a"), conn)
    with pytest.raises(RuntimeError, match="re-ingest"):
        ingest(chunks_v1(), HashEmbedder(name="model-b"), conn)
