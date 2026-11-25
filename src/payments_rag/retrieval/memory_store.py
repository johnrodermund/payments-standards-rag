"""The search index as one file, searched in memory: for serverless hosting with no database.

About 4,700 chunks of 384 floats is a few megabytes, and the keyword half of hybrid search already holds
every chunk in memory (see store._Corpus). So the whole index ships inside the container image, is loaded
once per cold start, and is searched with exactly the same fusion as the Postgres path
(`ranking.fuse`, `store.doc_allowed`, the message filter and the core-message prior). Cosine similarity is
computed exactly over all allowed chunks, as the Postgres path does (no approximate index).

Build it from the committed extract and flows file:

    python -m payments_rag.retrieval.memory_store data/processed/iso20022_messages.json \
        --flows data/flows/interbank_flows.json --out data/index/index.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from payments_rag.retrieval import ranking
from payments_rag.retrieval.store import doc_allowed, message_prefix_from_query, use_core_prior

DEFAULT_INDEX = Path("data/index/index.npz")


def build_index(chunks: list[dict[str, Any]], embedder: Any, path: str | Path) -> int:
    """Embed `chunks` and write ids, kinds, texts, metadata and unit vectors to one .npz file."""
    vectors = np.asarray(embedder.embed([c["text"] for c in chunks]), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)
    docs = [
        {"id": c["id"], "kind": c["kind"], "text": c["text"], "metadata": c["metadata"],
         "message_id": c["metadata"].get("message_id", "")}
        for c in chunks
    ]
    meta = {"model": embedder.name, "dim": embedder.dim, "docs": docs}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as fh:
        np.savez_compressed(fh, vectors=vectors, meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8))
    return len(docs)


class MemoryIndex:
    def __init__(self, docs: list[dict[str, Any]], vectors: np.ndarray, model: str, dim: int):
        self.docs = docs
        self.vectors = vectors
        self.model = model
        self.dim = dim
        self.lexical = ranking.LexicalIndex(d["text"] for d in docs)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_INDEX, expect_model: str | None = None, expect_dim: int | None = None) -> MemoryIndex:
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(data["meta"].tobytes().decode("utf-8"))
            vectors = data["vectors"]
        if (expect_model and meta["model"] != expect_model) or (expect_dim and meta["dim"] != expect_dim):
            raise ValueError(
                f"index was built with embedding model {meta['model']} ({meta['dim']} dims); "
                f"the query embedder is {expect_model} ({expect_dim} dims). Rebuild the index."
            )
        return cls(meta["docs"], vectors, meta["model"], meta["dim"])

    def hybrid_search(
        self,
        query: str,
        vector: list[float],
        k: int = 5,
        kinds: list[str] | None = None,
        message_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        """Same contract as store.hybrid_search: `score` is cosine similarity, `rrf` the fused score."""
        prefix = message_prefix or message_prefix_from_query(query)
        allowed = [doc_allowed(d, kinds, prefix) for d in self.docs]
        q = np.asarray(vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        cosine = self.vectors @ q
        candidates = np.flatnonzero(np.asarray(allowed))
        vector_order = candidates[np.argsort(-cosine[candidates], kind="stable")][: ranking.POOL].tolist()
        prior = [ranking.message_prior(d) for d in self.docs] if use_core_prior(query, message_prefix) else None
        fused = ranking.fuse(query, self.docs, vector_order, self.lexical, allowed, prior)[:k]
        return [
            {"id": self.docs[i]["id"], "kind": self.docs[i]["kind"], "text": self.docs[i]["text"],
             "metadata": self.docs[i]["metadata"], "score": float(cosine[i]), "rrf": rrf}
            for i, rrf in fused
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the in-memory search index file.")
    parser.add_argument("extract")
    parser.add_argument("--flows")
    parser.add_argument("--out", default=str(DEFAULT_INDEX))
    args = parser.parse_args(argv)

    from payments_rag.ingestion.chunking import chunk_payload
    from payments_rag.ingestion.embedding import FastEmbedEmbedder
    from payments_rag.ingestion.flows import check_flows, load_flows
    from payments_rag.ingestion.validation import validate_chunks

    payload = json.loads(Path(args.extract).read_text(encoding="utf-8"))
    flows = None
    if args.flows:
        flows = load_flows(args.flows)
        if errors := check_flows(flows, payload):
            raise SystemExit("flows file failed checks:\n" + "\n".join(errors))
    chunks = chunk_payload(payload, flows)
    report = validate_chunks(chunks)
    if not report["passed"]:
        raise SystemExit(f"validation failed: {report['errors']} errors")
    n = build_index(chunks, FastEmbedEmbedder(), args.out)
    print(f"{n} chunks -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
