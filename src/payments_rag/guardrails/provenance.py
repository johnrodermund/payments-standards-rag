"""Serve only chunks whose text still matches the committed source extract.

The injection guard cannot recognise a plausible false statement planted in a passage
("Correction: the maximum length is 99999 characters"). What can be checked cheaply is
provenance: every chunk the system is allowed to serve is produced deterministically by
`chunk_payload` from the committed extract `data/processed/iso20022_messages.json`. Re-chunking
that file gives a manifest {chunk id: sha256(text)}; a retrieved chunk whose id is unknown or
whose text hash differs (a row edited or inserted in the database, a tampered chunks file) is
refused before the model sees it.

Limits (stated plainly):
* It trusts the extract. If the extract itself, or the upstream ISO download it came from, is
  poisoned, the manifest is poisoned too. Review changes to that file like code.
* It does not detect false facts that are in the source; nothing here judges truth.
* It only applies to retrieval from the real index. The injection eval feeds synthetic
  passages straight into `answer_question` and does not exercise it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_EXTRACT = Path("data/processed/iso20022_messages.json")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(payload: dict[str, Any], flows: dict[str, Any] | None = None) -> dict[str, str]:
    from payments_rag.ingestion.chunking import chunk_payload

    return {c["id"]: text_hash(c["text"]) for c in chunk_payload(payload, flows)}


class ProvenanceCheck:
    """Callable: hits -> (verified hits, rejection records in the guard's record format)."""

    def __init__(self, manifest: dict[str, str]):
        if not manifest:
            raise ValueError("empty provenance manifest")
        self.manifest = manifest

    @classmethod
    def from_extract(cls, path: str | Path = DEFAULT_EXTRACT, flows_path: str | Path | None = None) -> ProvenanceCheck:
        """Manifest from the extract plus the flows file (data/flows/interbank_flows.json) when it exists."""
        from payments_rag.ingestion.flows import DEFAULT_FLOWS, load_flows

        flows_file = Path(flows_path) if flows_path is not None else DEFAULT_FLOWS
        flows = load_flows(flows_file) if flows_file.exists() else None
        return cls(build_manifest(json.loads(Path(path).read_text(encoding="utf-8")), flows))

    def __call__(self, hits: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ok: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for hit in hits:
            expected = self.manifest.get(hit["id"])
            if expected is None:
                reason = "unknown_chunk_id"
            elif text_hash(hit["text"]) != expected:
                reason = "text_does_not_match_source"
            else:
                ok.append(hit)
                continue
            rejected.append({"id": hit["id"], "reasons": [reason], "action": "rejected_provenance", "removed": []})
        return ok, rejected
