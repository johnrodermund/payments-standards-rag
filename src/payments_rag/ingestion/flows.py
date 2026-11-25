"""Hand-written interbank flows: load, check against the extract, and chunk for retrieval.

The flows file (data/flows/interbank_flows.json) is written for this project, not published by
ISO. Each step names the message it uses and cites the chunk ids of the ISO text that backs it.
`check_flows` makes sure every citation and message exists in the committed extract; whether the
text is *right* is a human review (see the design doc). Flow chunks are labelled as project
summaries in both their text and their `source` metadata so they are never passed off as ISO text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_FLOWS = Path("data/flows/interbank_flows.json")
FLOW_SOURCE = "Project flow summary based on ISO 20022 e-Repository definitions"
FLOW_ID = re.compile(r"^[a-z0-9-]+$")


def load_flows(path: str | Path = DEFAULT_FLOWS) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _prefix(message_id: str) -> str:
    return ".".join(message_id.split(".")[:2])


def _messages(flow: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    for step in flow["steps"]:
        if step["message"] not in seen:
            seen.append(step["message"])
    return seen


def _cites(flow: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for cid in [c for s in flow["steps"] for c in s.get("cites", [])] + flow.get("cites", []):
        if cid not in out:
            out.append(cid)
    return out


def check_flows(flows: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    """Error strings; empty when every flow id, message and citation is valid against `payload`."""
    from payments_rag.ingestion.chunking import chunk_payload

    errors: list[str] = []
    if not str(flows.get("source_note", "")).strip():
        errors.append("source_note is missing or empty")
    known_messages = {_prefix(m["id"]) for m in payload["messages"]}
    known_ids = {c["id"] for c in chunk_payload(payload)}
    seen: set[str] = set()
    for flow in flows.get("flows", []):
        fid = flow.get("id", "")
        if not FLOW_ID.match(fid):
            errors.append(f"flow id {fid!r} must match {FLOW_ID.pattern}")
        if fid in seen:
            errors.append(f"duplicate flow id {fid!r}")
        seen.add(fid)
        for step in flow.get("steps", []):
            if step.get("message") not in known_messages:
                errors.append(f"flow {fid!r} step {step.get('n')}: message {step.get('message')!r} is not indexed")
        for cid in _cites(flow):
            if cid not in known_ids:
                errors.append(f"flow {fid!r}: cited chunk {cid!r} does not exist")
    return errors


def flow_chunks(flows: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Chunks for every flow whose messages are all in `payload`, plus a `flow:index` overview."""
    from payments_rag.ingestion.chunking import MAX_CHARS, _balanced, _words

    names = {_prefix(m["id"]): _words(m["name"]) for m in payload["messages"]}
    included = [f for f in flows.get("flows", []) if all(m in names for m in _messages(f))]
    if not included:
        return []

    def make(chunk_id: str, title: str, text: str, messages: list[str], cites: list[str]) -> dict[str, Any]:
        return {
            "id": chunk_id,
            "kind": "flow",
            "text": text,
            "metadata": {"message_id": "flow", "message_name": title, "source": FLOW_SOURCE,
                         "messages": messages, "cites": cites},
        }

    chunks: list[dict[str, Any]] = []
    for flow in included:
        title = flow["title"]
        head = [
            f"Interbank flow: {title} (project summary; not ISO text)",
            flow["summary"],
            f"Parties: {', '.join(flow['parties'])}",
            "Steps:",
        ]
        body = [
            f"{s['n']}. {s['from']} → {s['to']}: {s['message']} ({names[s['message']]}). {s['purpose']}"
            for s in flow["steps"]
        ] + [f"Note: {n}" for n in flow.get("notes", [])]
        cont = f"Interbank flow: {title}, continued (project summary; not ISO text)"
        first_budget = MAX_CHARS - sum(len(h) + 1 for h in head)
        if sum(len(b) + 1 for b in body) <= first_budget:
            parts = [body]
        else:
            parts = _balanced(body, min(first_budget, MAX_CHARS - len(cont) - 1))
        for n, part in enumerate(parts, start=1):
            lines = (head if n == 1 else [cont]) + part
            suffix = "" if n == 1 else f":{n}"
            chunks.append(make(f"flow:{flow['id']}{suffix}", title, "\n".join(lines), _messages(flow), _cites(flow)))

    index_lines = ["Interbank flows this assistant knows (project summaries; not ISO text):"] + [
        f"- {f['title']}: {' → '.join(_messages(f))}. {f['summary']}" for f in included
    ]
    all_messages = list(dict.fromkeys(m for f in included for m in _messages(f)))
    all_cites = list(dict.fromkeys(c for f in included for c in f.get("cites", []) or _cites(f)))
    chunks.append(make("flow:index", "Interbank flows overview", "\n".join(index_lines), all_messages, all_cites))
    return chunks
