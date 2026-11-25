"""Read-only list of the interbank flows the assistant knows (for a UI and for people).

Served from the committed flows file and extract, loaded once on first use: no database, no LLM.
The flows are project summaries of ISO text, and every response carries the file's source note.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from payments_rag.guardrails.provenance import DEFAULT_EXTRACT
from payments_rag.ingestion.flows import DEFAULT_FLOWS, load_flows

router = APIRouter()
CACHE = {"Cache-Control": "public, max-age=300"}


@lru_cache(maxsize=1)
def _catalogue() -> dict[str, Any]:
    from payments_rag.ingestion.chunking import _words

    flows = load_flows(Path(os.environ.get("FLOWS_PATH", DEFAULT_FLOWS)))
    extract = json.loads(Path(os.environ.get("PROVENANCE_EXTRACT", DEFAULT_EXTRACT)).read_text(encoding="utf-8"))
    messages = {".".join(m["id"].split(".")[:2]): m for m in extract["messages"]}
    out = []
    for flow in flows["flows"]:
        steps = []
        for s in flow["steps"]:
            msg = messages.get(s["message"])
            steps.append({
                "n": s["n"], "from": s["from"], "to": s["to"], "message": s["message"],
                "message_id": msg["id"] if msg else None,
                "message_name": _words(msg["name"]) if msg else None,
                "purpose": s["purpose"],
            })
        out.append({"id": flow["id"], "title": flow["title"], "summary": flow["summary"],
                    "parties": flow["parties"], "steps": steps, "notes": flow.get("notes", [])})
    return {"source_note": flows["source_note"], "flows": out}


@router.get("/flows")
def list_flows() -> JSONResponse:
    return JSONResponse(_catalogue(), headers=CACHE)


@router.get("/flows/{flow_id}")
def get_flow(flow_id: str) -> JSONResponse:
    catalogue = _catalogue()
    flow = next((f for f in catalogue["flows"] if f["id"] == flow_id), None)
    if flow is None:
        raise HTTPException(status_code=404, detail=f"unknown flow {flow_id!r}")
    return JSONResponse({"source_note": catalogue["source_note"], **flow}, headers=CACHE)
