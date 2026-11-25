"""Data-quality validation for chunks, run before anything is embedded.

Errors fail the run (exit code 1); warnings are reported but do not.

Errors:    duplicate_id, empty_text, text_too_short, text_too_long, missing_metadata,
           invalid_kind, bad_encoding, orphan_children, missing_expected_message,
           no_definition_ratio, suspected_injection
Warnings:  duplicate_text, no_definition, placeholder_text

Usage:
    python -m payments_rag.ingestion.validation chunks.jsonl --expect pacs.008 \
        --expect pacs.002 --out validation_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from payments_rag.guardrails.injection import scan_passage
from payments_rag.ingestion.chunking import MAX_CHARS

VALID_KINDS = {"message", "element", "children", "rule", "flow"}
MIN_CHARS = 40
COMMON_METADATA = ("message_id", "message_name", "source")
KIND_METADATA = {
    "element": ("xml_path", "name"),
    "children": ("xml_path",),
    "rule": ("name",),
    "message": (),
    "flow": ("messages", "cites"),
}
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MOJIBAKE = re.compile("Ã.|â€|Â ")
PLACEHOLDER = re.compile(r"\b(TBD|TODO|lorem ipsum|to be defined)\b", re.IGNORECASE)


def _issue(severity: str, check: str, chunk_id: str | None, detail: str) -> dict[str, Any]:
    return {"severity": severity, "check": check, "chunk_id": chunk_id, "detail": detail}


def validate_chunks(
    chunks: list[dict[str, Any]],
    expected: list[str] | None = None,
    max_no_definition_ratio: float = 0.05,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    ids = Counter(c.get("id") for c in chunks)
    element_paths = {
        (c["metadata"].get("message_id"), c["metadata"].get("xml_path"))
        for c in chunks
        if c.get("kind") == "element" and "metadata" in c
    }
    texts: dict[str, list[str]] = defaultdict(list)

    for dup, n in ids.items():
        if n > 1:
            issues.append(_issue("error", "duplicate_id", dup, f"id appears {n} times"))

    for c in chunks:
        cid = c.get("id")
        kind = c.get("kind")
        text = c.get("text") or ""
        meta = c.get("metadata") or {}

        if kind not in VALID_KINDS:
            issues.append(_issue("error", "invalid_kind", cid, f"kind {kind!r} is not one of {sorted(VALID_KINDS)}"))

        stripped = text.strip()
        if not stripped:
            issues.append(_issue("error", "empty_text", cid, "text is empty or whitespace"))
        else:
            if len(stripped) < MIN_CHARS:
                issues.append(_issue("error", "text_too_short", cid, f"{len(stripped)} chars (< {MIN_CHARS})"))
            if len(text) > MAX_CHARS:
                issues.append(_issue("error", "text_too_long", cid, f"{len(text)} chars (> {MAX_CHARS})"))
            if "�" in text or CONTROL_CHARS.search(text) or MOJIBAKE.search(text):
                issues.append(_issue("error", "bad_encoding", cid, "replacement/control characters or mojibake"))
            # Same screen as at answer time, run once at source: a poisoned chunk never gets
            # embedded. Real corpus: 0 of 1,245 chunks flagged (tests/test_injection_guard.py).
            if reasons := scan_passage(text):
                issues.append(_issue("error", "suspected_injection", cid, ", ".join(reasons)))
            if PLACEHOLDER.search(text):
                issues.append(_issue("warning", "placeholder_text", cid, "contains a placeholder such as TBD/TODO"))
            texts[stripped].append(cid)

        required = COMMON_METADATA + KIND_METADATA.get(kind or "", ())
        missing = [k for k in required if not meta.get(k)]
        if missing:
            issues.append(_issue("error", "missing_metadata", cid, f"missing: {', '.join(missing)}"))

        if kind == "children":
            path = meta.get("xml_path") or ""
            is_root = path != "" and "/" not in path
            if path and not is_root and (meta.get("message_id"), path) not in element_paths:
                issues.append(_issue("error", "orphan_children", cid, f"no element chunk for path {path}"))

    for ids_with_text in texts.values():
        if len(ids_with_text) > 1:
            issues.append(
                _issue("warning", "duplicate_text", ids_with_text[0], f"same text as {len(ids_with_text) - 1} other chunk(s)")
            )

    elements = [c for c in chunks if c.get("kind") == "element"]
    no_def = [c for c in elements if (c.get("metadata") or {}).get("has_definition") is False]
    for c in no_def:
        issues.append(_issue("warning", "no_definition", c.get("id"), "element has no published definition"))
    ratio = len(no_def) / len(elements) if elements else 0.0
    if ratio > max_no_definition_ratio:
        issues.append(
            _issue("error", "no_definition_ratio", None, f"{ratio:.1%} of elements lack a definition (limit {max_no_definition_ratio:.1%})")
        )

    for prefix in expected or []:
        matching = [c for c in chunks if str((c.get("metadata") or {}).get("message_id", "")).startswith(prefix + ".")]
        if not any(c.get("kind") == "element" for c in matching):
            issues.append(_issue("error", "missing_expected_message", None, f"no element chunks for {prefix}"))

    per_message = Counter((c.get("metadata") or {}).get("message_id", "?") for c in chunks)
    per_kind = Counter(c.get("kind", "?") for c in chunks)
    errors = sum(1 for i in issues if i["severity"] == "error")
    warnings = sum(1 for i in issues if i["severity"] == "warning")
    return {
        "passed": errors == 0,
        "errors": errors,
        "warnings": warnings,
        "by_check": dict(Counter(f"{i['severity']}:{i['check']}" for i in issues)),
        "stats": {
            "chunks": len(chunks),
            "per_message": dict(per_message),
            "per_kind": dict(per_kind),
            "elements_without_definition": len(no_def),
        },
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate chunk quality before embedding.")
    parser.add_argument("chunks_path")
    parser.add_argument("--expect", action="append", default=[], help="message prefix, e.g. pacs.008")
    parser.add_argument("--max-no-definition-ratio", type=float, default=0.05)
    parser.add_argument("--out", help="write the full JSON report here")
    args = parser.parse_args(argv)

    lines = Path(args.chunks_path).read_text(encoding="utf-8").splitlines()
    chunks = [json.loads(line) for line in lines if line.strip()]
    report = validate_chunks(chunks, args.expect, args.max_no_definition_ratio)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {report['stats']['chunks']} chunks, "
          f"{report['errors']} errors, {report['warnings']} warnings")
    for key, n in sorted(report["by_check"].items()):
        print(f"  {key}: {n}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
