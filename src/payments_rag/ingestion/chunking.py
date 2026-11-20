"""Chunk extracted ISO 20022 message definitions for retrieval.

Strategy (why it looks like this):

* The source is a tree, not prose. Element definitions are short (median ~65
  characters) and heavily repeated: for pacs.008 only ~270 distinct definitions
  cover ~1,660 elements, and "Name" under Debtor reads the same as "Name" under
  Creditor. A chunk holding only a definition is therefore ambiguous, so every
  chunk carries its message, full path, type, cardinality and parent context.
* One chunk per element gives precise citations ("pacs.008.001.14 Debtor >
  Name") and keeps chunks small, so the embedding is about one thing.
* A "children" chunk per parent element lists its direct children, which
  answers structure questions ("what does Debtor contain?") that no single
  leaf chunk can.
* The message Scope/Usage prose is split by section and paragraph.
* Business rules (constraints) are their own chunks, deduplicated across the
  many places a shared component appears, with the paths they apply to kept
  in metadata.
* Every chunk is capped at MAX_CHARS (about 375 tokens), well inside any
  embedding model's input limit, and ids are stable, readable citations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

MAX_CHARS = 1500
PARENT_DEFINITION_CHARS = 300
SOURCE_LABEL = "ISO 20022 e-Repository"


def _occurs(element: dict[str, Any]) -> str:
    hi = element.get("max_occurs") or "1"
    return f"{element.get('min_occurs') or '0'}..{'*' if hi == 'unbounded' else hi}"


def _pack(lines: list[str], budget: int) -> list[list[str]]:
    """Group lines into parts whose joined length stays within budget."""
    parts: list[list[str]] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if len(line) > budget:
            line = line[: budget - 3] + "..."
        add = len(line) + 1
        if current and size + add > budget:
            parts.append(current)
            current, size = [], 0
        current.append(line)
        size += add
    if current:
        parts.append(current)
    return parts


def _display(element: dict[str, Any]) -> str:
    path = element["path"]
    return path.split(" > ", 1)[1] if " > " in path else path


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _chunk(chunk_id: str, kind: str, text: str, msg: dict[str, Any], **metadata: Any) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "kind": kind,
        "text": text,
        "metadata": {
            "message_id": msg["id"],
            "message_name": msg["name"],
            "source": SOURCE_LABEL,
            **metadata,
        },
    }


def _message_chunks(msg: dict[str, Any], header: str) -> list[dict[str, Any]]:
    sections: list[tuple[str, list[str]]] = [("Overview", [])]
    for raw in msg["definition"].splitlines():
        line = raw.strip()
        if not line:
            continue
        if len(line) <= 20 and "." not in line and line[0].isupper():
            sections.append((line, []))
        else:
            sections[-1][1].append(line)

    chunks = []
    for label, lines in sections:
        if not lines:
            continue
        overhead = len(header) + len(label) + 10
        for n, part in enumerate(_pack(lines, MAX_CHARS - overhead), start=1):
            suffix = "" if n == 1 else f":{n}"
            text = f"{header}\nMessage definition, {label}:\n" + "\n".join(part)
            chunks.append(
                _chunk(f"{msg['id']}#message:{label.lower()}{suffix}", "message", text, msg, section=label)
            )
    return chunks


def _element_chunks(msg: dict[str, Any], header: str, by_path: dict[str, dict[str, Any]]):
    chunks = []
    for el in msg["elements"]:
        xml_path = el["xml_path"]
        parent = by_path.get(xml_path.rsplit("/", 1)[0]) if "/" in xml_path else None
        lines = [
            header,
            f"Element: {_display(el)} (XML path: {xml_path})",
            f"Type: {el['type'] or 'n/a'}; occurrence: {_occurs(el)}",
        ]
        if parent:
            lines.append(
                f"Parent: {_display(parent)}. {_short(parent['definition'], PARENT_DEFINITION_CHARS)}"
            )
        lines.append(f"Definition: {el['definition'] or '(no definition published)'}")
        if el["constraints"]:
            names = ", ".join(c["name"] for c in el["constraints"])
            lines.append(f"Rules that apply: {names}")
        text = "\n".join(lines)
        if len(text) > MAX_CHARS:
            text = text[: MAX_CHARS - 3] + "..."
        chunks.append(
            _chunk(
                f"{msg['id']}#{xml_path}",
                "element",
                text,
                msg,
                xml_path=xml_path,
                path=_display(el),
                name=el["name"],
                type=el["type"],
                occurs=_occurs(el),
                has_definition=bool(el["definition"].strip()),
            )
        )
    return chunks


def _children_chunks(msg: dict[str, Any], header: str, root_tag: str):
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for el in msg["elements"]:
        parent_path = el["xml_path"].rsplit("/", 1)[0] if "/" in el["xml_path"] else root_tag
        children[parent_path].append(el)

    by_path = {e["xml_path"]: e for e in msg["elements"]}
    chunks = []
    for parent_path, kids in children.items():
        parent = by_path.get(parent_path)
        title = _display(parent) if parent else msg["name"]
        intro = f"{header}\nElement: {title} (XML path: {parent_path}) contains these direct children:"
        lines = [
            f"- {k['name']} ({k['xml_path'].rsplit('/', 1)[-1]}, {_occurs(k)}, {k['type'] or 'n/a'})"
            for k in kids
        ]
        for n, part in enumerate(_pack(lines, MAX_CHARS - len(intro) - 1), start=1):
            suffix = "" if n == 1 else f":{n}"
            chunks.append(
                _chunk(
                    f"{msg['id']}#{parent_path}#children{suffix}",
                    "children",
                    intro + "\n" + "\n".join(part),
                    msg,
                    xml_path=parent_path,
                    path=title,
                    child_count=len(kids),
                )
            )
    return chunks


def _rule_chunks(msg: dict[str, Any], header: str):
    rules: dict[tuple[str, str], list[str]] = {}
    for c in msg["constraints"]:
        rules.setdefault((c["name"], c["definition"]), [])
    for el in msg["elements"]:
        for c in el["constraints"]:
            paths = rules.setdefault((c["name"], c["definition"]), [])
            if el["xml_path"] not in paths:
                paths.append(el["xml_path"])

    seen_names: set[str] = set()
    chunks = []
    for (name, definition), paths in rules.items():
        suffix = ""
        if name in seen_names:
            suffix = ":" + hashlib.sha1(definition.encode("utf-8")).hexdigest()[:8]
        seen_names.add(name)
        lines = [header, f"Business rule {name}: {definition}"]
        if paths:
            shown = ", ".join(paths[:5]) + (f" (+{len(paths) - 5} more)" if len(paths) > 5 else "")
            lines.append(f"Applies at: {shown}")
        else:
            lines.append("Applies to the whole message.")
        text = "\n".join(lines)
        if len(text) > MAX_CHARS:
            text = text[: MAX_CHARS - 3] + "..."
        chunks.append(
            _chunk(f"{msg['id']}#rule:{name}{suffix}", "rule", text, msg, name=name, xml_paths=paths)
        )
    return chunks


def chunk_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = payload.get("messages") or []
    if not messages:
        raise ValueError("payload has no messages to chunk")

    chunks: list[dict[str, Any]] = []
    for msg in messages:
        header = f"ISO 20022 {msg['id']} ({msg['name']})"
        by_path = {e["xml_path"]: e for e in msg["elements"]}
        root_tag = msg["elements"][0]["xml_path"].split("/", 1)[0] if msg["elements"] else msg["name"]
        chunks += _message_chunks(msg, header)
        chunks += _element_chunks(msg, header, by_path)
        chunks += _children_chunks(msg, header, root_tag)
        chunks += _rule_chunks(msg, header)
    return chunks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chunk extracted ISO 20022 definitions into JSONL.")
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.in_path).read_text(encoding="utf-8"))
    chunks = chunk_payload(payload)
    with Path(args.out_path).open("w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)

    counts: dict[str, int] = defaultdict(int)
    for c in chunks:
        counts[c["kind"]] += 1
    print(f"{len(chunks)} chunks: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
