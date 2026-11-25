"""Chunk extracted ISO 20022 message definitions for retrieval.

Strategy (why it looks like this):

* The source is a tree, not prose. Element definitions are short (median ~65
  characters) and heavily repeated: for pacs.008 only ~270 distinct definitions
  cover ~1,660 elements, and "Name" under Debtor reads the same as "Name" under
  Creditor. A chunk holding only a definition is therefore ambiguous, so every
  chunk carries its message, full path, type, cardinality and parent context.
* One chunk per element gives precise citations ("pacs.008.001.14 Debtor >
  Name") and keeps chunks small, so the embedding is about one thing.
* Shared components are indexed once. The same component type (a party, an
  agent, a postal address) is expanded under many parents, so pacs.008 has
  ~1,660 element paths but only ~290 distinct (component, element) pairs. Every
  element whose parent sits on a "first use" path gets its own chunk, so each
  use keeps its direct children (Creditor > Name stays separate from Debtor >
  Name); deeper copies inside a repeated component are only indexed at the first
  use, whose chunks list the other places the component appears. That cuts the
  index to about a third and stops near-identical copies crowding out the
  element a question is about.
* A "children" chunk per parent element lists its direct children, which
  answers structure questions ("what does Debtor contain?") that no single
  leaf chunk can.
* The message Scope/Usage prose is split by section into balanced parts, each
  labelled with what the section holds.
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
import re
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


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _words(name: str) -> str:
    """'EndToEndIdentification' -> 'End To End Identification' (helps both embeddings and keyword search)."""
    return _CAMEL.sub(" ", name)


def _display(element: dict[str, Any]) -> str:
    path = element["path"]
    path = path.split(" > ", 1)[1] if " > " in path else path
    return " > ".join(_words(part) for part in path.split(" > "))


def _requirement(element: dict[str, Any]) -> str:
    lo = element.get("min_occurs") or "0"
    hi = element.get("max_occurs") or "1"
    text = "optional" if lo == "0" else "mandatory"
    if hi != "1":
        text += ", repeatable" if hi == "unbounded" else f", repeatable up to {hi} times"
    return text


def _type_description(element: dict[str, Any]) -> str:
    info = element.get("type_info")
    name = element.get("type") or "n/a"
    if not info:
        return name
    kind = info.get("kind", "")
    lo, hi = info.get("min_length"), info.get("max_length")
    if kind == "Text" and hi:
        span = f"{lo} to {hi}" if lo and lo != hi else hi
        return f"{name} (text, {span} characters)"
    if kind == "CodeSet":
        codes = ", ".join(_words(c) for c in info.get("codes", []))
        more = info.get("code_count", 0) - len(info.get("codes", []))
        extra = f" and {more} more" if more > 0 else ""
        return f"{name} (code set: {_short(info.get('definition', ''), 120)} Codes: {codes}{extra})".replace("  ", " ")
    return f"{name} ({_words(kind).lower()}: {_short(info.get('definition', ''), 120)})"


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


SECTION_LABELS = {
    "Scope": "Scope (purpose of the message: what it is for and who sends it to whom)",
    "Usage": "Usage (how and when the message is used)",
}


def _spell_out(text: str) -> str:
    """Spell out CamelCase identifiers in prose ('GroupHeader/TotalAmount' -> 'Group Header / Total Amount')."""
    text = re.sub(r"(?<=\w)/(?=\w)", " / ", text)
    return re.sub(r"\b[A-Za-z]*[a-z][A-Z]\w*", lambda m: _words(m.group(0)), text)


def _balanced(lines: list[str], budget: int) -> list[list[str]]:
    """Like _pack, but aims for parts of similar size so no part is a stray fragment."""
    total = sum(len(line) + 1 for line in lines)
    parts = -(-total // budget)  # ceiling division
    return _pack(lines, min(budget, -(-total // parts) + max(len(line) for line in lines) // 2))


def _message_chunks(msg: dict[str, Any], header: str) -> list[dict[str, Any]]:
    sections: list[tuple[str, list[str]]] = [("Overview", [])]
    for raw in msg["definition"].splitlines():
        line = raw.strip()
        if not line:
            continue
        if len(line) <= 20 and "." not in line and line[0].isupper():
            sections.append((line, []))
        else:
            sections[-1][1].append(_spell_out(line))

    chunks = []
    title = f"{_words(msg['name'])} message ({msg['id']})"
    for label, lines in sections:
        if not lines:
            continue
        intro = f"{header}\n{SECTION_LABELS.get(label, label)} of the {title}:"
        for n, part in enumerate(_balanced(lines, MAX_CHARS - len(intro) - 1), start=1):
            suffix = "" if n == 1 else f":{n}"
            chunks.append(
                _chunk(
                    f"{msg['id']}#message:{label.lower()}{suffix}",
                    "message",
                    intro + "\n" + "\n".join(part),
                    msg,
                    section=label,
                )
            )
    return chunks


IndexPlan = tuple[set[str], dict[str, list[str]], dict[str, str]]


def _index_plan(msg: dict[str, Any]) -> IndexPlan:
    """Decide which element paths get chunks (see module docstring).

    Returns (indexed paths, every use of each component type in document order, and for
    each element the closest enclosing use of a component that is used more than once).
    """
    elements = msg["elements"]
    by_path = {e["xml_path"]: e for e in elements}
    parents = {e["xml_path"].rsplit("/", 1)[0] for e in elements if "/" in e["xml_path"]}
    uses: dict[str, list[str]] = defaultdict(list)
    for e in elements:
        if e["xml_path"] in parents and e.get("type"):
            uses[e["type"]].append(e["xml_path"])

    def is_first_use(path: str) -> bool:
        el = by_path.get(path)
        return el is None or path not in parents or uses[el["type"]][0] == path

    indexed: set[str] = set()
    owner: dict[str, str] = {}
    for e in elements:
        parts = e["xml_path"].split("/")
        ancestors = ["/".join(parts[:i]) for i in range(2, len(parts))]
        if all(is_first_use(a) for a in ancestors[:-1]):
            indexed.add(e["xml_path"])
        shared = [a for a in ancestors if a in by_path and len(uses.get(by_path[a]["type"], [])) > 1]
        if shared:
            owner[e["xml_path"]] = shared[-1]
    return indexed, dict(uses), owner


def _list_paths(paths: list[str], by_path: dict[str, dict[str, Any]], limit: int = 4) -> str:
    """Short list of other places ('Debtor > Postal Address; ...'), so it does not swamp the chunk."""
    labels = list(dict.fromkeys(" > ".join(_display(by_path[p]).split(" > ")[-3:]) for p in paths))
    shown = "; ".join(labels[:limit])
    return shown + (f" (+{len(labels) - limit} more)" if len(labels) > limit else "")


def _element_chunks(
    msg: dict[str, Any], header: str, by_path: dict[str, dict[str, Any]], plan: IndexPlan
) -> list[dict[str, Any]]:
    indexed, uses, owner = plan
    chunks = []
    for el in msg["elements"]:
        xml_path = el["xml_path"]
        if xml_path not in indexed:
            continue
        parent = by_path.get(xml_path.rsplit("/", 1)[0]) if "/" in xml_path else None
        # The element's own words come first so they dominate the embedding; boilerplate goes last.
        lines = [
            f"{_words(el['name'])}: {el['definition'] or '(no definition published)'}",
            f"Path: {_display(el)} (XML path: {xml_path})",
            f"Type: {_type_description(el)}; occurrence {_occurs(el)} ({_requirement(el)})",
        ]
        if parent:
            lines.append(
                f"Parent: {_display(parent)}. {_short(parent['definition'], PARENT_DEFINITION_CHARS)}"
            )
        same_type = uses.get(el["type"], [])
        others = same_type[1:] if same_type[:1] == [xml_path] else []
        if others:
            lines.append(f"Also used at: {_list_paths(others, by_path)}")
        elif len(same_type) > 1:
            lines.append(f"Contents described in full at: {_display(by_path[same_type[0]])}")
        elif xml_path in owner:
            component = owner[xml_path]
            rel = xml_path[len(component) :]
            # Copies of this element that have no chunk of their own, named by where they sit.
            unindexed = [
                p
                for p in uses[by_path[component]["type"]]
                if p != component and p + rel in by_path and p + rel not in indexed
            ]
            if unindexed:
                lines.append(f"The same element also applies under: {_list_paths(unindexed, by_path)}")
        lines.append(header)
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
                **({"also_at": others} if others else {}),
            )
        )
    return chunks


def _children_chunks(msg: dict[str, Any], header: str, root_tag: str, indexed: set[str]):
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for el in msg["elements"]:
        parent_path = el["xml_path"].rsplit("/", 1)[0] if "/" in el["xml_path"] else root_tag
        children[parent_path].append(el)

    by_path = {e["xml_path"]: e for e in msg["elements"]}
    chunks = []
    for parent_path, kids in children.items():
        parent = by_path.get(parent_path)
        if parent and parent_path not in indexed:
            continue
        title = _display(parent) if parent else msg["name"]
        intro = f"{header}\nElement: {title} (XML path: {parent_path}) contains these direct children:"
        lines = [
            f"- {_words(k['name'])} ({k['xml_path'].rsplit('/', 1)[-1]}, {_requirement(k)})" for k in kids
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
        lines = [header, f"Business rule {name} ({_words(name)}): {_spell_out(definition)}"]
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


def chunk_payload(payload: dict[str, Any], flows: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    messages = payload.get("messages") or []
    if not messages:
        raise ValueError("payload has no messages to chunk")

    chunks: list[dict[str, Any]] = []
    for msg in messages:
        header = f"ISO 20022 {msg['id']} ({msg['name']})"
        by_path = {e["xml_path"]: e for e in msg["elements"]}
        root_tag = msg["elements"][0]["xml_path"].split("/", 1)[0] if msg["elements"] else msg["name"]
        chunks += _message_chunks(msg, header)
        plan = _index_plan(msg)
        chunks += _element_chunks(msg, header, by_path, plan)
        chunks += _children_chunks(msg, header, root_tag, plan[0])
        chunks += _rule_chunks(msg, header)
    if flows is not None:
        from payments_rag.ingestion.flows import flow_chunks

        chunks += flow_chunks(flows, payload)
    return chunks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chunk extracted ISO 20022 definitions into JSONL.")
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    parser.add_argument("--flows", help="flows file to check and chunk, e.g. data/flows/interbank_flows.json")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.in_path).read_text(encoding="utf-8"))
    flows = None
    if args.flows:
        from payments_rag.ingestion.flows import check_flows, load_flows

        flows = load_flows(args.flows)
        errors = check_flows(flows, payload)
        if errors:
            raise SystemExit("flows file failed checks:\n" + "\n".join(errors))
    chunks = chunk_payload(payload, flows)
    with Path(args.out_path).open("w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)

    counts: dict[str, int] = defaultdict(int)
    for c in chunks:
        counts[c["kind"]] += 1
    print(f"{len(chunks)} chunks: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
