"""Extract message and element definitions from the ISO 20022 e-Repository.

The e-Repository is a ~118 MB XML (EMF/XMI) file inside a zip. This module
stream-parses it once and keeps only the latest version of the requested
messages (for example ``pacs.008``), flattened to one record per element with
its full path, definition, cardinality and type.

Usage:
    python -m payments_rag.ingestion.iso20022_repository ZIP OUT.json \
        --message pacs.008 --message pacs.002
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

XMI_ID = "{http://www.omg.org/XMI}id"
XSI_TYPE = "{http://www.w3.org/2001/XMLSchema-instance}type"

# Attributes of a messageElement/messageBuildingBlock that point at a type or component.
REF_ATTRS = ("complexType", "simpleType", "type")


def _local(value: str | None) -> str:
    return (value or "").rsplit(":", 1)[-1]


def _constraints(elem: ET.Element) -> list[dict[str, str]]:
    return [
        {"name": c.get("name", ""), "definition": c.get("definition", "")}
        for c in elem.findall("constraint")
    ]


def _compact_element(elem: ET.Element) -> dict[str, Any]:
    ref = next((elem.get(a) for a in REF_ATTRS if elem.get(a)), None)
    return {
        "name": elem.get("name", ""),
        "definition": elem.get("definition", ""),
        "xml_tag": elem.get("xmlTag") or elem.get("name", ""),
        "min_occurs": elem.get("minOccurs"),
        # An absent maxOccurs means the element may repeat without limit (e.g. CdtTrfTxInf).
        "max_occurs": elem.get("maxOccurs") or "unbounded",
        "ref": ref,
    }


MAX_TYPE_DEFINITION = 200
MAX_CODES = 12


def _type_info(elem: ET.Element) -> dict[str, Any]:
    """Compact description of a data-dictionary type (text, code set, amount, ...)."""
    codes: list[str] = []
    for code in elem.findall("code"):
        name = code.get("name", "")
        if name and name not in codes:
            codes.append(name)
    definition = " ".join(elem.get("definition", "").split())
    return {
        "name": elem.get("name", ""),
        "kind": _local(elem.get(XSI_TYPE)),
        "definition": definition[:MAX_TYPE_DEFINITION],
        "min_length": elem.get("minLength"),
        "max_length": elem.get("maxLength"),
        "codes": codes[:MAX_CODES],
        "code_count": len(codes),
    }


def _message_key(identifier: ET.Element) -> tuple[str, int, int]:
    area = identifier.get("businessArea", "")
    func = identifier.get("messageFunctionality", "")
    return f"{area}.{func}", int(identifier.get("flavour", "0")), int(identifier.get("version", "0"))


def _scan(zip_path: Path, wanted: set[str]):
    components: dict[str, dict[str, Any]] = {}
    types: dict[str, str] = {}
    best: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}

    with zipfile.ZipFile(zip_path) as z:
        member = z.namelist()[0]
        with z.open(member) as fh:
            for _, elem in ET.iterparse(fh, events=("end",)):
                if elem.tag == "topLevelDictionaryEntry":
                    entry_id = elem.get(XMI_ID, "")
                    if _local(elem.get(XSI_TYPE)) == "MessageComponent":
                        components[entry_id] = {
                            "name": elem.get("name", ""),
                            "constraints": _constraints(elem),
                            "elements": [_compact_element(e) for e in elem.findall("messageElement")],
                        }
                    else:
                        types[entry_id] = _type_info(elem)
                    elem.clear()
                elif elem.tag == "messageDefinition":
                    ident = elem.find("messageDefinitionIdentifier")
                    if ident is not None:
                        key, flavour, version = _message_key(ident)
                        if key in wanted:
                            record = {
                                "id": f"{key}.{ident.get('flavour', '')}.{ident.get('version', '')}",
                                "name": elem.get("name", ""),
                                "definition": elem.get("definition", ""),
                                "registration_status": elem.get("registrationStatus", ""),
                                "xml_tag": elem.get("xmlTag") or elem.get("name", ""),
                                "constraints": _constraints(elem),
                                "blocks": [
                                    _compact_element(b) for b in elem.findall("messageBuildingBlock")
                                ],
                            }
                            rank = (flavour, version)
                            if key not in best or rank > best[key][0]:
                                best[key] = (rank, record)
                    elem.clear()
    return components, types, {k: v[1] for k, v in best.items()}


def _flatten(
    items: list[dict[str, Any]],
    components: dict[str, dict[str, Any]],
    types: dict[str, dict[str, Any]],
    names: tuple[str, ...],
    tags: tuple[str, ...],
    stack: frozenset[str],
    out: list[dict[str, Any]],
) -> None:
    for item in items:
        item_names = names + (item["name"],)
        item_tags = tags + (item["xml_tag"],)
        ref = item["ref"]
        component = components.get(ref) if ref else None
        info = None if component else types.get(ref or "")
        out.append(
            {
                "xml_path": "/".join(item_tags),
                "path": " > ".join(item_names),
                "name": item["name"],
                "definition": item["definition"],
                "min_occurs": item["min_occurs"],
                "max_occurs": item["max_occurs"],
                "type": component["name"] if component else (info["name"] if info else ""),
                "type_info": info,
                "constraints": component["constraints"] if component else [],
            }
        )
        if component and ref not in stack:
            _flatten(component["elements"], components, types, item_names, item_tags, stack | {ref}, out)


def extract_messages(zip_path: str | Path, message_keys: list[str]) -> list[dict[str, Any]]:
    """Return the latest version of each requested message, flattened to elements."""
    components, types, found = _scan(Path(zip_path), set(message_keys))
    missing = [k for k in message_keys if k not in found]
    if missing:
        raise ValueError(f"message(s) not found in repository: {', '.join(missing)}")

    messages = []
    for key in message_keys:
        record = found[key]
        elements: list[dict[str, Any]] = []
        _flatten(record["blocks"], components, types, (record["name"],), (record["xml_tag"],), frozenset(), elements)
        messages.append(
            {
                "id": record["id"],
                "name": record["name"],
                "definition": record["definition"],
                "registration_status": record["registration_status"],
                "constraints": record["constraints"],
                "elements": elements,
            }
        )
    return messages


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("zip_path")
    parser.add_argument("out_path")
    parser.add_argument("--message", action="append", required=True, help="e.g. pacs.008")
    args = parser.parse_args(argv)

    zip_path = Path(args.zip_path)
    messages = extract_messages(zip_path, args.message)
    payload = {
        "source": {
            "file": zip_path.name,
            "size_bytes": zip_path.stat().st_size,
            "sha256": _sha256(zip_path),
            "terms": "https://www.iso20022.org/terms-use",
        },
        "messages": messages,
    }
    Path(args.out_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
