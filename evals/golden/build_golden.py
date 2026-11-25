"""Generate evals/golden_set.jsonl from data/processed/iso20022_messages.json.

Every question has a reference answer and the chunk ids that contain the evidence, both derived
from the same ISO 20022 data the index is built from, so each answer can be checked at its source.
Questions are templated (several phrasings per type), not written independently by domain experts:
that is a stated limitation of this set.

A few questions are deliberately unanswerable from the corpus (off-topic, or about material that was
not ingested). They check that the assistant hands those to a human instead of answering.

Run:  uv run python evals/golden/build_golden.py     (deterministic)
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from payments_rag.ingestion.chunking import _words  # same name-splitting the chunk text uses

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "processed" / "iso20022_messages.json"
OUT = ROOT / "evals" / "golden_set.jsonl"
SEED = 7

MAXLEN = [
    "What is the maximum length of {n} in {m}?",
    "How many characters can {n} hold in {m}?",
    "What are the length limits for {n} in a {m} message?",
]
REQUIREMENT = [
    "Is {n} mandatory or optional in {m}?",
    "Does {m} require {n}?",
    "Can {n} be omitted from a {m} message?",
]
DEFINITION = [
    "What is {n} in {m}?",
    "What does {n} mean in {m}?",
    "Explain the {n} element of {m}.",
]
CODES = [
    "Which values can {n} take in {m}?",
    "What codes are allowed for {n} in {m}?",
]
CHILDREN = [
    "What does the {n} block of {m} contain?",
    "Which elements make up {n} in {m}?",
]

UNANSWERABLE = [
    ("What is the capital of France?", "off_topic"),
    ("How do I bake sourdough bread?", "off_topic"),
    ("Who won the football world cup in 2018?", "off_topic"),
    ("What is the maximum length of the BIC in a pacs.009 message?", "not_in_corpus"),
    ("What fees does FedNow charge per transaction?", "not_in_corpus"),
    ("How does a camt.053 statement report intraday balances?", "not_in_corpus"),
    ("What is the settlement cut-off time for SEPA Instant payments?", "not_in_corpus"),
    ("Which version of pacs.008 does CBPR+ require?", "not_in_corpus"),
]


def short_message(msg_id: str) -> str:
    return msg_id.rsplit(".", 2)[0]  # pacs.008.001.14 -> pacs.008


def first_sentences(text: str, limit: int = 320) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut[: cut.rfind(" ")] + "..."


def occurrence_phrase(el: dict) -> str:
    lo = el.get("min_occurs") or "0"
    hi = el.get("max_occurs") or "1"
    base = "optional" if lo == "0" else "mandatory"
    return base + (", and it may be repeated" if hi != "1" else "")


def build() -> list[dict]:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    cases: list[dict] = []

    def add(category, question, answer, ids, message):
        cases.append(
            {
                "id": f"gold-{len(cases) + 1:02d}",
                "category": category,
                "question": question,
                "reference_answer": answer,
                "reference_ids": ids,
                "answerable": True,
                "message": message,
            }
        )

    def pick(candidates, n):
        candidates = sorted(candidates, key=lambda e: e["xml_path"])
        rng.shuffle(candidates)
        return candidates[:n]

    for msg in payload["messages"]:
        mid, short = msg["id"], short_message(msg["id"])
        names = Counter(e["name"] for e in msg["elements"])
        unique = [e for e in msg["elements"] if names[e["name"]] == 1]
        child_counts = Counter(e["xml_path"].rsplit("/", 1)[0] for e in msg["elements"] if "/" in e["xml_path"])
        def eid(e, mid=mid):
            return f"{mid}#{e['xml_path']}"

        for k, e in enumerate(pick([e for e in unique if (e.get("type_info") or {}).get("kind") == "Text" and e["type_info"].get("max_length")], 4)):
            info = e["type_info"]
            span = f"{info['min_length']} to {info['max_length']}" if info.get("min_length") and info["min_length"] != info["max_length"] else info["max_length"]
            add("max_length", MAXLEN[k % 3].format(n=_words(e["name"]), m=short), f"{_words(e['name'])} is of type {e['type']} and allows {span} characters.", [eid(e)], mid)

        for k, e in enumerate(pick([e for e in unique if e.get("min_occurs") is not None], 4)):
            add("requirement", REQUIREMENT[k % 3].format(n=_words(e["name"]), m=short), f"{_words(e['name'])} is {occurrence_phrase(e)}.", [eid(e)], mid)

        for k, e in enumerate(pick([e for e in unique if len(e["definition"]) > 40], 5)):
            add("definition", DEFINITION[k % 3].format(n=_words(e["name"]), m=short), first_sentences(e["definition"]), [eid(e)], mid)

        for k, e in enumerate(pick([e for e in unique if (e.get("type_info") or {}).get("kind") == "CodeSet" and len(e["type_info"]["codes"]) >= 2], 3)):
            info = e["type_info"]
            more = " (and others)" if info["code_count"] > len(info["codes"]) else ""
            add("codes", CODES[k % 2].format(n=_words(e["name"]), m=short), f"{_words(e['name'])} takes values from the {e['type']} code set: " + ", ".join(_words(c) for c in info["codes"]) + more + ".", [eid(e)], mid)

        parents = [e for e in unique if 3 <= child_counts.get(e["xml_path"], 0) <= 8]
        for k, e in enumerate(pick(parents, 3)):
            kids = [c for c in msg["elements"] if c["xml_path"].rsplit("/", 1)[0] == e["xml_path"]]
            add("children", CHILDREN[k % 2].format(n=_words(e["name"]), m=short), f"{_words(e['name'])} contains: " + ", ".join(_words(c["name"]) for c in kids) + ".", [f"{eid(e)}#children"], mid)

        if msg["definition"]:
            add("message", f"What is the purpose of the {short} message?", first_sentences(msg["definition"].split("Usage")[0].replace("Scope", "", 1)), [f"{mid}#message:scope"], mid)

        rules = [c for c in msg["constraints"] if len(c["definition"]) > 30]
        for r in rng.sample(rules, min(2, len(rules))):
            add("rule", f"What does the rule {_words(r['name'])} require in {short}?", r["definition"], [f"{mid}#rule:{r['name']}"], mid)

    for question, kind in UNANSWERABLE:
        cases.append(
            {
                "id": f"gold-{len(cases) + 1:02d}",
                "category": f"unanswerable_{kind}",
                "question": question,
                "reference_answer": None,
                "reference_ids": [],
                "answerable": False,
                "message": None,
            }
        )
    return cases


def main() -> None:
    cases = build()
    OUT.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    counts = Counter(c["category"] for c in cases)
    print(f"wrote {len(cases)} questions to {OUT}")
    for category, n in sorted(counts.items()):
        print(f"  {category}: {n}")


if __name__ == "__main__":
    main()
