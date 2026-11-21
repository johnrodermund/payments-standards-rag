"""Ranking signals for hybrid retrieval, kept free of database code so they are easy to test.

Why these signals (measured on evals/retrieval_set.jsonl; ablations, dev MRR 0.92 with all on:
no name match 0.57, no intent demotion 0.79, no depth prior 0.83). The constants below were
set once and are on a plateau (OFF_INTENT_FACTOR 0.15-0.5 and DEPTH_PENALTY 0.05-0.1 score
the same); they were not tuned against evals/retrieval_holdout.jsonl.

* Vector similarity alone ranks badly here: element definitions are short and repeat across
  many paths, so a question like "what is the creditor agent?" is about as close to
  "Creditor Agent Account" or a list of children as to the element itself.
* BM25 over the chunk text catches exact terms (UETR, charge bearer) that embeddings blur.
* A name match is the strongest single signal: users ask about elements by name, and ISO
  20022 element names are descriptive ("Interbank Settlement Amount"). A chunk ranks high
  when the trailing components of its path appear in the question as whole phrases
  ("group header ... instructing agent" matches Group Header > Instructing Agent better than
  Credit Transfer Transaction Information > Instructing Agent).
* Question intent decides the chunk kind: "what does X contain" wants the children list,
  "what is the rule on X" wants a business rule, everything else wants the element or
  message text. Other kinds are demoted, not removed.
* Shallow elements win ties: the same definition appears at many depths, and the shallow
  one is the one people mean (and the one the deeper copies refer back to).

Tried and rejected: re-ranking the fused top 20 with a cross-encoder (fastembed
ms-marco-MiniLM-L-6-v2, bge-reranker-base) dropped dev MRR from 0.92 to about 0.67; general
web-trained rerankers do not weigh ISO element names the way this corpus needs.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

RRF_K = 60
POOL = 50
NAME_WEIGHT = 1.0
OFF_INTENT_FACTOR = 0.3
DEPTH_PENALTY = 0.05

STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who", "whom", "how",
    "when", "where", "why", "does", "do", "did", "of", "in", "on", "for", "to", "by", "with",
    "and", "or", "be", "can", "could", "its", "it", "this", "that", "these", "those", "as",
    "at", "from", "used", "use", "there", "any", "one", "i", "we", "you", "me", "my", "our",
    "your", "element", "elements", "field", "fields",
})
MESSAGE_MENTION = re.compile(r"\b([a-z]{4}\.\d{3})(?:\.\d{3}\.\d{2})?\b", re.IGNORECASE)
STRUCTURE_INTENT = re.compile(
    r"\b(contains?|consists?|fields|children|structure|made up|sub-?elements|composed|"
    r"elements (?:are )?in|elements of|parts)\b",
    re.IGNORECASE,
)
RULE_INTENT = re.compile(r"\b(rules?|guidelines?|constraints?|must|allowed|permitted)\b", re.IGNORECASE)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_WORD = re.compile(r"[A-Za-z0-9]+")


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """Lower-case word stems; CamelCase tokens also yield their parts ('CdtrAgt' -> cdtragt, cdtr, agt)."""
    out: list[str] = []
    for raw in _WORD.findall(text):
        parts = _CAMEL.sub(" ", raw).split()
        for word in [raw, *parts] if len(parts) > 1 else [raw]:
            word = word.lower()
            if keep_stopwords or word not in STOPWORDS:
                out.append(_stem(word))
    return out


def content_words(query: str) -> list[str]:
    """Question words that carry meaning: no stopwords, no message ids like 'pacs.008'."""
    return tokenize(MESSAGE_MENTION.sub(" ", query))


def intent(query: str) -> str:
    """'children' for structure questions, 'rule' for rule questions, else 'definition'."""
    if STRUCTURE_INTENT.search(query):
        return "children"
    if RULE_INTENT.search(query):
        return "rule"
    return "definition"


def kind_factor(kind: str, wanted: str) -> float:
    if wanted == "definition":
        return OFF_INTENT_FACTOR if kind == "children" else 1.0
    return 1.0 if kind == wanted else OFF_INTENT_FACTOR


def depth(doc: dict[str, Any]) -> int:
    path = (doc.get("metadata") or {}).get("xml_path") or ""
    return path.count("/")


class LexicalIndex:
    """Okapi BM25 over chunk texts. Small corpus (a few thousand chunks), so in memory."""

    def __init__(self, texts: Iterable[str], k1: float = 1.2, b: float = 0.75):
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.lengths.append(sum(counts.values()))
            for word, tf in counts.items():
                self.postings[word].append((i, tf))
        n = len(self.lengths)
        self.avg = (sum(self.lengths) / n) if n else 0.0
        self.idf = {w: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.postings.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for word in set(tokenize(query)):
            idf = self.idf.get(word)
            if idf is None:
                continue
            for i, tf in self.postings[word]:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


@dataclass(frozen=True)
class NameMatch:
    components: int  # trailing path components found in the question
    words: int  # words in those components
    coverage: float  # share of the question's content words they account for

    def key(self, doc_depth: int) -> tuple[int, int, int]:
        return (self.components, self.words, -doc_depth)


def _contains(seq: Sequence[str], sub: Sequence[str]) -> bool:
    n = len(sub)
    return n > 0 and any(list(seq[i : i + n]) == list(sub) for i in range(len(seq) - n + 1))


def name_match(query: str, doc: dict[str, Any]) -> NameMatch | None:
    """How well the element's path names (or XML tags) appear in the question."""
    meta = doc.get("metadata") or {}
    if doc.get("kind") not in ("element", "children") or not meta.get("path") or not meta.get("xml_path"):
        return None
    words = tokenize(query)
    raw = {w.lower() for w in _WORD.findall(query)}
    comps = [tokenize(c) for c in meta["path"].split(" > ")]
    tags = meta["xml_path"].split("/")[-len(comps) :]
    matched = matched_words = 0
    for comp, tag in zip(reversed(comps), reversed(tags), strict=False):
        if comp and _contains(words, comp):
            matched_words += len(comp)
        elif tag.lower() in raw:
            matched_words += max(1, len(comp))
        else:
            break
        matched += 1
    if not matched:
        return None
    total = len(content_words(query)) or 1
    return NameMatch(matched, matched_words, min(1.0, matched_words / total))


def fuse(
    query: str,
    docs: Sequence[dict[str, Any]],
    vector_order: Sequence[int],
    lexical: LexicalIndex,
    allowed: Sequence[bool] | None = None,
) -> list[tuple[int, float]]:
    """Weighted reciprocal rank fusion of vector, BM25 and name-match rankings.

    docs: the whole corpus; vector_order: doc indexes by descending cosine similarity
    (already restricted to allowed docs); allowed: optional mask (e.g. message filter).
    Returns (doc index, fused score) best first.
    """
    ok = allowed if allowed is not None else [True] * len(docs)
    wanted = intent(query)
    fused: dict[int, float] = defaultdict(float)

    for rank, i in enumerate(vector_order[:POOL], start=1):
        fused[i] += 1.0 / (RRF_K + rank)

    bm25 = lexical.scores(query)
    by_bm25 = sorted((i for i in bm25 if ok[i]), key=lambda i: -bm25[i])
    for rank, i in enumerate(by_bm25[:POOL], start=1):
        fused[i] += 1.0 / (RRF_K + rank)

    named = []
    for i, doc in enumerate(docs):
        if ok[i] and kind_factor(doc["kind"], wanted) == 1.0 and (m := name_match(query, doc)):
            named.append((m.key(depth(doc)), m.coverage, i))
    named.sort(reverse=True)
    for rank, (_, coverage, i) in enumerate(named[:POOL], start=1):
        fused[i] += NAME_WEIGHT * coverage / (RRF_K + rank)

    for i in fused:
        fused[i] *= kind_factor(docs[i]["kind"], wanted) / (1 + DEPTH_PENALTY * max(0, depth(docs[i]) - 1))
    return sorted(fused.items(), key=lambda kv: -kv[1])
