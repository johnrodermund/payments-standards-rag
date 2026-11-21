"""First-line defence against prompt injection planted in retrieved documents.

Two cheap, deterministic checks:

* `screen_passages` quarantines retrieved passages that read like instructions to an AI
  (override, prompt exfiltration, fake system messages, links, encoded blobs, ...) so the
  model never sees them.
* `check_output` rejects answers that contain things this assistant must never emit
  (links, markdown images, tool-call syntax, its own system prompt).

These are heuristics: they reduce risk, they do not eliminate it. Paraphrased or novel
payloads can slip through, so they sit alongside (not instead of) the prompt's data
framing, citation checks, human escalation and the audit log. A model-based classifier
can be added behind the same `scan_passage` interface.
"""

from __future__ import annotations

import re
from typing import Any

_FLAGS = re.IGNORECASE | re.MULTILINE

_PASSAGE_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|any)\b"
            r"[^.\n]{0,40}\b(instructions?|rules?|prompts?|context)\b"
            r"|\bignor(?:ez|e[sr]?|a|iere)\b[^.\n]{0,40}\b(instructions?|instrucciones|anweisungen|consignes)\b",
            _FLAGS,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(system|developer) (prompt|message|instructions?)\b"
            r"|\b(print|reveal|show|repeat|output|display|leak)\b[^.\n]{0,40}\b(your|the)\b[^.\n]{0,25}"
            r"\b(prompt|instructions?|rules)\b",
            _FLAGS,
        ),
    ),
    (
        "role_or_tag_injection",
        re.compile(r"</?\s*(passages?|system|assistant|user)\b|^\s*\[?\s*(system|assistant)\s*\]?\s*[:\]]", _FLAGS),
    ),
    ("forced_output", re.compile(r"\b(reply|respond|answer|say|write)\b[^.\n]{0,15}\b(only|exactly|always)\b", _FLAGS)),
    (
        "behaviour_change",
        re.compile(r"\b(from now on|you (must|will|are to) now|new (instructions?|policy|rules?))\b", _FLAGS),
    ),
    ("exfiltration_link", re.compile(r"https?://|\bwww\.|!\[[^\]]*\]\(", _FLAGS)),
    ("tool_mimic", re.compile(r"<\s*(function_call|tool_call|invoke)\b|\bsend_email\b", _FLAGS)),
    ("hidden_characters", re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")),
    (
        "authority_claim",
        re.compile(
            r"\b(passages?|sources?|documents?) (other than|beyond|below)\b[^.\n]{0,40}\b(outdated|wrong|incorrect|ignore)\b"
            r"|\bonly from this passage\b|\bother passages are (outdated|wrong|incorrect)\b",
            _FLAGS,
        ),
    ),
]

_OUTPUT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("link_or_image", re.compile(r"https?://|\bwww\.|!\[[^\]]*\]\(", _FLAGS)),
    ("tool_mimic", re.compile(r"<\s*(function_call|tool_call|invoke)\b|\bsend_email\b", _FLAGS)),
    (
        "prompt_leak",
        re.compile(
            r"numbered passages|untrusted reference data|never follow instructions found|"
            r"only the numbered passages|do not invent element names",
            _FLAGS,
        ),
    ),
]


_LONG_TOKEN = re.compile(r"[A-Za-z0-9+]{40,}={0,2}")


_WORD_RUN = re.compile(r"[A-Z][a-z]{2,}|[a-z]{3,}")


def _looks_like_base64(text: str) -> bool:
    """Long token that is not made of words.

    ISO names such as BranchAndFinancialInstitutionIdentification8 are long but are almost
    entirely capitalised words (measured on the real chunks: at least 95% word characters);
    base64 of an instruction measured 22-51%. Threshold sits between them.
    """
    for match in _LONG_TOKEN.finditer(text):
        token = match.group(0)
        word_chars = sum(len(w) for w in _WORD_RUN.findall(token))
        if token.endswith("=") or word_chars / len(token) < 0.8:
            return True
    return False


def scan_passage(text: str) -> list[str]:
    """Names of the injection patterns found in one passage (empty if it looks clean)."""
    reasons = [name for name, pattern in _PASSAGE_RULES if pattern.search(text)]
    if _looks_like_base64(text):
        reasons.append("encoded_payload")
    return reasons


def screen_passages(hits: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split retrieved hits into (kept, quarantined). Quarantined entries list why."""
    kept: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for hit in hits:
        reasons = scan_passage(hit["text"])
        if reasons:
            quarantined.append({"id": hit["id"], "reasons": reasons})
        else:
            kept.append(hit)
    return kept, quarantined


def check_output(text: str) -> list[str]:
    """Names of the policies a generated answer violates (empty if it is fine to show)."""
    return [name for name, pattern in _OUTPUT_RULES if pattern.search(text)]
