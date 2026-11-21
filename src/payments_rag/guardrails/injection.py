"""Deterministic first-line defence against prompt injection planted in retrieved documents.

What it does
------------
* `neutralise_passage` removes only the offending sentences of a passage and keeps the rest,
  so a poisoned passage that also holds the real definition can still answer the question
  (quarantining the whole passage handed attackers a cheap lever to force refusals). Before
  scanning it strips invisible characters (zero-width, bidi controls, Unicode "tag" smuggling,
  variation selectors) from the served text, and scans a length-preserving "skeleton" in
  which homoglyphs, accents and full-width forms are folded to ASCII. Each sentence is also
  scanned in de-obfuscated variants (spaced-out letters, leetspeak, ROT13, reversed) and any
  base64 / hex / percent / escape blob is decoded and scanned too.
* `screen_passages` applies that to every retrieved hit and returns (kept, interventions).
  A passage is dropped only when nothing usable is left after neutralising it.
* `scan_passage` lists the reasons for one text (used at ingest time by the validation step).
* `check_output` rejects answers that contain what this assistant must never emit (links,
  e-mail addresses, tool-call syntax, its own system prompt, role markers) and, when the
  passages are given, code-like tokens (e.g. ``AB-1234``) that appear in neither the passages
  nor the question: a sign the model followed a hidden or decoded instruction.

What it does NOT do (be honest)
-------------------------------
* It cannot tell a plausible false statement ("the maximum length is 99999") from a true one.
  That needs provenance (see `payments_rag.guardrails.provenance`), not pattern matching.
* It is a set of heuristics. Novel phrasings, other languages than the ones covered, or
  instructions disguised as domain prose can slip through. It sits alongside (not instead of)
  the prompt's data framing, citation checks, human escalation and the audit log.
* A model-based classifier can be plugged in through `screen_passages(classifier=...)`; it is
  off by default and nothing in this module calls a model.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_F = re.IGNORECASE | re.MULTILINE

# A neutralised passage with less text than this left is dropped instead of shown. Kept low on
# purpose: dropping is what lets an attacker force "not enough context" escalations.
MIN_USABLE_CHARS = 20
EXCERPT_CHARS = 160

# The answer protocol's own sentinel (see llm.generation.SYSTEM_PROMPT). Reference data has no
# reason to contain it; a passage that does is trying to steer the reply.
_SENTINELS = re.compile(r"\bINSUFFICIENT_CONTEXT\b")

# ---------------------------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------------------------

# Latin look-alikes from Cyrillic and Greek (the usual homoglyph substitutions).
_CONFUSABLES = {
    **dict(zip("аеорсухіјѕԁӏкмнтвгпАЕОРСУХІЈЅКМНТВ", "aeopcyxijsdlkmhtbrnAEOPCYXIJSKMHTB", strict=True)),
    **dict(zip("οαειкνρτυχΑΒΕΗΙΚΜΝΟΡΤΥΧΖ", "oaeiknptuxABEHIKMNOPTYXZ", strict=True)),
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
}


def _is_invisible(ch: str) -> bool:
    cp = ord(ch)
    return (
        unicodedata.category(ch) == "Cf"  # zero-width, bidi controls, BOM, soft hyphen, tag chars
        or 0xFE00 <= cp <= 0xFE0F  # variation selectors (used to smuggle bytes)
        or 0xE0100 <= cp <= 0xE01EF
        or cp in (0x180E, 0x3164, 0x115F, 0x1160, 0xFFA0)  # fillers that render as nothing
    )


def strip_invisible(text: str) -> tuple[str, int, str]:
    """(text without invisible characters, how many were removed, ASCII smuggled in tag chars)."""
    kept: list[str] = []
    smuggled: list[str] = []
    removed = 0
    for ch in text:
        if _is_invisible(ch):
            removed += 1
            if 0xE0020 <= ord(ch) <= 0xE007E:
                smuggled.append(chr(ord(ch) - 0xE0000))
        else:
            kept.append(ch)
    return "".join(kept), removed, "".join(smuggled)


def _fold(ch: str) -> str:
    if ch in _CONFUSABLES:
        return _CONFUSABLES[ch]
    if ch.isascii():
        return ch
    nfkc = unicodedata.normalize("NFKC", ch)
    if len(nfkc) == 1 and nfkc.isascii():
        return nfkc  # full-width and other compatibility forms
    nfkd = unicodedata.normalize("NFKD", ch)
    if nfkd and nfkd[0].isascii() and all(unicodedata.combining(c) for c in nfkd[1:]):
        return nfkd[0]  # accents: é -> e
    return ch


def skeleton(text: str) -> str:
    """Same length as `text`, with homoglyphs, accents and full-width forms folded to ASCII."""
    return "".join(_fold(ch) for ch in text)


_WORD = re.compile(r"\w+")


def _mixed_script(text: str) -> bool:
    """A word mixing ASCII letters with Cyrillic/Greek look-alikes (no real ISO text does)."""
    for m in _WORD.finditer(text):
        word = m.group(0)
        if any(c.isascii() and c.isalpha() for c in word) and any(
            c in _CONFUSABLES and not c.isascii() and c.isalpha() for c in word
        ):
            return True
    return False


# ---------------------------------------------------------------------------------------------
# Sentence-level rules (run on the skeleton)
# ---------------------------------------------------------------------------------------------

_EN_OVERRIDE_VERBS = (
    r"ignore|ignoring|disregard|forget|override|bypass|discard|abandon|neglect|set aside|put aside|"
    r"pay no attention to|do not follow|don'?t follow|stop following"
)
_EN_RULE_NOUNS = (
    r"instructions?|rules?|prompts?|guidance|guidelines?|directions?|directives?|orders?|"
    r"constraints?|polic(?:y|ies)|system message|programming|context|everything (?:above|before)"
)
_FOREIGN_OVERRIDE_VERBS = (
    r"ignor\w*|oubli\w*|olvid\w*|vergiss\w*|vergeet|negeer|missacht\w*|dimentic\w*|esquec\w*|"
    r"ne tenez pas compte|no hagas caso|non considerare"
)
_FOREIGN_RULE_NOUNS = (
    r"instructions|instrucciones|instructies|instruktionen|istruzioni|instrucoes|anweisungen|"
    r"consignes|regles|reglas|regeln|regole|regras|indicaciones|directives|directrices|vorgaben|regels|"
    r"anwisungen|indicazioni"
)

_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "instruction_override",
        re.compile(
            rf"\b(?:{_EN_OVERRIDE_VERBS})\b[^.\n]{{0,60}}\b(?:{_EN_RULE_NOUNS})\b"
            rf"|\b(?:{_FOREIGN_OVERRIDE_VERBS})\b[^.\n]{{0,60}}\b(?:{_FOREIGN_RULE_NOUNS})\b"
            r"|\b(?:instructions?|guidance|rules?|directions?) (?:you|that you) (?:were|have been|got) given\b"
            rf"|\b(?:{_EN_RULE_NOUNS})\b[^.\n]{{0,30}}\b(?:is|are|was|were|has been|have been)\s+(?:now\s+)?"
            r"(?:cancell?ed|void|revoked|withdrawn|suspended|lifted|obsolete|no longer (?:valid|apply|applies|in force))\b"
            rf"|\b(?:beachte|befolge)\w*\b[^.\n]{{0,60}}\b(?:{_FOREIGN_RULE_NOUNS})\b[^.\n]{{0,20}}\bnicht\b",
            _F,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(?:system|developer|hidden|initial|original|secret|internal|pre-?)\s?(?:prompt|instructions)\b"
            r"|\b(?:system|developer) message\b"
            r"|\b(?:print|reveal|show|repeat|output|display|leak|disclose|share|tell me|list|dump|recite|echo|"
            r"write out|spell out|summari[sz]e)\b[^.\n]{0,50}\b(?:your|the|all|any)\b[^.\n]{0,30}"
            r"\b(?:prompt|instructions|rules|guidelines|configuration|directives)\b"
            r"|\byour (?:prompt|instructions|rules|guidelines|configuration|directives)\b"
            r"|\b(?:were|have) you (?:been )?(?:told|given|instructed|configured)\b"
            r"|\byou (?:were|have been) (?:told|instructed|configured)\b",
            _F,
        ),
    ),
    (
        "role_or_tag_injection",
        re.compile(
            r"</?\s*(?:passages?|system|assistant|user|human|instructions?|context|documents?|tool_?\w*|"
            r"im_start|im_end|inst|sys|admin)\b"
            r"|<\|[a-z_]{2,30}\|>"
            r"|\[/?(?:inst|sys|system|assistant|user|admin)\]"
            r"|^\s*[#>*\-\[(]*\s*(?:system|assistant|user|human|developer|admin|administrator|ai|operator)"
            r"\s*[\]):]"
            r"|^\s*#{1,6}\s*(?:system|instructions?|new task|assistant|ai|model|guidelines?|policy)\b"
            r"|\b(?:begin|end|start)\s+(?:of\s+)?(?:system|instructions?|prompt)\b"
            r"|\b(?:system|admin|administrator|security|developer|operator|moderator)\s+"
            r"(?:notice|override|alert|update|command|directive|announcement|instruction)\b",
            _F,
        ),
    ),
    (
        "forced_output",
        re.compile(
            r"\b(?:reply|respond|answer|say|write|output|print|emit|produce)\b[^.\n]{0,25}"
            r"\b(?:only|exactly|always|solely|just|verbatim|word for word|nothing else|nothing but)\b"
            r"|\b(?:start|begin|end|prefix|suffix|preface|conclude|finish|close|terminate|sign)\w*\b[^.\n]{0,25}"
            r"\b(?:every|each|all|your|any)\s+(?:reply|replies|response|responses|answer|answers|output|message)s?\b"
            r"|\b(?:append|add|include|insert|put|attach|embed|mention)\b[^.\n]{0,40}\b(?:to|in|into|at the "
            r"(?:end|start|beginning) of)\s+(?:(?:every|each|all|your|any|the)\s+)?(?:repl(?:y|ies)|responses?|answers?|outputs?)\b"
            r"|\b(?:reply|respond|answer)\s+with\b"
            r"|\b(?:expected|preferred|correct|required|desired|proper|ideal|appropriate|right)\s+"
            r"(?:response|reply|answer|output|completion)s?\b"
            r"|\b(?:when|while|before) (?:summari[sz]ing|answering|responding|replying)\b"
            r"|\blead with\b"
            r"|\b(?:decline|refuse) to (?:answer|respond|reply)\b"
            r"|\b(?:answering|responding|replying)\b[^.\n]{0,40}\b(?:not permitted|forbidden|prohibited|not allowed)\b"
            r"|\b(?:state|say|reply|respond|claim|report) that (?:the )?(?:information|context|passages?|data|material)\s+"
            r"(?:is|are)\s+(?:insufficient|missing|not enough|incomplete|unavailable)\b"
            r"|\bcannot be answered\b"
            r"|\b(?:output|print|emit|repeat|say|write|type|produce)\s+(?:the|this|that)\s+"
            r"(?:token|code|word|string|phrase|sentence|value)\b"
            r"|\bnothing else\b"
            r"|\bwhatever (?:the|you are|you're) (?:question|asked|user)\b"
            r"|\bregardless of (?:the|what) (?:question|user|is asked)\b"
            r"|\bno matter what (?:the )?(?:question|user|is asked)\b",
            _F,
        ),
    ),
    (
        "addressed_to_model",
        re.compile(
            r"\b(?:you|your|yourself)\b[^.\n]{0,40}\b(?:must|should|will|shall|are to|need to|have to|"
            r"are required|answer|reply|respond|output|say|print|write|ignore|forget|cite|include|start|begin|"
            r"objective|goal|task|role|purpose|job)\b"
            r"|\b(?:note|message|instruction|notice|reminder|memo|directive|warning)s?\s+(?:to|for)\s+(?:the\s+|all\s+|any\s+)?"
            r"(?:ai|assistants?|models?|llms?|chatbots?|bots?|language models?|automated|readers?|agents?)\b"
            r"|\b(?:dear|attention|hey|hi|hello)\s+(?:ai|assistant|model|llm|chatgpt|claude|gpt|bot)\b"
            r"|\b(?:if|when|since) you(?:'re| are) an? (?:ai|assistant|language model|llm|bot|automated)"
            r"|\b(?:ai|llm|language model|chatbot)\s+(?:assistants?|readers?|agents?|systems?|models?)\b"
            r"|\breader of this (?:passage|document|text|chunk|page)\b"
            r"|\bautomated (?:readers?|agents?|assistants?)\b"
            r"|\b(?:the|an?|this)\s+(?:ai|assistant|model|llm|chatbot|bot)\b[^.\n]{0,30}\b(?:should|must|will|shall|"
            r"is to|needs? to|has to|may not|cannot|is required)\b"
            r"|\b(?:whoever|anyone|anything|whatever) (?:processes|reads|parses|summari[sz]es|handles|ingests) this\b",
            _F,
        ),
    ),
    (
        "behaviour_change",
        re.compile(
            r"\bfrom now on\b|\bhenceforth\b|\bgoing forward,? (?:you|the assistant)\b"
            r"|\bnew (?:instructions?|policy|rules?|task|objective|directive|guidelines?|role|persona)\b"
            r"|\byour (?:real|true|actual|new|only|primary|main|hidden|secret|ultimate) "
            r"(?:objective|goal|task|job|purpose|instructions?|role|mission|priority)\b"
            r"|\bpretend (?:to be|you)\b|\brole-?play\b|\byou are (?:now|no longer|actually)\b"
            r"|\bjailbreak\b|\bdeveloper mode\b",
            _F,
        ),
    ),
    (
        "authority_claim",
        re.compile(
            r"\b(?:other|remaining|previous|preceding|following|above|below|earlier|rest of the|all other|every other|"
            r"neighbou?ring|adjacent|surrounding)\s+"
            r"(?:passages?|sources?|documents?|context|results?|chunks?|excerpts?|snippets?|entr(?:y|ies))\b[^.\n]{0,40}"
            r"\b(?:outdated|out of date|wrong|incorrect|invalid|obsolete|superseded|unreliable|fake|deprecated|"
            r"ignored?|disregard\w*|false|untrusted|inaccurate|errors?|not apply|withdrawn|revoked|retracted|void)\b"
            r"|\bsupersedes? (?:every|all|any) (?:other|previous)\b"
            r"|\b(?:rely on|trust|use|answer from|cite) (?:this|the present) (?:entry|excerpt|passage|record|source|document|text)\s+"
            r"(?:alone|only|exclusively)\b"
            r"|\bonly this (?:entry|excerpt|passage|record|source|document|text|chunk) (?:is|reflects|contains|has|gives)\s+"
            r"(?:the )?(?:current|correct|valid|accurate|up-to-date|authoritative|latest)\b"
            r"|\b(?:only|exclusively|solely)\s+(?:use|trust|rely on|answer from|cite|consider|believe)\s+"
            r"(?:this|the following|these)\s+(?:passage|source|document|text|chunk|section|note)\b"
            r"|\b(?:answer|respond|reply)\s+(?:only|exclusively|solely)\s+(?:from|using|based on)\s+this\b"
            r"|\bonly from this (?:passage|source|document|text)\b"
            r"|\b(?:passages?|sources?|documents?) (?:other than|beyond|besides|below)\b[^.\n]{0,40}"
            r"\b(?:outdated|wrong|incorrect|ignore)\b"
            r"|\b(?:this|the following) (?:passage|source|document|text|chunk|note) (?:is|takes|has) (?:the )?"
            r"(?:authoritative|official|only valid|precedence|priority)\b",
            _F,
        ),
    ),
    (
        "citation_spoof",
        re.compile(r"\b(?:cite|reference|attribute|label|number)\w*\b[^.\n]{0,40}\[\d+\]|\bcite (?:it|this|me) first\b", _F),
    ),
    (
        "exfiltration_link",
        re.compile(
            r"https?://|\bwww\.\w|!\[[^\]]*\]\(|\]\(\s*[a-z]+:"
            r"|\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b"
            r"|\b(?:[a-z0-9-]{2,}\.)+(?:com|net|org|io|info|biz|xyz|ru|cn|tk|top|example|site|online)\b"
            r"|\b(?:data|javascript):[^\s]{8,}",
            _F,
        ),
    ),
    (
        "tool_mimic",
        re.compile(
            r"<\s*/?\s*(?:function_calls?|tool_calls?|tool_use|invoke|antml|parameter)\b"
            r"|\b[a-z]+_[a-z_]+\s*\(\s*[a-z_]+\s*="
            r'|"(?:name|tool|function|tool_name|arguments|parameters)"\s*:\s*["{]'
            r"|\b(?:call|invoke|execute|run|trigger)\s+(?:the\s+)?(?:function|tool|plugin)s?\b"
            r"|\bsend_email\b",
            _F,
        ),
    ),
    (
        "encoding_reference",
        re.compile(
            r"\b(?:decode|decrypt|deobfuscate|unscramble|base ?64|base ?32|rot ?-?13|hex[- ]?encoded|"
            r"url[- ]?encoded|caesar cipher|read (?:it|this) backwards)\b",
            _F,
        ),
    ),
    ("sentinel_token", _SENTINELS),
]

_NON_LATIN_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "instruction_override",
        re.compile(
            r"(?:игнорир\w*|забудь\w*|не обращай\w*)[^.\n]{0,60}(?:инструкц\w*|правил\w*|указани\w*)"
            r"|(?:忽略|无视|無視|忽視|忘记|忘記|忘れ)[^。\n]{0,30}(?:指令|指示|说明|規則|规则|提示|命令|ルール)"
            r"|(?:指令|指示|说明|規則|规则|提示|命令|ルール)[^。\n]{0,10}(?:忽略|无视|無視|忽視|忘记|忘記|忘れ)",
            _F,
        ),
    ),
]

# Rules worth re-running on de-obfuscated variants (cheap and with no false positives on ISO text).
_CORE = {"instruction_override", "prompt_exfiltration", "forced_output", "addressed_to_model", "behaviour_change"}

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})
_SPACED = re.compile(r"(?:\b\w[ ._*\-]){4,}\w\b")

# Encoded blobs.
_LONG_TOKEN = re.compile(r"[A-Za-z0-9+]{40,}={0,2}")
_WORD_RUN = re.compile(r"[A-Z][a-z]{2,}|[a-z]{3,}")
_B64_CANDIDATE = re.compile(r"[A-Za-z0-9+/_-]{16,}={0,2}")
_HEX_CANDIDATE = re.compile(r"\b(?:[0-9a-fA-F]{2}){16,}\b")
_PERCENT = re.compile(r"(?:%[0-9a-fA-F]{2}){6,}")
_ESCAPES = re.compile(r"(?:\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}){6,}|(?:&#x?[0-9a-fA-F]+;){6,}")


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


def _printable_text(raw: bytes) -> str | None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if len(text) < 12 or " " not in text:
        return None
    printable = sum(c.isprintable() for c in text)
    return text if printable / len(text) >= 0.9 else None


def _decoded_blobs(text: str) -> list[str]:
    """Readable text hidden in base64 / hex / percent / escape encodings inside `text`."""
    found: list[str] = []
    for m in _B64_CANDIDATE.finditer(text):
        blob = m.group(0).replace("-", "+").replace("_", "/")
        blob += "=" * (-len(blob) % 4)
        try:
            decoded = _printable_text(base64.b64decode(blob, validate=True))
        except (binascii.Error, ValueError):
            decoded = None
        if decoded:
            found.append(decoded)
    for m in _HEX_CANDIDATE.finditer(text):
        decoded = _printable_text(bytes.fromhex(m.group(0)))
        if decoded:
            found.append(decoded)
    for m in _PERCENT.finditer(text):
        from urllib.parse import unquote

        found.append(unquote(m.group(0)))
    for m in _ESCAPES.finditer(text):
        s = m.group(0)
        if s.startswith("&"):
            import html

            found.append(html.unescape(s))
        else:
            found.append(codecs.decode(s, "unicode_escape"))
    return found


def _match_rules(text: str, rules: list[tuple[str, re.Pattern[str]]], only: set[str] | None = None) -> list[str]:
    return [name for name, pattern in rules if (only is None or name in only) and pattern.search(text)]


def _scan_segment(original: str, skel: str, depth: int = 0) -> list[str]:
    """Reasons one sentence looks like an instruction to the model (empty if clean)."""
    reasons = _match_rules(skel, _RULES) + _match_rules(original, _NON_LATIN_RULES)
    if _mixed_script(original):
        reasons.append("homoglyph")

    variants = [_SPACED.sub(lambda m: re.sub(r"[ ._*\-]", "", m.group(0)), skel), skel.translate(_LEET),
                codecs.encode(skel, "rot13"), skel[::-1]]
    plain = set(_match_rules(skel, _RULES, _CORE))
    for variant in variants:
        if variant != skel and set(_match_rules(variant, _RULES, _CORE)) - plain:
            reasons.append("obfuscated_text")
            break

    blobs = _decoded_blobs(skel)
    if blobs or _looks_like_base64(skel) or _PERCENT.search(skel) or _ESCAPES.search(skel):
        reasons.append("encoded_payload")
    if depth == 0:
        for blob in blobs:
            reasons += _scan_segment(blob, skeleton(blob), depth=1)
    return list(dict.fromkeys(reasons))


# ---------------------------------------------------------------------------------------------
# Passage-level API
# ---------------------------------------------------------------------------------------------

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def _segments(text: str) -> list[tuple[int, int, int]]:
    """(line number, start, end) of each sentence; lines are never merged."""
    out: list[tuple[int, int, int]] = []
    for line_no, line in enumerate(re.finditer(r"[^\n]+", text)):
        base, s, pos = line.start(), line.group(0), 0
        for m in _SENTENCE_BREAK.finditer(s):
            out.append((line_no, base + pos, base + m.start()))
            pos = m.end()
        if pos < len(s):
            out.append((line_no, base + pos, base + len(s)))
    return out


@dataclass
class Neutralised:
    text: str  # what may be shown to the model
    reasons: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)  # excerpts of what was cut, for the audit log

    @property
    def changed(self) -> bool:
        return bool(self.reasons)

    @property
    def usable(self) -> bool:
        return len(self.text.strip()) >= MIN_USABLE_CHARS


def neutralise_passage(text: str) -> Neutralised:
    """Strip invisible characters and remove the sentences that read like instructions.

    Removal is per sentence. A flagged sentence also takes the rest of its line with it:
    planted payloads are contiguous and often put the bait ("... code XYZ-12.") in a short
    unflagged sentence after the instruction, while ISO lines lead with the definition.
    Known blind spot: unflagged payload text placed *before* the instruction on the same
    line survives.
    """
    clean, invisible, smuggled = strip_invisible(text)
    reasons: list[str] = []
    if invisible:
        reasons.append("hidden_characters")
        if smuggled.strip():
            reasons += _scan_segment(smuggled, skeleton(smuggled))
    skel = skeleton(clean)

    drop: list[tuple[int, int]] = []
    removed: list[str] = []
    cut_line: int | None = None
    for line_no, start, end in _segments(clean):
        found = _scan_segment(clean[start:end], skel[start:end])
        if found or cut_line == line_no:
            reasons += found
            drop.append((start, end))
            removed.append(clean[start:end].strip()[:EXCERPT_CHARS])
            cut_line = line_no
    if not drop:
        return Neutralised(clean, list(dict.fromkeys(reasons)), removed)

    pieces, pos = [], 0
    for start, end in drop:
        pieces.append(clean[pos:start])
        pos = end
    pieces.append(clean[pos:])
    lines = [ln.rstrip() for ln in "".join(pieces).split("\n")]
    out = "\n".join(ln for ln in lines if ln.strip())
    return Neutralised(out, list(dict.fromkeys(reasons)), removed)


def scan_passage(text: str) -> list[str]:
    """Names of the injection patterns found in one passage (empty if it looks clean)."""
    return neutralise_passage(text).reasons


Classifier = Callable[[str], float]


def screen_passages(
    hits: list[dict[str, Any]],
    classifier: Classifier | None = None,
    threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (hits to show the model, one record per passage the guard changed or dropped).

    Each record: {"id", "reasons", "action": "neutralised" | "dropped", "removed": [excerpts]}.
    Kept hits are copies whose text is the neutralised text.

    `classifier` (optional, off by default) maps a passage's neutralised text to a probability
    that it is still an injection attempt, e.g. a model-based judge. A passage at or above
    `threshold` is dropped. If the classifier raises, the passage is kept (so an outage cannot be
    used to empty every context) and the failure is recorded as "classifier_unavailable".
    """
    kept: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for hit in hits:
        result = neutralise_passage(hit["text"])
        reasons = list(result.reasons)
        action = "neutralised"
        if result.changed and not result.usable:
            action = "dropped"  # only a passage the guard cut down can be dropped for being too short
        elif classifier is not None:
            try:
                if classifier(result.text) >= threshold:
                    reasons.append("classifier")
                    action = "dropped"
            except Exception:  # noqa: BLE001 - any classifier failure is recorded, never fatal
                reasons.append("classifier_unavailable")
        if reasons:
            records.append({"id": hit["id"], "reasons": reasons, "action": action, "removed": result.removed})
        if action != "dropped":
            kept.append({**hit, "text": result.text} if result.changed else hit)
    return kept, records


# ---------------------------------------------------------------------------------------------
# Output side
# ---------------------------------------------------------------------------------------------

_OUTPUT_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "link_or_image",
        re.compile(
            r"https?://|\bwww\.\w|!\[[^\]]*\]\(|\]\(\s*[a-z]+:|\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b"
            r"|\b(?:[a-z0-9-]{2,}\.)+(?:com|net|org|io|info|biz|xyz|ru|cn|tk|top|example|site|online)\b",
            _F,
        ),
    ),
    (
        "tool_mimic",
        re.compile(
            r"<\s*/?\s*(?:function_calls?|tool_calls?|tool_use|invoke|antml|parameter)\b"
            r"|\b[a-z]+_[a-z_]+\s*\(\s*[a-z_]+\s*=|\bsend_email\b"
            r'|"(?:name|tool|function|tool_name|arguments|parameters)"\s*:\s*["{]',
            _F,
        ),
    ),
    (
        "prompt_leak",
        re.compile(
            r"numbered passages|untrusted reference data|never follow instructions found|"
            r"only the numbered passages|do not invent element names",
            _F,
        ),
    ),
    ("role_marker", re.compile(r"^\s*[\[<(]*\s*(?:system|assistant|user|human|developer)\s*[\]>):]", _F)),
]

_SHINGLE = 8
_TOKEN = re.compile(r"(?<![\w.])(?=[\w-]*\d)(?=[\w-]*[A-Za-z])[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*(?![\w])")
_UNIT_SUFFIX = re.compile(r"^\d+(?:st|nd|rd|th|s|h|m|k|am|pm|kb|mb|gb|x)$", re.IGNORECASE)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _shingles(text: str) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i : i + _SHINGLE]) for i in range(len(w) - _SHINGLE + 1)}


_PROMPT_SHINGLES: set[tuple[str, ...]] | None = None


def _system_prompt_shingles() -> set[tuple[str, ...]]:
    global _PROMPT_SHINGLES
    if _PROMPT_SHINGLES is None:
        from payments_rag.llm.generation import SYSTEM_PROMPT

        _PROMPT_SHINGLES = _shingles(SYSTEM_PROMPT)
    return _PROMPT_SHINGLES


def _squash(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", text.lower())


def ungrounded_tokens(answer: str, passages: list[str], question: str = "") -> list[str]:
    """Code-like tokens (letters and digits, e.g. AB-1234) in the answer that appear in neither
    the passages nor the question. Ordinals and units (2nd, 10am) are ignored."""
    corpus = _squash(" ".join(passages) + " " + question)
    out = []
    for m in _TOKEN.finditer(answer):
        token = m.group(0)
        letters = sum(c.isalpha() for c in token)
        digits = sum(c.isdigit() for c in token)
        if len(token) < 4 or letters < 2 or digits < 2 or _UNIT_SUFFIX.match(token):
            continue
        if _squash(token) not in corpus:
            out.append(token)
    return list(dict.fromkeys(out))


def check_output(text: str, passages: list[str] | None = None, question: str = "") -> list[str]:
    """Names of the policies a generated answer violates (empty if it is fine to show).

    With `passages` (the texts the model was shown) it also flags `ungrounded_token`.
    """
    clean, invisible, _ = strip_invisible(text)
    violations = [name for name, pattern in _OUTPUT_RULES if pattern.search(clean)]
    if invisible:
        violations.append("hidden_characters")
    if "prompt_leak" not in violations and _shingles(clean) & _system_prompt_shingles():
        violations.append("prompt_leak")
    if passages is not None and ungrounded_tokens(clean, passages, question):
        violations.append("ungrounded_token")
    return violations
