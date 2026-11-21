"""Judges for DeepEval's faithfulness and answer-relevancy metrics.

Two options, chosen with `build_metrics(judge=...)`:

* "jev"    DeepEval's built-in System One mode: TypeSafe Jev runs each metric as typed questions.
           Independent of the Claude generator, very cheap, needs TYPESAFE_API_KEY and `typesafe-sdk`.
* "claude" An LLM judge (default claude-sonnet-5-5, low effort) behind the same disk cache as
           generation, so reruns are free. Needs ANTHROPIC_API_KEY.

Judge calls send the question, the answer and the retrieved public ISO passages to the judge's
provider. No private data is involved.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")  # never send usage data to a third party

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM

from payments_rag.llm.budget import Budget
from payments_rag.llm.cache import ResponseCache

DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"
JUDGE_SYSTEM = "You are a careful evaluator. Follow the instructions exactly. When asked for JSON, reply with JSON only."


def _extract_json(text: str) -> str:
    """Pull a JSON object out of a reply that may be wrapped in code fences or prose."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self, client: Any | None = None, model: str | None = None, cache: ResponseCache | None = None):
        self._client = client
        self.budget = Budget.from_env() if client is None else None  # real client => project spend cap
        self.model_name = model or os.environ.get("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        self.cache = cache
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cache_hits": 0}
        super().__init__(model=self.model_name)

    def load_model(self) -> ClaudeJudge:
        return self

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(timeout=90.0, max_retries=2)
        return self._client

    def get_model_name(self) -> str:
        return self.model_name

    def _ask(self, prompt: str) -> str:
        request = {
            "model": self.model_name,
            "max_tokens": 2000,
            "system": JUDGE_SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": "low"},
        }
        key = ResponseCache.key(request) if self.cache else None
        if key and (hit := self.cache.get(key)) is not None:
            self.usage["cache_hits"] += 1
            return hit["text"]
        if self.budget:
            self.budget.check(self.model_name, request)
        response = self.client.messages.create(**request)
        if self.budget:
            self.budget.record(self.model_name, response.usage.input_tokens, response.usage.output_tokens)
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        self.usage["calls"] += 1
        self.usage["input_tokens"] += response.usage.input_tokens
        self.usage["output_tokens"] += response.usage.output_tokens
        if key:
            self.cache.put(key, {"text": text})
        return text

    def generate(self, prompt: str, schema: Any | None = None) -> Any:
        if schema is None:
            return self._ask(prompt)
        wanted = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        full = f"{prompt}\n\nReply with a single JSON object that matches this JSON schema, and nothing else:\n{wanted}"
        last_error: Exception | None = None
        for attempt in range(2):
            reply = self._ask(full if attempt == 0 else f"{full}\n\nYour previous reply was not valid ({last_error}). Reply with valid JSON only.")
            try:
                return schema.model_validate_json(_extract_json(reply))
            except ValueError as exc:  # pydantic.ValidationError is a ValueError
                last_error = exc
        raise ValueError(f"judge did not return valid JSON for {schema.__name__}: {last_error}")

    async def a_generate(self, prompt: str, schema: Any | None = None) -> Any:
        return await asyncio.to_thread(self.generate, prompt, schema)


def build_metrics(judge: str, threshold: float = 0.5, claude_judge: ClaudeJudge | None = None) -> dict[str, Any]:
    """Return {"faithfulness": metric, "answer_relevancy": metric} for the chosen judge.

    DeepEval's own thresholds only affect its pass/fail flag; the real gate is evals/thresholds.json.
    """
    common = {"threshold": threshold, "include_reason": False, "async_mode": False}
    if judge == "jev":
        common["eval_mode"] = "system_one"
    elif judge == "claude":
        common["model"] = claude_judge or ClaudeJudge(cache=ResponseCache(os.environ.get("EVAL_CACHE_DIR", ".cache/eval")))
    else:
        raise ValueError(f"unknown judge {judge!r}; use 'jev' or 'claude'")
    return {
        "faithfulness": FaithfulnessMetric(**common),
        "answer_relevancy": AnswerRelevancyMetric(**common),
    }
