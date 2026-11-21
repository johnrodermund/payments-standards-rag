"""A hard, project-wide cap on Claude API spend.

Every real (non-cached) call is checked against a ledger file before it is sent and recorded after.
When the running total plus the worst case for the next call would pass the limit, the call is refused
with BudgetExceededError instead of being made. Cached and replayed answers cost nothing and are never
counted.

The limit comes from LLM_BUDGET_USD (default 0.50). The ledger is a plain JSON file
(LLM_LEDGER, default .cache/llm/spend.json): delete it to reset, or raise the limit deliberately.
Prices are list prices per million tokens as recorded in this project (or supplied with LLM_PRICES); a model
with no known price is charged at a deliberately high fallback so it can only make the cap trip early.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("payments_rag.llm")

DEFAULT_LIMIT_USD = 0.50
DEFAULT_LEDGER = Path(".cache/llm/spend.json")

# (input, output) USD per million tokens.
PRICES: dict[str, tuple[float, float]] = {"claude-haiku-5-5": (0.10, 0.50)}
FALLBACK_PRICE = (5.0, 25.0)
CHARS_PER_TOKEN = 3  # pessimistic for English/technical text, so input is over- rather than under-estimated


class BudgetExceededError(RuntimeError):
    """The project's API spend limit would be exceeded by this call."""


def price_for(model: str) -> tuple[float, float]:
    """USD per million (input, output) tokens: the built-in table, then LLM_PRICES, then a high fallback.

    LLM_PRICES is JSON such as {"us.anthropic.claude-haiku-...": [0.10, 0.50]}. Set it for any model not in the
    table (every Bedrock model id), from the provider's current price list; prices are not guessed here.
    """
    if model in PRICES:
        return PRICES[model]
    try:
        override = json.loads(os.environ.get("LLM_PRICES") or "{}").get(model)
        if override:
            return float(override[0]), float(override[1])
    except (ValueError, TypeError, IndexError, AttributeError):
        log.warning("LLM_PRICES is not valid JSON of {model: [input, output]}; using the high fallback price")
    return FALLBACK_PRICE


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = price_for(model)
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


class Budget:
    def __init__(self, limit_usd: float = DEFAULT_LIMIT_USD, ledger: Path | str = DEFAULT_LEDGER):
        self.limit_usd = limit_usd
        self.ledger = Path(ledger)
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> Budget:
        return cls(float(os.environ.get("LLM_BUDGET_USD", DEFAULT_LIMIT_USD)), os.environ.get("LLM_LEDGER", DEFAULT_LEDGER))

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.ledger.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"spent_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}

    def spent(self) -> float:
        return float(self._read()["spent_usd"])

    def check(self, model: str, request: dict[str, Any], max_tokens: int | None = None) -> None:
        """Refuse the call if the worst case for it would take the total past the limit."""
        est_input = len(json.dumps(request, default=str)) // CHARS_PER_TOKEN + 1
        worst = cost_usd(model, est_input, int(max_tokens if max_tokens is not None else request.get("max_tokens", 0)))
        with self._lock:
            spent = float(self._read()["spent_usd"])
        if spent + worst > self.limit_usd:
            raise BudgetExceededError(
                f"API budget ${self.limit_usd:.2f} would be exceeded: spent ${spent:.4f}, this call could cost up to ${worst:.4f}. "
                f"Raise LLM_BUDGET_USD or delete {self.ledger} to continue."
            )

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = cost_usd(model, input_tokens, output_tokens)
        with self._lock:
            data = self._read()
            data["spent_usd"] = round(float(data["spent_usd"]) + cost, 8)
            data["calls"] += 1
            data["input_tokens"] += input_tokens
            data["output_tokens"] += output_tokens
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.ledger.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.ledger)
        return cost
