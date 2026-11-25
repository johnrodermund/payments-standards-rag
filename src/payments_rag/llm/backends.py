"""Provider backends for answer generation: the Claude API, Amazon Bedrock (Converse API) and OpenAI (Responses API).

`LLM_PROVIDER` picks one ("anthropic", the default, "bedrock" or "openai"). Both sit behind the same small interface so
everything around them (response cache, replay-only mode, spend cap, citation checks, audit, tracing) is shared:

    build_request(system, messages, max_tokens) -> dict    exactly what is sent; also the cache key
    send(request) -> BackendResponse                       the one network call

The Anthropic request is byte-for-byte what it was before this module existed, so earlier cached and
recorded answers still replay. The Bedrock request has its own shape, so the two never share a cache entry.

Differences to be aware of (not hidden):
* Bedrock is called through the Converse API with `inferenceConfig.maxTokens` always set (an unset value
  reserves the model's maximum against quota). The Anthropic path also passes `output_config={"effort": "low"}`;
  that parameter is not sent to Bedrock, so answers from the two providers are not guaranteed identical.
* Credentials come from the standard AWS chain (profile, SSO, or an ECS task role). Nothing is read from code.
* The model id is not defaulted: set BEDROCK_GENERATION_MODEL_ID to an inference-profile id that
  `aws bedrock list-inference-profiles` shows for your account and region. Prices are not guessed either:
  see llm/budget.py (LLM_PRICES).
* OpenAI is called through the Responses API with `reasoning.effort` "low" (matching the Claude path's effort)
  and `store` false (no server-side copy of the conversation). `OPENAI_MODEL` picks the model (default
  gpt-6-luna, priced like claude-haiku-5-5). Its reasoning tokens count against the same output cap.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

log = logging.getLogger("payments_rag.llm")

ANTHROPIC_DEFAULT_MODEL = "claude-haiku-5-5"
PROVIDERS = ("anthropic", "bedrock", "openai")
OPENAI_DEFAULT_MODEL = "gpt-6-luna"
# Responses API incomplete reasons -> the Claude API stop-reason vocabulary.
OPENAI_INCOMPLETE = {"max_output_tokens": "max_tokens", "content_filter": "refusal"}

# Bedrock stop reasons -> the vocabulary the rest of the app uses (the Claude API's).
STOP_REASONS = {
    "end_turn": "end_turn",
    "stop_sequence": "stop_sequence",
    "max_tokens": "max_tokens",
    "tool_use": "tool_use",
    "content_filtered": "refusal",
    "guardrail_intervened": "refusal",
}
# Errors that mean "try again later" or "the provider cannot serve this now" (boto already retried).
TRANSIENT_CODES = {"ThrottlingException", "ServiceUnavailableException", "ModelTimeoutException", "InternalServerException", "ModelNotReadyException"}
# Errors that mean the configuration is wrong (credentials, model access, model id, request shape).
CONFIG_CODES = {"AccessDeniedException", "ResourceNotFoundException", "ValidationException", "ModelErrorException"}


class LLMUnavailableError(RuntimeError):
    """The model provider could not serve the request. `str()` is for logs and the audit log; the API
    returns only a generic message to callers (the detail can contain account or resource names)."""


@dataclass(frozen=True)
class BackendResponse:
    text: str
    model: str
    usage: dict[str, int]
    stop_reason: str | None
    request_id: str | None = None


class LLMBackend(Protocol):
    name: str
    model: str

    def build_request(self, system: str, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]: ...

    def send(self, request: dict[str, Any]) -> BackendResponse: ...


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, client: Any | None = None, model: str | None = None):
        self._client = client
        self.model = model or os.environ.get("CLAUDE_MODEL", ANTHROPIC_DEFAULT_MODEL)

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            # Reads ANTHROPIC_API_KEY from the environment; the SDK retries 429/5xx itself.
            self._client = anthropic.Anthropic(timeout=60.0, max_retries=2)
        return self._client

    def build_request(self, system: str, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "output_config": {"effort": "low"},
        }

    def send(self, request: dict[str, Any]) -> BackendResponse:
        response = self.client.messages.create(**request)
        return BackendResponse(
            text="".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip(),
            model=getattr(response, "model", self.model),
            usage={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
            stop_reason=response.stop_reason,
            request_id=getattr(response, "_request_id", None),
        )


class BedrockBackend:
    name = "bedrock"

    def __init__(self, client: Any | None = None, model_id: str | None = None, region: str | None = None):
        self.model = model_id or os.environ.get("BEDROCK_GENERATION_MODEL_ID", "")
        if not self.model:
            raise ValueError(
                "BEDROCK_GENERATION_MODEL_ID is not set. Use an inference-profile id from "
                "`aws bedrock list-inference-profiles --region <region>`; model ids are not guessed."
            )
        self._client = client
        self._region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3
            from botocore.config import Config

            # bedrock-runtime (not bedrock) is the inference endpoint; adaptive retry handles throttling.
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region,
                config=Config(retries={"max_attempts": 5, "mode": "adaptive"}, read_timeout=60),
            )
        return self._client

    def build_request(self, system: str, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        return {
            "modelId": self.model,
            "system": [{"text": system}],
            "messages": [
                {"role": m["role"], "content": [{"text": m["content"]}] if isinstance(m["content"], str) else m["content"]}
                for m in messages
            ],
            "inferenceConfig": {"maxTokens": max_tokens},  # always explicit: an unset value reserves the model maximum
        }

    def send(self, request: dict[str, Any]) -> BackendResponse:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            response = self.client.converse(**request)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "Unknown")
            if code in TRANSIENT_CODES:
                raise LLMUnavailableError(f"Bedrock {code} (after retries)") from exc
            if code in CONFIG_CODES:
                raise LLMUnavailableError(f"Bedrock {code}: {exc.response.get('Error', {}).get('Message', '')[:300]}") from exc
            raise
        except BotoCoreError as exc:  # no credentials, no network, bad region
            raise LLMUnavailableError(f"Bedrock client error: {type(exc).__name__}") from exc

        content = response.get("output", {}).get("message", {}).get("content", [])
        usage = response.get("usage", {})
        stop = response.get("stopReason")
        return BackendResponse(
            text="".join(block["text"] for block in content if "text" in block).strip(),
            model=self.model,
            usage={"input_tokens": usage.get("inputTokens", 0), "output_tokens": usage.get("outputTokens", 0)},
            stop_reason=STOP_REASONS.get(stop, stop),
            request_id=response.get("ResponseMetadata", {}).get("RequestId"),
        )


class OpenAIBackend:
    name = "openai"

    def __init__(self, client: Any | None = None, model: str | None = None):
        self._client = client
        self.model = model or os.environ.get("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)

    @property
    def client(self) -> Any:
        if self._client is None:
            import openai

            # Reads OPENAI_API_KEY from the environment; the SDK retries 429/5xx and connection errors itself.
            self._client = openai.OpenAI(timeout=60.0, max_retries=2)
        return self._client

    def build_request(self, system: str, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        return {
            "model": self.model,
            "instructions": system,
            "input": messages,
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": "low"},
            "store": False,
        }

    def send(self, request: dict[str, Any]) -> BackendResponse:
        import openai

        try:
            response = self.client.responses.create(**request)
        except (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise LLMUnavailableError(f"OpenAI {type(exc).__name__} (after retries)") from exc
        except openai.APIStatusError as exc:  # 400/401/403/404: key, model access, model id or request shape
            raise LLMUnavailableError(f"OpenAI {type(exc).__name__} ({exc.status_code}): {str(exc.message)[:300]}") from exc

        stop = "end_turn"
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", None)
            stop = OPENAI_INCOMPLETE.get(reason, reason or "incomplete")
        usage = response.usage
        return BackendResponse(
            text=(response.output_text or "").strip(),
            model=response.model or self.model,
            usage={"input_tokens": usage.input_tokens if usage else 0, "output_tokens": usage.output_tokens if usage else 0},
            stop_reason=stop,
            request_id=getattr(response, "_request_id", None),
        )


def make_backend(provider: str | None = None, **kwargs: Any) -> LLMBackend:
    """Backend for `provider` (default: LLM_PROVIDER, else "anthropic")."""
    provider = (provider or os.environ.get("LLM_PROVIDER") or "anthropic").lower()
    if provider == "anthropic":
        return AnthropicBackend(**kwargs)
    if provider == "bedrock":
        return BedrockBackend(**kwargs)
    if provider == "openai":
        return OpenAIBackend(**kwargs)
    raise ValueError(f"unknown LLM_PROVIDER {provider!r}; use one of {PROVIDERS}")
