"""One-call smoke test for the Bedrock backend. MAKES A REAL (BILLED) BEDROCK CALL: run it deliberately.

    python -m payments_rag.llm.bedrock_check

It sends a single tiny Converse request (about 30 input and at most 20 output tokens, well under a cent)
through the same backend, spend cap and ledger the service uses, and prints what came back or why it failed.
Needs AWS credentials (AWS_PROFILE or the usual chain), AWS_REGION, BEDROCK_GENERATION_MODEL_ID and, for the
spend cap to price the call, an entry for that model in LLM_PRICES (otherwise the high fallback price is used).
"""

from __future__ import annotations

import sys

from payments_rag.llm.backends import BedrockBackend, LLMUnavailableError
from payments_rag.llm.budget import Budget

HINTS = {
    "ThrottlingException": "quota for this model/region is probably 0 or exhausted: check Service Quotas (Bedrock) and the account's model-access status",
    "AccessDeniedException": "the identity lacks bedrock:InvokeModel for this model or inference profile, or the Anthropic use-case form was not submitted",
    "ResourceNotFoundException": "the model id is wrong for this region: use an inference-profile id from `aws bedrock list-inference-profiles`",
    "ValidationException": "the model id may need an inference profile (a bare model id can be rejected for on-demand use)",
}
MAX_TOKENS = 20


def main() -> int:
    try:
        backend = BedrockBackend()
    except ValueError as exc:
        print(f"NOT CONFIGURED: {exc}")
        return 2
    budget = Budget.from_env()
    request = backend.build_request("Reply with the single word OK.", [{"role": "user", "content": "Are you there?"}], MAX_TOKENS)
    try:
        budget.check(backend.model, request, max_tokens=MAX_TOKENS)
        response = backend.send(request)
    except LLMUnavailableError as exc:
        print(f"FAILED: {exc}")
        for code, hint in HINTS.items():
            if code in str(exc):
                print(f"  hint: {hint}")
        return 1
    budget.record(backend.model, response.usage["input_tokens"], response.usage["output_tokens"])
    print(f"OK: model={response.model} stop={response.stop_reason} usage={response.usage} request_id={response.request_id}")
    print(f"  reply: {response.text!r}")
    print(f"  recorded in the spend ledger: total ${budget.spent():.6f} of ${budget.limit_usd:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
