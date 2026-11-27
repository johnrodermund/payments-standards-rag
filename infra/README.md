# AWS deployment (CDK, Python): serverless

One stack, `PaymentsRag`, in `us-east-2`. No VPC, no servers, no database to run.

## What runs where

```
browser ── HTTPS ──► CloudFront ──(Origin Access Control, SigV4)──► Lambda Function URL (AWS_IAM auth)
                                                                      │  container image (Dockerfile.lambda), x86_64, 3 GB
                                                                      │  search index baked into the image
                                                                      ├─ DynamoDB: AuditEntries, AuditHead, AuditAnchor, Spend, RateLimits
                                                                      └─ SSM Parameter Store: model API key (SecureString)
EventBridge, daily ──► AuditVerify function ──► re-walks the audit hash chain, moves the anchor
CloudWatch alarms (API crashes, Lambda errors, throttles, audit verify failed, spend ≥ 80% of cap) ──► SNS topic ──► email
```

- **CloudFront is the only way in.** The Function URL rejects unsigned calls. For POST, CloudFront has to sign the body, so the caller must send its SHA-256 in `x-amz-content-sha256`. The web page does this.
  ```
  body='{"question":"What is the creditor agent in pacs.008?"}'
  curl -s https://<cloudfront-domain>/ask -H 'content-type: application/json' \
       -H "x-amz-content-sha256: $(printf %s "$body" | sha256sum | cut -d' ' -f1)" -d "$body"
  ```
- **Content updates rebuild the image.** `POST /ingest` is disabled here. Change the extract or flows file, then deploy again.
- **Audit log:**
  - Every question is a hash-chained entry in `AuditEntries`; `AuditHead` holds the chain's length and last hash, so deleted entries are caught too.
  - The daily verify writes an anchor (seq and hash) to `AuditAnchor`, which only the verify function can write, and checks that yesterday's anchor is still in the chain.
  - What this does not stop: IAM cannot force a write to be conditional, so anyone holding the API function's role could rewrite entries added since the last verify. That window is at most a day.
  - The three audit tables, and `Spend`, have deletion protection and point-in-time recovery, and are kept if the stack is deleted.
- **Spend cap:** `LLM_BUDGET_USD`, $5 per calendar month by default, enforced atomically in the `Spend` table. At 80% the `SpendNearCap` alarm emails you; at 100% visitors see "monthly question budget reached".
- **Tracing:** with the LangSmith key in place, each question is one LangSmith trace: the request, retrieval (passages and scores), screening, the model call (model, tokens, cost) and the output check. Spans are sent before each invocation ends, because Lambda freezes the function after it responds. Full text goes to LangSmith (a third party), as decided on 2026-11-27; set `TRACE_CONTENT=0` to send only ids, scores, counts and timings.
- **Rate limit:** 10 questions a minute and 100 a day per visitor, keyed on CloudFront's viewer address (hashed). IPv6 visitors are counted per /64, because one client usually holds a whole /64.
- **Model call timeout:** 12 s per attempt and one retry, so a slow provider fails inside Lambda's 29 s limit and the question is still logged and its budget reservation returned.

## Settings (`-c key=value`)

| Key | Default | Notes |
|---|---|---|
| `llm_provider` | `anthropic` in code; **`openai` in `cdk.json`** (the live deployment) | `openai` runs GPT-6 Luna (see `docs/model-comparison-2026-11-25.md`) |
| `model` | `claude-haiku-5-5` / `gpt-6-luna` | per provider |
| `api_key_parameter` | `/payments-rag/anthropic-api-key` / `/payments-rag/openai-api-key` | SSM SecureString name |
| `llm_budget_usd` | `5` | monthly cap; the alarm fires at 80% |
| `rate_per_minute`, `rate_per_day` | `10`, `100` | per visitor |
| `memory` | `3008` | MB; more memory also means more CPU for the embedding model |
| `reserved_concurrency` | unset | optional hard ceiling on parallel invocations. New accounts often have a total of only 10, and AWS refuses a reservation that leaves fewer than 10 unreserved, so leave it unset unless your quota is higher |
| `alarm_email` | unset | pass at deploy time; never write it into `cdk.json` |
| `langsmith_key_parameter` | unset in code; `/payments-rag/langsmith-api-key` in `cdk.json` | SSM SecureString holding a LangSmith API key. Set: every question is traced to LangSmith **with its full text** (question, passages, answer). If the parameter is missing, the app runs with tracing off |
| `langsmith_project` | `payments-assistant` | LangSmith project the traces land in |

## Before the first deploy (John, in the consoles)

1. **AWS Billing:**
   - Check the plan. On the Free plan, the account closes after 6 months or when the credits run out, and its data is erased 90 days after that. Upgrade before then to keep it.
   - Create a **Budget** of about $20/month with email alerts.
2. **Model provider console** (OpenAI for the live deployment):
   - Create a project (OpenAI) or workspace (Anthropic) for this service, an API key in it, and a monthly spend limit (for example $10, above the app's $5 cap, so the app's own message shows first).
3. **AWS Systems Manager → Parameter Store** (us-east-2):
   - Create `/payments-rag/openai-api-key` (the live provider), type **SecureString**, key source **My current account**, key **alias/aws/ssm**, data type **text**, value: the API key only.
   - Check the prefix: OpenAI keys start with `sk-proj-`, Anthropic keys with `sk-ant-api03-`. The wrong one fails with `invalid x-api-key`.
   - Paste it in the console, not in a shell command, so it never lands in shell history.
   - For Anthropic (`-c llm_provider=anthropic`), create `/payments-rag/anthropic-api-key` instead.
4. **LangSmith (optional):** create `/payments-rag/langsmith-api-key` the same way, with a LangSmith API key (starts with `lsv2_`). Leave it out to run without tracing.
5. **Sign in:** `aws login` with the `payments-rag` profile, and keep Docker Desktop running: the deploy builds the image.

## Commands

```powershell
cd infra
uv run pytest -q
npx -y aws-cdk@2 bootstrap --profile payments-rag
npx -y aws-cdk@2 diff --profile payments-rag -c alarm_email=you@example.com
npx -y aws-cdk@2 deploy --profile payments-rag -c alarm_email=you@example.com
```

Confirm the SNS subscription email. Then check that the `Url` output serves the page and answers a question.

To check the audit log from your machine:

```powershell
$env:AUDIT_TABLE = "<AuditEntriesTable output>"; $env:AUDIT_HEAD_TABLE = "<AuditHeadTable output>"; $env:AUDIT_ANCHOR_TABLE = "<AuditAnchorTable output>"; $env:AWS_PROFILE = "payments-rag"; $env:AWS_REGION = "us-east-2"
uv run python -m payments_rag.audit.log verify
```

This checks the chain and the anchor but never moves the anchor; only the daily function does.

## Cost (estimate, list prices)

| Item | $/month at low traffic |
|---|---|
| Lambda, DynamoDB on demand, CloudFront, CloudWatch (4 alarms), SNS, EventBridge | about $0–1 |
| ECR image storage (about 1–1.5 GB) | about $0.10–0.15 |
| SSM Parameter Store (standard parameter) | $0 |
| Model (Claude Haiku 5.5 or GPT-6 Luna) | at most `llm_budget_usd` ($5) |

These are estimates, not bills, and how much the Free plan's credits cover was not checked service by service. Read the first month's Cost Explorer figure.

## Not covered

- CI deploy (GitHub Actions with OIDC), a custom domain, a WAF, and a scheduled drift check.
- CloudFront 5xx alarms: CloudFront metrics live in us-east-1, so they would need a second stack.
- Cold starts: the first question after a quiet spell loads the embedding model and the index, which takes a few seconds.
