# AWS deployment (CDK, Python): serverless

One stack, `PaymentsRag`, in `us-east-2`. No VPC, no servers, no database to run.

## What runs where

```
browser ── HTTPS ──► CloudFront ──(Origin Access Control, SigV4)──► Lambda Function URL (AWS_IAM auth)
                                                                      │  container image (Dockerfile.lambda), x86_64, 3 GB
                                                                      │  search index baked into the image
                                                                      ├─ DynamoDB: AuditEntries (add-only), AuditHead, Spend, RateLimits
                                                                      └─ SSM Parameter Store: model API key (SecureString)
EventBridge, daily ──► AuditVerify function ──► re-walks the audit hash chain
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
  - The API role may only `PutItem` on `AuditEntries` and `GetItem`/`UpdateItem` on `AuditHead`, so it cannot change or delete an entry.
  - Both audit tables, and `Spend`, have deletion protection and point-in-time recovery, and are kept if the stack is deleted.
- **Spend cap:** `LLM_BUDGET_USD`, $5 per calendar month by default, enforced atomically in the `Spend` table. At 80% the `SpendNearCap` alarm emails you; at 100% visitors see "monthly question budget reached".
- **Rate limit:** 10 questions a minute and 100 a day per visitor, keyed on CloudFront's viewer address (hashed).

## Settings (`-c key=value`)

| Key | Default | Notes |
|---|---|---|
| `llm_provider` | `anthropic` | or `openai` (see the comparison in `docs/model-comparison-2026-11-25.md`) |
| `model` | `claude-haiku-5-5` / `gpt-6-luna` | per provider |
| `api_key_parameter` | `/payments-rag/anthropic-api-key` / `/payments-rag/openai-api-key` | SSM SecureString name |
| `llm_budget_usd` | `5` | monthly cap; the alarm fires at 80% |
| `rate_per_minute`, `rate_per_day` | `10`, `100` | per visitor |
| `memory` | `3008` | MB; more memory also means more CPU for the embedding model |
| `reserved_concurrency` | unset | optional hard ceiling on parallel invocations. New accounts often have a total of only 10, and AWS refuses a reservation that leaves fewer than 10 unreserved, so leave it unset unless your quota is higher |
| `alarm_email` | unset | pass at deploy time; never write it into `cdk.json` |

## Before the first deploy (John, in the consoles)

1. **AWS Billing:**
   - Check the plan. On the Free plan, the account closes after 6 months or when the credits run out, and its data is erased 90 days after that. Upgrade before then to keep it.
   - Create a **Budget** of about $20/month with email alerts.
2. **Anthropic Console:**
   - Create a workspace for this service, an API key in it, and a monthly spend limit (for example $10, above the app's $5 cap, so the app's own message shows first).
3. **AWS Systems Manager → Parameter Store** (us-east-2):
   - Create `/payments-rag/anthropic-api-key`, type **SecureString**, key **aws/ssm**, value: that API key.
   - Paste it in the console, not in a shell command, so it never lands in shell history.
   - For OpenAI, create `/payments-rag/openai-api-key` instead.
4. **Sign in:** `aws login` with the `payments-rag` profile, and keep Docker Desktop running: the deploy builds the image.

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
$env:AUDIT_TABLE = "<AuditEntriesTable output>"; $env:AUDIT_HEAD_TABLE = "<AuditHeadTable output>"; $env:AWS_PROFILE = "payments-rag"; $env:AWS_REGION = "us-east-2"
uv run python -m payments_rag.audit.log verify
```

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
