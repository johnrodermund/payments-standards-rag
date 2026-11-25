# Serverless go-live (Lambda + CloudFront + DynamoDB)

**Goal:** A public, always-on HTTPS address with the web page, cited answers and escalation, costing close to nothing while the AWS account is on the Free plan. Replaces the Fargate + ALB + RDS design in the go-live plan on `plan/go-live` (about $93/month).

**Decided by John (2026-11-25):** serverless, all on AWS, an agent writes everything (the spend cap and rate limiter are no longer hand-written), start straight away. John will create a dedicated Anthropic API key with its own spend limit in the Anthropic Console as the backstop.

## Shape

```
browser ── HTTPS ──► CloudFront ──(OAC, SigV4)──► Lambda Function URL (AWS_IAM auth)
                                                      │  container image, x86_64, 2 GB, reserved concurrency 5
                                                      │  FastAPI app via Mangum
                                                      ├─ search index: file baked into the image (built at docker build)
                                                      ├─ DynamoDB: audit entries (put-only), audit head, spend (monthly), rate limits (TTL)
                                                      └─ SSM SecureString: model API key (read once per cold start)
EventBridge (daily) ──► same image, verify handler ──► audit chain check (fails = Errors alarm)
CloudWatch alarms ──► SNS email
```

## Calls (Claude's, with reasons)

- **x86_64, not ARM:** building an ARM image on an x86 Windows PC runs under emulation, and the image build embeds about 4,700 chunks. Same price inside the free allowance.
- **Index built during `docker build`** from the committed extract and flows file. No binary is committed and it is reproducible. The Postgres path stays for local development and CI (`RETRIEVAL_BACKEND=postgres`, the default); Lambda uses `RETRIEVAL_BACKEND=memory`.
- **Audit log on DynamoDB:**
  - Two tables, written in one transaction: entries (`pk="chain"`, `sk=seq`) and head (`pk="head"`: last seq and hash).
  - The Lambda role may only `PutItem` on entries and `GetItem`/`UpdateItem` on head, so it cannot edit or delete entries.
  - Because the head records the last seq, deleting entries from the tail is also detected, which the Postgres version could not do.
  - Point-in-time recovery and deletion protection are on; the tables are retained on stack delete.
- **Spend cap on DynamoDB:**
  - One item per UTC month holds a `committed` amount (settled spend plus outstanding reservations).
  - Reserve the worst-case cost with a conditional update (`committed <= limit - worst`); settle on success (adjust to the actual cost); release on failure.
  - The `Generator` gains reserve/settle/release and the file `Budget` keeps working.
  - Callers get a generic "monthly question budget reached" message, with no figures or paths.
  - Spend is emitted as a CloudWatch EMF metric for the 80% alarm.
- **Rate limiter on DynamoDB:**
  - Per client, keyed on a SHA-256 of the `CloudFront-Viewer-Address` IP, so raw IPs are not stored and a forged `X-Forwarded-For` changes nothing.
  - Per-minute and per-day counters with TTL; 10/min and 100/day by default. Over the limit returns 429 with `Retry-After`.
  - With no table configured (local), there is no limit.
- **Function URL with AWS_IAM + CloudFront OAC:** the URL cannot be called directly. For POST, CloudFront needs the body hash in `x-amz-content-sha256`; the page computes it, and API callers must send it too.
- **API key in SSM Parameter Store** (SecureString, standard tier: free) instead of Secrets Manager ($0.40/month). It is read once per cold start into the environment.
- **Lambda's filesystem is read-only** apart from `/tmp`: the response cache goes to `/tmp/llm`, the embedding model is baked into the image and loaded offline.
- **Page:**
  - One HTML file served at `/`: question box, cited answer with source ids, an "escalated to a human" state, the flows list from `GET /flows`, and a notice that questions are logged verbatim.
  - Text is inserted with `textContent` only, never as HTML; strict CSP.
  - It does not promise a reply to escalations.
- **Ingest endpoint:** stays disabled in Lambda (no token). Content updates mean rebuilding the image.
- **Not in this change:** CI deploy (OIDC), custom domain, drift-check schedule, WAF.

## Tasks

1. **In-memory retriever:**
   - `retrieval/memory_store.py`: `build_index(chunks, embedder, path)`, `MemoryIndex.load(path)`, `MemoryIndex.hybrid_search(query, vector, k, kinds, message_prefix)` with the same fusion as `store.hybrid_search`. A CLI builds it from extract + flows.
   - `service.build_retriever` picks the backend by `RETRIEVAL_BACKEND`.
   - Tests on a small corpus; a retrieval-only eval on the memory backend must meet the same gate as Postgres.
2. **DynamoDB audit log** (`audit/dynamo.py`): `record(entry) -> hash` and `verify()`. Tests with moto: chain, concurrent conflict retry, edit detected, tail deletion detected.
3. **Spend cap:**
   - `llm/spend_cap.py` `DynamoSpendCap` with `reserve`, `settle` and `release`, plus `Budget` adapters.
   - The `Generator` reserves, settles, and releases on failure.
   - The `/ask` budget error is generic.
   - Tests: limit, concurrent reservations never pass the limit, release on failure, month rollover.
4. **Rate limiter:** `api/ratelimit.py` with `client_key(request)` and `DynamoRateLimiter.hit(key)`, as a dependency on `/ask` that returns 429 with `Retry-After`. Tests: keying (a forged XFF is ignored), minute and day limits, no table means no limit.
5. **Page and Lambda entry:**
   - `api/static/index.html` served at `/` with a CSP header.
   - `api/lambda_handler.py` (Mangum, SSM key loading, a verify handler).
   - `Dockerfile.lambda`.
   - Tests: page served with CSP, no `innerHTML` in the page script, handler round trip through a Function URL v2 event.
6. **CDK stack:** rewrite `infra/stack.py` as Lambda + Function URL + CloudFront OAC + DynamoDB + SSM read + EventBridge verify + alarms (Lambda errors and throttles, verify errors, spend ≥ 80%) to SNS email. `alarm_email` is passed at deploy time and never committed. Template tests replace the Fargate ones.
7. **Docs:**
   - README status, architecture and cost.
   - `infra/README.md` deploy runbook, including John's console steps: SSM parameter, Anthropic key with a limit, AWS Budget, alarm email.
   - Then a PR. The deploy itself waits for John's go-ahead and AWS sign-in.
