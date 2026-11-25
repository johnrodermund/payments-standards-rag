# Payments standards Q&A assistant

A small retrieval-augmented (RAG) service that answers questions about six ISO 20022 interbank credit transfer messages (**pain.001.001.13**, **pacs.008.001.14**, **pacs.009.001.13**, **pacs.002.001.16**, **pain.002.001.15**, **pacs.004.001.15**) and four interbank flows built from them (serial credit transfer, cover method, bank-to-bank transfer, return), cites the passages it used, and hands off to a human when it is not confident. It is a personal portfolio project built on **public documents only**.

This README states only what has been built and run. What is missing is listed plainly in [Not built yet](#not-built-yet).

## Status

| Area | State |
|---|---|
| Ingestion: parse, chunk, validate, embed, store | Built and run |
| Hybrid retrieval (vector + BM25 + name/intent ranking) | Built and measured |
| Interbank flows (4, hand-written, cited) and `GET /flows` | Built and tested; the flows file awaits a fact review by its owner (see [Flows](#design-decisions)) |
| `POST /ask` with cited answers and human escalation | Built and run live against Claude |
| Audit log (every question, answer and source) | Built and tested; verified on a real database |
| Prompt-injection defence and a test set | Built and run live, results below |
| Golden set, scoring and a CI gate | Built; the retrieval gate runs in CI, the LLM-scored part ran locally only |
| Spend cap and replay mode for LLM calls | Built and tested |
| Drift check (retrieval score, answer length, escalation rate over time) | Built; tested by simulation only, **not yet run on real traffic** |
| Phoenix tracing (OpenTelemetry spans per request) | Built; verified against a local Phoenix container |
| Async ingest endpoint (`POST /ingest`, job status) | Built; verified against the real database |
| Container image | Built and run locally: starts healthy, ingests through `POST /ingest`, answers a recorded question in replay-only mode |
| Serverless AWS stack (CDK: CloudFront, Lambda container, DynamoDB) | Built: template tests pass and the app synthesizes; **not yet deployed** (see [infra/README.md](infra/README.md)). Replaces the Fargate stack below to run at close to no fixed cost |
| Web page (question box, cited answer, escalation state, flows list) | Built and tested (strict CSP, text-only rendering); not yet served publicly |
| In-memory search index (for Lambda) | Built; the retrieval gate on it reproduces the Postgres numbers exactly |
| DynamoDB audit log, monthly spend cap, per-visitor rate limit | Built and tested against a DynamoDB emulator (moto); not yet run in AWS |
| OpenAI generation backend (`gpt-6-luna`) | Built; compared with Claude Haiku on the golden set ([comparison](docs/model-comparison-2026-11-25.md)); Claude stays the default |
| Earlier AWS stack (CDK: ECS Fargate, ALB, RDS PostgreSQL) | Deployed once on 2026-11-21 (us-east-2, replay mode, open to one IP only), exercised, then destroyed; replaced by the serverless stack (see [Deployed run](#deployed-run-2026-11-21)) |
| Public, always-on deployment | **Not done**: there is no live URL yet (see the [recorded demo](evals/demo/DEMO.md)) |
| Recorded run against the deployed service | `evals/demo/deployed_run.json` (replay mode; one question had no recording) |
| Amazon Bedrock generation backend (Converse API) | Built; tested with a stubbed client only, **never run against live Bedrock** (see [Provider](#provider-and-why)) |
| Azure OpenAI backend | **Not built** |

## Architecture

```
ISO 20022 e-Repository (zip, pinned by sha256)
        │  stream-parse (ET.iterparse)
        ▼
  iso20022_messages.json ─┬► chunking ──► validation ──► local embeddings ──► Postgres + pgvector
  interbank_flows.json ───┘                  │ fails the run on bad data      (chunks table)
                                             │ or suspected injection
                                             ▼
question ─► FastAPI /ask ─► hybrid retrieval ─► passage screening ─► Claude (cited answer)
                │              (top 5)            (neutralise)            │
                │                  │                                      ▼
                │           low score? ─────────────► escalate to a human ◄─ no valid [n] citation,
                │                                                             model declined, output
                ▼                                                             policy violation
        audit log (hash-chained, append-only)  ◄──────────────── every outcome
```

- **Language and stack:** Python 3.12, FastAPI, Postgres 16 with pgvector, `fastembed` for embeddings, the `anthropic` SDK for generation, `uv`, `pytest`, `ruff`, GitHub Actions.
- **Source:** the ISO 20022 e-Repository (`20260904_ISO20022_2013_eRepository.zip`, 10.3 MB, sha256 `fb1504abc9df0d445880faedeb0aad208047820a61bd1754bb571c1419ae1c88`). The 118 MB XML inside is stream-parsed; only the two messages above are kept. CI downloads the file and refuses to continue if the hash differs from the one recorded in the committed extract.

## Provider and why

- **Generation:** Claude API, `claude-haiku-5-5`, effort low, 700 max output tokens.
- **Embeddings:** local `BAAI/bge-small-en-v1.5` (quantised, 384 dimensions) through `fastembed`. No API, no per-call cost. The model name and dimension are stored in the database and mixing is refused.
- **Eval judge:** TypeSafe JEV (a separate model family from the generator, so the generator does not grade itself). A Claude judge is also implemented but was not used.
- **Bedrock:** the original plan was AWS Bedrock. On this new AWS account every Bedrock inference quota is 0 (an account-level restriction that needs an Anthropic use-case form and a support case), so **no Bedrock call has ever been made and no Bedrock result is reported here**. The backend itself exists and is selected with `LLM_PROVIDER=bedrock`: it calls the Converse API on `bedrock-runtime` with `maxTokens` always set, adaptive retries, credentials from the standard AWS chain (profile, SSO or an ECS task role, so no keys in code), and the same response cache, replay-only mode, spend cap, citation checks, audit log and tracing as the Claude API path. It is tested with botocore's `Stubber` on a real boto3 client, which validates each request and response against AWS's API model, so the request and response shapes are checked but real behaviour is not. Known differences from the Claude API path: the `effort: low` setting is not sent to Bedrock, so answers from the two providers may differ; the model id is not defaulted (`BEDROCK_GENERATION_MODEL_ID`, an inference-profile id from `aws bedrock list-inference-profiles`); and prices are not guessed (set `LLM_PRICES`, otherwise a deliberately high fallback price makes the spend cap trip early). Embeddings stay local; there is no Bedrock embedding backend.
- **Azure OpenAI:** not built.
- Keys are read from environment variables only (`ANTHROPIC_API_KEY`, `TYPESAFE_API_KEY`).

## Design decisions

**Chunking.** The unit of retrieval is one XML element definition, because that is the unit users ask about ("what is the maximum length of X"). Chunk kinds, with the counts in the current index (4,661 chunks):

- `element`: 3,244. One element: its own name and definition first, then requirement, length or code set, parent and message. Boilerplate goes last so it does not dominate the embedding.
- `children`: 1,177. A short list of an element's direct children, for "what does X contain" questions.
- `message` and `rule`: 234. Message scope and usage sections, split into parts of similar size, and the message's business rules.
- `flow`: 6. One chunk per flow (the serial flow is split in two) plus a `flow:index` overview, built from `data/flows/interbank_flows.json`. See Flows below.
- Ids are stable and readable (for example `pacs.008.001.14#FIToFICstmrCdtTrf/CdtTrfTxInf/CdtrAgt`) and chunks are capped at 1,500 characters. CamelCase names are split into words ("Total Interbank Settlement Amount") because the embedding model does not understand `TtlIntrBkSttlmAmt`.
- Shared components that repeat under several parents are indexed once; each chunk lists the other places it applies. For the original two messages this cut the index from 3,211 to 1,245 chunks and a third off the embedding time, with no measurable accuracy change.

**Retrieval.** Exact vector search (an HNSW index silently lost true nearest neighbours here, so it was removed; a scan of about 4,700 chunks still takes milliseconds) fused with BM25 and three ranking signals: whole-phrase match of an element's path or tag in the question, question intent (a "what fields does X contain" question prefers a children list, a "rule" question prefers rule chunks), and a depth prior for shallower elements. When the question names one message (for example "pacs.008") results are filtered to it, plus the flows that use it. A fourth intent, "flow" ("what is sent after", "who sends", "cover method", "walk me through"), prefers flow chunks and ISO scope/usage text. When the question names no message, pacs.008, pacs.002 and the flows win ties against identical elements in the other four messages (factor 0.6; the dev score is flat from 0.4 to 0.7), and each signal's candidate pool is 100 (it was 50, which lost the pacs.008 copy of shared elements once six messages were indexed). Retrieval p50 is 85.5 ms per question including query embedding, against 77.5 ms for the two-message index on the same machine (2026-11-25). Rebuilding the whole index takes about 9 minutes on a CPU (532 s to embed 3,416 new chunks).

**Flows.** `data/flows/interbank_flows.json` is written for this project, not published by ISO (the e-Repository has no flow data). Each step names who sends which message to whom and cites the chunk ids of the ISO scope, usage or element text that backs it. Where the ISO text does not say something (for example, that a reject happens before settlement), the flow says less and a note records why. The build fails if a cited chunk or message does not exist; whether the text is right is a human review. Flow chunks are labelled "project summary; not ISO text" in their text and their `source`, which `/ask` returns with every cited passage. An ingest includes a flow only when all of its messages are ingested. `GET /flows` and `GET /flows/{id}` list them for a UI, from the committed files, with no database or model call.

**Escalation.** The answer is shown only if it carries a valid `[n]` citation that matches a retrieved passage. Otherwise the system escalates, with one of these reasons: `low_retrieval_confidence`, `all_passages_quarantined`, `insufficient_context`, `no_valid_citation`, `model_declined`, `truncated`, `output_policy_violation`. The retrieval cutoff is the cosine similarity of the top result, `ESCALATION_MIN_SCORE`, default 0.6. On the current question sets, on-topic questions score 0.69 or higher at the top result (median about 0.78) and clearly off-topic ones score 0.41 to 0.54. A question that names something the corpus does define (for example "euro to dollar exchange rate", where an Exchange Rate element exists) scores 0.68 and passes. That is correct behaviour, not a bug, but it means the score cutoff alone cannot catch in-domain questions the corpus does not answer; the model's `INSUFFICIENT_CONTEXT` reply does (see results).

**Audit log.** Every `/ask` call writes one row to a Postgres `audit_log` table before the response returns: question (verbatim), answer, source ids and scores, outcome, escalation reason, model, latency and the blocked answer if the output check stopped one. Triggers block UPDATE, DELETE and TRUNCATE. Each row stores the SHA-256 of the previous row plus its own canonical JSON, serialised with an advisory lock, and `python -m payments_rag.audit.log verify` re-walks the chain. It fails closed: if the log cannot be written the endpoint returns 503, unless `AUDIT_FAIL_OPEN=1`. Limits are in [Risks and controls](#risks-and-controls).

**Async ingest.** `POST /ingest` starts a background re-index and returns `202` with a job id; `GET /ingest/{id}` reports `queued`, `running`, `succeeded` or `failed`, the current stage (`loading`, `chunking`, `storing`) and the result (chunks total, embedded, skipped, deleted). `GET /ingest` lists recent jobs. It reuses the same pipeline as the command line, so unchanged chunks are not re-embedded and a validation failure (including suspected injection in the source text) writes nothing. The endpoint is closed by default:

- It is disabled (503) unless `INGEST_API_TOKEN` is set to at least 16 characters, and every call needs `Authorization: Bearer <token>` (constant-time comparison).
- The caller chooses only from an allowlist: which messages (`pain.001`, `pacs.008`, `pacs.009`, `pacs.002`, `pain.002`, `pacs.004`) and whether to re-chunk the committed extract (default) or re-parse the ISO e-Repository zip configured on the server. The request can never name a path or URL, and unknown fields are rejected with 422. The zip is refused unless its sha256 matches the one pinned in the committed extract.
- One job at a time, enforced with a Postgres advisory lock so it holds across several app replicas; a second request gets 409 with the active job id.
- Job state is in the `ingest_jobs` table. The work runs in a thread of the process that accepted the request; if that process dies the job is left `running` and, after 30 minutes without progress (`INGEST_STALE_SECONDS`), the next request marks it `interrupted` so a new job can start. It is never resumed automatically.

Verified: the real API against the real database ingested the unchanged corpus in about 5 seconds (1,245 chunks skipped, 0 embedded). Limits: a job killed mid-embedding leaves the index partly updated (chunks are written in batches, not one transaction), and rerunning it completes the update; progress is by stage, not by chunk; there is no cancel endpoint and no per-caller rate limit; one shared token means no per-user audit of who started a job.

**Drift check.** `python -m payments_rag.observability.drift check` reads the most recent audit-log rows (default 200) and compares three signals with a stored baseline: top retrieval score, answer length in words, and escalation rate. It uses a two-sample Kolmogorov-Smirnov test for the first two and a two-proportion test for the third, and alarms only when the p-value is below 0.01 **and** the effect is large enough to matter (KS statistic at least 0.25, or an escalation-rate change of at least 10 points). With fewer than 30 observations on either side it reports `insufficient_data` and exits 2; it never reports "ok" on too little evidence. Exit codes: 0 ok, 1 drift, 2 insufficient data. Limits:

- The committed baseline (`evals/drift_baseline.json`) comes from the golden-set run, which is not production traffic (it has 15% unanswerable questions and templated phrasing). Until a baseline is taken from real traffic (`drift baseline --from-audit`), an alarm may reflect that difference and not real drift.
- It has only been tested on simulated data: with no drift it alarmed in at most 6% of 300 simulated checks (about 3% expected); a 0.10 drop in mean top score or a doubled answer length was caught at least 95% of the time; escalations rising from 15% to 45% were caught about 91% of the time at 60 baseline and 100 recent requests. Run on the real audit log today it correctly reports `insufficient_data`, since the log holds only a handful of test rows.
- It detects shifts in behaviour, not wrong answers. Answer quality is the eval's job. Nothing runs it on a schedule yet.

**Tracing.** When `PHOENIX_COLLECTOR_ENDPOINT` is set, each `/ask` request exports OpenTelemetry spans in the OpenInference format that Arize Phoenix displays: a root `POST /ask` span (outcome, escalation reason, top score, request id) with `retrieve` (retrieved passages with ids, scores and text), `screen` (what the injection and provenance checks removed), `generate` (model, token counts, cached or not, stop reason) and `output_check` spans. The audit-log row for each request stores the `trace_id`, so a row can be opened as its trace. Tracing is off unless configured, uses its own tracer provider, and every tracing call is guarded, so a collector outage cannot break a request (tested with an exporter that always raises). Verified end to end: the real API with the local Phoenix container received 2 traces (8 spans) in the `payments-assistant` project, with no failed exports. Notes:

- **Question, passage and answer text is attached to spans** so traces are readable. Set `TRACE_CONTENT=0` to send ids, scores, counts and timings only. The audit log already holds the same text verbatim; a collector is a second copy, so do not send traces to a collector you do not control.
- **Phoenix 21+ requires authentication by default in Docker**, which first made the collector reject our spans with 401. The compose file turns authentication off and binds port 6006 to localhost, for local development only. For anything shared or deployed, keep authentication on and give the app a key through `PHOENIX_API_KEY` (sent as a bearer token; tested against a local HTTP server, not against an authenticated Phoenix).
- Only `/ask` is traced. Ingestion, the evals and the embedding step are not.

**LLM cost controls.** Every real call is cached on disk (`.cache/llm`, keyed by the hash of the full request), so identical requests are free. `LLM_REPLAY_ONLY=1` makes no network calls at all and returns 503 on a cache miss. A project-wide spend cap (`LLM_BUDGET_USD`, default $0.50) is enforced from a ledger file: a call that could take the total past the cap is refused before it is sent.

## Results

All LLM runs below used `claude-haiku-5-5` for generation on 2026-11-20. **They were run locally**; CI runs the retrieval gate on every push but skips the LLM-scored steps unless the API keys are configured as repository secrets, and they are not configured. So the CI badge does not vouch for the numbers in this section.

### Six messages and flows (2026-11-25)

Same method, `claude-haiku-5-5` generation and the JEV judge, run locally on the six-message index. The golden set now has 65 questions: the 52 above (`original` cohort; one not-in-corpus question changed from pacs.009, now indexed, to camt.056) and 13 new (`flows-v1`: 10 answerable flow and new-message questions, 3 about direct debits, recalls and status requests, which are not indexed).

| Metric | Measured | CI minimum |
|---|---|---|
| Retrieval hit@1 / hit@5 / MRR, all answerable | 0.852 / 0.907 / 0.880 | 0.76 / 0.80 / 0.79 |
| Retrieval hit@1 = hit@5 = MRR, original 44 only | 0.886 (unchanged) | 0.80 |
| Answerable questions actually answered | 0.944 | 0.85 |
| Unanswerable questions escalated (all 11) | 1.00 | 0.85 |
| Faithfulness (JEV, 51 answers) | 0.882 | 0.79 |
| Answer relevancy (JEV, 51 answers) | 0.876 | 0.78 |

- The original questions have their own CI gate (`retrieval_original.*` at 0.80), so a drop there cannot hide behind the new questions.
- All 10 new answerable questions were answered; the 3 unanswered are original questions, as before.
- Retrieval sets, against the two-message index rebuilt on the same machine:

| Set | Cohort | n | hit@1 | hit@5 | MRR | two-message index hit@1 / hit@5 / MRR |
|---|---|---|---|---|---|---|
| Dev | original | 26 | 0.885 | 0.962 | 0.917 | 0.846 / 0.962 / 0.897 |
| Dev | flows-v1 | 23 | 0.826 | 1.000 | 0.913 | n/a |
| Holdout | original | 15 | 0.800 | 1.000 | 0.863 | 0.800 / 1.000 / 0.867 |
| Holdout | flows-v1 | 7 | 0.429 | 0.714 | 0.529 | n/a |

- **Weak spot:** flow questions without flow wording ("what happens when a settled transfer has to be undone?") are read as field questions and miss the flows. This shows on the new holdout and was not tuned against.
- Prompt size per question is unchanged: the top-5 passages for the original golden questions average 2,667 characters, against 2,755 before; the system prompt grew by 8 words.

### Golden set (52 questions, 2026-11-20, two messages)

44 answerable questions (max length, requirement, definition, code set, children, message scope, business rule) and 8 unanswerable (3 off-topic, 5 about things outside the corpus, such as FedNow fees). Reference answers and reference chunk ids are derived from the parsed ISO data.

| Metric | Measured | CI minimum |
|---|---|---|
| Retrieval hit@1 = hit@5 = MRR | 0.886 | 0.80 |
| Answerable questions that pass the retrieval cutoff | 1.00 | 0.90 |
| Off-topic questions escalated by the cutoff | 1.00 | 0.99 |
| Answerable questions actually answered | 0.932 | 0.85 |
| Unanswerable questions escalated (all 8, after generation) | 1.00 | 0.85 |
| Faithfulness (JEV judge, mean of 41 answers) | 0.891 | 0.80 |
| Answer relevancy (JEV judge, mean of 41 answers) | 0.895 | 0.80 |

- The 3 answerable questions that were not answered are 3 of the 5 retrieval misses; the model replied that the context did not contain the answer instead of inventing one. The 5 misses are "codes allowed" questions that land on rule chunks, a "make up" phrasing that does not trigger the children-list intent, and one ambiguous "Agent" question. These are known and unfixed. They were deliberately not tuned against, so the gate stays meaningful.
- The retrieval cutoff alone escalated only 3 of the 5 not-in-corpus questions (0.6); the model's own refusal covered the rest.
- Minimums are the measured value minus about 0.09, from this single run. One question moves a retrieval metric by 0.023. Re-measure after any retrieval, chunking or prompt change.
- **Judge sanity check (12 cases):** JEV scored the real answers 0.91 faithful and 0.92 relevant, answers swapped between questions 0.29 faithful and 0.15 relevant, and a real answer with an invented false claim appended 0.20 faithful (relevancy barely moves for that case, 0.73, as expected).
- **The golden set is easier than real questions.** It is templated from exact element names. On an earlier, more natural dev set, hybrid retrieval before the rewrite scored hit@5 of 0.35.

### Retrieval, before and after the rewrite

Hybrid mode, same embedding model, same question sets, original system rebuilt in a throwaway database for the "before" numbers.

| Set | hit@5 before | hit@5 after | MRR before | MRR after |
|---|---|---|---|---|
| Dev (26 questions; tuned on) | 0.35 | 0.96 | 0.33 | 0.90 |
| Holdout (15 questions; written after) | 0.67 | 1.00 | 0.52 | 0.87 |

The largest single effects: removing the lossy HNSW index, matching element names in the question, and routing by question intent. The holdout numbers were printed once early in diagnosis, and tuning afterwards used the dev set only. Cross-encoder rerankers and a larger embedding model (`bge-base`) were tried and rejected because they hurt or barely helped.

### Prompt-injection test set

36 cases: 32 attacks in 15 categories (override, system-message spoofing, tag breakout, exfiltration through a link, prompt leak, tool-call mimicry, encoded payloads, hidden characters, multilingual, citation spoofing, authority claims, denial of service, false facts, paraphrase evasion) and 4 benign controls. Each attack plants a passage in the retrieved context carrying a unique canary; success is decided by the canary, not by a person reading the answer. Outcomes are compromised, safe-answered or safe-escalated.

| Set | Guard | Attack success | Attacks escalated | Controls correct |
|---|---|---|---|---|
| Original 32 attacks | none | 9.4% (3) | 6% | 100% |
| Original 32 attacks | first guard (whole-passage quarantine) | 12.5% (4) | 47% | 100% |
| Original 32 attacks | **reworked guard** | **6.2% (2, both false facts)** | **0%** | 100% |
| Holdout of 21 new attacks, written after the guard design was frozen | none | 9.5% (2) | 5% | 100% |
| Holdout of 21 new attacks | **reworked guard** | **9.5% (2)** | 5% | 100% |

What this does and does not show:

- **The first guard was worse than no guard on this measure.** It quarantined whole passages, which let an attacker force refusals ("reply INSUFFICIENT_CONTEXT" in one passage emptied the context). The rework removes only the flagged sentences and escalates only when nothing usable is left, which brought attack escalations from 47% to 0%.
- **On the new, out-of-sample attacks the guard made no measured difference** to how often Claude was compromised. It screened 43% of them, but the model resisted most injections on its own. The two that succeeded in both runs were a paraphrased "override" instruction that was not screened, and a plausible false fact.
- **The model itself is the main defence here.** `claude-haiku-5-5` ignored most of the injected instructions with the guard off. A different or smaller model could behave differently, and the guard has not been shown to change outcomes for this one beyond the cases above.
- **Offline, with stand-in models (not evidence about Claude):** the reworked guard flags 0 of 1,245 real chunks and 0 of the benign controls. Out of sample, it screens roughly half of novel attacks, reliably the structural ones (tags, links, tool calls, hidden characters, homoglyphs, encodings) and mostly not semantic paraphrases or new override wordings.
- The sample is small (32 and 21 attacks, one model, one run, test cases written by the same author as the guard). Percentages are not tight and this is not an independent red team.

## Risks and controls

| Risk | Control | Limit, stated plainly |
|---|---|---|
| Prompt injection through retrieved text | Passage screening that removes flagged sentences, strips hidden characters and decodes obfuscation; passages are escaped and wrapped as data; the system prompt tells the model to treat them as untrusted; an output check for links, tool-call shapes, role markers, leaked system prompt and tokens that appear in no passage. Also checked at ingest, where a flagged chunk fails validation. | Heuristic. Misses paraphrased semantic attacks (roughly half of novel ones offline). The corpus is trusted ISO text written in the third person, so rules tuned on it may misfire on another corpus. |
| Wrong or invented answers | Answers must cite a retrieved passage `[n]`; the model must reply `INSUFFICIENT_CONTEXT` when the passages lack the answer; escalation below the retrieval cutoff. | A citation proves the passage was shown, not that the answer is correct. Faithfulness of 0.89 means some answers are not fully supported. |
| A false statement inside a passage | A provenance check refuses any served chunk whose text no longer matches the hash computed from the committed extract (`PROVENANCE_CHECK=0` disables it). | Catches tampering after extraction only. It trusts the extract and cannot judge truth. A plausible false fact in a passage succeeded in every run. |
| Silent confident error on a question the corpus does not cover | Escalation on a low top score, on `INSUFFICIENT_CONTEXT`, and on a missing citation. | The score cutoff cannot separate in-domain questions the corpus does not answer; that relies on the model declining. All 8 unanswerable golden questions escalated, but that is 8 questions. |
| Tampering with the record of what was answered | Append-only audit table (triggers), SHA-256 hash chain, `verify` command, fail-closed writes. | A database superuser can bypass the triggers. Deleting the newest rows cannot be detected without an external anchor of the latest hash. |
| Personal data in questions | The service is built for public standards questions. | Questions are stored verbatim in the audit log. Do not ask it anything private. There is no PII redaction. |
| Runaway API cost | Disk cache, replay mode, and a hard spend cap enforced from a ledger. | A local safeguard, not a billing control: it does not stop another script using the same key. |
| Secrets | Keys come from environment variables only; none are committed; `.env` is git-ignored. | |
| Data licensing | Only the ISO 20022 e-Repository is ingested, under its terms of use (see below). FedNow documents were rejected because their terms prohibit reproduction. | |

## Run locally

Prerequisites: Docker Desktop, [uv](https://docs.astral.sh/uv/), Python 3.12.

```
docker compose up -d db
uv sync --group eval
```

Download the ISO source and verify it against the hash recorded in the committed extract (the site rejects requests without a browser user agent):

```
mkdir -p data/raw/iso20022
curl --fail --location --user-agent "Mozilla/5.0" --output data/raw/iso20022/20260904_ISO20022_2013_eRepository.zip "https://www.iso20022.org/sites/default/files/media/file/20260904_ISO20022_2013_eRepository.zip"
```

Build the index (extract, chunk, validate, embed, store):

```
uv run python -m payments_rag.ingestion.iso20022_repository data/raw/iso20022/20260904_ISO20022_2013_eRepository.zip data/processed/iso20022_messages.json --message pain.001 --message pacs.008 --message pacs.009 --message pacs.002 --message pain.002 --message pacs.004
uv run python -m payments_rag.ingestion.chunking data/processed/iso20022_messages.json data/processed/chunks.jsonl --flows data/flows/interbank_flows.json
uv run python -m payments_rag.ingestion.validation data/processed/chunks.jsonl --expect pain.001 --expect pacs.008 --expect pacs.009 --expect pacs.002 --expect pain.002 --expect pacs.004
uv run python -m payments_rag.ingestion.pipeline data/processed/chunks.jsonl --expect pain.001 --expect pacs.008 --expect pacs.009 --expect pacs.002 --expect pain.002 --expect pacs.004
```

Run the checks that need no API key (tests, lint, retrieval gate):

```
uv run pytest -q
uv run ruff check .
uv run python -m payments_rag.evals.rag_eval --retrieval-only --out evals/results/retrieval_only.json
uv run python -m payments_rag.evals.gate evals/results/retrieval_only.json evals/thresholds.json --only retrieval,confidence
```

Re-index through the API (optional; disabled unless a token is set):

```
export INGEST_API_TOKEN=<at least 16 random characters>
curl -s -X POST localhost:8000/ingest -H "authorization: Bearer $INGEST_API_TOKEN" -H "content-type: application/json" -d '{}'
curl -s localhost:8000/ingest/<job_id> -H "authorization: Bearer $INGEST_API_TOKEN"
```

To use Bedrock instead of the Claude API (not yet exercised live; the check below makes one billed call):

```
export LLM_PROVIDER=bedrock AWS_REGION=<region> BEDROCK_GENERATION_MODEL_ID=<inference-profile id>
export LLM_PRICES='{"<that id>": [<input USD per M tokens>, <output USD per M tokens>]}'
uv run python -m payments_rag.llm.bedrock_check
```

Serve and ask (needs `ANTHROPIC_API_KEY` in the environment; every call is capped by `LLM_BUDGET_USD`):

```
uv run uvicorn payments_rag.api.main:app --port 8000
curl -s localhost:8000/ask -H "content-type: application/json" -d '{"question": "What is the maximum length of the Message Identification in pacs.008?"}'
uv run python -m payments_rag.audit.log verify
```

Live evals (these spend API credit, so check the cap first):

```
uv run python -m payments_rag.evals.rag_eval --judge jev --out evals/results/latest.json
uv run python -m payments_rag.security.injection_eval evals/injection_set.jsonl --guard on --out evals/injection/results.json
```

To see traces, start Phoenix (`docker compose up -d phoenix`), run the API with `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006` set, ask a question and open http://localhost:6006.

The compose `app` service was only verified to serve `/health` at the scaffold stage; running the API through `uv` as above is the path that has been exercised end to end.

## Tests and CI

309 tests. CI (`.github/workflows/ci.yml`) runs lint and tests, then downloads and hash-verifies the ISO source, builds the index and runs the retrieval and confidence gate on every push. The LLM-scored eval (faithfulness, relevancy, behaviour) runs only when `ANTHROPIC_API_KEY` and the judge key are repository secrets, and is skipped with a warning otherwise. The gate fails the build if any metric is below its minimum in `evals/thresholds.json`. A deliberately reversed-ranking retriever failed the retrieval gate (MRR 0.23), which is why hit@1 and MRR are gated and not hit@5 alone.

## Deployed run (2026-11-21)

**Demo:** there is no live URL (the stack was torn down to stop costs). The [recorded demo transcript](evals/demo/DEMO.md) shows real answers, sources and escalations from the deployed service, and the [raw responses](evals/demo/deployed_run.json) are in the repository.

The stack in `infra/` was deployed once to AWS (us-east-2), used for about 25 minutes, and destroyed; nothing from it is running. It ran in **replay mode**, so the service made no model calls: answers were replayed from earlier recordings of live Claude runs. The load balancer accepted traffic only from the operator's IP.

| Measurement | Result |
|---|---|
| Ingest on Fargate (1 vCPU, 2 GB), `POST /ingest` from the committed extract | 1,245 chunks embedded in 4 min 41 s; pgvector worked on RDS PostgreSQL 16 |
| Golden set through the deployed `/ask` (44 answerable) | 41 answered, 3 escalated (`insufficient_context`), no errors; hit@5 39/44 (0.886, same as the local run) |
| Questions that should escalate (8 golden unanswerable + 12 off-topic) | 19 of 20 escalated; 0 answered; 1 returned 503 because replay mode had no recording for it (a euro/dollar rate question) |

The full responses are in `evals/demo/deployed_run.json`, produced by `evals/demo/record_run.py`. What this does not show: live model calls from AWS (neither Claude API mode nor Bedrock was run there), behaviour under a task restart, or the audit log's contents (a write failure would have made `/ask` return 503, and none did, but the table was not inspected). The injection test set is passage-level and is not reproducible through `/ask`, so it was not part of this run.

**Cost.** Billing data had not posted when this was written, so there is no billed figure yet. From the runtime and list prices (load balancer, one Fargate task, one `db.t4g.micro`, public IPv4 addresses, two secrets, all for well under an hour) the estimate is under $0.10, drawn from AWS Free Plan credits. Left behind: the CDK bootstrap bucket and image repository (a 338 MB image, a few cents a month) and one automated database snapshot that expires by itself.

## Not built yet

- **Traces for ingestion and the evals.** Only `/ask` is traced; tracing against an authenticated Phoenix is untested.
- **Scheduled drift runs and a real-traffic baseline.** The drift check exists but nothing runs it on a schedule, and it has no real traffic to check yet.
- **A standing cloud deployment.** The stack was deployed and verified once, then destroyed (see above); there is no public URL. Untested on AWS: Claude API mode, Bedrock mode, HTTPS, task restarts.
- **Live Bedrock verification and results.** The backend is built but unproven against the real service until the account quotas are fixed; `python -m payments_rag.llm.bedrock_check` makes one tiny billed call to confirm it. A golden-set run on both providers is the planned comparison and does not exist yet. Also not built: Azure OpenAI and Bedrock embeddings.
- **Other sources.** A BIS report on harmonised ISO 20022 data requirements was approved but not ingested; FedNow and CBPR+ material is out of scope for licensing reasons (see `data/SOURCES.md`).
- **Known quality gaps:** the 5 retrieval misses above, flow questions without flow wording, and the guard's paraphrase evasion.
- **Re-recorded replay answers.** The system prompt changed with the six messages, so answers recorded before 2026-11-25 (`.cache/llm`, used by replay-only mode and the recorded demo) no longer match and need re-recording.
- **More flows and messages:** recalls (camt.056/camt.029), statements and notifications (camt.053/camt.054), direct debits and status requests are not indexed.
- **Not independent:** the golden set and the injection sets were written by the same author as the system.

## Sources and notices

This is a non-commercial personal project and is **not the official ISO 20022 website**. The sole source of up-to-date ISO 20022 message standards and the repository is https://www.iso20022.org/.

`data/processed/iso20022_messages.json` is derived from the ISO 20022 e-Repository (pain.001.001.13, pacs.008.001.14, pacs.009.001.13, pacs.002.001.16, pain.002.001.15 and pacs.004.001.15 definitions), reproduced under the [ISO 20022 terms of use](https://www.iso20022.org/terms-use). The source file's SHA-256 is recorded in the JSON. Answers quote short extracts and cite the source. `data/flows/interbank_flows.json` is written for this project from the ISO definitions it cites; it is not an ISO publication. Other sources and their licence notes are listed in [data/SOURCES.md](data/SOURCES.md).
