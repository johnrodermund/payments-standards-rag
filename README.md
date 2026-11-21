# Payments standards Q&A assistant

A small retrieval-augmented (RAG) service that answers questions about ISO 20022 payment message definitions (**pacs.008.001.14** and **pacs.002.001.16**), cites the passages it used, and hands off to a human when it is not confident. It is a personal portfolio project built on **public documents only**.

This README states only what has been built and run. What is missing is listed plainly in [Not built yet](#not-built-yet).

## Status

| Area | State |
|---|---|
| Ingestion: parse, chunk, validate, embed, store | Built and run |
| Hybrid retrieval (vector + BM25 + name/intent ranking) | Built and measured |
| `POST /ask` with cited answers and human escalation | Built and run live against Claude |
| Audit log (every question, answer and source) | Built and tested; verified on a real database |
| Prompt-injection defence and a test set | Built and run live, results below |
| Golden set, scoring and a CI gate | Built; the retrieval gate runs in CI, the LLM-scored part ran locally only |
| Spend cap and replay mode for LLM calls | Built and tested |
| Drift check, Phoenix tracing, async ingest endpoint | **Not built** |
| Cloud deployment, recorded demo | **Not built** |
| Bedrock / Azure OpenAI backend | **Not built** (see [Provider](#provider-and-why)) |

## Architecture

```
ISO 20022 e-Repository (zip, pinned by sha256)
        │  stream-parse (ET.iterparse)
        ▼
  iso20022_pacs.json ──► chunking ──► validation ──► local embeddings ──► Postgres + pgvector
                                         │ fails the run on bad data          (chunks table)
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
- **Why not Bedrock or Azure OpenAI:** the original plan was AWS Bedrock. On this new AWS account every Bedrock inference quota is 0, and the account-level restriction needs an Anthropic use-case form and a support case before it lifts. No Bedrock call was ever made. The generation code is one small module (`llm/generation.py`), so a Bedrock backend would be a contained addition, but **it does not exist**.
- Keys are read from environment variables only (`ANTHROPIC_API_KEY`, `TYPESAFE_API_KEY`).

## Design decisions

**Chunking.** The unit of retrieval is one XML element definition, because that is the unit users ask about ("what is the maximum length of X"). Chunk kinds, with the counts in the current index (1,245 chunks):

- `element`: 879. One element: its own name and definition first, then requirement, length or code set, parent and message. Boilerplate goes last so it does not dominate the embedding.
- `children`: 291. A short list of an element's direct children, for "what does X contain" questions.
- `message` and `rule`: 75. Message scope and usage sections, split into parts of similar size, and the message's business rules.
- Ids are stable and readable (for example `pacs.008.001.14#FIToFICstmrCdtTrf/CdtTrfTxInf/CdtrAgt`) and chunks are capped at 1,500 characters. CamelCase names are split into words ("Total Interbank Settlement Amount") because the embedding model does not understand `TtlIntrBkSttlmAmt`.
- Shared components that repeat under several parents are indexed once; each chunk lists the other places it applies. This cut the index from 3,211 to 1,245 chunks and a third off the embedding time, with no measurable accuracy change.

**Retrieval.** Exact vector search (an HNSW index silently lost true nearest neighbours here, so it was removed; a scan of about 1,200 chunks takes milliseconds) fused with BM25 and three ranking signals: whole-phrase match of an element's path or tag in the question, question intent (a "what fields does X contain" question prefers a children list, a "rule" question prefers rule chunks), and a depth prior for shallower elements. When the question names one message (for example "pacs.008") results are filtered to it. Retrieval is about 85 to 90 ms per question including query embedding. Rebuilding the whole index takes about 65 to 90 seconds on a CPU.

**Escalation.** The answer is shown only if it carries a valid `[n]` citation that matches a retrieved passage. Otherwise the system escalates, with one of these reasons: `low_retrieval_confidence`, `all_passages_quarantined`, `insufficient_context`, `no_valid_citation`, `model_declined`, `truncated`, `output_policy_violation`. The retrieval cutoff is the cosine similarity of the top result, `ESCALATION_MIN_SCORE`, default 0.6. On the current question sets, on-topic questions score 0.69 or higher at the top result (median about 0.78) and clearly off-topic ones score 0.41 to 0.54. A question that names something the corpus does define (for example "euro to dollar exchange rate", where an Exchange Rate element exists) scores 0.68 and passes. That is correct behaviour, not a bug, but it means the score cutoff alone cannot catch in-domain questions the corpus does not answer; the model's `INSUFFICIENT_CONTEXT` reply does (see results).

**Audit log.** Every `/ask` call writes one row to a Postgres `audit_log` table before the response returns: question (verbatim), answer, source ids and scores, outcome, escalation reason, model, latency and the blocked answer if the output check stopped one. Triggers block UPDATE, DELETE and TRUNCATE. Each row stores the SHA-256 of the previous row plus its own canonical JSON, serialised with an advisory lock, and `python -m payments_rag.audit.log verify` re-walks the chain. It fails closed: if the log cannot be written the endpoint returns 503, unless `AUDIT_FAIL_OPEN=1`. Limits are in [Risks and controls](#risks-and-controls).

**LLM cost controls.** Every real call is cached on disk (`.cache/llm`, keyed by the hash of the full request), so identical requests are free. `LLM_REPLAY_ONLY=1` makes no network calls at all and returns 503 on a cache miss. A project-wide spend cap (`LLM_BUDGET_USD`, default $0.50) is enforced from a ledger file: a call that could take the total past the cap is refused before it is sent.

## Results

All LLM runs below used `claude-haiku-5-5` for generation on 2026-11-20. **They were run locally**; CI runs the retrieval gate on every push but skips the LLM-scored steps unless the API keys are configured as repository secrets, and they are not configured. So the CI badge does not vouch for the numbers in this section.

### Golden set (52 questions)

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
uv run python -m payments_rag.ingestion.iso20022_repository data/raw/iso20022/20260904_ISO20022_2013_eRepository.zip data/processed/iso20022_pacs.json --message pacs.008 --message pacs.002
uv run python -m payments_rag.ingestion.chunking data/processed/iso20022_pacs.json data/processed/chunks.jsonl
uv run python -m payments_rag.ingestion.validation data/processed/chunks.jsonl --expect pacs.008 --expect pacs.002
uv run python -m payments_rag.ingestion.pipeline data/processed/chunks.jsonl --expect pacs.008 --expect pacs.002
```

Run the checks that need no API key (tests, lint, retrieval gate):

```
uv run pytest -q
uv run ruff check .
uv run python -m payments_rag.evals.rag_eval --retrieval-only --out evals/results/retrieval_only.json
uv run python -m payments_rag.evals.gate evals/results/retrieval_only.json evals/thresholds.json --only retrieval,confidence
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

The compose `app` service was only verified to serve `/health` at the scaffold stage; running the API through `uv` as above is the path that has been exercised end to end.

## Tests and CI

220 tests. CI (`.github/workflows/ci.yml`) runs lint and tests, then downloads and hash-verifies the ISO source, builds the index and runs the retrieval and confidence gate on every push. The LLM-scored eval (faithfulness, relevancy, behaviour) runs only when `ANTHROPIC_API_KEY` and the judge key are repository secrets, and is skipped with a warning otherwise. The gate fails the build if any metric is below its minimum in `evals/thresholds.json`. A deliberately reversed-ranking retriever failed the retrieval gate (MRR 0.23), which is why hit@1 and MRR are gated and not hit@5 alone.

## Not built yet

- **Drift check** (retrieval score and answer length over time) and **Phoenix tracing**. The compose file lists a Phoenix container, but nothing sends traces to it.
- **Async ingest endpoint.** Ingestion is a command-line pipeline only.
- **Cloud deployment.** The service is not deployed and is not reachable from the internet. The plan was AWS ECS on Fargate; the Dockerfile exists but the cloud path is untested.
- **Recorded demo.** Replay mode exists, but no recording has been made.
- **Bedrock or Azure OpenAI backend** (see [Provider](#provider-and-why)).
- **Other sources.** A BIS report on harmonised ISO 20022 data requirements was approved but not ingested; FedNow and CBPR+ material is out of scope for licensing reasons (see `data/SOURCES.md`).
- **Known quality gaps:** the 5 retrieval misses above, and the guard's paraphrase evasion.
- **Not independent:** the golden set and the injection sets were written by the same author as the system.

## Sources and notices

This is a non-commercial personal project and is **not the official ISO 20022 website**. The sole source of up-to-date ISO 20022 message standards and the repository is https://www.iso20022.org/.

`data/processed/iso20022_pacs.json` is derived from the ISO 20022 e-Repository (pacs.008.001.14 and pacs.002.001.16 definitions), reproduced under the [ISO 20022 terms of use](https://www.iso20022.org/terms-use). The source file's SHA-256 is recorded in the JSON. Answers quote short extracts and cite the source. Other sources and their licence notes are listed in [data/SOURCES.md](data/SOURCES.md).
