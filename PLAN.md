# Payments-standards Q&A assistant: build plan (middle path)

Public documents only (ISO 20022, FedNow, public SWIFT/CBPR+). No proprietary or employer data.

## Decisions so far
| Area | Choice |
|---|---|
| Generation | Claude API (Anthropic credit), behind a provider adapter |
| Eval judge | Claude API, a different model from the generator where possible |
| Escalation / injection guard | JEV API (calibrated probabilities), similarity score as fallback signal |
| Embeddings | Local open model (e.g. `bge-small`) in the container; no API cost |
| Vector store | pgvector |
| Tracing | Arize Phoenix |
| Hosting | Azure Container Apps (container only; no Azure OpenAI) |
| Bedrock | Final optional phase; README claims it only after it has actually run |
| Raspberry Pi | Not used in the deployed path (dev experiments only) |

Rule: the README states only what has been built and run. Keys come from environment variables, never hardcoded.

## Cost minimisation (applies to every phase)
- Phases 0-5 run entirely in docker-compose. No Azure account or card until phase 6.
- Every LLM and judge call goes through a disk cache keyed by a hash of (model, prompt, params). Reruns, CI and the demo replay cost $0. Only changed prompts hit the API.
- Cheapest adequate model (Haiku-class) for generation. Judge model also small. Anthropic prompt caching for the fixed system prompt.
- CI default: replay from cached responses. A live run is a manual, labelled workflow (`workflow_dispatch`), and its results are what the README quotes.
- Retrieval, chunking, validation and drift code are tested without any LLM.
- Budget alerts set on any cloud account before deploying; tear down the resource group after demos.

## Recorded demo
- The same cache gives a replayable run: `make demo` replays a saved session (questions, retrieved sources, answers, escalations, a planted-injection case) with no API key and no cost.
- Record one live run once, commit the cache and a short screen recording or terminal cast. The README labels it "recorded run", not live.
- The deployed URL stays the live proof. The recording is the fallback if it is scaled to zero or taken down.

## Phases

### 0. Scaffold (about 1 hour)
- Repo layout, `pyproject.toml`, `docker-compose.yml` (app + pgvector + Phoenix), lint/test setup, `data/SOURCES.md` listing every public doc with URL and licence.
- Done when: `docker compose up` starts all three services and `/health` returns 200.

### 1. Ingestion (weekend 1)
- Loaders for ISO 20022 message definitions, FedNow docs and CBPR+ material.
- Structure-aware chunking: one chunk per message element or section, parent path kept as metadata. Prose is split by heading with overlap. Write down why in the README.
- Data-quality validation step: empty or duplicate chunks, size bounds, missing metadata, encoding errors, and a report that fails the run on violations.
- Embed locally and store in pgvector.
- Done when: a validation report is produced and a retrieval smoke test passes.

### 2. Service
- FastAPI `/ask`: retrieve, score confidence, then generate with citations. `/ingest` runs as an async job with a status endpoint.
- Provider adapter (`llm/`) is the only module that calls an LLM. Claude API backend first.
- Escalation: JEV decides whether the retrieved context supports an answer. Below threshold, return "escalated to human" with the sources found, and no generated answer.
- Done when: citations resolve to real chunks and a low-confidence question escalates in a test.

### 3. Observability
- Phoenix tracing with spans for retrieval, orchestration and generation.
- Drift check: record retrieval score and answer length per request, compare against a baseline window, and expose a simple report.
- Done when: one request shows all three spans in Phoenix and the drift report runs.

### 4. Evals in CI (weekend 2)
- Golden set of 40-60 questions with reference sources, hand-checked against the documents.
- RAGAS or DeepEval for faithfulness and answer relevance; Claude judge.
- CI job fails below a threshold. Set the threshold from a first measured run, not guessed.
- Done when: CI is green above the threshold and red on a deliberately broken retriever.

### 5. Security
- Injection test set: malicious text planted inside documents (instruction override, exfiltration, citation spoofing). Measure attack success rate with and without the JEV guard.
- Audit log: append-only record of every question, answer, sources and escalation decision.
- README section on risks and controls, listing the failures actually observed.
- Done when: the injection runner produces a results file that the README quotes.

### 6. Deploy to Azure Container Apps
- Dockerfile, IaC in `infra/`, GitHub Actions deploy workflow, secrets via Container Apps secrets. Set an Azure budget alert first.
- Postgres/pgvector as a container, or the cheapest managed option if persistence is needed.
- Done when: a public URL answers `/ask` and the deployment survives a restart.

### 7. Bedrock backend (optional, last)
- Open an AWS account, set a budget alert, submit the Anthropic access form, and use the inference-profile ID for Haiku.
- Add a Bedrock backend to the adapter. Rerun the golden set and injection set on it and record both result sets.
- Open question: Azure-hosted containers have no native AWS IAM role. Options are a narrowly scoped IAM user with keys stored as a container secret (works, documented as a limitation), or hosting this phase on ECS Fargate instead (a few dollars a month, clean IAM roles). Decide when reaching this phase.
- Done when: both backends have run the evals and the README reports each result. If skipped, the README says generation uses the Claude API only.

### 8. README and wrap-up
- Architecture diagram, providers actually used, eval results, injection results, drift check description, risks and controls, known limitations, cost notes.
- Done when: every claim in the README maps to a command that was run.

## Risks to watch
- Golden-set quality: bad reference answers make the eval meaningless. Hand-check each one.
- Judge bias: judge and generator should differ, and the README should say so.
- Cost drift: cap eval frequency in CI (full run on pull requests to main only).
- Licence: confirm each source document permits redistribution, or ingest from a download script and don't commit the files.
