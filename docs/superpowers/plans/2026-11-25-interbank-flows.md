# Interbank Flows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow the assistant from two messages to six (pain.001, pacs.008, pacs.009, pacs.002, pain.002, pacs.004) plus four hand-written, cited interbank flows. The flows can be retrieved through `/ask` and listed through `GET /flows`, with no extra cost or latency per question.

**Architecture:** The committed extract gains four messages and is renamed. A new `ingestion/flows.py` loads `data/flows/interbank_flows.json`, checks it against the extract and turns it into `flow` chunks, which go through the same validation, provenance, embedding and storage path as every other chunk. Retrieval gains a flow intent, and the message filter keeps the flows that use the named message. The API gains read-only flow endpoints, and the eval gate gains a separate cohort for the original questions.

**Tech Stack:** Python 3.12, FastAPI, psycopg 3, Postgres 16 + pgvector, fastembed, pytest, ruff, uv, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-25-interbank-flows-design.md` (approved 2026-11-25). Read it before starting any task.

## Global Constraints

- Messages, exactly these six latest versions from the pinned zip (sha256 `fb1504abc9df0d445880faedeb0aad208047820a61bd1754bb571c1419ae1c88`): `pain.001.001.13`, `pacs.008.001.14`, `pacs.009.001.13`, `pacs.002.001.16`, `pain.002.001.15`, `pacs.004.001.15`. No new download, no other messages, no older versions.
- Message prefixes, in this order wherever a tuple is listed: `("pain.001", "pacs.008", "pacs.009", "pacs.002", "pain.002", "pacs.004")`.
- Committed extract path: `data/processed/iso20022_messages.json` (replaces `data/processed/iso20022_pacs.json` everywhere; ignore `infra/cdk.out/`, which is build output).
- Flows file path: `data/flows/interbank_flows.json`. JSON only; do not add a YAML dependency.
- Flow chunk kind: `flow`. Ids `flow:<flow id>` (split parts `flow:<id>:2`, …), overview id `flow:index`. Metadata: `message_id="flow"`, `message_name=<flow title>`, `source="Project flow summary based on ISO 20022 e-Repository definitions"`, `messages=[prefixes]`, `cites=[chunk ids]`.
- Every chunk is ≤ 1,500 characters (`chunking.MAX_CHARS`).
- `/ask` still sends the top 5 passages. The system prompt does not grow by more than a few words, and no flow text goes into it.
- No flow step may state something its cited ISO text does not support. Soften or drop the step and add a note saying why.
- Keys only from environment variables. No LLM provider or model changes.
- Never use `gh pr merge --auto`. Merge only after `test-and-build` passes, and only after John has reviewed the flows commit.

## Review Focus

1. **A question naming one message that asks what comes next** ("what is sent after a pacs.008?"): the message filter must not hide flow chunks. Covered by a test in Task 5.
2. **A pacs.004 field question containing the word "return"** ("what is the return reason in pacs.004?") should stay a definition question and not be pulled to flow chunks. Covered by a test in Task 5.
3. **A partial ingest** (`messages=["pacs.008"]`) must succeed, store no flow chunks, and delete any flow chunks left by an earlier full ingest. Covered by a test in Task 4.
4. **A tampered or hand-edited flow chunk in the database** must be refused by the provenance check, like any other chunk, and a genuine flow chunk must pass. Covered by a test in Task 4.
5. **`GET /flows/<unknown id>`** returns 404, not 500, and `GET /flows` works when the database is down. Covered by tests in Task 6.

---

### Task 0: Branch

- [ ] **Step 1:** `git switch -c feat/interbank-flows main`. The untracked spec and plan come along with the switch.
- [ ] **Step 2:** Commit `docs/superpowers/specs/2026-11-25-interbank-flows-design.md` and `docs/superpowers/plans/2026-11-25-interbank-flows.md` with the message `docs: interbank flows design and plan`.

### Task 1: Six-message extract and allowlists

**Files:**
- Create: `data/processed/iso20022_messages.json` (generated). Delete: `data/processed/iso20022_pacs.json`
- Modify: `src/payments_rag/ingestion/jobs.py:11,39,41`, `src/payments_rag/api/ingest.py:61`, `src/payments_rag/guardrails/provenance.py:6,26`, `Dockerfile:3,26`, `.github/workflows/ci.yml:79,86-89`, `evals/golden/build_golden.py:1,24`, `tests/test_golden_and_gate.py:19`, `tests/test_ingest_jobs.py:17`, `tests/test_injection_guard.py:242`
- Test: `tests/test_ingest_api.py`, `tests/test_ingest_jobs.py`

**Interfaces:**
- Produces: `jobs.ALLOWED_MESSAGES: tuple[str, ...]` (the six prefixes); `jobs.DEFAULT_EXTRACT` and `provenance.DEFAULT_EXTRACT` both `Path("data/processed/iso20022_messages.json")`; `IngestRequest.messages: list[Literal[<six prefixes>]]` defaulting to all six.

- [ ] **Step 1: Write failing tests** in `tests/test_ingest_api.py`:
  - `test_default_request_asks_for_all_six_messages`: POST `/ingest` with `{}` and assert that the submitted params' `messages == sorted(ALLOWED_MESSAGES)` and `len == 6`.
  - `test_unknown_message_is_rejected`: POST with `{"messages": ["camt.056"]}` and assert 422.
- [ ] **Step 2:** Run `uv run pytest tests/test_ingest_api.py -v`. Expected: the first test fails.
- [ ] **Step 3: Regenerate the extract:**
  ```
  uv run python -m payments_rag.ingestion.iso20022_repository data/raw/iso20022/20260904_ISO20022_2013_eRepository.zip data/processed/iso20022_messages.json --message pain.001 --message pacs.008 --message pacs.009 --message pacs.002 --message pain.002 --message pacs.004
  ```
  Then `git rm data/processed/iso20022_pacs.json`. Check that the message ids in the new file are exactly the six in Global Constraints and that `source.sha256` is unchanged.
- [ ] **Step 4: Update every reference** listed under Files. Set the CI `--message`/`--expect` arguments to all six. Set `ALLOWED_MESSAGES` and the `Literal`. The `Literal` must list the six prefixes explicitly; a test asserts it equals `ALLOWED_MESSAGES`.
- [ ] **Step 5:** Run `uv run python -m payments_rag.ingestion.chunking data/processed/iso20022_messages.json data/processed/chunks.jsonl`. Expected: about 4,650 chunks (±5%). If the count is far off, stop and report.
- [ ] **Step 6:** Run `uv run pytest -q && uv run ruff check .`. Expected: all pass. Update the existing tests that assert two messages, or 1,245 chunks, to the new numbers.
- [ ] **Step 7:** Commit with the message `feat: index six interbank messages from the pinned e-Repository`.

### Task 2: Flow chunks, checks and validation

**Files:**
- Create: `src/payments_rag/ingestion/flows.py`, `tests/test_flows.py`
- Modify: `src/payments_rag/ingestion/chunking.py:350-385` (`chunk_payload`, `main`), `src/payments_rag/ingestion/validation.py:28-35`

**Interfaces:**
- Produces:
  - `flows.DEFAULT_FLOWS = Path("data/flows/interbank_flows.json")`
  - `flows.FLOW_SOURCE = "Project flow summary based on ISO 20022 e-Repository definitions"`
  - `flows.load_flows(path: str | Path = DEFAULT_FLOWS) -> dict[str, Any]`
  - `flows.check_flows(flows: dict, payload: dict) -> list[str]`: returns error strings and an empty list when the file is valid. Checks:
    - flow ids are unique and match `^[a-z0-9-]+$`;
    - every step `message` is a prefix of a message in `payload`;
    - every id in step `cites` and flow `cites` is in `{c["id"] for c in chunk_payload(payload)}`;
    - `source_note` is non-empty.
  - `flows.flow_chunks(flows: dict, payload: dict) -> list[dict]`: chunks for the flows whose every step message is in `payload`, plus `flow:index` listing those flows. Returns `[]` when no flow qualifies. Message names in the step lines come from `payload` (`_words(msg["name"])`).
  - `chunking.chunk_payload(payload: dict, flows: dict | None = None) -> list[dict]`: unchanged when `flows is None`; otherwise appends `flow_chunks(flows, payload)`.
  - `chunking.main` gains an optional `--flows PATH`. When it is given, the CLI runs `check_flows` and exits non-zero, printing the errors, if any exist.
  - `validation.VALID_KINDS` includes `"flow"`, and `KIND_METADATA["flow"] = ("messages", "cites")`.

- [ ] **Step 1: Write failing tests** in `tests/test_flows.py`, using the small `payload()` helper style from `tests/test_chunking.py` with two fake messages `pacs.008.001.14` and `pacs.002.001.16`, and a two-step fixture flow citing their `#message:scope` ids:
  - `test_flow_chunk_has_id_kind_and_metadata`: the chunk has id `flow:demo`, kind `flow`, `message_id == "flow"`, `source == FLOW_SOURCE`, `messages == ["pacs.008", "pacs.002"]`.
  - `test_flow_text_lists_numbered_steps_with_parties_and_message_names`: assert `"1. Debtor agent → Creditor agent: pacs.008"` and the spelled-out message name are in the text.
  - `test_flow_skipped_when_a_message_is_missing`: with a payload that has only pacs.008, `flow_chunks == []`.
  - `test_index_chunk_lists_included_flows`: `flow:index` exists and names the flow title.
  - `test_long_flow_is_split_below_max_chars`: every chunk is at most `MAX_CHARS`, and ids get a `:2` suffix.
  - `test_check_flows_reports_unknown_cite_unknown_message_duplicate_id_and_missing_note`: assert four distinct errors.
  - `test_check_flows_accepts_valid_file`: assert `[]`.
  - `test_chunk_payload_without_flows_is_unchanged`: the ids equal the ids from the old call.
  - `test_validation_accepts_flow_kind_and_requires_messages_and_cites`: a flow chunk without `cites` fails with `missing_metadata`.
- [ ] **Step 2:** Run `uv run pytest tests/test_flows.py -v`. Expected: FAIL (the module does not exist).
- [ ] **Step 3: Implement** `flows.py` and the `chunk_payload`/`main`/`validation` changes. Reuse `chunking._balanced`, `_words` and `_chunk` (pass a msg-like dict `{"id": "flow", "name": title}` and then overwrite `source` in the metadata). Flow chunk text layout:
  ```
  Interbank flow: <title> (project summary; not ISO text)
  <summary>
  Parties: <a, b, c>
  Steps:
  1. <from> → <to>: <prefix> (<Message Name In Words>). <purpose>
  Notes: <note>; <note>
  ```
- [ ] **Step 4:** Run `uv run pytest tests/test_flows.py tests/test_chunking.py tests/test_validation.py -v`. Expected: PASS.
- [ ] **Step 5:** Commit with the message `feat: flow chunks with citation checks`.

### Task 3: Author the flows file (own commit, for John's review)

**Files:**
- Create: `data/flows/interbank_flows.json`
- Test: `tests/test_flows.py` (add committed-file tests)

**Interfaces:**
- Consumes: `load_flows`, `check_flows`, `flow_chunks` (Task 2); the extract (Task 1).

- [ ] **Step 1: Write failing tests:**
  - `test_committed_flows_pass_checks`: `check_flows(load_flows(), <committed extract>) == []`.
  - `test_committed_flows_cover_the_four_flows`: the flow ids are exactly `{"serial-credit-transfer", "cover-method", "fi-credit-transfer", "return"}`.
  - `test_every_indexed_message_appears_in_some_flow`: the six prefixes are a subset of all step messages.
  - `test_committed_flow_chunks_fit_and_pass_validation`: `validate_chunks(chunk_payload(extract, load_flows()))["passed"]`.
- [ ] **Step 2:** Run them. Expected: FAIL (no file yet).
- [ ] **Step 3: Write the file.** Use the shape in the spec. Before writing each step, read the cited chunks' text (`uv run python -m payments_rag.ingestion.chunking … --flows …` and grep `chunks.jsonl`), and write only what that text supports.
  - Prefer `#message:scope`/`#message:usage` ids.
  - For the cover method, cite the pacs.009 element that carries the underlying customer credit transfer (find its `xml_path` in the extract).
  - For the return flow, cite the pacs.004 return reason element.
  - Put the reject (pacs.002, before settlement) versus return (pacs.004, after settlement) distinction in `notes`, backed by citations.
  - Where the text does not support a step in the spec's list, soften or drop it and add a note saying what was left out.
- [ ] **Step 4:** Run `uv run pytest tests/test_flows.py -v`. Expected: PASS.
- [ ] **Step 5:** Commit **only this file and its tests** with the message `data: hand-written interbank flows (needs John's fact review)`. Record the commit SHA for the PR description.

### Task 4: Ingest flows end to end (pipeline, jobs, provenance, CI)

**Files:**
- Modify: `src/payments_rag/ingestion/pipeline.py:32-99`, `src/payments_rag/ingestion/jobs.py:150-200`, `src/payments_rag/guardrails/provenance.py:33-50`, `.github/workflows/ci.yml:84-89`
- Test: `tests/test_ingest_pipeline.py`, `tests/test_ingest_jobs.py`, `tests/test_ask.py` (provenance)

**Interfaces:**
- Consumes: `chunk_payload(payload, flows)`, `load_flows`, `check_flows`, `DEFAULT_FLOWS` (Task 2).
- Produces:
  - `pipeline.ingest(chunks, embedder, conn, expected=None)`: always includes `"flow"` in the `message_id` groups it reads hashes for and deletes from, so flow chunks absent from `chunks` are deleted.
  - `provenance.build_manifest(payload: dict, flows: dict | None = None) -> dict[str, str]`.
  - `ProvenanceCheck.from_extract(path=DEFAULT_EXTRACT, flows_path: str | Path | None = DEFAULT_FLOWS)` loads the flows when the file exists.
  - `jobs.default_runner(..., flows_path: Path | None = None)`, falling back to env `INGEST_FLOWS`, then `DEFAULT_FLOWS`. It runs `check_flows(flows, full_extract)` against the **whole** committed extract and raises `ValueError` listing the errors, then chunks `chunk_payload(selected_payload, flows)`.

- [ ] **Step 1: Write failing tests:**
  - `test_partial_ingest_stores_no_flows_and_deletes_old_ones` (pipeline, same DB fixture style as the existing pipeline tests): ingest all six messages with flows and assert flow rows exist; then ingest `pacs.008` chunks only and assert there are 0 rows with `kind='flow'`.
  - `test_runner_refuses_flows_with_bad_cites` (jobs): with a temp flows file citing `nope#x`, `run()` raises a `ValueError` containing `nope#x`.
  - `test_provenance_accepts_flow_chunks_and_rejects_edited_ones` (`tests/test_ask.py` style): manifest from `build_manifest(payload, flows)`; a genuine flow hit passes, and the same hit with altered text is rejected with `text_does_not_match_source`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3: Implement** the interfaces above. In CI, add `--flows data/flows/interbank_flows.json` to the chunking command.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: PASS. Then rebuild the local index with the four CI commands (extract step skipped, the extract is committed), and record the chunk total and the wall time of the embed step.
- [ ] **Step 5:** Commit with the message `feat: ingest and verify flow chunks`.

### Task 5: Retrieval: flow intent and flow-aware message filter

**Files:**
- Modify: `src/payments_rag/retrieval/ranking.py:50-96`, `src/payments_rag/retrieval/store.py:98-108,156-180`
- Test: `tests/test_ranking.py`, `tests/test_retrieval_store.py` (create if there is no pure-function test file for the store)

**Interfaces:**
- Produces:
  - `ranking.FLOW_INTENT: re.Pattern`.
  - `ranking.intent(query) -> Literal["flow", "children", "rule", "definition"]`. `flow` is checked first, but a question that names a single message **and** has no flow word other than "return" is not a flow question.
  - `kind_factor(kind, "flow")` returns 1.0 for `flow` and `OFF_INTENT_FACTOR` otherwise.
  - `store.doc_allowed(doc: dict, kinds: list[str] | None, prefix: str | None) -> bool`: true when the kind is allowed and either there is no prefix, `doc["message_id"]` starts with `prefix + "."`, or `doc["kind"] == "flow"` and `prefix in doc["metadata"]["messages"]`.
  - `store._where` emits `(message_id LIKE %s OR (kind = 'flow' AND metadata->'messages' ? %s))` when a prefix is given.

- [ ] **Step 1: Write failing tests:**
  - `test_flow_questions_get_flow_intent`, parametrized over: "What is sent after a pacs.008?", "Which message does the creditor agent use to send funds back?", "Walk me through the cover method", "Who sends a pain.002 and to whom?", "What is the difference between a reject and a return?"
  - `test_field_questions_keep_their_intent`: "What is the return reason in pacs.004?" → `definition`; "What does the group header contain in pacs.009?" → `children`; "What is the rule on settlement method?" → `rule`.
  - `test_doc_allowed_keeps_flows_that_use_the_named_message` and `test_doc_allowed_drops_flows_that_do_not`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3: Implement.** Use `doc_allowed` in `hybrid_search` for the `allowed` mask. The `?` operator needs `metadata` as jsonb, which it already is (`store.py:43`). Do not change `RRF_K`, `POOL`, `OFF_INTENT_FACTOR` or `DEPTH_PENALTY`.
- [ ] **Step 4:** Run `uv run pytest -q`, then `uv run python -m payments_rag.retrieval.evaluate evals/retrieval_set.jsonl --mode hybrid`. Expected: tests pass, and today's questions keep hit@5 ≥ 0.8.
- [ ] **Step 5:** Commit with the message `feat: flow intent and flow-aware message filter`.

### Task 6: API: `GET /flows`, source labels, prompt wording

**Files:**
- Create: `src/payments_rag/api/flows.py`, `tests/test_flows_api.py`
- Modify: `src/payments_rag/api/main.py:10-20`, `src/payments_rag/api/rag.py:32-39`, `src/payments_rag/llm/generation.py:27`
- Test: `tests/test_ask.py`, `tests/test_generation.py`

**Interfaces:**
- Consumes: `load_flows`, `DEFAULT_FLOWS` (Task 2); the extract for message names.
- Produces:
  - `api.flows.router` with `GET /flows` returning `{"source_note": str, "flows": [{"id", "title", "summary", "parties", "steps": [{"n", "from", "to", "message", "message_id", "message_name", "purpose"}], "notes"}]}`.
  - `GET /flows/{flow_id}` returns one flow or 404.
  - The data is loaded once (cached on first use) from `FLOWS_PATH` (default `DEFAULT_FLOWS`) and `PROVENANCE_EXTRACT` (default the extract). There is no DB access, and the response carries the header `Cache-Control: public, max-age=300`.
  - `_source(hit)` adds `"source": hit.get("metadata", {}).get("source")`.
  - `SYSTEM_PROMPT`'s opening line names "ISO 20022 interbank credit transfer messages (pain.001, pacs.008, pacs.009, pacs.002, pain.002, pacs.004) and project flow summaries". It must stay within about 10 words of today's length, and no rule is added.

- [ ] **Step 1: Write failing tests:**
  - `test_list_flows_returns_four_flows_with_message_names`
  - `test_get_flow_by_id`
  - `test_unknown_flow_is_404`
  - `test_flows_work_without_a_database`: point `DATABASE_URL` at an unreachable host and assert 200.
  - `test_flows_response_is_cacheable`
  - `test_ask_sources_carry_source_label` (in `tests/test_ask.py`)
  - `test_system_prompt_names_the_six_messages` (in `tests/test_generation.py`)
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3: Implement** and register the router in `main.py`.
- [ ] **Step 4:** Run `uv run pytest -q && uv run ruff check .`. Expected: PASS.
- [ ] **Step 5:** Commit with the message `feat: GET /flows and source labels on cited passages`.

### Task 7: Evals: cohorts, new questions, re-measure

**Files:**
- Modify: `evals/golden_set.jsonl`, `evals/retrieval_set.jsonl`, `evals/retrieval_holdout.jsonl`, `src/payments_rag/evals/rag_eval.py:164-190`, `evals/thresholds.json`, `.github/workflows/ci.yml:94-95`, `evals/golden/build_golden.py` (only if it rewrites the set; keep the hand-added rows)
- Test: `tests/test_golden_and_gate.py`

**Interfaces:**
- Produces:
  - Golden and retrieval rows gain `"cohort": "original" | "flows-v1"`. All existing rows are `original`.
  - `rag_eval` summaries gain a `retrieval_original` group (`retrieval_metrics` over `cohort == "original"` rows) in both the full and the retrieval-only runs.
  - `thresholds.json` gains `retrieval_original.hit@1`, `retrieval_original.hit@5` and `retrieval_original.mrr`, each `{"min": 0.8}`.
  - CI `--only` becomes `retrieval,retrieval_original,confidence` for the retrieval gate and `retrieval,retrieval_original,behaviour,judged` for the LLM gate.

- [ ] **Step 1: Write a failing test**, `test_summary_has_original_cohort_metrics`: rows from both cohorts, where the original cohort has a miss, and assert `summary["retrieval_original"]["hit@5"]` reflects only the original rows.
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3: Implement** the cohort summary and the gate/CI changes.
- [ ] **Step 4: Add the `flows-v1` questions.** Reference ids must be real chunk ids.
  - Retrieval set and holdout:
    - ≥ 3 per new message (pain.001, pacs.009, pain.002, pacs.004);
    - ≥ 8 flow questions, ≥ 3 of which name no message;
    - ≥ 3 ambiguity checks (an element name shared across messages, asked with one message named).
  - Golden set:
    - ≥ 8 answerable flow and new-message questions with reference answers;
    - ≥ 3 `unanswerable_not_in_corpus` questions: direct debit flow, camt.056 recall, pacs.028 status request.
- [ ] **Step 5: Measure.** Run the retrieval-only eval and the CI gate locally. Expected: the `original` cohort is ≥ 0.8 on all three keys. If not, go back to Task 5 tuning and do not lower the thresholds.
- [ ] **Step 6: Run the full LLM eval** (needs `ANTHROPIC_API_KEY` and `TYPESAFE_API_KEY`; if they are absent, say so and skip). Then set every overall threshold to measured − 0.09, following the existing rule, and rewrite `thresholds.json`'s `_comment` with the new run (date, index size, counts, known misses).
- [ ] **Step 7: Measure latency.** Measure retrieval p50 over the retrieval set the same way as the README's 85–90 ms figure, before (on `main`) and after. Also confirm the per-question input tokens on a few replayed questions are unchanged within ±5%. Write both numbers in the PR description. If p50 grew by more than 25 ms, profile `ranking.fuse`/`name_match` and report before optimising.
- [ ] **Step 8:** Commit with the message `test: flow and new-message eval questions; original-cohort gate`.

### Task 8: Docs

**Files:**
- Modify: `README.md`, `data/SOURCES.md`

- [ ] **Step 1: Update the README:**
  - the opening scope line (six messages, four flows);
  - the Status table (new row "Interbank flows and `GET /flows`: built");
  - the architecture diagram's extract name;
  - chunk counts per kind (from Task 4);
  - a "Flows" paragraph: hand-written, cited, labelled as not ISO text, checked at build time, reviewed by John;
  - the latency and eval numbers from Task 7;
  - the commands at lines 186–187.
- [ ] **Step 2: Update `data/SOURCES.md`:** add the four messages under the existing e-Repository entry, and add a line for `data/flows/interbank_flows.json` as project-authored and derived from the cited ISO text.
- [ ] **Step 3:** Run `uv run pytest -q && uv run ruff check .`, then commit with the message `docs: six messages and interbank flows`.
- [ ] **Step 4:** Push the branch and open a PR against `main`. Say in the description that the flows commit (SHA from Task 3) needs John's fact review, and add the latency and eval numbers. Do not merge.

**Follow-up outside this plan:** after merge, update `docs/go-live-plan.md` on `plan/go-live`: the UI title (no longer "pacs.008 / pacs.002"), a UI section rendering `GET /flows`, and the first-ingest time from Task 4.
