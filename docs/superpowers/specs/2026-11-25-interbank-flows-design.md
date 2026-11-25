# Interbank flows: a small, useful knowledge base

Date: 2026-11-25. Status: reviewed by a builder check; awaiting John's approval.

## Intent

Today the assistant knows two messages (pacs.008, pacs.002) element by element. That makes it a proof of concept: it can say what a field means, but not how a payment moves. The goal is to make it **potentially useful to someone working on interbank credit transfers**, while keeping it small and **without making any single question slower or more expensive**.

It becomes an assistant for a handful of common interbank flows. It knows the messages in those flows field by field, and it knows the flows themselves: who sends which message to whom, and why. The flows it knows are listed (for the agent and for a UI), so a user can see what it covers and what it does not.

## Goals

1. Index six messages, each at its latest registered version in the already-pinned e-Repository zip (`20260904_ISO20022_2013_eRepository.zip`, sha256 `fb1504ab…c88`, no new download):

   | Message | Version in the zip | Role | Chunks (measured with today's chunker) |
   |---|---|---|---|
   | pain.001 | pain.001.001.13 | Customer instructs its bank to pay | 622 |
   | pacs.008 | pacs.008.001.14 | Bank-to-bank customer credit transfer | 732 (already indexed) |
   | pacs.009 | pacs.009.001.13 | Bank-to-bank transfer (own account, or cover) | 1,071 |
   | pacs.002 | pacs.002.001.16 | Bank-to-bank status report | 513 (already indexed) |
   | pain.002 | pain.002.001.15 | Bank-to-customer status report | 540 |
   | pacs.004 | pacs.004.001.15 | Payment return | 1,177 |

   About 4,650 chunks in total, up from 1,245.

2. Add a hand-written **flows file** describing these flows (the builder writes the text; this list is the scope):
   - **Serial customer credit transfer:** debtor → debtor agent (pain.001); debtor agent → (intermediary agents) → creditor agent (pacs.008); status back between agents (pacs.002); status back to the debtor (pain.002).
   - **Cover method:** debtor agent → creditor agent directly (pacs.008); settlement through correspondents (pacs.009 COV).
   - **Bank-to-bank transfer (own account):** pacs.009 core, between financial institutions.
   - **Return:** after settlement, the creditor agent or an intermediary sends the funds back (pacs.004). Include the distinction between a *reject* before settlement (reported with pacs.002) and a *return* after settlement (pacs.004).

3. Make the flows **answerable** (they are retrieved like any other passage) and **listable** (`GET /flows`).

4. Keep the quality gates honest: the eval sets cover the new messages and the flows, and existing pacs.008 and pacs.002 questions do not regress.

## Non-goals

- Any message not in the table: no camt.056/camt.029 recalls, no camt.053/054 statements, no direct debits (pacs.003, pain.008), no pacs.028.
- Older message versions. Only the latest registered version of each is indexed.
- Market practice (CBPR+, FedNow, SEPA usage guidelines): out of scope because of their terms, as already recorded in `data/SOURCES.md`.
- ISO External Code Sets.
- Building the UI. The go-live plan (`docs/go-live-plan.md` on the `plan/go-live` branch) already has a UI task. This work provides `GET /flows` for it, and that UI task is updated to render it.

## Anti-goals (a failure even if it technically works)

- **A slower or more expensive question.** The model still receives the top 5 passages and the prompt does not grow. No flow text goes into the system prompt.
- **Invented payment facts.** A flow step that is not backed by the ISO scope or definition text of the message it names. The flows file is our writing, not ISO's, and must be labelled that way wherever it is shown or cited.
- **Worse answers about fields.** The new messages share many element names with pacs.008 (Debtor Agent, Instructed Agent, …). If adding them makes today's pacs.008 and pacs.002 questions retrieve the wrong message, that is a regression, even if the new questions pass.

## Constraints found in the code

- **Search size is fine.** At about 4,700 chunks, exact pgvector search and the in-memory BM25 corpus (`retrieval/store.py` `_corpus`) stay small. No HNSW index is needed.
- **The provenance guard** (`guardrails/provenance.py`) builds its allowlist manifest by re-chunking the committed extract. It would refuse flow chunks unless the manifest also covers the flows file.
- **The message filter** (`store.message_prefix_from_query`, `_where`) restricts results to `message_id LIKE '<prefix>.%'` when a question names exactly one message. Flow chunks span several messages, so this filter would hide them for a question like "what is sent after a pacs.008?".
- **Validation** (`ingestion/validation.py`) allows only the kinds `message`, `element`, `children`, `rule`. It requires `message_id`, `message_name` and `source` metadata on every chunk.
- **The pipeline** (`ingestion/pipeline.py`) groups upserts and deletes by `message_id`.
- **Allowlists** for messages are in `ingestion/jobs.py` (`ALLOWED_MESSAGES`) and `api/ingest.py` (`IngestRequest.messages` Literal). Both currently list only pacs.008 and pacs.002.
- **The committed extract** is `data/processed/iso20022_pacs.json` (2.3 MB). It is referenced by CI, the Dockerfile, `jobs.py` and `provenance.py`. At six messages it grows to roughly 9 MB.
- **CI** rebuilds and embeds the whole index on every run (`.github/workflows/ci.yml`, about 1.5 minutes today). At about 4,700 chunks that becomes about 4–5 minutes.
- **No YAML library** is a dependency. The project uses JSON throughout.

## Approaches considered

- **A. Curated messages plus a flows file (chosen).** Six messages plus a structured, hand-written flows file. The file is indexed as passages and also served through an endpoint. It answers both field questions and "what happens next" questions at no extra cost per question. It costs a small amount of authored content that has to be kept honest, which is handled with citations and validation.
- **B. Curated messages only.** Less work, but the assistant still cannot explain how a payment moves. That is the part that makes it useful.
- **C. Flows in the system prompt or only in the UI.** The prompt version adds tokens to every question, which breaks the cost constraint. The UI-only version means the assistant cannot answer flow questions.
- **Rejected earlier: index the whole payments family** (pacs/pain/camt/head/admi/remt, about 26k chunks), or all 806 messages (about 203k chunks). Much larger, ambiguous across messages, slower to build, and not needed for the goal.

John picked A and included the return flow (pacs.004).

## Design

### Flows file

`data/flows/interbank_flows.json`, committed and reviewed like code. Shape (the builder may refine the field names):

```json
{
  "source_note": "Written for this project from the ISO 20022 message definitions it cites. Not an ISO publication.",
  "flows": [
    {
      "id": "serial-credit-transfer",
      "title": "Serial customer credit transfer",
      "summary": "…one or two sentences…",
      "parties": ["Debtor", "Debtor agent", "Intermediary agent", "Creditor agent", "Creditor"],
      "steps": [
        {"n": 1, "from": "Debtor", "to": "Debtor agent", "message": "pain.001",
         "purpose": "…", "cites": ["pain.001.001.13#message:scope"]}
      ],
      "notes": ["…edge cases, e.g. reject vs return…"],
      "cites": ["…chunk ids backing the summary and notes…"]
    }
  ]
}
```

Every `message` must be one of the six indexed messages. Every id in `cites` must be a chunk id that `chunk_payload` produces from the committed extract. Prefer `#message:scope` / `#message:usage` chunks and specific element chunks (for example the pacs.009 settlement method or the pacs.004 return reason).

### Chunks

- **New chunk kind `flow`.** One chunk per flow: title, summary, numbered steps ("1. Debtor → Debtor agent: pain.001 (Customer Credit Transfer Initiation), to …"), and notes, all within the 1,500-character cap. If a flow is longer, split it with the existing balanced-parts helper, as message chunks do.
- **Ids:** `flow:<flow id>` (for example `flow:serial-credit-transfer`), with `:2`… suffixes for split parts.
- **Metadata:** `message_id = "flow"` (one group for the pipeline's per-message upsert and delete), `message_name` = the flow title, `source` = "Project flow summary based on ISO 20022 e-Repository definitions" (so it is never presented as ISO text), plus `messages` (the list of message prefixes the flow uses) and `cites`.
- A small **flows overview chunk** (`flow:index`) lists every flow the assistant knows. This answers "what flows do you know?" and "can you help with direct debits?" (the second should then fall through to `INSUFFICIENT_CONTEXT`).

### Ingestion

- **Extract the six messages.** Rename the committed extract to `data/processed/iso20022_messages.json`, and update every reference (CI, Dockerfile, `jobs.py`, `provenance.py`, README).
- **Chunk the flows** in the same step that chunks the extract, so one manifest and one chunks file cover everything.
- **Validation** accepts kind `flow` and adds these checks, each of which fails the build:
  - every `cites` id exists among the generated chunks;
  - every step's `message` is an indexed message;
  - flow text passes the existing injection screening.

  `--expect` gets all six prefixes.
- **The provenance manifest** includes flow chunks (built from the extract and the flows file).
- **Ingest allowlists** (`ALLOWED_MESSAGES` and the `IngestRequest` Literal) list all six. Flows are always ingested with whatever messages are requested, because they are not a choice the caller makes.

### Retrieval

- **Flow intent.** Add an intent pattern for flow questions ("flow", "sequence", "after", "before", "next", "who sends", "reply", "respond", "which message do I use", "cover", "reject … return"). A flow question prefers `flow` chunks and demotes other kinds, using the same factor mechanism as `children` and `rule`. The pattern must not hijack field questions about pacs.004 elements (for example "what is the return reason in pacs.004?" stays a definition question). The builder tunes this against the eval sets.
- **Message filter keeps relevant flows.** When a question names one message, results include element, message and rule chunks of that message, **plus** flow chunks whose `messages` include it.
- **Cost unchanged.** Top-5 passages and the same prompt; the escalation threshold is unchanged unless re-measurement says otherwise.

### API

- `GET /flows` returns the list of flows (id, title, summary, parties, steps with message ids and their ISO names, and `source_note`). `GET /flows/{id}` returns one flow. Both are public, read-only and need no LLM call or database (they read the committed file once at startup). Responses are cacheable.
- `/ask` is unchanged. A cited flow chunk appears in `sources` like any other passage, carrying its `source` label.

### Evals

- **Retrieval set and holdout:**
  - at least 3 questions per new message (pain.001, pacs.009, pain.002, pacs.004);
  - at least 8 flow questions, a few of which do not name a message;
  - at least 3 ambiguity checks, where the same element name exists in several messages and the question names one of them.
- **Golden set:**
  - flow questions with reference answers;
  - in-domain questions the assistant should decline, for example a direct debit flow or a camt.056 recall, which should escalate with `insufficient_context`.
- **Off-topic set unchanged.**
- **Re-measure, then set thresholds.** Re-measure everything after the change. Existing pacs.008/pacs.002 questions must still meet today's gates. Update `evals/thresholds.json` from the new measurement, using the same "measured minus margin" rule, and record the run in its comment.
- **Latency check.** Retrieval p50 before and after, measured the same way as the README's 85–90 ms figure. Measured and reported, not gated (John's decision, below). Token counts per question must be unchanged.

### Docs

- README: scope ("six messages and four flows"), status table, chunk counts, the flows file and its "not ISO text" label, and `GET /flows`.
- `data/SOURCES.md`: the four added messages under the existing e-Repository entry. There is no new source.
- Go-live plan (`plan/go-live` branch): the UI title and its "pacs.008 / pacs.002" wording, the UI showing the flows list from `GET /flows`, and the first-ingest time.

## Calls made after the builder check

These are Claude's calls; John can override any of them.

- **Partial ingests.** `cites` ids are validated against the chunks of the **whole committed extract**, not only the messages in one ingest request. An ingest includes a flow only when every message the flow uses is in the request. The default request is all six, so a default ingest includes every flow. For the `"flow"` group, the pipeline's delete-missing removes a flow whose messages were not all requested. That is intended: the index never holds a flow pointing at messages it lacks.
- **Unbacked steps.** If the ISO text in the extract does not support a step in scope, the step is softened to what the text does say, or dropped. A note in the flow records what was left out and why. Element ids are never cited "loosely".
- **No-regression gate.** Questions in the retrieval set and golden set get a `cohort` field: `original` for today's pacs.008/pacs.002 questions, `flows-v1` for the new ones. The gate reports metrics per cohort. `thresholds.json` gets keys for the `original` cohort set at today's minimums (0.8), and these run in CI with the existing retrieval gate, so a drop on old questions cannot hide behind gains on new ones.
- **"Not ISO text" label.** `/ask` `sources` entries gain a `source` field (the chunk's `source` metadata), which a UI shows next to each citation. The prompt line that names the corpus is reworded to name the six messages and "project flow summaries", at about the same length. No new prompt rule asks the model to label flow passages in its answer text.

## Decided by John after the builder check

- **Fact review of the flows file:** John reviews it before the PR merges. The flows file lands in its own commit so it can be read on its own.
- **Latency:** measure and report, not a hard gate. Measure retrieval p50 before and after, and report it in the PR and README. Optimise only if p50 grows by more than 25 ms.

## Left to the builder

- The exact flow wording, field names in the JSON, the flow intent pattern and factor, and the final eval questions (within the counts above).
- Whether CI caches embeddings to claw back build time. Not required at about 4–5 minutes.
- How `GET /flows` loads the file (startup vs lazy), as long as it does no per-request disk or database work.
