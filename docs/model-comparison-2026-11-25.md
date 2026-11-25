# Generator comparison: claude-haiku-5-5 vs gpt-6-luna (2026-11-25)

**Question:** does OpenAI's `gpt-6-luna`, which costs the same as `claude-haiku-5-5` ($0.10 / $0.50 per 1M tokens), answer this assistant's questions as well?

**Setup (identical for both):**
- **Questions:** the 52-question golden set (44 answerable, 8 unanswerable).
- **Index:** the two-message index (1,245 chunks), built in a separate database. Retrieval matched the 20 Nov run exactly (hit@5 0.886), so both models saw the same top-5 passages.
- **Generation:** same system prompt, 700 output tokens, effort `low` (Claude `output_config.effort`, OpenAI `reasoning.effort`), injection guard on.
- **Judge:** TypeSafe JEV, a third model family, so neither generator grades itself.
- **Live vs replayed:** Haiku replayed 46 of its 52 answers from the 20 Nov cache, because the requests were byte-identical; the other 6 were live calls. All Luna answers were live.

## Results

| | claude-haiku-5-5 | gpt-6-luna |
|---|---|---|
| Answerable questions answered (not escalated) | 41 / 44 (0.932) | 40 / 44 (0.909) |
| Unanswerable questions escalated | 8 / 8 (1.0) | 8 / 8 (1.0) |
| Faithfulness, mean (JEV) | 0.891 | **0.920** |
| Answer relevancy, mean (JEV) | 0.893 | **0.927** |
| Tokens for the 52 questions, in / out | 82,185 / 6,004 | 51,280 / 1,504 |
| Cost for the 52 questions | ~$0.0112 | ~$0.0059 |
| Live latency p50 (10 questions, interleaved, no cache) | 853 ms (525–5,382) | 872 ms (657–2,118) |

**Reading it:**
- **Quality:** Luna scores slightly higher on faithfulness and relevancy, but one question moves a mean by about 0.02, so these are a single small run, not a proven gap.
- **The one changed outcome:** `gold-15`, "What codes are allowed for Settlement Method in pacs.008?". Luna escalated with `insufficient_context`. Haiku answered, while saying the passages don't hold a complete list. The retrieved passages really don't list the codes (this is a known retrieval miss recorded in `evals/thresholds.json`), so Luna's refusal is defensible.
- **Style:** Luna's answers are terser. It usually drops the type name and XML path that Haiku includes, such as "Max35Text" or "FIToFICstmrCdtTrf/GrpHdr/MsgId". JEV marked Haiku up for that detail on a couple of questions (gold-12, gold-03) and Luna up for brevity on others.
- **Cost:** Luna was about half the cost per question here, because it used fewer input tokens (a different tokenizer) and wrote shorter answers.
- **Speed:** the same at the median; Haiku had one slow outlier.

**Verdict:** Luna is good enough, so `gpt-6.1-terra` (10× the price) was not tried. Both are usable; the choice can rest on cost, provider preference and answer style.

**Not measured:** the six-message index and flow questions (on the `feat/interbank-flows` branch). Re-run this comparison once that merges.

Reproduce:

```
LLM_PROVIDER=openai OPENAI_MODEL=gpt-6-luna uv run python -m payments_rag.evals.rag_eval --judge jev --cache-dir .cache/cmp/luna --out .cache/cmp/luna.json
```
