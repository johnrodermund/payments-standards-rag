# Payments standards assistant: API image (FastAPI on uvicorn).
#
# Build context must contain data/processed/iso20022_messages.json and data/flows/interbank_flows.json (committed) and, for a replay-only demo,
# .cache/llm (recorded answers; gitignored, so build from a checkout that has them).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed

# Dependencies first (cached layer). --frozen: install exactly what uv.lock pins.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# Bake the embedding model in so the container never downloads anything at start-up.
RUN python -c "from payments_rag.ingestion.embedding import FastEmbedEmbedder; FastEmbedEmbedder()"

# Data the service reads at run time: the pinned extract and the flows file (ingest source, provenance
# check and GET /flows) and the
# recorded model answers (used with LLM_REPLAY_ONLY=1).
COPY data/processed/iso20022_messages.json ./data/processed/iso20022_messages.json
COPY data/flows/interbank_flows.json ./data/flows/interbank_flows.json
COPY .cache/llm ./.cache/llm

# Run as an unprivileged user; it needs to write only the spend ledger / response cache.
RUN useradd --system --uid 10001 --home /app app && chown -R app /app/.cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "payments_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
