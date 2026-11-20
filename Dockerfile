FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

COPY src ./src
RUN uv sync --no-dev

EXPOSE 8000
CMD ["uvicorn", "payments_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
