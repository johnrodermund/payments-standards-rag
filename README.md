# Payments standards Q&A assistant

A small RAG service over public payments documents (ISO 20022, FedNow, public SWIFT/CBPR+ material). Public documents only.

**Status: scaffold only.** Only the `/health` endpoint exists. Ingestion, retrieval, generation, tracing, evals, injection tests and deployment are not built yet. See [PLAN.md](PLAN.md).

## Run locally

```
docker compose up --build
curl http://localhost:8000/health
```

## Results

None yet. Eval and injection-test results will be added here only after they have been run.
