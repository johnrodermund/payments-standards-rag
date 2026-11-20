# Payments standards Q&A assistant

A small RAG service over public payments documents (ISO 20022, FedNow, public SWIFT/CBPR+ material). Public documents only.

**Status: scaffold only.** Only the `/health` endpoint exists. Ingestion, retrieval, generation, tracing, evals, injection tests and deployment are not built yet. See [PLAN.md](PLAN.md).

## Sources and notices

This is a non-commercial personal project and is **not the official ISO 20022 website**. The sole source of up-to-date ISO 20022 message standards and the repository is https://www.iso20022.org/.

`data/processed/iso20022_pacs.json` is derived from the ISO 20022 e-Repository (pacs.008.001.14 and pacs.002.001.16 definitions), reproduced under the [ISO 20022 terms of use](https://www.iso20022.org/terms-use). The source file's SHA-256 is recorded in the JSON. Other sources and their licence notes are listed in [data/SOURCES.md](data/SOURCES.md).

## Run locally

```
docker compose up --build
curl http://localhost:8000/health
```

## Results

None yet. Eval and injection-test results will be added here only after they have been run.
