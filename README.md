# DocuPilot

**An evidence-grounded technical documentation assistant.** DocuPilot indexes documentation URLs, retrieves relevant passages, and answers questions with verifiable citations. It can abstain when the indexed evidence is insufficient.

## Demo

The Streamlit interface lets you register documentation sources, check ingestion status, ask questions, and inspect citations.

> **Demo media:** Add a real screen recording after running the UI. Save the GIF as `docs/assets/docupilot-demo.gif` and the short video as `docs/assets/docupilot-demo.mp4`. The README references are intentionally not embedded until those files exist.

## Features

- Documentation URL registration and configurable crawling.
- Redis Queue worker for background ingestion.
- Hybrid retrieval with cross-encoder reranking.
- Evidence-grounded answers with verifiable citations and abstention.
- MongoDB metadata and version tracking; Weaviate vector search.
- FastAPI backend, CLI, and optional Streamlit UI.

## Architecture

```text
Browser (Streamlit :8501) ──HTTP──> FastAPI (:8000)
                                      ├── Question → retrieval/reranking → answer + citations
                                      └── Source URL → Redis/RQ worker → crawl and index
                                                               ├── MongoDB metadata
                                                               └── Weaviate vectors
```

## Run with Docker Compose

### 1. Configure environment

```bash
cp .env.example .env
```

Set the LLM provider variables required by the backend in `.env`. Never commit API keys or the `.env` file.

### 2. Start backend services

```bash
docker compose up --build -d
docker compose ps
```

This starts the API, worker, Redis, MongoDB, and Weaviate. API docs: [http://localhost:8000/docs](http://localhost:8000/docs).

### 3. Start the optional UI

In another terminal:

```bash
docker compose --profile ui up --build -d ui
```

Open [http://localhost:8501](http://localhost:8501).

1. Enter a documentation URL in the sidebar and submit it for indexing.
2. Check the source status and wait for the background job to finish.
3. Ask a question. Optionally select sources to restrict retrieval.
4. Expand citations to inspect the source links.

### CLI

The CLI uses the same API:

```bash
docker compose exec api python -m docupilot.cli
```

### Logs and shutdown

```bash
docker compose logs -f api worker
docker compose logs -f ui
docker compose --profile ui stop ui
docker compose down
```

The UI is opt-in and is not started by the default Compose command. Named volumes persist MongoDB, Weaviate, and Redis data. `docker compose down -v` deletes those volumes and their stored data.

## Optional evaluation

A five-question reranking comparison is available under `evaluation/`. It is an exploratory engineering check, not proof that reranking always improves answer quality. See [evaluation/README.md](evaluation/README.md) for commands, outputs, dataset format, and limitations.

## Limitations and future work

- The evaluation dataset is small; broader evaluation is needed to determine when reranking justifies its latency and compute cost.
- Crawl coverage depends on site structure and crawl limits.
- LLM provider availability and quotas can affect answer generation.
- Future work includes broader retrieval/answer evaluation, repeated latency measurements, and evaluating when reranking should be enabled.

## Reproducibility

Use Docker Compose and the same `.env` configuration to reproduce the service setup. The named volumes preserve indexed data between runs. Record the commit SHA, provider/model, and dataset version when comparing results. Only remove volumes when you intentionally want to delete stored data.
