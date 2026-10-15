# Architecture

## Current implementation

- FastAPI application with a health endpoint
- Dockerized Python environment
- Automated health endpoint test

## Planned components

- Scrapy-based documentation ingestion
- MongoDB for source and ingestion metadata
- Weaviate for vector retrieval
- Pretrained embeddings and cross-encoder reranking
- Evidence-grounded answer generation
- Streamlit user interface
- Redis-backed background ingestion jobs

## Design principles

- Reproducible development and deployment
- Source and version isolation
- Explicit error handling
- Measurable retrieval and answer quality
- Resource-aware model selection