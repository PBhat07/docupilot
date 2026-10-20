
from __future__ import annotations

import time
from typing import Any

from docupilot.indexing.embedder import Embedder
from docupilot.indexing.weaviate_store import WeaviateStore
from docupilot.retrieval.reranker import Reranker
from docupilot.retrieval.version_filter import CurrentVersionFilter
from collections.abc import Sequence


class RetrievalPipeline:
    """Coordinate hybrid retrieval and optional cross-encoder reranking."""

    def __init__(
        self,
        embedder: Embedder,
        store: WeaviateStore,
        version_filter: CurrentVersionFilter,
        reranker: Reranker | None = None,
        rerank_enabled: bool = True,
        candidate_limit: int = 20,
        final_limit: int = 3,
        alpha: float = 0.5,
    ) -> None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be at least 1")

        if final_limit < 1:
            raise ValueError("final_limit must be at least 1")

        if final_limit > candidate_limit:
            raise ValueError("final_limit cannot exceed candidate_limit")

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")

        if rerank_enabled and reranker is None:
            raise ValueError(
                "A reranker must be provided when rerank_enabled=True"
            )

        self.embedder = embedder
        self.store = store
        self.version_filter = version_filter
        self.reranker = reranker
        self.rerank_enabled = rerank_enabled
        self.candidate_limit = candidate_limit
        self.final_limit = final_limit
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        source_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant chunks and report query-stage timings."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        query = query.strip()

        # Stage 1: Embed the question.
        start = time.perf_counter()
        vectors = self.embedder.embed_texts([query])
        embedding_ms = (time.perf_counter() - start) * 1000

        if len(vectors) != 1:
            raise RuntimeError("Expected exactly one query embedding")

        # Stage 2: Retrieve candidates using hybrid search.
        start = time.perf_counter()
        candidates = self.store.search_hybrid(
            query_text=query,
            query_vector=vectors[0],
            limit=self.candidate_limit,
            alpha=self.alpha,
            source_ids=source_ids,
        )
        retrieval_ms = (time.perf_counter() - start) * 1000
        version_filter_start = time.perf_counter()

        current_candidates = self.version_filter.filter_current(
            candidates
        )

        version_filter_ms = (
            time.perf_counter() - version_filter_start
        ) * 1000

        # Stage 3: Optionally rerank candidates.
        reranking_ms = 0.0

        if self.rerank_enabled and current_candidates:
            assert self.reranker is not None

            start = time.perf_counter()
            chunks = self.reranker.rerank(
                query=query,
                candidates=current_candidates,
                top_k=self.final_limit,
            )
            reranking_ms = (time.perf_counter() - start) * 1000
        else:
            chunks = current_candidates[: self.final_limit]

        return {
            "query": query,
            "chunks": chunks,
            "candidate_count": len(candidates),
            "current_candidate_count": len(current_candidates),
            "stale_candidate_count": (len(candidates) - len(current_candidates)),
            "reranking_enabled": self.rerank_enabled,
            "timings_ms": {
                "embedding": round(embedding_ms, 3),
                "retrieval": round(retrieval_ms, 3),
                "version_filter": round(version_filter_ms, 3),
                "reranking": round(reranking_ms, 3),
                "retrieval_pipeline": round(
                    embedding_ms + retrieval_ms + version_filter_ms + reranking_ms,
                    3,
                ),
            },
        }
