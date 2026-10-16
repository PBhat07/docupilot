
from __future__ import annotations

from typing import Any, Sequence

from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Rerank retrieved documentation chunks for a query."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return the top-k candidates ranked by cross-encoder score."""
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if not candidates:
            return []

        pairs = []

        for candidate in candidates:
            properties = candidate.get("properties")

            if not isinstance(properties, dict):
                raise ValueError("Each candidate requires properties")

            text = properties.get("chunk_text")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "Each candidate requires non-empty chunk_text"
                )

            pairs.append((query.strip(), text))

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        if len(scores) != len(candidates):
            raise RuntimeError("Reranker score count does not match candidates")

        ranked = []

        for candidate, score in zip(candidates, scores):
            result = dict(candidate)
            result["rerank_score"] = float(score)
            ranked.append(result)

        ranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return ranked[:top_k]
