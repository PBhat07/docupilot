
from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384


class Embedder:
    """Generate normalized embeddings for documentation chunks."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        self.model.eval()

        self.dimension = self.model.get_sentence_embedding_dimension()

        if self.dimension != EXPECTED_DIMENSION:
            raise ValueError(
                f"Expected {EXPECTED_DIMENSION}-dimensional embeddings, "
                f"but model returned dimension {self.dimension}"
            )

    def _validate_texts(self, texts: Sequence[str]) -> None:
        max_length = self.model.max_seq_length

        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Text at index {index} must be a non-empty string"
                )

            token_ids = self.model.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
            )

            if len(token_ids) > max_length:
                raise ValueError(
                    f"Text at index {index} contains {len(token_ids)} tokens; "
                    f"model limit is {max_length}. Reduce the chunk size."
                )

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Return one normalized embedding vector per input text."""
        if not texts:
            return []

        self._validate_texts(texts)

        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        expected_shape = (len(texts), self.dimension)

        if embeddings.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected embedding shape {embeddings.shape}; "
                f"expected {expected_shape}"
            )

        if not np.isfinite(embeddings).all():
            raise RuntimeError("Embeddings contain non-finite values")

        return embeddings.tolist()
    def count_tokens(self, text: str) -> int:
        """Count tokens including special tokens, without truncation."""
        return len(
            self.model.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
            )
        )