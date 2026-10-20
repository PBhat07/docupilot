
from __future__ import annotations

from typing import Any

from docupilot.indexing.chunker import (
    DEFAULT_MAX_TOKENS,
    chunk_page,
)
from docupilot.indexing.embedder import Embedder
from docupilot.indexing.weaviate_store import WeaviateStore


class Indexer:
    """Coordinate incremental indexing from MongoDB into Weaviate."""

    def __init__(
        self,
        db: Any,
        embedder: Embedder,
        store: WeaviateStore,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.store = store

    @staticmethod
    def _is_current(
        chunks: list[dict[str, Any]],
        existing: list[dict[str, Any]],
        embedding_model: str,
    ) -> bool:
        """Check whether the expected chunks and metadata are already indexed."""
        if not chunks or len(chunks) != len(existing):
            return False

        expected_by_id = {
            chunk["chunk_id"]: chunk for chunk in chunks
        }
        existing_by_id = {
            chunk.get("chunk_id"): chunk for chunk in existing
        }

        if expected_by_id.keys() != existing_by_id.keys():
            return False

        fields_to_compare = (
            "source_id",
            "canonical_url",
            "title",
            "content_hash",
            "chunk_index",
            "chunk_text",
            "headings",
            "chunker_version",
            "chunk_size",
            "chunk_overlap",
            "max_tokens",
        )

        for chunk_id, expected in expected_by_id.items():
            stored = existing_by_id[chunk_id]

            for field in fields_to_compare:
                if stored.get(field) != expected.get(field):
                    return False

            if stored.get("embedding_model") != embedding_model:
                return False

        return True

    def index_pages(
        self,
        source_id: str | None = None,
    ) -> dict[str, int]:
        """Index current pages, optionally restricted to one source."""
        if source_id is not None and not source_id.strip():
            raise ValueError("source_id must not be empty")

        stats = {
            "pages_seen": 0,
            "pages_indexed": 0,
            "pages_skipped": 0,
            "chunks_written": 0,
            "chunks_deleted": 0,
            "empty_pages_skipped": 0,
        }

        query: dict[str, Any] = {
            "text": {"$type": "string"},
        }

        if source_id is not None:
            query["source_id"] = source_id

        pages = self.db.pages.find(query)

        # Keep chunks within the embedding model's token limit.
        model_limit = self.embedder.model.max_seq_length
        token_budget = min(DEFAULT_MAX_TOKENS, model_limit - 1)

        for page in pages:
            stats["pages_seen"] += 1

            chunks = chunk_page(
                page,
                token_counter=self.embedder.count_tokens,
                max_tokens=token_budget,
            )

            # Preserve existing indexed content if extraction is empty.
            if not chunks:
                stats["empty_pages_skipped"] += 1
                continue

            page_source_id = page["source_id"]
            canonical_url = page["canonical_url"]

            existing = self.store.get_page_chunks(
                source_id=page_source_id,
                canonical_url=canonical_url,
            )

            if self._is_current(
                chunks,
                existing,
                self.embedder.model_name,
            ):
                stats["pages_skipped"] += 1
                continue

            # Generate every vector before modifying existing chunks.
            vectors = self.embedder.embed_texts(
                [chunk["chunk_text"] for chunk in chunks]
            )

            written = self.store.upsert_chunks(
                chunks,
                vectors,
                embedding_model=self.embedder.model_name,
            )
            stats["chunks_written"] += written

            # Delete obsolete chunks only after successful upserts.
            keep_ids = {chunk["chunk_id"] for chunk in chunks}

            deleted = self.store.delete_page_chunks_except(
                source_id=page_source_id,
                canonical_url=canonical_url,
                keep_chunk_ids=keep_ids,
            )

            stats["chunks_deleted"] += deleted
            stats["pages_indexed"] += 1

        return stats
