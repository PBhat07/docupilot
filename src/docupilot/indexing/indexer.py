
from __future__ import annotations

from typing import Any

from docupilot.indexing.chunker import chunk_page
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
        )

        for chunk_id, expected in expected_by_id.items():
            stored = existing_by_id[chunk_id]

            for field in fields_to_compare:
                if stored.get(field) != expected.get(field):
                    return False

            if stored.get("embedding_model") != embedding_model:
                return False

        return True

    def index_pages(self) -> dict[str, int]:
        """Index current pages and clean up stale chunks after successful writes."""
        stats = {
            "pages_seen": 0,
            "pages_indexed": 0,
            "pages_skipped": 0,
            "chunks_written": 0,
            "chunks_deleted": 0,
            "empty_pages_skipped": 0,
        }

        pages = self.db.pages.find(
            {"text": {"$type": "string"}}
        )

        for page in pages:
            stats["pages_seen"] += 1

            chunks = chunk_page(page)

            # Empty extraction may indicate a crawler or parser problem.
            # Preserve any previously indexed content in that case.
            if not chunks:
                stats["empty_pages_skipped"] += 1
                continue

            source_id = page["source_id"]
            canonical_url = page["canonical_url"]

            existing = self.store.get_page_chunks(
                source_id=source_id,
                canonical_url=canonical_url,
            )

            if self._is_current(
                chunks,
                existing,
                self.embedder.model_name,
            ):
                stats["pages_skipped"] += 1
                continue

            # Generate every vector before modifying the existing index.
            vectors = self.embedder.embed_texts(
                [chunk["chunk_text"] for chunk in chunks]
            )

            # Write the new/current chunks before removing old ones.
            written = self.store.upsert_chunks(
                chunks,
                vectors,
                embedding_model=self.embedder.model_name,
            )
            stats["chunks_written"] += written

            # Remove obsolete chunks only after all upserts return successfully.
            keep_ids = {chunk["chunk_id"] for chunk in chunks}

            deleted = self.store.delete_page_chunks_except(
                source_id=source_id,
                canonical_url=canonical_url,
                keep_chunk_ids=keep_ids,
            )

            stats["chunks_deleted"] += deleted
            stats["pages_indexed"] += 1

        return stats
