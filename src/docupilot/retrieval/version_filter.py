
from __future__ import annotations

from typing import Any


class CurrentVersionFilter:
    """Keep only chunks matching the latest version of each page in MongoDB."""

    def __init__(self, pages_collection: Any) -> None:
        self.pages_collection = pages_collection

    def filter_current(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Discard stale or unverifiable chunks before reranking."""
        if not candidates:
            return []

        # Identify pages represented in the retrieved candidates.
        page_keys: set[tuple[str, str]] = set()

        for candidate in candidates:
            properties = candidate.get("properties", {})

            if not isinstance(properties, dict):
                continue

            source_id = properties.get("source_id")
            canonical_url = properties.get("canonical_url")
            content_hash = properties.get("content_hash")

            if (
                isinstance(source_id, str)
                and source_id.strip()
                and isinstance(canonical_url, str)
                and canonical_url.strip()
                and isinstance(content_hash, str)
                and content_hash.strip()
            ):
                page_keys.add((source_id, canonical_url))

        if not page_keys:
            return []

        # Fetch the authoritative current version for these exact pages.
        query = {
            "$or": [
                {
                    "source_id": source_id,
                    "canonical_url": canonical_url,
                }
                for source_id, canonical_url in page_keys
            ]
        }

        current_pages = self.pages_collection.find(
            query,
            {
                "_id": 0,
                "source_id": 1,
                "canonical_url": 1,
                "content_hash": 1,
            },
        )

        current_hashes = {
            (page["source_id"], page["canonical_url"]): page["content_hash"]
            for page in current_pages
            if isinstance(page.get("source_id"), str)
            and isinstance(page.get("canonical_url"), str)
            and isinstance(page.get("content_hash"), str)
            and page["content_hash"].strip()
        }

        # Preserve retrieval order; only remove stale or unverifiable chunks.
        filtered: list[dict[str, Any]] = []

        for candidate in candidates:
            properties = candidate.get("properties", {})

            if not isinstance(properties, dict):
                continue

            source_id = properties.get("source_id")
            canonical_url = properties.get("canonical_url")
            content_hash = properties.get("content_hash")

            if (
                not isinstance(source_id, str)
                or not isinstance(canonical_url, str)
                or not isinstance(content_hash, str)
            ):
                continue

            page_key = (source_id, canonical_url)

            if current_hashes.get(page_key) == content_hash:
                filtered.append(candidate)

        return filtered
