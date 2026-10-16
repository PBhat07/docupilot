
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


CHUNKER_VERSION = "1"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def _stable_chunk_id(
    *,
    source_id: str,
    canonical_url: str,
    content_hash: str,
    chunk_index: int,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """Create a deterministic ID for a chunk and its indexing configuration."""
    identity = {
        "source_id": source_id,
        "canonical_url": canonical_url,
        "content_hash": content_hash,
        "chunk_index": chunk_index,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunker_version": CHUNKER_VERSION,
    }

    serialized = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    """Normalize whitespace while retaining useful line boundaries."""
    lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


def _find_chunk_end(
    text: str,
    start: int,
    max_end: int,
    min_end: int,
) -> int:
    """Prefer a newline or word boundary near the target chunk size."""
    if max_end >= len(text):
        return len(text)

    newline = text.rfind("\n", min_end, max_end + 1)
    if newline >= min_end:
        return newline

    space = text.rfind(" ", min_end, max_end + 1)
    if space >= min_end:
        return space

    # A single unbroken string may not contain a usable boundary.
    return max_end


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text into bounded, overlapping character windows."""
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        max_end = min(start + chunk_size, text_length)

        min_end = min(
            start + max(1, int(chunk_size * 0.6)),
            max_end,
        )

        end = _find_chunk_end(text, start, max_end, min_end)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Move the next window backwards to retain context.
        next_start = max(start + 1, end - chunk_overlap)

        # Avoid beginning the next chunk in the middle of a word.
        if next_start > 0 and text[next_start - 1] != "\n":
            boundary = text.find(" ", next_start, end)
            newline_boundary = text.find("\n", next_start, end)

            candidates = [
                position
                for position in (boundary, newline_boundary)
                if position != -1
            ]

            if candidates:
                next_start = min(candidates) + 1

        # Guarantee progress, even for unusual inputs.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def chunk_page(
    page: dict[str, Any],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Convert a stored documentation page into metadata-rich chunks.

    The page must contain source_id, canonical_url, content_hash and text.
    Empty pages return no chunks.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    required_fields = (
        "source_id",
        "canonical_url",
        "content_hash",
    )

    for field in required_fields:
        value = page.get(field)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Page requires a non-empty {field}")
        
    text_value = page.get("text")

    if not isinstance(text_value, str):
        raise ValueError("Page requires text as a string")     

    text = _normalize_text(text_value)

    if not text:
        return []

    raw_chunks = _split_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []

    for index, chunk_text in enumerate(raw_chunks):
        chunk_id = _stable_chunk_id(
            source_id=page["source_id"],
            canonical_url=page["canonical_url"],
            content_hash=page["content_hash"],
            chunk_index=index,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks.append(
            {
                "chunk_id": chunk_id,
                "source_id": page["source_id"],
                "canonical_url": page["canonical_url"],
                "title": page.get("title", ""),
                "content_hash": page["content_hash"],
                "headings": list(page.get("headings") or []),
                "chunk_index": index,
                "chunk_text": chunk_text,
                "chunker_version": CHUNKER_VERSION,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        )

    return chunks
