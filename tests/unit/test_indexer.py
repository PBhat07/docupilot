
from unittest.mock import MagicMock

import pytest

from docupilot.indexing.indexer import Indexer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def page():
    return {
        "source_id": "source-123",
        "canonical_url": "https://docs.example.com/guide",
        "title": "Installation Guide",
        "content_hash": "hash-abc",
        "headings": ["Installation"],
        "text": "Install the package using pip.",
    }


@pytest.fixture
def setup_indexer(page):
    db = MagicMock()
    db.pages.find.return_value = [page]

    embedder = MagicMock()
    embedder.model_name = MODEL_NAME
    embedder.embed_texts.return_value = [[0.1] * 384]

    store = MagicMock()
    store.get_page_chunks.return_value = []
    store.upsert_chunks.return_value = 1
    store.delete_page_chunks_except.return_value = 0

    indexer = Indexer(
        db=db,
        embedder=embedder,
        store=store,
    )

    return indexer, db, embedder, store, page


def test_indexes_new_page(setup_indexer):
    indexer, _, embedder, store, _ = setup_indexer

    stats = indexer.index_pages()

    embedder.embed_texts.assert_called_once()
    store.upsert_chunks.assert_called_once()
    store.delete_page_chunks_except.assert_called_once()

    assert stats["pages_seen"] == 1
    assert stats["pages_indexed"] == 1
    assert stats["pages_skipped"] == 0
    assert stats["chunks_written"] == 1


def test_skips_page_when_already_current(setup_indexer):
    indexer, _, embedder, store, page = setup_indexer

    # Simulate the same chunk already existing in Weaviate.
    from docupilot.indexing.chunker import chunk_page

    existing_chunks = chunk_page(page)
    existing_chunks[0]["embedding_model"] = MODEL_NAME
    store.get_page_chunks.return_value = existing_chunks

    stats = indexer.index_pages()

    embedder.embed_texts.assert_not_called()
    store.upsert_chunks.assert_not_called()
    store.delete_page_chunks_except.assert_not_called()

    assert stats["pages_skipped"] == 1
    assert stats["pages_indexed"] == 0


def test_changed_content_is_reindexed(setup_indexer):
    indexer, _, embedder, store, page = setup_indexer

    # Existing chunk has a different content hash.
    old_chunks = [
        {
            "chunk_id": "old-chunk",
            "source_id": page["source_id"],
            "canonical_url": page["canonical_url"],
            "title": page["title"],
            "content_hash": "old-hash",
            "chunk_index": 0,
            "chunk_text": "Old page content",
            "headings": page["headings"],
            "chunker_version": "1",
            "chunk_size": 800,
            "chunk_overlap": 120,
            "embedding_model": MODEL_NAME,
        }
    ]
    store.get_page_chunks.return_value = old_chunks

    stats = indexer.index_pages()

    embedder.embed_texts.assert_called_once()
    store.upsert_chunks.assert_called_once()
    store.delete_page_chunks_except.assert_called_once()

    assert stats["pages_indexed"] == 1


def test_embedding_failure_does_not_delete_existing_chunks(setup_indexer):
    indexer, _, embedder, store, _ = setup_indexer
    embedder.embed_texts.side_effect = RuntimeError("Embedding failed")

    with pytest.raises(RuntimeError, match="Embedding failed"):
        indexer.index_pages()

    store.upsert_chunks.assert_not_called()
    store.delete_page_chunks_except.assert_not_called()


def test_storage_failure_does_not_trigger_cleanup(setup_indexer):
    indexer, _, _, store, _ = setup_indexer
    store.upsert_chunks.side_effect = RuntimeError("Weaviate unavailable")

    with pytest.raises(RuntimeError, match="Weaviate unavailable"):
        indexer.index_pages()

    store.delete_page_chunks_except.assert_not_called()


def test_empty_page_is_skipped_without_deleting_chunks(setup_indexer):
    indexer, _, embedder, store, page = setup_indexer
    page["text"] = " \n  \n "

    stats = indexer.index_pages()

    embedder.embed_texts.assert_not_called()
    store.upsert_chunks.assert_not_called()
    store.delete_page_chunks_except.assert_not_called()

    assert stats["empty_pages_skipped"] == 1
    assert stats["pages_indexed"] == 0


def test_multiple_pages_are_processed(setup_indexer):
    indexer, db, embedder, store, page = setup_indexer

    second_page = {
        **page,
        "canonical_url": "https://docs.example.com/configuration",
        "content_hash": "hash-def",
        "text": "Configure the database connection.",
    }
    db.pages.find.return_value = [page, second_page]

    stats = indexer.index_pages()

    assert stats["pages_seen"] == 2
    assert stats["pages_indexed"] == 2
    assert embedder.embed_texts.call_count == 2
    assert store.upsert_chunks.call_count == 2



# Test coverage and purpose:
# 1. New page indexing: verifies chunking, embedding, and storage are connected.
# 2. Unchanged page: avoids unnecessary embedding and writes.
# 3. Changed content: ensures updated documentation is re-indexed.
# 4. Embedding failure: prevents cleanup when vector generation fails.
# 5. Storage failure: prevents cleanup when indexing fails.
# 6. Empty page: preserves existing indexed content if extraction is empty.
# 7. Multiple pages: verifies the coordinator processes more than one page.
#
# Why mocks?
# These tests isolate Indexer logic without starting MongoDB or Weaviate
# or loading the embedding model, making them fast and repeatable.
