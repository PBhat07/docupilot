
import pytest

from docupilot.indexing.chunker import chunk_page


@pytest.fixture
def page():
    return {
        "source_id": "source-123",
        "canonical_url": "https://docs.example.com/guide",
        "title": "Installation Guide",
        "content_hash": "hash-abc",
        "headings": ["Installation"],
        "text": (
            "Installation\n"
            "Install the package using pip.\n"
            "pip install example-package"
        ),
    }


def test_short_page_produces_one_chunk(page):
    chunks = chunk_page(page)

    assert len(chunks) == 1
    assert chunks[0]["chunk_text"] == page["text"]
    assert chunks[0]["chunk_index"] == 0


def test_empty_page_produces_no_chunks(page):
    page["text"] = " \n  \n "

    assert chunk_page(page) == []


def test_long_text_produces_bounded_chunks(page):
    page["text"] = " ".join(f"word{i}" for i in range(250))

    chunks = chunk_page(
        page,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(len(chunk["chunk_text"]) <= 100 for chunk in chunks)


def test_adjacent_chunks_preserve_overlap(page):
    page["text"] = " ".join(f"word{i}" for i in range(100))

    chunks = chunk_page(
        page,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1

    first_words = chunks[0]["chunk_text"].split()
    second_words = chunks[1]["chunk_text"].split()

    # Verify that adjacent chunks share a consecutive sequence of words.
    overlap_found = any(
        first_words[-length:] == second_words[:length]
        for length in range(1, min(len(first_words), len(second_words)) + 1)
    )

    assert overlap_found


def test_chunk_ids_are_deterministic(page):
    first = chunk_page(page)
    second = chunk_page(page)

    assert [chunk["chunk_id"] for chunk in first] == [
        chunk["chunk_id"] for chunk in second
    ]


def test_changed_content_hash_changes_chunk_ids(page):
    original = chunk_page(page)

    page["content_hash"] = "hash-def"
    updated = chunk_page(page)

    assert original[0]["chunk_id"] != updated[0]["chunk_id"]


def test_chunk_preserves_source_metadata(page):
    chunk = chunk_page(page)[0]

    assert chunk["source_id"] == page["source_id"]
    assert chunk["canonical_url"] == page["canonical_url"]
    assert chunk["title"] == page["title"]
    assert chunk["content_hash"] == page["content_hash"]
    assert chunk["headings"] == page["headings"]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_invalid_chunk_settings_raise(page, chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        chunk_page(
            page,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


@pytest.mark.parametrize(
    "field",
    ["source_id", "canonical_url", "content_hash"],
)
def test_missing_required_metadata_raises(page, field):
    page[field] = ""

    with pytest.raises(ValueError):
        chunk_page(page)


def test_missing_text_raises(page):
    del page["text"]

    with pytest.raises(ValueError):
        chunk_page(page)


def test_non_string_text_raises(page):
    page["text"] = None

    with pytest.raises(ValueError):
        chunk_page(page)


# Test coverage:
# 1. Empty and whitespace-only input produces no chunks.
# 2. Short pages produce a single chunk.
# 3. Long text produces multiple chunks within the configured size limit.
# 4. Adjacent chunks preserve overlapping context.
# 5. Identical inputs produce deterministic chunk IDs.
# 6. Changed content hashes produce different chunk IDs.
# 7. Chunks preserve source metadata: URL, title, content hash, and headings.
# 8. Invalid chunk sizes and overlap settings raise ValueError.
# 9. Missing or empty required metadata raises ValueError.
# 10. Missing or non-string page text raises ValueError.