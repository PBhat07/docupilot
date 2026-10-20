
from docupilot.retrieval.version_filter import CurrentVersionFilter


class FakePagesCollection:
    def __init__(self, pages):
        self.pages = pages
        self.last_query = None

    def find(self, query, projection=None):
        self.last_query = query

        clauses = query["$or"]

        return [
            page
            for page in self.pages
            if any(
                page.get("source_id") == clause["source_id"]
                and page.get("canonical_url") == clause["canonical_url"]
                for clause in clauses
            )
        ]


def make_chunk(source_id, url, content_hash, text):
    return {
        "properties": {
            "source_id": source_id,
            "canonical_url": url,
            "content_hash": content_hash,
            "chunk_text": text,
        }
    }


def test_keeps_chunk_matching_current_version():
    pages = FakePagesCollection([
        {
            "source_id": "source-a",
            "canonical_url": "https://docs.example.com/guide",
            "content_hash": "hash-new",
        }
    ])

    current_chunk = make_chunk(
        "source-a",
        "https://docs.example.com/guide",
        "hash-new",
        "Updated documentation",
    )

    result = CurrentVersionFilter(pages).filter_current([current_chunk])

    assert result == [current_chunk]


def test_rejects_stale_chunk():
    pages = FakePagesCollection([
        {
            "source_id": "source-a",
            "canonical_url": "https://docs.example.com/guide",
            "content_hash": "hash-new",
        }
    ])

    stale_chunk = make_chunk(
        "source-a",
        "https://docs.example.com/guide",
        "hash-old",
        "Old documentation",
    )

    result = CurrentVersionFilter(pages).filter_current([stale_chunk])

    assert result == []


def test_same_url_from_different_sources_is_isolated():
    url = "https://docs.example.com/guide"

    pages = FakePagesCollection([
        {
            "source_id": "source-a",
            "canonical_url": url,
            "content_hash": "hash-a",
        },
        {
            "source_id": "source-b",
            "canonical_url": url,
            "content_hash": "hash-b",
        },
    ])

    chunk_a = make_chunk("source-a", url, "hash-a", "Source A")
    stale_chunk_a = make_chunk("source-a", url, "hash-b", "Wrong version")
    chunk_b = make_chunk("source-b", url, "hash-b", "Source B")

    result = CurrentVersionFilter(pages).filter_current(
        [chunk_a, stale_chunk_a, chunk_b]
    )

    assert result == [chunk_a, chunk_b]


def test_empty_candidates_return_empty_list():
    pages = FakePagesCollection([])

    result = CurrentVersionFilter(pages).filter_current([])

    assert result == []
    assert pages.last_query is None


def test_rejects_chunk_with_missing_version_metadata():
    pages = FakePagesCollection([])

    chunk = {
        "properties": {
            "source_id": "source-a",
            "canonical_url": "https://docs.example.com/guide",
            "chunk_text": "Missing content hash",
        }
    }

    result = CurrentVersionFilter(pages).filter_current([chunk])

    assert result == []
