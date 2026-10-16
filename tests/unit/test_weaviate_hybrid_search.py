
from unittest.mock import MagicMock

import pytest

from docupilot.indexing.weaviate_store import WeaviateStore


def make_store():
    """Create a store with a mocked Weaviate collection."""
    store = WeaviateStore.__new__(WeaviateStore)
    store.collection = MagicMock()
    store.expected_dimension = 384
    return store


def test_hybrid_search_returns_result_structure():
    store = make_store()

    obj = MagicMock()
    obj.uuid = "test-uuid"
    obj.metadata.score = 0.85
    obj.metadata.explain_score = "hybrid score explanation"
    obj.properties = {
        "chunk_text": "Install the package using pip.",
        "source_id": "source-1",
    }

    response = MagicMock()
    response.objects = [obj]
    store.collection.query.hybrid.return_value = response

    results = store.search_hybrid(
        query_text="How do I install the package?",
        query_vector=[0.1] * 384,
        limit=5,
        alpha=0.5,
    )

    assert len(results) == 1
    assert results[0]["uuid"] == "test-uuid"
    assert results[0]["score"] == 0.85
    assert results[0]["explain_score"] == "hybrid score explanation"
    assert results[0]["properties"]["chunk_text"] == (
        "Install the package using pip."
    )


def test_hybrid_search_passes_parameters_to_weaviate():
    store = make_store()
    response = MagicMock()
    response.objects = []
    store.collection.query.hybrid.return_value = response

    vector = [0.1] * 384

    store.search_hybrid(
        query_text="  installation instructions  ",
        query_vector=vector,
        limit=10,
        alpha=0.3,
    )

    store.collection.query.hybrid.assert_called_once()

    kwargs = store.collection.query.hybrid.call_args.kwargs
    assert kwargs["query"] == "installation instructions"
    assert kwargs["vector"] == vector
    assert kwargs["limit"] == 10
    assert kwargs["alpha"] == 0.3


@pytest.mark.parametrize("query", ["", "   ", None, 123])
def test_hybrid_search_rejects_invalid_query(query):
    store = make_store()

    with pytest.raises(ValueError):
        store.search_hybrid(
            query_text=query,
            query_vector=[0.1] * 384,
            limit=5,
            alpha=0.5,
        )


@pytest.mark.parametrize("limit", [0, -1])
def test_hybrid_search_rejects_invalid_limit(limit):
    store = make_store()

    with pytest.raises(ValueError):
        store.search_hybrid(
            query_text="installation",
            query_vector=[0.1] * 384,
            limit=limit,
            alpha=0.5,
        )


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_hybrid_search_rejects_invalid_alpha(alpha):
    store = make_store()

    with pytest.raises(ValueError):
        store.search_hybrid(
            query_text="installation",
            query_vector=[0.1] * 384,
            limit=5,
            alpha=alpha,
        )

# Test coverage:
# 1. Verifies the structure and contents of hybrid-search results.
# 2. Verifies query trimming and parameter forwarding to Weaviate.
# 3. Checks rejection of empty, whitespace-only, None, and non-string queries.
# 4. Checks rejection of invalid search limits (zero or negative).
# 5. Checks rejection of alpha values outside the valid range [0.0, 1.0].
# 6. Uses a mocked Weaviate collection to run tests without a live database.
