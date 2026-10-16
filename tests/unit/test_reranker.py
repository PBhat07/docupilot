
from unittest.mock import MagicMock

import pytest

from docupilot.retrieval.reranker import Reranker


@pytest.fixture
def mock_reranker(monkeypatch):
    """Create a reranker with a mocked cross-encoder model."""
    mock_model = MagicMock()
    mock_cross_encoder = MagicMock(return_value=mock_model)

    monkeypatch.setattr(
        "docupilot.retrieval.reranker.CrossEncoder",
        mock_cross_encoder,
    )

    reranker = Reranker()

    return reranker, mock_model


def make_candidates():
    """Create sample retrieval candidates."""
    return [
        {
            "uuid": "chunk-1",
            "properties": {"chunk_text": "Install the package using pip."},
        },
        {
            "uuid": "chunk-2",
            "properties": {"chunk_text": "Configure the database connection."},
        },
        {
            "uuid": "chunk-3",
            "properties": {"chunk_text": "Start the application server."},
        },
    ]


def test_reranker_sorts_by_score_and_returns_top_k(mock_reranker):
    reranker, model = mock_reranker
    candidates = make_candidates()

    model.predict.return_value = [0.2, 0.9, 0.5]

    results = reranker.rerank(
        query="How do I configure the database?",
        candidates=candidates,
        top_k=2,
    )

    assert len(results) == 2
    assert [item["uuid"] for item in results] == ["chunk-2", "chunk-3"]
    assert results[0]["rerank_score"] == pytest.approx(0.9)
    assert results[1]["rerank_score"] == pytest.approx(0.5)


def test_reranker_passes_query_and_chunks_to_model(mock_reranker):
    reranker, model = mock_reranker
    candidates = make_candidates()

    model.predict.return_value = [0.1, 0.2, 0.3]

    reranker.rerank(
        query="  Find database configuration  ",
        candidates=candidates,
        top_k=3,
    )

    model.predict.assert_called_once_with(
        [
            ("Find database configuration", "Install the package using pip."),
            ("Find database configuration", "Configure the database connection."),
            ("Find database configuration", "Start the application server."),
        ],
        show_progress_bar=False,
    )


def test_reranker_returns_empty_list_for_no_candidates(mock_reranker):
    reranker, model = mock_reranker

    results = reranker.rerank(
        query="How do I install the package?",
        candidates=[],
    )

    assert results == []
    model.predict.assert_not_called()


@pytest.mark.parametrize("query", ["", "   ", None])
def test_reranker_rejects_empty_query(mock_reranker, query):
    reranker, _ = mock_reranker

    with pytest.raises(ValueError):
        reranker.rerank(
            query=query,
            candidates=make_candidates(),
        )


@pytest.mark.parametrize("top_k", [0, -1])
def test_reranker_rejects_invalid_top_k(mock_reranker, top_k):
    reranker, _ = mock_reranker

    with pytest.raises(ValueError):
        reranker.rerank(
            query="Find installation instructions",
            candidates=make_candidates(),
            top_k=top_k,
        )


def test_reranker_rejects_candidate_without_properties(mock_reranker):
    reranker, _ = mock_reranker

    with pytest.raises(ValueError, match="requires properties"):
        reranker.rerank(
            query="Find installation instructions",
            candidates=[{"uuid": "chunk-1"}],
        )


@pytest.mark.parametrize(
    "properties",
    [
        {},
        {"chunk_text": ""},
        {"chunk_text": "   "},
        {"chunk_text": None},
    ],
)
def test_reranker_rejects_missing_or_empty_chunk_text(
    mock_reranker,
    properties,
):
    reranker, _ = mock_reranker

    with pytest.raises(ValueError, match="chunk_text"):
        reranker.rerank(
            query="Find installation instructions",
            candidates=[{"properties": properties}],
        )


def test_reranker_rejects_mismatched_score_count(mock_reranker):
    reranker, model = mock_reranker
    model.predict.return_value = [0.5]

    with pytest.raises(RuntimeError, match="score count"):
        reranker.rerank(
            query="Find installation instructions",
            candidates=make_candidates(),
            top_k=2,
        )


# Test coverage:
# 1. Verifies descending reranker-score ordering and top-k selection.
# 2. Verifies query trimming and query/chunk pairs sent to the model.
# 3. Verifies empty candidate lists return an empty result without inference.
# 4. Checks rejection of empty queries and invalid top-k values.
# 5. Checks rejection of candidates with missing properties or chunk text.
# 6. Checks handling of mismatched model-score and candidate counts.
# 7. Uses a mocked cross-encoder to avoid loading or downloading the model.
