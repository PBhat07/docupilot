
from unittest.mock import MagicMock

import pytest

from docupilot.retrieval.pipeline import RetrievalPipeline
from docupilot.retrieval.version_filter import CurrentVersionFilter

@pytest.fixture
def dependencies():
    """Create mocked dependencies for the retrieval pipeline."""
    embedder = MagicMock()
    store = MagicMock()
    reranker = MagicMock()

    embedder.embed_texts.return_value = [[0.1] * 384]
    store.search_hybrid.return_value = [
        {
            "uuid": "chunk-1",
            "score": 0.8,
            "properties": {"chunk_text": "Install using pip."},
        },
        {
            "uuid": "chunk-2",
            "score": 0.6,
            "properties": {"chunk_text": "Configure the database."},
        },
    ]
    reranker.rerank.return_value = [
        {
            "uuid": "chunk-2",
            "rerank_score": 0.9,
            "properties": {"chunk_text": "Configure the database."},
        }
    ]

    version_filter = MagicMock(spec=CurrentVersionFilter)
    version_filter.filter_current.side_effect = lambda candidates: candidates

    return embedder, store, reranker, version_filter


def test_pipeline_returns_reranked_chunks(dependencies):
    embedder, store, reranker, version_filter = dependencies

    pipeline = RetrievalPipeline(
        embedder=embedder,
        store=store,
        version_filter=version_filter,
        reranker=reranker,
        rerank_enabled=True,
        candidate_limit=20,
        final_limit=3,
        alpha=0.5,
    )

    result = pipeline.retrieve("  How do I configure the database?  ")

    assert result["query"] == "How do I configure the database?"
    assert result["chunks"][0]["uuid"] == "chunk-2"
    assert result["candidate_count"] == 2
    assert result["reranking_enabled"] is True

    embedder.embed_texts.assert_called_once_with(
        ["How do I configure the database?"]
    )
    store.search_hybrid.assert_called_once_with(
        query_text="How do I configure the database?",
        query_vector=[0.1] * 384,
        limit=20,
        alpha=0.5,
        source_ids=None,
    )
    reranker.rerank.assert_called_once()


def test_pipeline_skips_reranking_when_disabled(dependencies):
    embedder, store, reranker, version_filter = dependencies

    pipeline = RetrievalPipeline(
        embedder=embedder,
        store=store,
        version_filter=version_filter,
        reranker=None,
        rerank_enabled=False,
        candidate_limit=20,
        final_limit=1,
    )

    result = pipeline.retrieve("Find installation instructions")

    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["uuid"] == "chunk-1"
    assert result["reranking_enabled"] is False
    reranker.rerank.assert_not_called()


def test_pipeline_returns_empty_chunks_when_no_candidates(dependencies):
    embedder, store, reranker, version_filter = dependencies
    store.search_hybrid.return_value = []

    pipeline = RetrievalPipeline(
        embedder=embedder,
        store=store,
        version_filter=version_filter,
        reranker=reranker,
        rerank_enabled=True,
    )

    result = pipeline.retrieve("Find an answer")

    assert result["chunks"] == []
    assert result["candidate_count"] == 0
    reranker.rerank.assert_not_called()


@pytest.mark.parametrize("query", ["", "   ", None, 123])
def test_pipeline_rejects_invalid_query(dependencies, query):
    embedder, store, reranker, version_filter = dependencies

    pipeline = RetrievalPipeline(
        embedder=embedder,
        store=store,
        version_filter=version_filter,
        reranker=reranker,
    )

    with pytest.raises(ValueError):
        pipeline.retrieve(query)

    embedder.embed_texts.assert_not_called()
    store.search_hybrid.assert_not_called()


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("candidate_limit", 0),
        ("candidate_limit", -1),
        ("final_limit", 0),
        ("final_limit", -1),
        ("alpha", -0.1),
        ("alpha", 1.1),
    ],
)
def test_pipeline_rejects_invalid_configuration(
    dependencies, setting, value
):
    embedder, store, reranker, version_filter = dependencies

    kwargs = {
        "embedder": embedder,
        "store": store,
        "version_filter": version_filter,
        "reranker": reranker,
    }
    kwargs[setting] = value

    with pytest.raises(ValueError):
        RetrievalPipeline(**kwargs)


def test_pipeline_rejects_final_limit_greater_than_candidate_limit(
    dependencies,
):
    embedder, store, reranker, version_filter = dependencies

    with pytest.raises(ValueError, match="final_limit"):
        RetrievalPipeline(
            embedder=embedder,
            store=store,
            version_filter=version_filter,
            reranker=reranker,
            candidate_limit=2,
            final_limit=3,
        )


def test_pipeline_requires_reranker_when_enabled(dependencies):
    embedder, store,_, version_filter = dependencies

    with pytest.raises(ValueError, match="reranker"):
        RetrievalPipeline(
            embedder=embedder,
            store=store,
            version_filter=version_filter,
            rerank_enabled=True,
        )


def test_pipeline_rejects_unexpected_embedding_count(dependencies):
    embedder, store, reranker, version_filter = dependencies
    embedder.embed_texts.return_value = []

    pipeline = RetrievalPipeline(
        embedder=embedder,
        store=store,
        reranker=reranker,
        version_filter=version_filter,
    )

    with pytest.raises(RuntimeError, match="embedding"):
        pipeline.retrieve("Find installation instructions")

    store.search_hybrid.assert_not_called()


def test_pipeline_reports_stage_timings(dependencies):
    embedder, store, reranker, version_filter = dependencies

    pipeline = RetrievalPipeline(
        embedder=embedder,
        store=store,
        reranker=reranker,
        rerank_enabled=True,
        version_filter=version_filter,
    )

    result = pipeline.retrieve("Find installation instructions")
    timings = result["timings_ms"]

    assert set(timings) == {
        "embedding",
        "retrieval",
        "version_filter",
        "reranking",
        "retrieval_pipeline",
    }
    assert all(isinstance(value, float) for value in timings.values())
    assert all(value >= 0 for value in timings.values())

    expected_total = (
        timings["embedding"]
        + timings["retrieval"]
        + timings["version_filter"]
        + timings["reranking"]
    )
    assert timings["retrieval_pipeline"] == pytest.approx(
        expected_total, abs=0.003
    )


# Test coverage:
# 1. Verifies the complete embedding -> hybrid retrieval -> reranking flow.
# 2. Verifies query trimming and correct dependency arguments.
# 3. Verifies retrieval works with reranking disabled.
# 4. Verifies empty candidate results do not invoke the reranker.
# 5. Checks invalid queries and pipeline configuration.
# 6. Checks missing reranker configuration and invalid embedding counts.
# 7. Verifies query-stage timing fields are present and non-negative.
# 8. Uses mocks to avoid requiring a live Weaviate instance or real models.
