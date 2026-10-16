
import math

import pytest

from docupilot.indexing.weaviate_store import WeaviateStore


@pytest.fixture
def store():
    instance = WeaviateStore.__new__(WeaviateStore)
    instance.expected_dimension = 3
    return instance


def test_valid_vector_is_accepted(store):
    vector = store._validate_vector([0.1, 0.2, 0.3])

    assert vector == [0.1, 0.2, 0.3]


def test_invalid_vector_dimension_raises(store):
    with pytest.raises(ValueError, match="dimensions"):
        store._validate_vector([0.1, 0.2])


@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, -math.inf])
def test_non_finite_vector_values_are_rejected(store, invalid_value):
    with pytest.raises(ValueError, match="non-finite"):
        store._validate_vector([0.1, invalid_value, 0.3])


def test_object_uuid_is_deterministic():
    first = WeaviateStore._object_uuid("chunk-123")
    second = WeaviateStore._object_uuid("chunk-123")

    assert first == second


def test_different_chunk_ids_produce_different_uuids():
    first = WeaviateStore._object_uuid("chunk-123")
    second = WeaviateStore._object_uuid("chunk-456")

    assert first != second


def test_mismatched_chunk_and_vector_counts_raise(store):
    with pytest.raises(ValueError, match="number of chunks"):
        store.upsert_chunks(
            chunks=[{"chunk_id": "chunk-1"}],
            vectors=[],
        )


def test_empty_upsert_returns_zero(store):
    assert store.upsert_chunks([], []) == 0
