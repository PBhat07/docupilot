
from __future__ import annotations

import math
import uuid
from typing import Any, Sequence

from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery
from weaviate.classes.query import Filter


COLLECTION_NAME = "DocuPilotChunks"
EXPECTED_DIMENSION = 384
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class WeaviateStore:
    """Store and retrieve documentation chunks with custom embeddings."""

    def __init__(
        self,
        client: Any,
        collection_name: str = COLLECTION_NAME,
        expected_dimension: int = EXPECTED_DIMENSION,
    ) -> None:
        if expected_dimension < 1:
            raise ValueError("expected_dimension must be positive")

        self.client = client
        self.collection_name = collection_name
        self.expected_dimension = expected_dimension

        self._ensure_collection()
        self.collection = self.client.collections.get(collection_name)

    def _ensure_collection(self) -> None:
        """Create the collection only if it does not already exist."""
        if self.client.collections.exists(self.collection_name):
            return

        self.client.collections.create(
            name=self.collection_name,
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="source_id", data_type=DataType.TEXT),
                Property(name="canonical_url", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="content_hash", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="chunk_text", data_type=DataType.TEXT),
                Property(name="headings", data_type=DataType.TEXT_ARRAY),
                Property(name="chunker_version", data_type=DataType.TEXT),
                Property(name="chunk_size", data_type=DataType.INT),
                Property(name="chunk_overlap", data_type=DataType.INT),
                Property(name="embedding_model", data_type=DataType.TEXT),
            ],
        )

    @staticmethod
    def _object_uuid(chunk_id: str) -> uuid.UUID:
        """Generate the same Weaviate UUID for the same chunk ID."""
        return uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"docupilot:{chunk_id}",
        )

    def _validate_vector(self, vector: Sequence[float]) -> list[float]:
        if len(vector) != self.expected_dimension:
            raise ValueError(
                f"Expected {self.expected_dimension} vector dimensions, "
                f"received {len(vector)}"
            )

        values = [float(value) for value in vector]

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Vector contains non-finite values")

        return values

    def upsert_chunks(
        self,
        chunks: Sequence[dict[str, Any]],
        vectors: Sequence[Sequence[float]],
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> int:
        """Insert new chunks or replace existing objects using stable IDs."""
        if len(chunks) != len(vectors):
            raise ValueError(
                "The number of chunks must match the number of vectors"
            )

        for chunk, vector in zip(chunks, vectors):
            chunk_id = chunk.get("chunk_id")

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("Each chunk requires a non-empty chunk_id")

            required = (
                "source_id",
                "canonical_url",
                "content_hash",
                "chunk_text",
                "chunk_index",
            )
            for field in required:
                if field not in chunk or chunk[field] is None:
                    raise ValueError(f"Chunk requires {field}")

            properties = {
                "chunk_id": chunk_id,
                "source_id": chunk["source_id"],
                "canonical_url": chunk["canonical_url"],
                "title": chunk.get("title", ""),
                "content_hash": chunk["content_hash"],
                "chunk_index": chunk["chunk_index"],
                "chunk_text": chunk["chunk_text"],
                "headings": list(chunk.get("headings") or []),
                "chunker_version": chunk.get("chunker_version", "1"),
                "chunk_size": chunk.get("chunk_size", 800),
                "chunk_overlap": chunk.get("chunk_overlap", 120),
                "embedding_model": embedding_model,
            }

            object_uuid = self._object_uuid(chunk_id)
            validated_vector = self._validate_vector(vector)

            if self.collection.data.exists(object_uuid):
                self.collection.data.replace(
                    uuid=object_uuid,
                    properties=properties,
                    vector=validated_vector,
                )
            else:
                self.collection.data.insert(
                    uuid=object_uuid,
                    properties=properties,
                    vector=validated_vector,
                )

        return len(chunks)

    def search_by_vector(
        self,
        query_vector: Sequence[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the nearest chunks for a supplied query embedding."""
        if limit < 1:
            raise ValueError("limit must be at least 1")

        vector = self._validate_vector(query_vector)

        response = self.collection.query.near_vector(
            near_vector=vector,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )

        return [
            {
                "uuid": str(obj.uuid),
                "distance": obj.metadata.distance,
                "properties": obj.properties,
            }
            for obj in response.objects
        ]
        
    
    def search_hybrid(
        self,
        query_text: str,
        query_vector: Sequence[float],
        limit: int = 20,
        alpha: float = 0.5,
        source_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks using combined BM25 and vector search.

        alpha=0.0 uses lexical search only.
        alpha=1.0 uses vector search only.
        alpha=0.5 balances both retrieval signals.
        """
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string")

        if limit < 1:
            raise ValueError("limit must be at least 1")

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        
        if source_ids is not None:
            if (
                isinstance(source_ids, (str, bytes))
                or not isinstance(source_ids, Sequence)
                or not source_ids
            ):
                raise ValueError(
                    "source_ids must be a non-empty sequence of source IDs"
                )

            if any(
                not isinstance(source_id, str) or not source_id.strip()
                for source_id in source_ids
            ):
                raise ValueError(
                    "source_ids must contain non-empty strings"
                )

            source_ids = list(
                dict.fromkeys(source_id.strip() for source_id in source_ids)
            )

        vector = self._validate_vector(query_vector)

        search_kwargs: dict[str, Any] = {
            "query": query_text.strip(),
            "vector": vector,
            "alpha": alpha,
            "limit": limit,
            "return_metadata": MetadataQuery(
                score=True,
                explain_score=True,
            ),
        }

        if source_ids is not None:
            search_kwargs["filters"] = Filter.any_of(
                [
                    Filter.by_property("source_id").equal(source_id)
                    for source_id in source_ids
                ]
            )

        response = self.collection.query.hybrid(**search_kwargs)

        return [
            {
                "uuid": str(obj.uuid),
                "score": obj.metadata.score,
                "explain_score": obj.metadata.explain_score,
                "properties": obj.properties,
            }
            for obj in response.objects
        ]
    

    def delete_chunk(self, chunk_id: str) -> None:
        """Delete one chunk by its stable ID."""
        object_uuid = self._object_uuid(chunk_id)

        if self.collection.data.exists(object_uuid):
            self.collection.data.delete_by_id(object_uuid)
            
    
    def get_page_chunks(
        self,
        source_id: str,
        canonical_url: str,
    ) -> list[dict[str, Any]]:
        """Fetch indexed chunks belonging to one source page."""
        filters = Filter.all_of(
            [
                Filter.by_property("source_id").equal(source_id),
                Filter.by_property("canonical_url").equal(canonical_url),
            ]
        )

        response = self.collection.query.fetch_objects(
            filters=filters,
            limit=10_000,
            return_properties=[
                "chunk_id",
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
                "embedding_model",
            ],
        )

        return [
            {
                "uuid": str(obj.uuid),
                **obj.properties,
            }
            for obj in response.objects
        ]

    def delete_page_chunks_except(
        self,
        source_id: str,
        canonical_url: str,
        keep_chunk_ids: set[str],
    ) -> int:
        """Remove obsolete chunks from one page, preserving the supplied IDs."""
        existing_chunks = self.get_page_chunks(
            source_id=source_id,
            canonical_url=canonical_url,
        )

        deleted = 0

        for chunk in existing_chunks:
            if chunk.get("chunk_id") not in keep_chunk_ids:
                self.collection.data.delete_by_id(chunk["uuid"])
                deleted += 1

        return deleted
    
    
            
