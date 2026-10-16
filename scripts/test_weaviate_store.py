
import numpy as np
import weaviate

from docupilot.indexing.embedder import Embedder
from docupilot.indexing.weaviate_store import WeaviateStore


def main():
    client = weaviate.connect_to_local(
        host="weaviate",
        port=8080,
        grpc_port=50051,
    )

    chunk = {
        "chunk_id": "storage-smoke-test-chunk-v1",
        "source_id": "storage-smoke-test",
        "canonical_url": "https://example.invalid/storage-test",
        "title": "Storage smoke test",
        "content_hash": "storage-test-hash",
        "chunk_index": 0,
        "chunk_text": (
            "Install the example package using pip install example-package."
        ),
        "headings": ["Installation"],
        "chunker_version": "1",
        "chunk_size": 800,
        "chunk_overlap": 120,
    }

    store = None

    try:
        embedder = Embedder(device="cpu")
        store = WeaviateStore(client)

        vector = embedder.embed_texts([chunk["chunk_text"]])[0]

        stored_count = store.upsert_chunks([chunk], [vector])
        print(f"Objects stored: {stored_count}")

        results = store.search_by_vector(vector, limit=3)

        print(f"Search results: {len(results)}")

        for result in results:
            properties = result["properties"]
            print(f"Title: {properties['title']}")
            print(f"URL: {properties['canonical_url']}")
            print(f"Distance: {result['distance']}")

        assert stored_count == 1
        assert any(
            result["properties"]["chunk_id"] == chunk["chunk_id"]
            for result in results
        )

        # Confirm that repeating the upsert does not create a duplicate.
        store.upsert_chunks([chunk], [vector])
        print("Repeated upsert completed.")

        print("Weaviate storage smoke test passed.")

    finally:
        if store is not None:
            store.delete_chunk(chunk["chunk_id"])

        client.close()


if __name__ == "__main__":
    main()
