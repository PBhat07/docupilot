
from __future__ import annotations

import weaviate
from pymongo import MongoClient

from docupilot.indexing.embedder import Embedder
from docupilot.indexing.weaviate_store import WeaviateStore
from docupilot.retrieval.pipeline import RetrievalPipeline
from docupilot.retrieval.reranker import Reranker
from docupilot.retrieval.version_filter import CurrentVersionFilter


MONGO_URI = "mongodb://mongodb:27017"
MONGO_DATABASE = "docupilot"

QUERY = "What sentence was added for the next crawl?"


def main() -> None:
    weaviate_client = None
    mongo_client = None

    try:
        weaviate_client = weaviate.connect_to_local(
            host="weaviate",
            port=8080,
            grpc_port=50051,
        )

        mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
        )
        mongo_client.admin.command("ping")

        database = mongo_client[MONGO_DATABASE]

        embedder = Embedder()
        store = WeaviateStore(weaviate_client)
        version_filter = CurrentVersionFilter(
            pages_collection=database["pages"],
        )
        reranker = Reranker(device="cpu")

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

        result = pipeline.retrieve(QUERY)

        print("\n=== VERSION-AWARE RETRIEVAL SMOKE TEST ===")
        print(f"Query: {result['query']}")
        print(f"Candidates retrieved: {result['candidate_count']}")
        print(
            "Current candidates: "
            f"{result['current_candidate_count']}"
        )
        print(
            "Stale/unverifiable candidates removed: "
            f"{result['stale_candidate_count']}"
        )
        print(f"Reranking enabled: {result['reranking_enabled']}")
        print(f"Timings (ms): {result['timings_ms']}")

        for index, chunk in enumerate(result["chunks"], start=1):
            properties = chunk.get("properties", {})

            print(f"\n--- Result {index} ---")
            print(f"Title: {properties.get('title')}")
            print(f"URL: {properties.get('canonical_url')}")
            print(f"Source ID: {properties.get('source_id')}")
            print(f"Content hash: {properties.get('content_hash')}")
            print(f"Chunk ID: {properties.get('chunk_id')}")
            print(f"Text: {properties.get('chunk_text')}")

    finally:
        if mongo_client is not None:
            mongo_client.close()

        if weaviate_client is not None:
            weaviate_client.close()


if __name__ == "__main__":
    main()
