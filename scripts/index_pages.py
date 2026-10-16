
import weaviate
from pymongo import MongoClient

from docupilot.indexing.embedder import Embedder
from docupilot.indexing.indexer import Indexer
from docupilot.indexing.weaviate_store import WeaviateStore


def main():
    mongo_client = MongoClient(
        "mongodb://mongodb:27017",
        serverSelectionTimeoutMS=5000,
    )
    weaviate_client = None

    try:
        mongo_client.admin.command("ping")

        weaviate_client = weaviate.connect_to_local(
            host="weaviate",
            port=8080,
            grpc_port=50051,
        )

        if not weaviate_client.is_ready():
            raise RuntimeError("Weaviate is not ready")

        db = mongo_client["docupilot"]
        embedder = Embedder(device="cpu")
        store = WeaviateStore(weaviate_client)

        indexer = Indexer(
            db=db,
            embedder=embedder,
            store=store,
        )

        stats = indexer.index_pages()

        print("Indexing completed.")
        for name, value in stats.items():
            print(f"{name}: {value}")

    finally:
        if weaviate_client is not None:
            weaviate_client.close()

        mongo_client.close()


if __name__ == "__main__":
    main()
