
from __future__ import annotations

import logging
import os
import subprocess
import sys
import uuid
import weaviate
from pathlib import Path

from pymongo import MongoClient

from docupilot.crawler.pipelines import make_source_id
from docupilot.crawler.url_utils import normalize_url
from docupilot.indexing.embedder import Embedder
from docupilot.indexing.indexer import Indexer
from docupilot.indexing.weaviate_store import WeaviateStore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_ingestion(
    start_url: str,
    max_pages: int = 10,
    max_depth: int = 1,
) -> dict:
    """Crawl a documentation source and verify persisted results."""

    root_url = normalize_url(start_url)
    if root_url is None:
        raise ValueError("start_url must be a valid HTTP(S) URL")

    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")

    if max_depth < 0:
        raise ValueError("max_depth cannot be negative")

    source_id = make_source_id(root_url)
    crawl_id = uuid.uuid4().hex

    env = os.environ.copy()
    env["MONGODB_URI"] = env.get(
        "MONGODB_URI", "mongodb://mongodb:27017"
    )
    env["MONGODB_DATABASE"] = env.get(
        "MONGODB_DATABASE", "docupilot"
    )

    command = [
        sys.executable,
        "-m",
        "scrapy",
        "runspider",
        "src/docupilot/crawler/spiders/docs_spider.py",
        "-a",
        f"start_url={root_url}",
        "-a",
        f"max_pages={max_pages}",
        "-a",
        f"max_depth={max_depth}",
        "-a",
        f"crawl_id={crawl_id}",
    ]

    logger.info(
        "Starting crawl %s for source %s",
        crawl_id,
        source_id,
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        logger.error("Crawler failed:\n%s", result.stderr[-5000:])
        raise RuntimeError(
            f"Crawler process failed with exit code {result.returncode}"
        )

    client = MongoClient(
        env["MONGODB_URI"],
        serverSelectionTimeoutMS=5000,
    )
    weaviate_client = None

    try:
        client.admin.command("ping")
        db = client[env["MONGODB_DATABASE"]]

        crawl = db["crawls"].find_one({"_id": crawl_id})
        if crawl is None:
            raise RuntimeError(
                f"Crawl {crawl_id} finished without a MongoDB crawl record"
            )

        pages_seen = db["pages"].count_documents(
            {"source_id": source_id}
        )
        versions_seen = db["page_versions"].count_documents(
            {"source_id": source_id}
        )
        errors_seen = db["crawl_errors"].count_documents(
            {"crawl_id": crawl_id}
        )

        if crawl.get("status") != "completed":
            raise RuntimeError(
                f"Crawl {crawl_id} did not complete successfully: "
                f"{crawl.get('status')!r}"
            )

        if pages_seen == 0:
            raise RuntimeError(
                f"Crawl {crawl_id} completed but source "
                f"{source_id} has no persisted current pages "
                f"(errors: {errors_seen})"
            )

        # Connect to Weaviate and prepare the existing indexer.
        weaviate_client = weaviate.connect_to_local(
            host="weaviate",
            port=8080,
            grpc_port=50051,
        )

        if not weaviate_client.is_ready():
            raise RuntimeError("Weaviate is not ready")

        embedder = Embedder(device="cpu")
        store = WeaviateStore(weaviate_client)

        indexer = Indexer(
            db=db,
            embedder=embedder,
            store=store,
        )

        # Restrict indexing to the source that was just crawled.
        indexing_stats = indexer.index_pages(source_id=source_id)

        logger.info(
            "Ingestion completed for source %s, crawl %s",
            source_id,
            crawl_id,
        )

        return {
            "status": "completed",
            "source_id": source_id,
            "crawl_id": crawl_id,
            "root_url": root_url,
            "pages_seen": pages_seen,
            "versions_seen": versions_seen,
            "errors_seen": errors_seen,
            "crawl_status": crawl.get("status"),
            "indexing": indexing_stats,
        }

    finally:
        if weaviate_client is not None:
            weaviate_client.close()

        client.close()
