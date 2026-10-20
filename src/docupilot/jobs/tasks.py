
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from pymongo import MongoClient
from rq import get_current_job

from docupilot.crawler.pipelines import make_source_id
from docupilot.crawler.url_utils import normalize_url
from docupilot.jobs.ingestion import run_ingestion

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def worker_smoke_test() -> dict[str, str]:
    """Verify that the RQ worker can execute a DocuPilot task."""
    return {
        "status": "ok",
        "worker": "docupilot",
    }


def ingest_source(
    start_url: str,
    max_pages: int = 10,
    max_depth: int = 1,
) -> dict:
    """Run ingestion as an RQ job and persist the source's job status."""
    root_url = normalize_url(start_url)
    if root_url is None:
        raise ValueError("start_url must be a valid HTTP(S) URL")

    source_id = make_source_id(root_url)
    job = get_current_job()
    job_id = job.id if job is not None else None

    mongo_uri = os.getenv(
        "MONGODB_URI",
        os.getenv("MONGO_URI", "mongodb://mongodb:27017"),
    )
    database_name = os.getenv(
        "MONGODB_DATABASE",
        os.getenv("MONGO_DATABASE", "docupilot"),
    )

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=5000,
    )

    try:
        db = client[database_name]
        db.command("ping")

        db.sources.update_one(
            {"_id": source_id},
            {
                "$set": {
                    "status": "running",
                    "active_job_id": job_id,
                    "last_error": None,
                    "updated_at": utc_now(),
                }
            },
        )

        result = run_ingestion(
            start_url=root_url,
            max_pages=max_pages,
            max_depth=max_depth,
        )

        db.sources.update_one(
            {"_id": source_id},
            {
                "$set": {
                    "status": "completed",
                    "active_job_id": None,
                    "last_job_id": job_id,
                    "last_crawl_id": result["crawl_id"],
                    "last_ingested_at": utc_now(),
                    "last_result": result,
                    "last_error": None,
                    "updated_at": utc_now(),
                }
            },
        )

        return result

    except Exception as exc:
        logger.exception("Ingestion failed for source %s", source_id)

        db.sources.update_one(
            {"_id": source_id},
            {
                "$set": {
                    "status": "failed",
                    "active_job_id": None,
                    "last_job_id": job_id,
                    "last_error": f"{type(exc).__name__}: {exc}",
                    "updated_at": utc_now(),
                }
            },
        )
        raise

    finally:
        client.close()
