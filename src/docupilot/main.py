
from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import weaviate
from fastapi import FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from pymongo import MongoClient, ReturnDocument
from redis import Redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job

from docupilot.crawler.pipelines import make_source_id
from docupilot.crawler.url_utils import normalize_url
from docupilot.generation.answer_service import AnswerService
from docupilot.generation.client import (
    LLMAuthenticationError,
    LLMClient,
    LLMConfig,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from docupilot.indexing.embedder import Embedder
from docupilot.indexing.weaviate_store import WeaviateStore
from docupilot.jobs.tasks import ingest_source
from docupilot.retrieval.pipeline import RetrievalPipeline
from docupilot.retrieval.reranker import Reranker
from docupilot.retrieval.version_filter import CurrentVersionFilter


logger = logging.getLogger(__name__)

QUEUE_NAME = "docupilot"
ACTIVE_SOURCE_STATUSES = {"queued", "running"}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    source_ids: list[str] | None = None

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class SourceRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    max_pages: int = Field(default=10, ge=1, le=500)
    max_depth: int = Field(default=1, ge=0, le=10)


class RefreshRequest(BaseModel):
    max_pages: int = Field(default=10, ge=1, le=500)
    max_depth: int = Field(default=1, ge=0, le=10)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_mongo_uri() -> str:
    """Support both legacy and current MongoDB environment variable names."""
    return (
        os.getenv("MONGO_URI")
        or os.getenv("MONGODB_URI")
        or "mongodb://mongodb:27017"
    )


def get_database_name() -> str:
    return (
        os.getenv("MONGO_DATABASE")
        or os.getenv("MONGODB_DATABASE")
        or "docupilot"
    )


def get_database(request: Request) -> Any:
    database = getattr(request.app.state, "database", None)
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is unavailable.",
        )
    return database


def get_queue(request: Request) -> Queue:
    queue = getattr(request.app.state, "ingestion_queue", None)
    if queue is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion queue is unavailable.",
        )
    return queue


def serialize_source(source: dict[str, Any]) -> dict[str, Any]:
    """Return public source fields without exposing MongoDB internals."""
    return {
        "source_id": source["_id"],
        "url": source.get("root_url"),
        "status": source.get("status", "registered"),
        "created_at": source.get("created_at"),
        "updated_at": source.get("updated_at"),
        "last_ingested_at": source.get("last_ingested_at"),
        "active_job_id": source.get("active_job_id"),
        "last_job_id": source.get("last_job_id"),
        "last_crawl_id": source.get("last_crawl_id"),
        "last_error": source.get("last_error"),
        "last_result": source.get("last_result"),
    }


def build_answer_service() -> tuple[AnswerService, Any, MongoClient]:
    """Build retrieval and generation services and their clients."""
    llm_config = LLMConfig.from_env()

    weaviate_client = weaviate.connect_to_local(
        host=os.getenv("WEAVIATE_HOST", "weaviate"),
        port=int(os.getenv("WEAVIATE_PORT", "8080")),
        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
    )

    mongo_client = MongoClient(
        get_mongo_uri(),
        serverSelectionTimeoutMS=5000,
    )

    try:
        mongo_client.admin.command("ping")
        database = mongo_client[get_database_name()]

        retrieval = RetrievalPipeline(
            embedder=Embedder(),
            store=WeaviateStore(weaviate_client),
            version_filter=CurrentVersionFilter(
                pages_collection=database["pages"],
            ),
            reranker=Reranker(device="cpu"),
            rerank_enabled=True,
            candidate_limit=20,
            final_limit=3,
            alpha=0.5,
        )

        answer_service = AnswerService(
            retrieval=retrieval,
            llm_client=LLMClient(llm_config),
        )

        return answer_service, weaviate_client, mongo_client

    except Exception:
        mongo_client.close()
        weaviate_client.close()
        raise


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    weaviate_client = None
    mongo_client = None
    redis_client = None

    try:
        answer_service, weaviate_client, mongo_client = (
            build_answer_service()
        )

        database = mongo_client[get_database_name()]

        redis_client = Redis.from_url(
            os.getenv("REDIS_URL", "redis://redis:6379/0"),
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        redis_client.ping()

        app.state.answer_service = answer_service
        app.state.mongo_client = mongo_client
        app.state.database = database
        app.state.redis_client = redis_client
        app.state.ingestion_queue = Queue(
            QUEUE_NAME,
            connection=redis_client,
        )

        logger.info("DocuPilot API initialized successfully.")
        yield

    finally:
        if redis_client is not None:
            redis_client.close()
        if mongo_client is not None:
            mongo_client.close()
        if weaviate_client is not None:
            weaviate_client.close()


app = FastAPI(
    title="DocuPilot API",
    description="An evidence-grounded technical documentation assistant.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Background ingestion
# ---------------------------------------------------------------------------

def enqueue_ingestion(
    request: Request,
    source_id: str,
    root_url: str,
    max_pages: int,
    max_depth: int,
) -> dict[str, Any]:
    """Reserve a source and enqueue ingestion without blocking the API."""
    database = get_database(request)
    queue = get_queue(request)

    source = database.sources.find_one({"_id": source_id})
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source not found.",
        )

    active_job_id = source.get("active_job_id")
    source_status = source.get("status", "registered")

    if source_status in ACTIVE_SOURCE_STATUSES:
        if active_job_id:
            return {
                "source_id": source_id,
                "job_id": active_job_id,
                "status": source_status,
                "message": "An ingestion job is already active.",
            }

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "The source is marked active but has no active job ID. "
                "Inspect the source before retrying ingestion."
            ),
        )

    if active_job_id:
        database.sources.update_one(
            {
                "_id": source_id,
                "active_job_id": active_job_id,
                "status": {"$nin": list(ACTIVE_SOURCE_STATUSES)},
            },
            {"$set": {"active_job_id": None}},
        )

    job_id = uuid.uuid4().hex
    now = utc_now()

    reserved = database.sources.find_one_and_update(
        {
            "_id": source_id,
            "status": {"$nin": list(ACTIVE_SOURCE_STATUSES)},
            "active_job_id": None,
        },
        {
            "$set": {
                "status": "queued",
                "active_job_id": job_id,
                "updated_at": now,
                "last_error": None,
            }
        },
        return_document=ReturnDocument.AFTER,
    )

    if reserved is None:
        latest = database.sources.find_one({"_id": source_id})

        if latest and latest.get("active_job_id"):
            return {
                "source_id": source_id,
                "job_id": latest["active_job_id"],
                "status": latest.get("status", "queued"),
                "message": "An ingestion job is already active.",
            }

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Could not reserve this source for ingestion.",
        )

    try:
        queue.enqueue(
            ingest_source,
            root_url,
            max_pages,
            max_depth,
            job_id=job_id,
            job_timeout=1800,
        )

    except Exception as exc:
        logger.exception("Could not enqueue source %s", source_id)

        database.sources.update_one(
            {"_id": source_id, "active_job_id": job_id},
            {
                "$set": {
                    "status": "failed",
                    "active_job_id": None,
                    "last_error": "Failed to enqueue ingestion job.",
                    "updated_at": utc_now(),
                }
            },
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not enqueue ingestion.",
        ) from exc

    return {
        "source_id": source_id,
        "job_id": job_id,
        "status": "queued",
        "message": "Ingestion queued successfully.",
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Source management
# ---------------------------------------------------------------------------

@app.post(
    "/sources",
    tags=["sources"],
    status_code=status.HTTP_202_ACCEPTED,
)
def register_source(
    payload: SourceRequest,
    request: Request,
) -> dict[str, Any]:
    root_url = normalize_url(payload.url)

    if root_url is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A valid HTTP(S) documentation URL is required.",
        )

    source_id = make_source_id(root_url)
    database = get_database(request)
    now = utc_now()

    database.sources.update_one(
        {"_id": source_id},
        {
            "$set": {
                "root_url": root_url,
                "updated_at": now,
            },
            "$setOnInsert": {
                "created_at": now,
                "status": "registered",
                "active_job_id": None,
            },
        },
        upsert=True,
    )

    return enqueue_ingestion(
        request=request,
        source_id=source_id,
        root_url=root_url,
        max_pages=payload.max_pages,
        max_depth=payload.max_depth,
    )


@app.get("/sources", tags=["sources"])
def list_sources(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    database = get_database(request)

    total = database.sources.count_documents({})
    cursor = (
        database.sources.find({})
        .sort("updated_at", -1)
        .skip(offset)
        .limit(limit)
    )

    sources = [serialize_source(source) for source in cursor]

    return {
        "items": sources,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/sources/{source_id}", tags=["sources"])
def get_source(
    source_id: str,
    request: Request,
) -> dict[str, Any]:
    database = get_database(request)

    source = database.sources.find_one({"_id": source_id})
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source not found.",
        )

    result = serialize_source(source)
    result["page_count"] = database.pages.count_documents(
        {"source_id": source_id}
    )
    result["version_count"] = database.page_versions.count_documents(
        {"source_id": source_id}
    )

    latest_crawl = database.crawls.find_one(
        {"source_id": source_id},
        sort=[("started_at", -1)],
    )

    if latest_crawl is not None:
        result["latest_crawl"] = {
            "crawl_id": latest_crawl.get("_id"),
            "status": latest_crawl.get("status"),
            "started_at": latest_crawl.get("started_at"),
            "finished_at": latest_crawl.get("finished_at"),
            "pages_seen": latest_crawl.get("pages_seen", 0),
            "errors_seen": latest_crawl.get("errors_seen", 0),
        }

    return result


@app.post(
    "/sources/{source_id}/refresh",
    tags=["sources"],
    status_code=status.HTTP_202_ACCEPTED,
)
def refresh_source(
    source_id: str,
    request: Request,
    payload: RefreshRequest | None = None,
) -> dict[str, Any]:
    database = get_database(request)

    source = database.sources.find_one({"_id": source_id})
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source not found.",
        )

    root_url = source.get("root_url")
    if not isinstance(root_url, str) or normalize_url(root_url) is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The stored source URL is invalid.",
        )

    options = payload or RefreshRequest()

    return enqueue_ingestion(
        request=request,
        source_id=source_id,
        root_url=root_url,
        max_pages=options.max_pages,
        max_depth=options.max_depth,
    )


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}", tags=["jobs"])
def get_job_status(
    job_id: str,
    request: Request,
) -> dict[str, Any]:
    redis_client = getattr(request.app.state, "redis_client", None)

    if redis_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job service is unavailable.",
        )

    try:
        job = Job.fetch(job_id, connection=redis_client)
        raw_status = job.get_status(refresh=True)

    except NoSuchJobError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        ) from exc

    job_status = (
        raw_status.value
        if hasattr(raw_status, "value")
        else str(raw_status)
    )

    response: dict[str, Any] = {
        "job_id": job.id,
        "status": job_status,
        "created_at": job.created_at,
        "enqueued_at": job.enqueued_at,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
        "result": None,
        "error": None,
    }

    if job_status == "finished":
        response["result"] = job.result
    elif job_status == "failed":
        response["error"] = (
            "The background job failed. Check the worker logs for details."
        )

    return response


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------

@app.post("/questions", tags=["questions"])
async def ask_question(
    payload: QuestionRequest,
    request: Request,
) -> dict[str, Any]:
    service = getattr(request.app.state, "answer_service", None)

    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "question_service_unavailable",
                "message": "Question-answering service is unavailable.",
                "retryable": True,
            },
        )

    try:
        return await service.answer(
            payload.question,
            source_ids=payload.source_ids,
        )

    except LLMRateLimitError as exc:
        logger.warning("LLM provider rate limit reached.")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "llm_rate_limited",
                "message": (
                    "The LLM provider rate limit was reached. "
                    "Please retry after a short delay."
                ),
                "retryable": True,
            },
        ) from exc

    except LLMUnavailableError as exc:
        logger.warning("LLM provider is temporarily unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "llm_unavailable",
                "message": (
                    "The LLM provider is temporarily unavailable. "
                    "Please try again later."
                ),
                "retryable": True,
            },
        ) from exc

    except LLMTimeoutError as exc:
        logger.warning("LLM provider request timed out.")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "code": "llm_timeout",
                "message": (
                    "The LLM provider did not respond in time. "
                    "Please try again."
                ),
                "retryable": True,
            },
        ) from exc

    except LLMAuthenticationError as exc:
        logger.error(
            "LLM provider rejected credentials or permissions."
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "llm_authentication_failed",
                "message": (
                    "The LLM provider rejected its configured credentials "
                    "or permissions. Check the server configuration."
                ),
                "retryable": False,
            },
        ) from exc

    except LLMConfigurationError as exc:
        logger.error("LLM provider configuration/request error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "llm_configuration_error",
                "message": (
                    "The LLM provider configuration or request is invalid. "
                    "Check the server configuration."
                ),
                "retryable": False,
            },
        ) from exc

    except LLMProviderError as exc:
        logger.warning("LLM provider request failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "llm_provider_error",
                "message": "The LLM provider could not complete the request.",
                "retryable": False,
            },
        ) from exc
