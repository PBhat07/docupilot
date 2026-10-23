"""Run a minimal paired DocuPilot evaluation: reranking off vs on.

Stores only the information needed for a small, inspectable comparison:
answer, error, successful-answer latency, and the final retrieved evidence.
No LLM-as-judge metrics or extra summary CSVs are generated.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import weaviate
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("docupilot.evaluation")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evaluation" / "dataset.json"
DEFAULT_OUTPUT = ROOT / "data" / "evaluation" / "results"


class CapturingRetrieval:
    """Capture the retrieval result used by AnswerService."""
    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.last_result: dict[str, Any] | None = None

    def retrieve(self, query: str, source_ids: list[str] | None = None) -> dict[str, Any]:
        self.last_result = self.pipeline.retrieve(query, source_ids=source_ids)
        return self.last_result


def as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "page_content", "answer", "response"):
            if isinstance(value.get(key), str):
                return value[key]
    return "" if value is None else str(value)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return jsonable(value.dict())
    return str(value)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("cases", payload.get("questions"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty JSON list in {path}")

    cases, seen = [], set()
    for index, item in enumerate(payload, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} must be an object")
        case_id = str(item.get("id") or item.get("case_id") or f"Q{index:03d}")
        question = item.get("question") or item.get("query") or item.get("user_input")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{case_id}: question is required")
        if case_id in seen:
            raise ValueError(f"Duplicate dataset id: {case_id}")
        seen.add(case_id)
        reference = item.get("reference_answer", item.get("reference", item.get("expected_answer", "")))
        answerable = item.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError(f"{case_id}.answerable must be true or false")
        cases.append({
            "case_id": case_id,
            "question": question.strip(),
            "reference_answer": as_text(reference).strip(),
            "answerable": answerable,
            "source_ids": item.get("source_ids"),
        })
    return cases


def build_runtime():
    import weaviate
    from pymongo import MongoClient
    from docupilot.generation.answer_service import AnswerService
    from docupilot.generation.client import LLMClient, LLMConfig
    from docupilot.indexing.embedder import Embedder
    from docupilot.indexing.weaviate_store import WeaviateStore
    from docupilot.retrieval.pipeline import RetrievalPipeline
    from docupilot.retrieval.reranker import Reranker
    from docupilot.retrieval.version_filter import CurrentVersionFilter

    mongo = MongoClient(
        os.getenv("MONGODB_URI", os.getenv("MONGO_URI", "mongodb://mongodb:27017")),
        serverSelectionTimeoutMS=8000,
    )
    mongo.admin.command("ping")
    db = mongo[os.getenv("MONGODB_DATABASE", os.getenv("MONGO_DATABASE", "docupilot"))]

    wclient = weaviate.connect_to_local(
        host=os.getenv("WEAVIATE_HOST", "weaviate"),
        port=int(os.getenv("WEAVIATE_PORT", "8080")),
        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
    )
    if not wclient.is_ready():
        raise RuntimeError("Weaviate is not ready")

    embedder = Embedder(device=os.getenv("EMBEDDING_DEVICE", "cpu"))
    store = WeaviateStore(wclient)
    version_filter = CurrentVersionFilter(pages_collection=db["pages"])
    reranker = Reranker(device=os.getenv("RERANKER_DEVICE", "cpu"))
    llm_client = LLMClient(LLMConfig.from_env())

    services = {}
    candidate_limit = int(os.getenv("EVAL_CANDIDATE_LIMIT", "20"))
    final_limit = int(os.getenv("EVAL_FINAL_LIMIT", "6"))
    alpha = float(os.getenv("EVAL_ALPHA", "0.5"))
    if candidate_limit < 1 or final_limit < 1 or final_limit > candidate_limit:
        raise ValueError("Require 1 <= EVAL_FINAL_LIMIT <= EVAL_CANDIDATE_LIMIT")

    for enabled in (False, True):
        pipeline = RetrievalPipeline(
            embedder=embedder,
            store=store,
            version_filter=version_filter,
            reranker=reranker,
            rerank_enabled=enabled,
            candidate_limit=candidate_limit,
            final_limit=final_limit,
            alpha=alpha,
        )
        capture = CapturingRetrieval(pipeline)
        services[enabled] = (AnswerService(capture, llm_client), capture)

    return wclient, mongo, services


def capture_chunks(retrieval: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(retrieval, dict):
        return []
    chunks = []
    for rank, chunk in enumerate(retrieval.get("chunks") or [], 1):
        props = chunk.get("properties", chunk) if isinstance(chunk, dict) else getattr(chunk, "properties", {})
        props = props if isinstance(props, dict) else {}
        text = as_text(props.get("text") or props.get("content") or props.get("chunk_text"))
        chunks.append({
            "rank": rank,
            "chunk_id": props.get("chunk_id") or props.get("id"),
            "source_id": props.get("source_id"),
            "title": props.get("title"),
            "url": props.get("canonical_url") or props.get("url"),
            "text_preview": text[:350],
        })
    return chunks


async def run_case(case: dict[str, Any], enabled: bool, service: Any, capture: CapturingRetrieval) -> dict[str, Any]:
    capture.last_result = None
    started = time.perf_counter()
    response: dict[str, Any] = {}
    error = None

    try:
        result = await service.answer(case["question"], source_ids=case.get("source_ids"))
        response = result if isinstance(result, dict) else {"answer": as_text(result)}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        LOG.exception("Answer failed for %s (reranking=%s)", case["case_id"], enabled)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    retrieval = capture.last_result or {}
    answer = as_text(response.get("answer") or response.get("response")).strip()
    chunks = capture_chunks(retrieval)
    citations = response.get("citations") or response.get("sources") or []

    return {
        "case_id": case["case_id"],
        "reranking": "on" if enabled else "off",
        "question": case["question"],
        "reference_answer": case["reference_answer"],
        "answerable_expected": case["answerable"],
        "answer": answer,
        "abstained": bool(response.get("abstained", False)),
        "latency_ms": elapsed_ms,
        "success": bool(answer) and error is None,
        "error": error,
        "retrieved_chunks": chunks,
        "citations": jsonable(citations),
    }


async def async_main(args: argparse.Namespace) -> int:
    dataset = load_dataset(Path(args.dataset).resolve())
    if args.limit:
        dataset = dataset[:args.limit]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wclient = mongo = None

    try:
        wclient, mongo, services = build_runtime()
        options = [False, True] if args.reranking == "both" else [args.reranking == "on"]
        rows = []
        for case in dataset:
            for enabled in options:
                service, capture = services[enabled]
                LOG.info("Evaluating %s; reranking %s", case["case_id"], "on" if enabled else "off")
                rows.append(await run_case(case, enabled, service, capture))

        payload = {
            "metadata": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_path": str(Path(args.dataset).resolve()),
                "question_count": len(dataset),
                "configurations": ["on" if x else "off" for x in options],
                "candidate_limit": int(os.getenv("EVAL_CANDIDATE_LIMIT", "20")),
                "final_limit": int(os.getenv("EVAL_FINAL_LIMIT", "6")),
                "alpha": float(os.getenv("EVAL_ALPHA", "0.5")),
                "note": "Small paired engineering evaluation; inspect answers and retrieved evidence manually.",
            },
            "results": rows,
        }
        path = output_dir / "results.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        successful = sum(row["success"] for row in rows)
        print(f"Saved {len(rows)} runs ({successful} successful answers) to {path}")
        print("Next: run scripts/plot_results.py to create the paired review CSV, latency chart, and findings.")
        return 0
    finally:
        if wclient is not None:
            wclient.close()
        if mongo is not None:
            mongo.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=os.getenv("EVAL_DATASET_PATH", str(DEFAULT_DATASET)))
    parser.add_argument("--output-dir", default=os.getenv("EVAL_OUTPUT_DIR", str(DEFAULT_OUTPUT)))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reranking", choices=("off", "on", "both"), default="both")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    main()
