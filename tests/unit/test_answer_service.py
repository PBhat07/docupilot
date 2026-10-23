
import asyncio

import pytest

from docupilot.generation.answer_service import (
    ABSTENTION_MESSAGE,
    AnswerService,
)


class FakeRetrieval:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, source_ids=None):
        return {
            "query": query,
            "chunks": self.chunks,
            "candidate_count": len(self.chunks),
            "current_candidate_count": len(self.chunks),
            "stale_candidate_count": 0,
            "reranking_enabled": True,
            "timings_ms": {"retrieval_pipeline": 12.0},
        }


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.called = False

    async def generate(self, messages, **kwargs):
        self.called = True
        return self.response


def make_chunk(
    chunk_id="chunk-001",
    text="The API uses port 8000.",
):
    return {
        "uuid": "weaviate-object-001",
        "properties": {
            "chunk_id": chunk_id,
            "source_id": "source-001",
            "canonical_url": "https://docs.example.com/api",
            "title": "API Configuration",
            "content_hash": "hash-001",
            "chunk_index": 0,
            "chunk_text": text,
        },
    }


def test_answer_returns_verified_citation():
    retrieval = FakeRetrieval([make_chunk()])
    llm = FakeLLM("The API uses port 8000. [S1]")
    service = AnswerService(retrieval, llm)

    result = asyncio.run(service.answer("Which port does the API use?"))

    assert result["abstained"] is False
    assert result["answer"] == "The API uses port 8000. [S1]"
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "chunk-001"
    assert result["citations"][0]["url"] == (
        "https://docs.example.com/api"
    )
    assert result["timings_ms"]["generation"] >= 0


def test_answer_abstains_when_no_evidence():
    retrieval = FakeRetrieval([])
    llm = FakeLLM("This should never be generated.")
    service = AnswerService(retrieval, llm)

    result = asyncio.run(service.answer("What is the API port?"))

    assert result["abstained"] is True
    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["abstention_reason"] == "no_usable_evidence"
    assert result["citations"] == []
    assert llm.called is False


@pytest.mark.parametrize(
    ("response", "expected_reason"),
    [
        ("The API uses port 8000.", "missing_citations"),
        (
            "The API uses port 8000. [S9]",
            "citation_validation_failed",
        ),
    ],
)
def test_answer_abstains_when_citations_are_invalid(
    response, expected_reason
):
    retrieval = FakeRetrieval([make_chunk()])
    llm = FakeLLM(response)
    service = AnswerService(retrieval, llm)

    result = asyncio.run(
        service.answer("Which port does the API use?")
    )

    assert result["abstained"] is True
    assert result["abstention_reason"] == expected_reason
    assert result["citations"] == []


def test_answer_rejects_empty_question():
    service = AnswerService(
        FakeRetrieval([]),
        FakeLLM("Unused"),
    )

    with pytest.raises(ValueError, match="question"):
        asyncio.run(service.answer("  "))
