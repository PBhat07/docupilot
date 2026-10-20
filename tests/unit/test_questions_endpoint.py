
import asyncio

import httpx
import pytest

from docupilot.generation.client import LLMProviderError
from docupilot.main import app


class FakeAnswerService:
    async def answer(self, question, source_ids=None):
        return {
            "question": question,
            "answer": "The API uses port 8000. [S1]",
            "citations": [
                {
                    "ref": "S1",
                    "chunk_id": "chunk-001",
                    "source_id": "source-001",
                    "url": "https://docs.example.com/api",
                    "title": "API Configuration",
                }
            ],
            "abstained": False,
            "abstention_reason": None,
            "retrieval": {
                "candidate_count": 3,
                "current_candidate_count": 3,
                "stale_candidate_count": 0,
                "reranking_enabled": True,
            },
            "timings_ms": {
                "retrieval": 12.0,
                "generation": 100.0,
                "total": 112.0,
            },
        }


class FailingAnswerService:
    async def answer(self, question, source_ids=None):
        raise LLMProviderError("Simulated provider failure")


def request(method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_health_endpoint():
    response = request("GET", "/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_questions_endpoint_returns_answer_and_citations():
    app.state.answer_service = FakeAnswerService()

    response = request(
        "POST",
        "/questions",
        json={
            "question": "Which port does the API use?",
            "source_ids": ["source-001"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The API uses port 8000. [S1]"
    assert body["citations"][0]["ref"] == "S1"
    assert body["abstained"] is False


@pytest.mark.parametrize("question", ["", "   "])
def test_questions_endpoint_rejects_blank_question(question):
    response = request(
        "POST",
        "/questions",
        json={"question": question},
    )

    assert response.status_code == 422


def test_questions_endpoint_maps_provider_failure():
    app.state.answer_service = FailingAnswerService()

    response = request(
        "POST",
        "/questions",
        json={"question": "Which port does the API use?"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The LLM provider could not complete the request."
    )

#These tests use a fake answer service, so they won't call the real LLM, MongoDB, or Weaviate.