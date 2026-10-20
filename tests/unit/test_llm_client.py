
import asyncio
import json

import httpx
import pytest

from docupilot.generation.client import (
    LLMClient,
    LLMConfig,
    LLMProviderError,
)


def test_generate_returns_model_text():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["Authorization"] == "Bearer test-key"

        payload = json.loads(request.content)
        assert payload["model"] == "test-model"

        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Evidence-based answer."}}
                ]
            },
        )

    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.test/v1/",
        model="test-model",
    )
    client = LLMClient(
        config,
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        client.generate(
            [{"role": "user", "content": "Test question"}]
        )
    )

    assert result == "Evidence-based answer."


def test_generate_raises_for_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limit"})

    config = LLMConfig(
        api_key="test-key",
        base_url="https://example.test/v1/",
        model="test-model",
    )
    client = LLMClient(
        config,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMProviderError, match="HTTP 429"):
        asyncio.run(
            client.generate(
                [{"role": "user", "content": "Test question"}]
            )
        )
