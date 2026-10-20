
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMConfigurationError(RuntimeError):
    """Raised when LLM configuration or request parameters are invalid."""


class LLMProviderError(RuntimeError):
    """Base exception for LLM provider failures."""


class LLMRateLimitError(LLMProviderError):
    """Raised when the provider rate limit is reached."""


class LLMUnavailableError(LLMProviderError):
    """Raised when the provider is temporarily unavailable."""


class LLMTimeoutError(LLMProviderError):
    """Raised when the provider does not respond in time."""


class LLMAuthenticationError(LLMProviderError):
    """Raised when credentials or access permissions are rejected."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> LLMConfig:
        api_key = os.getenv("LLM_API_KEY", "").strip()
        base_url = os.getenv(
            "LLM_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ).strip()
        model = os.getenv("LLM_MODEL", "gemini-3.8-flash").strip()

        if not api_key:
            raise LLMConfigurationError(
                "LLM_API_KEY is missing. Configure it in the environment."
            )

        if not base_url or not model:
            raise LLMConfigurationError(
                "LLM_BASE_URL and LLM_MODEL must not be empty."
            )

        try:
            timeout_seconds = float(
                os.getenv("LLM_TIMEOUT_SECONDS", "60")
            )
        except ValueError as exc:
            raise LLMConfigurationError(
                "LLM_TIMEOUT_SECONDS must be a number."
            ) from exc

        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise LLMConfigurationError(
                "LLM_TIMEOUT_SECONDS must be finite and greater than zero."
            )

        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/") + "/",
            model=model,
            timeout_seconds=timeout_seconds,
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Call an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        config: LLMConfig,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport

    async def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> str:
        """Generate text and translate provider failures into typed errors."""

        if not messages:
            raise ValueError("messages must not be empty")

        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")

        url = f"{self.config.base_url}chat/completions"

        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        started_at = time.perf_counter()

        try:
            async with httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                )

        except httpx.TimeoutException as exc:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.warning(
                "LLM request timed out: model=%s elapsed_ms=%.1f",
                self.config.model,
                elapsed_ms,
            )
            raise LLMTimeoutError(
                "The LLM provider did not respond within the configured timeout."
            ) from exc

        except httpx.RequestError as exc:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.warning(
                "LLM connection error: type=%s elapsed_ms=%.1f",
                type(exc).__name__,
                elapsed_ms,
            )
            raise LLMProviderError(
                f"Could not connect to the LLM provider "
                f"({type(exc).__name__})."
            ) from exc

        elapsed_ms = (time.perf_counter() - started_at) * 1000

        logger.info(
            "LLM HTTP response: model=%s status=%d elapsed_ms=%.1f",
            self.config.model,
            response.status_code,
            elapsed_ms,
        )

        try:
            response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            status_code = response.status_code

            # Keep the diagnostic in server logs, never log request headers
            # or the API key.
            logger.warning(
                "LLM provider HTTP error: status=%d body=%s",
                status_code,
                response.text[:1000],
            )

            if status_code == 429:
                raise LLMRateLimitError(
                    "The LLM provider rate limit was reached."
                ) from exc

            if status_code in (401, 403):
                raise LLMAuthenticationError(
                    "The LLM provider rejected the credentials "
                    "or access permissions."
                ) from exc

            if status_code in (408, 504):
                raise LLMTimeoutError(
                    "The LLM provider timed out while processing the request."
                ) from exc

            if status_code in (502, 503) or status_code >= 500:
                raise LLMUnavailableError(
                    "The LLM provider is temporarily unavailable."
                ) from exc

            if status_code in (400, 404, 422):
                raise LLMConfigurationError(
                    "The LLM provider rejected the model or request "
                    "configuration. Check the model name and request parameters."
                ) from exc

            raise LLMProviderError(
                f"The LLM provider returned HTTP {status_code}."
            ) from exc

        # Parse the successful response only after HTTP errors are handled.
        try:
            data = response.json()

            if not isinstance(data, dict):
                raise ValueError("Response JSON must be an object")

            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("Response contains no choices")

            choice = choices[0]
            if not isinstance(choice, dict):
                raise ValueError("Response choice must be an object")

            message = choice.get("message")
            if not isinstance(message, dict):
                raise ValueError("Response message is missing or invalid")

            content = message.get("content")
            finish_reason = choice.get("finish_reason")

        except (ValueError, TypeError, KeyError, IndexError) as exc:
            logger.warning(
                "Invalid LLM response format: type=%s",
                type(exc).__name__,
            )
            raise LLMProviderError(
                "The LLM provider returned an unexpected response format."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            logger.warning(
                "LLM returned empty text: model=%s finish_reason=%s usage=%s",
                self.config.model,
                finish_reason,
                data.get("usage"),
            )
            raise LLMProviderError(
                "The LLM provider returned empty text "
                f"(finish_reason={finish_reason!r})."
            )

        return content.strip()
