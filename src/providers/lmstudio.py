"""LM Studio provider (local OpenAI-compatible server)."""

from __future__ import annotations

import os

from .base import CappedOpenAIProvider

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "local-model"


class LMStudioProvider(CappedOpenAIProvider):
    """LM Studio or any local OpenAI-compatible server.

    Configuration (environment variables win over config values):

    - ``LMSTUDIO_BASE_URL`` — server base URL (default ``http://localhost:1234/v1``)
    - ``LMSTUDIO_MODEL`` — model id to use
    - ``LMSTUDIO_API_KEY`` — optional token for servers started with an API key
    - ``LMSTUDIO_MAX_TOKENS`` — reply cap (default: agent-mini's 4096)
    """

    label = "LM Studio"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(
            model or os.getenv("LMSTUDIO_MODEL") or DEFAULT_MODEL,
            base_url or os.getenv("LMSTUDIO_BASE_URL") or DEFAULT_BASE_URL,
            api_key or os.getenv("LMSTUDIO_API_KEY", ""),
            reasoning_effort,
            name="lmstudio",
            max_tokens=max_tokens,
        )
