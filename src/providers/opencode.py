"""OpenCode Go provider (OpenAI-compatible chat completions)."""

from __future__ import annotations

import os
import uuid

from agent_mini import __version__

from .base import CappedOpenAIProvider

DEFAULT_BASE_URL = "https://opencode.ai/inference/go/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"


class OpenCodeProvider(CappedOpenAIProvider):
    """OpenCode Go.

    Configuration (environment variables win over config values):

    - ``OPENCODE_API_KEY`` — bearer token (required by the server)
    - ``OPENCODE_MODEL`` — model id (default ``deepseek-v4.1-flash``)
    - ``OPENCODE_BASE_URL`` — base URL (default ``https://opencode.ai/inference/go/v1``)

    Go requires a stable ``x-opencode-session`` header per conversation so it
    can route requests and cache prompts; one provider instance is one session.
    """

    label = "OpenCode Go"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(
            model or os.getenv("OPENCODE_MODEL") or DEFAULT_MODEL,
            base_url or os.getenv("OPENCODE_BASE_URL") or DEFAULT_BASE_URL,
            api_key or os.getenv("OPENCODE_API_KEY", ""),
            reasoning_effort,
            name="opencode",
        )
        self.headers["User-Agent"] = f"agent-mini/{__version__} (codigo-temporario)"
        self.headers["x-opencode-session"] = f"agent-mini-{uuid.uuid4().hex}"
