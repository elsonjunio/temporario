"""Shared base for the custom OpenAI-compatible providers."""

from __future__ import annotations

import re
from typing import Any

from agent_mini.prompt import MAX_REPLY_TOKENS
from agent_mini.providers import OpenAIProvider

# OpenAI reasoning models reject max_tokens (same rule agent-mini uses for temperature).
_NO_MAX_TOKENS = re.compile(r"^(?:o\d|gpt-5)", re.IGNORECASE)


class CappedOpenAIProvider(OpenAIProvider):
    """OpenAI-compatible provider with a reply-length cap.

    agent-mini only caps replies when the provider is called ``local``; the
    custom providers use their own names, so they cap explicitly here.
    """

    def payload(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload = super().payload(messages, tools, temperature, stream)
        if not _NO_MAX_TOKENS.match(self.model.rsplit("/", 1)[-1]):
            payload.setdefault("max_tokens", MAX_REPLY_TOKENS)
        return payload
