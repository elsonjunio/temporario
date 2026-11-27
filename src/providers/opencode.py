"""OpenCode Go provider (OpenAI-compatible chat completions)."""

from __future__ import annotations

import os
import uuid
from typing import Any

from agent_mini import __version__
from agent_mini.providers import ModelInfo

from .base import CappedOpenAIProvider

DEFAULT_BASE_URL = "https://opencode.ai/inference/go/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"

# Go only serves cloud-class models. agent-mini guesses the tier from the model
# name, which misclassifies many Go ids (kimi-k3, glm-5.3, ...) as "small"; this
# hint maps them to the cloud tier. An explicit agent.tier in the config wins.
_CLOUD_CLASS_PARAMS_B = 80.0


class OpenCodeProvider(CappedOpenAIProvider):
    """OpenCode Go.

    Configuration (environment variables win over config values):

    - ``OPENCODE_API_KEY`` — bearer token (required by the server)
    - ``OPENCODE_MODEL`` — model id (default ``deepseek-v4.1-flash``)
    - ``OPENCODE_BASE_URL`` — base URL (default ``https://opencode.ai/inference/go/v1``)
    - ``OPENCODE_MAX_TOKENS`` — reply cap (default: agent-mini's 4096)

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
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(
            model or os.getenv("OPENCODE_MODEL") or DEFAULT_MODEL,
            base_url or os.getenv("OPENCODE_BASE_URL") or DEFAULT_BASE_URL,
            api_key or os.getenv("OPENCODE_API_KEY", ""),
            reasoning_effort,
            name="opencode",
            max_tokens=max_tokens,
        )
        self.headers["User-Agent"] = f"agent-mini/{__version__} (codigo-temporario)"
        self.headers["x-opencode-session"] = f"agent-mini-{uuid.uuid4().hex}"
        self._model_info: ModelInfo | None = None
        self._model_info_loaded = False

    async def model_info(self) -> ModelInfo | None:
        """Context length and capabilities from Go's ``/v1/models`` endpoint.

        agent-mini uses this to shrink the context budget and to warn when the
        model cannot call tools. When the endpoint is unavailable, the cloud
        tier hint is still returned so budgets stay reasonable.
        """
        if self._model_info_loaded:
            return self._model_info
        self._model_info_loaded = True
        info = ModelInfo(params_b=_CLOUD_CLASS_PARAMS_B)
        try:
            # The model list is public; sending an empty/invalid bearer token
            # would make Go reject it (401), so leave Authorization out.
            headers = {k: v for k, v in self.headers.items() if k.lower() != "authorization"}
            response = await self._client.get(f"{self.base_url}/models", headers=headers)
            if response.status_code < 400:
                entry = self._find_model(response.json())
                if entry is not None:
                    info = self._model_info_from(entry)
        except Exception:  # unreachable/invalid endpoint: keep the tier hint
            pass
        self._model_info = info
        return info

    def _find_model(self, payload: Any) -> dict | None:
        data = payload.get("data") if isinstance(payload, dict) else None
        for entry in data or []:
            if isinstance(entry, dict) and self.model in (entry.get("id"), entry.get("canonical_id")):
                return entry
        return None

    @staticmethod
    def _model_info_from(entry: dict) -> ModelInfo:
        limit = entry.get("limit") or {}
        context = limit.get("context")
        capabilities = ["completion"]
        caps = entry.get("capabilities") or {}
        if isinstance(caps, dict):
            if (caps.get("tools") or {}).get("supported"):
                capabilities.append("tools")
            if (caps.get("reasoning") or {}).get("supported"):
                capabilities.append("reasoning")
        modalities = (entry.get("modalities") or {}).get("input") or []
        if any(kind != "text" for kind in modalities):
            capabilities.append("vision")
        return ModelInfo(
            params_b=_CLOUD_CLASS_PARAMS_B,
            context_length=int(context) if isinstance(context, (int, float)) and context > 0 else None,
            capabilities=capabilities,
        )
