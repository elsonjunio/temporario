from __future__ import annotations

from typing import Any

from src.providers.base import BaseProvider, ProviderError

__all__ = ["LMStudioProvider", "ProviderError"]


class LMStudioProvider(BaseProvider):
    """Client for LM Studio's local OpenAI-compatible server.

    Config is read from the environment (via ``.env``) but every value can be
    overridden by constructor arguments:

    - ``LMSTUDIO_BASE_URL`` — server base URL (default ``http://localhost:1234/v1``)
    - ``LMSTUDIO_MODEL`` — model id to use
    - ``LMSTUDIO_API_KEY`` — optional bearer token for servers started with an
      API key; when unset, requests carry no Authorization header
    - ``LMSTUDIO_TIMEOUT`` — request timeout in seconds (default ``60``)
    """

    label = "LM Studio"
    default_base_url = "http://localhost:1234/v1"
    api_key_env = "LMSTUDIO_API_KEY"
    model_env = "LMSTUDIO_MODEL"
    base_url_env = "LMSTUDIO_BASE_URL"
    timeout_env = "LMSTUDIO_TIMEOUT"
    default_timeout = 60
    require_api_key = False

    def models(self) -> list[str]:
        """List the model ids currently loaded in the local server."""
        payload = self._request(
            f"{self.base_url}/models",
            headers={"User-Agent": "opencode-agent/1.0"},
        )
        return [
            entry.get("id", "")
            for entry in payload.get("data", [])
            if isinstance(entry, dict)
        ]
