"""StackSpot GenAI Inference provider (agent chat API).

StackSpot's ``/v1/agent/<id>/chat`` endpoint is not OpenAI-compatible:

- it takes a single ``user_prompt``, not a chat message list;
- it has no ``tools`` parameter, so tool schemas go inside the prompt and the
  model is asked to answer with ``<tool_call>{...}</tool_call>`` blocks;
- authentication is OAuth2 ``client_credentials`` against
  ``https://idm.stackspot.com/<realm>/oidc/oauth/token``.

This adapter authenticates (caching and refreshing the token), flattens
agent-mini's history into ``user_prompt`` and parses the tool calls back into
``ToolCall`` objects.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any
from urllib.parse import quote

import httpx

from agent_mini.providers import ChatResponse, ModelInfo, Provider, ToolCall
from agent_mini.providers.base import new_call_id, raise_for_status
from agent_mini.providers.parsing import parse_arguments, repair_json

DEFAULT_BASE_URL = "https://genai-inference-app.stackspot.com"
DEFAULT_AUTH_URL = "https://idm.stackspot.com"
DEFAULT_REALM = "stackspot-freemium"

# Renew the access token a little before it actually expires.
_TOKEN_MARGIN = 300

_TOOL_CALL = re.compile(r"<tool_call>", re.IGNORECASE)
_BRACKET_TOOL = re.compile(r"\[TOOL:([\w.-]+)\]", re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*", re.IGNORECASE)


def _brace_block(text: str, start: int) -> str | None:
    """The JSON object starting at *start*, nested braces and strings included."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _tagged_blocks(text: str):
    """Yield ``(json_text, start, end)`` for each tool-call block in *text*."""
    cursor = 0
    while match := _TOOL_CALL.search(text, cursor):
        start = match.start()
        brace = text.find("{", match.end())
        if brace < 0:
            break
        raw = _brace_block(text, brace)
        if raw is None:
            cursor = brace + 1
            continue
        end = brace + len(raw)
        close = text.find("</tool_call>", end)
        if close >= 0:
            end = close + len("</tool_call>")
        yield raw, start, end
        cursor = end

    for match in _BRACKET_TOOL.finditer(text):
        brace = text.find("{", match.end())
        if brace < 0:
            continue
        raw = _brace_block(text, brace)
        if raw is None:
            continue
        end = brace + len(raw)
        close = text.find("[/TOOL]", end)
        if close >= 0:
            end = close + len("[/TOOL]")
        yield json.dumps({"name": match.group(1), "arguments": raw}), match.start(), end

    for match in _FENCE.finditer(text):
        brace = text.find("{", match.end())
        if brace < 0:
            continue
        raw = _brace_block(text, brace)
        if raw is None:
            continue
        end = brace + len(raw)
        close = text.find("```", end)
        if close >= 0:
            end = close + 3
        yield raw, match.start(), end


class StackSpotAuthError(RuntimeError):
    """StackSpot refused the OAuth2 credentials."""


class StackSpotProvider(Provider):
    """StackSpot agent chat: OAuth2 auth, history flattened to ``user_prompt``."""

    name = "stackspot"

    def __init__(
        self,
        agent_id: str,
        client_id: str,
        client_secret: str,
        realm: str = DEFAULT_REALM,
        base_url: str = DEFAULT_BASE_URL,
        auth_url: str = DEFAULT_AUTH_URL,
        *,
        knowledge: bool = True,
        return_ks: bool = False,
        timeout: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not agent_id:
            raise ValueError("StackSpot: set STACKSPOT_AGENT_ID (or providers.stackspot.model).")
        if not client_id or not client_secret:
            raise ValueError(
                "StackSpot: set STACKSPOT_CLIENT_ID and STACKSPOT_CLIENT_SECRET "
                "(or providers.stackspot.clientId/clientSecret)."
            )
        super().__init__(agent_id)
        self.client_id = client_id
        self.client_secret = client_secret
        self.realm = realm or DEFAULT_REALM
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.auth_url = (auth_url or DEFAULT_AUTH_URL).rstrip("/")
        self.knowledge = knowledge
        self.return_ks = return_ks
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._refresh_token_value: str | None = None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def model_info(self) -> ModelInfo | None:
        """A StackSpot agent runs a cloud model; tools are handled by prompt."""
        return ModelInfo(params_b=80.0, capabilities=["completion", "tools"])

    # ── Authentication ───────────────────────────────────────────────

    async def authenticate(self) -> str:
        """Return a cached access token or fetch a new one."""
        if self._token and time.monotonic() < self._token_expires_at:
            return self._token
        return await self.refresh_token()

    async def refresh_token(self) -> str:
        """Force a new access token.

        StackSpot's documented flow is ``client_credentials``. If the server
        happens to return a ``refresh_token``, the refresh grant is tried first
        and client credentials are used as a fallback.
        """
        if self._refresh_token_value:
            token = await self._request_token(
                {
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token_value,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                required=False,
            )
            if token:
                return token
        return await self._request_token(
            {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            required=True,
        )

    async def _request_token(self, data: dict[str, str], *, required: bool) -> str | None:
        response = await self._client.post(
            f"{self.auth_url}/{self.realm}/oidc/oauth/token",
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "agent-mini/0.7.0 (codigo-temporario)",
            },
        )
        if response.status_code >= 400:
            if not required:
                return None
            raise StackSpotAuthError(
                f"StackSpot auth HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise StackSpotAuthError("StackSpot auth returned a non-JSON response.") from exc
        token = payload.get("access_token")
        if not token:
            if not required:
                return None
            raise StackSpotAuthError("StackSpot auth response has no access_token.")
        try:
            expires_in = int(payload.get("expires_in") or 3600)
        except (TypeError, ValueError):
            expires_in = 3600
        self._token = token
        self._token_expires_at = time.monotonic() + max(0, expires_in - _TOKEN_MARGIN)
        self._refresh_token_value = payload.get("refresh_token") or self._refresh_token_value
        return token

    # ── Chat ─────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        on_delta: Any = None,
        on_thinking: Any = None,
    ) -> ChatResponse:
        body = {
            "streaming": False,
            "user_prompt": self._build_prompt(messages, tools),
            "stackspot_knowledge": self.knowledge,
            "return_ks_in_response": self.return_ks,
            "use_conversation": False,
        }
        response = await self._post_chat(body)
        data = self._decode(response)
        text = self._extract_text(data)
        calls, spans = self._collect(text) if tools else ([], [])
        content = text
        if calls:
            content = self._strip_spans(text, spans)
            if not spans and self._calls_from_json(text.strip()):
                content = ""
            content = content.strip() or None
        if on_delta and content:
            await on_delta(content)
        return ChatResponse(content=content, tool_calls=calls or None, usage=self._usage(data))

    async def _post_chat(self, body: dict[str, Any]) -> httpx.Response:
        url = f"{self.base_url}/v1/agent/{quote(self.model, safe='')}/chat"
        response: httpx.Response | None = None
        for attempt in range(2):
            token = await self.authenticate() if attempt == 0 else await self.refresh_token()
            response = await self._client.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "User-Agent": "agent-mini/0.7.0 (codigo-temporario)",
                },
                json=body,
            )
            if response.status_code != 401:
                break
        assert response is not None
        if response.status_code >= 400:
            # Gives the loop an httpx.HTTPStatusError, so 429/5xx are retried.
            await raise_for_status(response)
        return response

    @staticmethod
    def _decode(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return response.text

    @staticmethod
    def _extract_text(data: Any) -> str:
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in ("message", "response", "content"):
                value = data.get(key)
                if isinstance(value, str):
                    return value
                if value is not None:
                    return json.dumps(value, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _usage(data: Any) -> dict | None:
        tokens = data.get("tokens") if isinstance(data, dict) else None
        if not isinstance(tokens, dict):
            return None
        prompt = tokens.get("input") or tokens.get("user") or 0
        completion = tokens.get("output") or 0
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    # ── Prompt: messages + tools as conversation ─────────────────────

    def _build_prompt(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "system":
                parts.append(f"### Instruções do sistema\n{content}")
            elif role == "user":
                parts.append(f"### Usuário\n{content}")
            elif role == "assistant":
                lines = [f"### Assistente\n{content}"] if content else ["### Assistente"]
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    arguments = function.get("arguments") or "{}"
                    try:
                        arguments = json.loads(arguments)
                    except (TypeError, ValueError):
                        pass
                    lines.append(
                        "<tool_call>"
                        + json.dumps(
                            {"name": function.get("name"), "arguments": arguments},
                            ensure_ascii=False,
                        )
                        + "</tool_call>"
                    )
                parts.append("\n".join(lines))
            elif role == "tool":
                parts.append(
                    f"### Resultado da ferramenta {message.get('name')} "
                    f"(id={message.get('tool_call_id')})\n{content}"
                )
        if tools:
            parts.append(self._tools_block(tools))
        return "\n\n".join(parts)

    @staticmethod
    def _tools_block(tools: list[dict]) -> str:
        schemas = json.dumps(tools, ensure_ascii=False, indent=2)
        return (
            "## Ferramentas disponíveis\n"
            "Para chamar uma ou mais ferramentas, responda apenas com blocos:\n"
            '<tool_call>{"name":"nome_da_ferramenta","arguments":{...}}</tool_call>\n'
            "Regras:\n"
            "- Use exatamente os nomes e parâmetros dos schemas abaixo.\n"
            "- Não invente ferramentas nem argumentos.\n"
            "- Se não precisar de ferramenta, responda normalmente, sem blocos.\n"
            "- Depois de receber um '### Resultado da ferramenta', continue a tarefa.\n\n"
            "Schemas (JSON):\n" + schemas
        )

    # ── Response: text → ToolCall ────────────────────────────────────

    def _collect(self, text: str) -> tuple[list[ToolCall], list[tuple[int, int]]]:
        calls: list[ToolCall] = []
        spans: list[tuple[int, int]] = []
        for raw, start, end in _tagged_blocks(text or ""):
            # A fenced block may wrap a <tool_call> tag already handled above.
            if any(start < other_end and end > other_start for other_start, other_end in spans):
                continue
            found = self._calls_from_json(raw)
            if found:
                calls.extend(found)
                spans.append((start, end))
        if not calls:
            calls.extend(self._calls_from_json((text or "").strip()))
        return calls, spans

    def _parse_calls(self, text: str) -> list[ToolCall] | None:
        calls, _ = self._collect(text)
        return calls or None

    @staticmethod
    def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
        for start, end in sorted(spans, reverse=True):
            text = text[:start] + text[end:]
        return text.strip()

    def _calls_from_json(self, raw: str) -> list[ToolCall]:
        try:
            obj = json.loads(raw)
        except (TypeError, ValueError):
            obj = repair_json(raw)
        if isinstance(obj, list):
            return [c for c in (self._call_from_object(item) for item in obj) if c]
        return [c for c in (self._call_from_object(item) for item in self._split_calls(obj)) if c]

    @staticmethod
    def _split_calls(obj: Any) -> list[dict]:
        if not isinstance(obj, dict):
            return []
        if isinstance(obj.get("tool_calls"), list):
            return [item for item in obj["tool_calls"] if isinstance(item, dict)]
        return [obj]

    def _call_from_object(self, obj: dict) -> ToolCall | None:
        function = obj.get("function") if isinstance(obj.get("function"), dict) else {}
        name = obj.get("name") or obj.get("tool") or function.get("name")
        arguments = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
        if "arguments" in function:
            arguments = function["arguments"]
        if not isinstance(name, str) or not name:
            return None
        return ToolCall(new_call_id(), name, parse_arguments(arguments))
