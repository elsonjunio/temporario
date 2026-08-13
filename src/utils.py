from __future__ import annotations

import json
import os
import platform
import re
from typing import Any

from src.memory import Memory

DEFAULT_INSTRUCTIONS = (
    "You are a helpful agent with access to the tools below. "
    "When you need to inspect the filesystem, pick one tool and respond with a "
    "single JSON block exactly like:\n"
    "```json\n"
    '{"tool": "<tool_name>", "action": "<action>", "params": {<arguments>}}\n'
    "```\n"
    "The user message that follows will contain the tool result. Keep issuing "
    "tool calls until you can answer, then reply without a JSON block.\n"
    "Confirmation flow: when a tool returns 'awaiting_confirmation', stop "
    "immediately and call no more tools; the plan is relayed to the user "
    "automatically. Never execute the plan yourself. When the user approves, "
    "resume through the tool's confirm mechanism (e.g. orchestrator execute "
    "with confirm=true) -- do not re-plan the same request and do not switch "
    "to other tools for the same task."
)

_FENCED_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)

_APPROVAL_TOKENS = {
    "sim",
    "s",
    "ss",
    "ok",
    "oka",
    "okay",
    "claro",
    "pode",
    "ir",
    "seguir",
    "podemos",
    "vai",
    "vamos",
    "segue",
    "siga",
    "agora",
    "continue",
    "continuar",
    "continua",
    "prossiga",
    "prosseguir",
    "execute",
    "executar",
    "aprovo",
    "aprovado",
    "confirm",
    "confirmar",
    "confirma",
    "confirmado",
    "por",
    "favor",
    "please",
    "yes",
    "y",
    "yeah",
    "sure",
    "go",
    "ahead",
    "goahead",
    "proceed",
    "approved",
    "adiante",
}

_ABORT_TOKENS = {
    "nao",
    "não",
    "no",
    "n",
    "nope",
    "cancel",
    "cancela",
    "cancelar",
    "abort",
    "aborta",
    "abortar",
    "para",
    "pare",
    "parar",
    "stop",
    "esquece",
    "esquecer",
    "descarta",
    "descartar",
    "deixa",
}

# Unambiguous cancellation verbs; their presence marks an abort even when the
# message is not made of abort tokens alone (e.g. "não, cancela").
_STRONG_CANCEL_TOKENS = {
    "cancela",
    "cancelar",
    "cancel",
    "aborta",
    "abortar",
    "abort",
    "para",
    "pare",
    "parar",
    "stop",
    "esquece",
    "esquecer",
    "descarta",
    "descartar",
}

_PUNCTUATION_RE = re.compile(r"[^0-9a-z\u00e0-\u00ff]+", re.IGNORECASE)


def normalize_message(message: str) -> str:
    """Lowercase a message and strip punctuation/emoji for token matching."""
    return _PUNCTUATION_RE.sub(" ", message.lower()).strip()


def is_approval(message: str) -> bool:
    """True only for a pure approval message (approval tokens only)."""
    tokens = [t for t in normalize_message(message).split() if t]
    if not tokens:
        return False
    return all(t in _APPROVAL_TOKENS for t in tokens)


def is_abort(message: str) -> bool:
    """True for an explicit cancellation/refusal.

    Either every token is an abort token ("não", "no") or the message contains
    an unambiguous cancellation verb ("cancela", "para", "abortar", ...).
    "não sei" is NOT an abort.
    """
    tokens = [t for t in normalize_message(message).split() if t]
    if not tokens:
        return False
    if any(t in _STRONG_CANCEL_TOKENS for t in tokens):
        return True
    return all(t in _ABORT_TOKENS for t in tokens)


def classify_followup(message: str) -> str:
    """Classify a user reply to an awaiting_confirmation prompt.

    Returns "approve" (pure approval), "abort" (explicit cancellation) or
    "other" (ambiguous, conditional or a brand-new request). "sim, mas muda X"
    yields "other" so the agent can re-plan instead of executing blindly.
    """
    if is_abort(message):
        return "abort"
    if is_approval(message):
        return "approve"
    return "other"


_INVOKE_RE = re.compile(
    r"<invoke\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>(.*?)</invoke>",
    re.DOTALL,
)

_PARAM_RE = re.compile(
    r"<parameter(?:\s+name\s*=\s*[\"']([^\"']*)[\"'])?\s*>(.*?)</parameter>",
    re.DOTALL,
)

# Tools expose a single primary action; used when a call names the tool only
# (e.g. Claude-style <invoke> blocks have no action concept).
_DEFAULT_ACTIONS = {
    "read_file": "read",
    "list_dir": "list",
    "search_files": "search",
    "grep_files": "search",
    "write_file": "write",
    "patch_file": "apply",
    "delete_file": "delete",
    "move_file": "move",
    "run_command": "run",
    "orchestrator": "run",
}


def build_environment_info(workspace: str | None = None) -> str:
    """Describe the runtime environment (OS, python, workspace) for the LLM."""
    lines = [
        f"OS: {platform.system()} ({os.name}, {platform.release()})",
        f"Python: {platform.python_version()}",
    ]
    if workspace:
        lines.append(f"Workspace: {workspace}")
    return "\n".join(lines)


def build_config_prompt(
    tool_manuals: str | list[str],
    memory: Memory | None = None,
    instructions: str | None = None,
    max_history_entries: int = 20,
    environment: str | None = None,
) -> str:
    """Assemble the configuration/system prompt for the LLM.

    It contains the runtime environment (OS, workspace), the tool list
    (manuals), interaction instructions and the recent conversation history
    from ``memory``.
    """
    if isinstance(tool_manuals, str):
        manual_block = tool_manuals
    else:
        manual_block = "\n\n".join(tool_manuals)

    sections: list[str] = []

    if environment:
        sections.extend(["## Environment", environment, ""])

    sections.extend(
        [
            "## Instructions",
            instructions or DEFAULT_INSTRUCTIONS,
            "",
            "## Available tools",
            manual_block if manual_block else "(none)",
        ]
    )

    if memory is not None:
        sections.extend(
            ["", "## Conversation history", memory.get_context(max_history_entries)]
        )

    return "\n".join(sections)


def _extract_balanced_json(text: str) -> str | None:
    """Return the first top-level JSON object literal found in ``text``."""
    for start in re.finditer(r"\{", text):
        depth = 0
        in_string = False
        escape = False
        for i in range(start.start(), len(text)):
            char = text[i]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start.start() : i + 1]
    return None


def extract_json_object(text: str) -> Any | None:
    """Return the first top-level JSON value (object/array) found in ``text``.

    Uses balanced-brace scanning, so it tolerates prose around the JSON.
    Returns None when nothing parseable is found.
    """
    balanced = _extract_balanced_json(text)
    if balanced is None:
        return None
    try:
        return json.loads(balanced)
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_invoke_xml(text: str) -> dict[str, Any] | None:
    """Parse a Claude-style ``<invoke name=...>`` tool call into the normalized
    ``{"tool", "action", "params"}`` dict, or None if none is present.

    Supports ``<parameter name="key">value</parameter>`` and, as a fallback for
    malformed output, bare ``<parameter>key</parameter><parameter>value</parameter>``
    pairs.
    """
    matches = list(_INVOKE_RE.finditer(text))
    if not matches:
        return None

    for match in matches:
        tool = match.group(1).strip()
        if not tool:
            continue

        body = match.group(2)
        named: dict[str, str] = {}
        unnamed: list[str] = []
        for param in _PARAM_RE.finditer(body):
            key = param.group(1)
            value = param.group(2).strip()
            if key:
                named[key] = value
            else:
                unnamed.append(value)

        if named:
            params: dict[str, Any] = dict(named)
        elif unnamed:
            params = {}
            it = iter(unnamed)
            for key in it:
                value = next(it, None)
                if value is None:
                    break
                params[key] = value
        else:
            params = {}

        action = params.pop("action", "") or _DEFAULT_ACTIONS.get(tool, "run")
        return {"tool": tool, "action": action, "params": params}

    return None


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Extract a tool call ``{"tool": ..., "action": ..., "params": {...}}``.

    Accepts either a JSON block (fenced or bare) or a Claude-style
    ``<invoke name="tool">`` XML block. Returns None when the response is a
    plain answer without a tool call.
    """
    candidates: list[str] = []

    fenced = _FENCED_BLOCK_RE.findall(text)
    candidates.extend(fenced)

    balanced = _extract_balanced_json(text)
    if balanced is not None:
        candidates.append(balanced)

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

        if isinstance(data, dict) and isinstance(data.get("tool"), str):
            params = data.get("params", {})
            if not isinstance(params, dict):
                params = {}
            return {
                "tool": data["tool"],
                "action": data.get("action", ""),
                "params": params,
            }

    return _parse_invoke_xml(text)
