from __future__ import annotations

import os
import platform
import re

from src.memory import Memory
from src.toolparse import extract_json_object, is_tool_attempt, parse_tool_call

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
