from __future__ import annotations

import json
import re
from typing import Any

_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

_INVOKE_RE = re.compile(
    r"<invoke\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>(.*?)</invoke>",
    re.DOTALL,
)

_PARAM_RE = re.compile(
    r"<parameter(?:\s+name\s*=\s*[\"']([^\"']*)[\"'])?\s*>(.*?)</parameter>",
    re.DOTALL,
)

# Tool -> default action, used when the model omits the "action" key.
DEFAULT_ACTIONS = {
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

# Common spellings small models use instead of the registered tool names.
TOOL_ALIASES = {
    "listdir": "list_dir",
    "readfile": "read_file",
    "read": "read_file",
    "writefile": "write_file",
    "search": "search_files",
    "grep": "grep_files",
    "grep_file": "grep_files",
    "move": "move_file",
    "rename": "move_file",
    "delete": "delete_file",
    "remove": "delete_file",
    "run": "run_command",
    "patch": "patch_file",
    "orchestrator": "orchestrator",
}

_RESERVED = {"tool", "action", "params", "name"}

# Marker words that look like key: value pairs but are not params.
_MARKER_KEYS = {"call", "tool_call", "tool_name", "tool"}

_MARKER_PATTERNS = [
    # <|tool_call|>read_file / <|tool_call>call:read_file
    r"<\|tool_call\|?>\s*(?:call\s*[:=]\s*)?([A-Za-z_][A-Za-z0-9_.-]*)",
    # call:read_file / call = read_file
    r"\bcall\s*[:=]\s*([A-Za-z_][A-Za-z0-9_.-]*)",
    # tool_call: read_file
    r"\btool_call\s*[:=]\s*([A-Za-z_][A-Za-z0-9_.-]*)",
    # tool_name: "read_file"
    r"\btool_name\s*[:=]\s*[\"']?([A-Za-z_][A-Za-z0-9_.-]*)[\"']?",
    # tool: "read_file"
    r"\btool\s*[:=]\s*[\"']?([A-Za-z_][A-Za-z0-9_.-]*)[\"']?",
    # <invoke name="read_file">
    r"<invoke\s+name\s*=\s*[\"']([^\"']+)[\"']",
]

_DQ_STRING_RE = re.compile(r'("[^"\\]*(?:\\.[^"\\]*)*")')

_KEYVAL_QUOTED_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*[\"']([^\"']*)[\"']")
_KEYVAL_RAW_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*([^,\s\]\}]+)")


def _segments(text: str) -> list[str]:
    """Split text into alternating segments; odd indices are inside double
    quotes (protected from transformations)."""
    return _DQ_STRING_RE.split(text)


def _transform_outside_strings(text: str, transform) -> str:
    parts = _segments(text)
    out: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)
        else:
            out.append(transform(part))
    return "".join(out)


def _swap_quotes(s: str) -> str:
    """Turn single quotes (keys and values) into double quotes, skipping any
    text already inside double-quoted strings."""
    out: list[str] = []
    in_dq = False
    escaped = False
    for ch in s:
        if in_dq:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_dq = False
            continue
        if ch == '"':
            in_dq = True
            out.append(ch)
        elif ch == "'":
            out.append('"')
        else:
            out.append(ch)
    return "".join(out)


def _quote_unquoted_keys(s: str) -> str:
    def transform(part: str) -> str:
        return re.sub(
            r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)(\s*)(:)",
            lambda m: '"%s"%s%s' % (m.group(1), m.group(2), m.group(3)),
            part,
        )

    return _transform_outside_strings(s, transform)


def _json_literals(s: str) -> str:
    def transform(part: str) -> str:
        part = re.sub(r"\bNone\b", "null", part)
        part = re.sub(r"\bTrue\b", "true", part)
        part = re.sub(r"\bFalse\b", "false", part)
        return part

    return _transform_outside_strings(s, transform)


def _strip_trailing_commas(s: str) -> str:
    return _transform_outside_strings(
        s, lambda part: re.sub(r",\s*([}\]])", r"\1", part)
    )


def _normalize_candidate(candidate: str) -> str:
    s = _swap_quotes(candidate)
    s = _quote_unquoted_keys(s)
    s = _json_literals(s)
    s = _strip_trailing_commas(s)
    return s


def _repaired_variants(candidate: str) -> list[str]:
    """Produce candidate strings for truncated/malformed JSON: normalized form
    plus closed-open truncations (single and nested levels)."""
    base = _normalize_candidate(candidate)
    variants = [base]
    for closer in (
        '"}',
        '"',
        "}",
        "]",
        "}]",
        "}}",
        '"}}',
        "}]}",
        '"]}}',
        "]]}",
    ):
        variants.append(base + closer)
        variants.append(base.rstrip() + closer)
    return list(dict.fromkeys(variants))


def _scan_json_object(text: str, start: int) -> str:
    """Return the balanced JSON object starting at ``start`` (or the unbalanced
    tail when the object is truncated)."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
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
                return text[start : i + 1]
    return text[start:]


def _json_candidates(text: str) -> list[str]:
    """Collect JSON object literals (fenced, balanced or truncated) from text."""
    candidates: list[str] = []
    for block in _FENCED_BLOCK_RE.findall(text):
        candidates.append(block)
    for match in re.finditer(r"\{", text):
        candidates.append(_scan_json_object(text, match.start()))
    return list(dict.fromkeys(candidates))


def extract_json_object(text: str) -> Any | None:
    """Return the first top-level JSON value (object/array) found in ``text``."""
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            repaired = _normalize_candidate(candidate)
            try:
                return json.loads(repaired)
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def _normalize_tool(name: str) -> str | None:
    candidate = name.strip().strip('"').strip("'")
    if candidate in DEFAULT_ACTIONS:
        return candidate
    if candidate in TOOL_ALIASES:
        return TOOL_ALIASES[candidate]
    base = candidate.split(".")[-1].replace("-", "_").lower()
    if base in DEFAULT_ACTIONS:
        return base
    if base in TOOL_ALIASES:
        return TOOL_ALIASES[base]
    return None


def _tool_from_text(text: str) -> str | None:
    """Find a known tool name in tool-call markers (``call:read_file``,
    ``<|tool_call|>read_file``, ``tool_name: ...``, ``<invoke name=...>``)."""
    for pattern in _MARKER_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            normalized = _normalize_tool(match.group(1))
            if normalized:
                return normalized
    return None


def _has_call_signal(text: str) -> bool:
    """True when the text carries an explicit tool-call marker or a brace, so a
    bare mention of a tool name in prose does not count as a call."""
    if "{" in text or "}" in text:
        return True
    for pattern in _MARKER_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _params_from_text(text: str) -> dict[str, str]:
    """Extract ``key: value`` / ``key = value`` pairs from free text as params."""
    params: dict[str, str] = {}
    for match in _KEYVAL_QUOTED_RE.finditer(text):
        params[match.group(1)] = match.group(2)
    for match in _KEYVAL_RAW_RE.finditer(text):
        key, value = match.group(1), match.group(2)
        if key not in params and key not in _RESERVED and key not in _MARKER_KEYS:
            params[key] = value
    params.pop("tool", None)
    params.pop("action", None)
    return params


def _call_from_data(data: dict[str, Any], text: str) -> dict[str, Any] | None:
    """Build a normalized call from a parsed JSON object. The tool name may
    live in the JSON ``tool`` key or in call markers inside the text."""
    tool = data.get("tool")
    tool = tool if isinstance(tool, str) else None
    if tool:
        if tool.strip():
            tool = _normalize_tool(tool) or tool.strip()
    if not tool:
        tool = _tool_from_text(text)
    if not tool:
        return None

    action = data.get("action")
    action = action if isinstance(action, str) else ""

    params = data.get("params")
    if not isinstance(params, dict):
        params = {}
    for key, value in data.items():
        if key in _RESERVED:
            continue
        params[key] = value
    params.pop("action", None)

    return {
        "tool": tool,
        "action": action or DEFAULT_ACTIONS.get(tool, "run"),
        "params": params,
    }


def _parse_invoke_xml(text: str) -> dict[str, Any] | None:
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

        action = params.pop("action", "") or DEFAULT_ACTIONS.get(tool, "run")
        return {"tool": tool, "action": action, "params": params}

    return None


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Extract a tool call ``{"tool": ..., "action": ..., "params": {...}}``
    from a model response, tolerating malformed output.

    Layers, in order of preference:

    1. ``<invoke name=...>`` XML blocks (supports unknown tool names).
    2. Strict JSON (fenced or bare), filling the tool name from call markers
       when the ``tool`` key is missing.
    3. Syntax-repaired JSON (single quotes, unquoted keys, trailing commas,
       truncation).
    4. Fuzzy fallback: tool name from markers + ``key: value`` pairs.

    Returns None when the response looks like a plain answer.
    """
    xml_call = _parse_invoke_xml(text)
    if xml_call is not None:
        return xml_call

    candidates = _json_candidates(text)

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        call = _call_from_data(data, text)
        if call is not None:
            return call

    for candidate in candidates:
        for variant in _repaired_variants(candidate):
            try:
                data = json.loads(variant)
            except (json.JSONDecodeError, TypeError):
                continue
            call = _call_from_data(data, text)
            if call is not None:
                return call

    tool = _tool_from_text(text)
    if tool is not None and _has_call_signal(text):
        return {
            "tool": tool,
            "action": DEFAULT_ACTIONS.get(tool, "run"),
            "params": _params_from_text(text),
        }

    return None


def is_tool_attempt(text: str) -> bool:
    """True when the response looks like an attempted (possibly malformed) tool
    call rather than a plain final answer. Used to decide whether to re-prompt
    the model after a failed parse instead of trusting the text as an answer.

    A plain answer may still contain JSON (e.g. echoing a file), so braces
    alone are not enough: there must be a call marker or a tool-call signature
    (``tool``/``action``/``params`` keys or a known tool name in call syntax).
    """
    if re.search(
        r"<\|?tool_call|<invoke|\bcall\s*[:=]|\btool(_name)?\s*[:=]",
        text,
        re.IGNORECASE,
    ):
        return True
    if "{" not in text and "}" not in text:
        return False
    for candidate in _json_candidates(text):
        if re.search(r'"(?:tool|action|params|tool_name)"\s*:', candidate):
            return True
    return _tool_from_text(text) is not None
