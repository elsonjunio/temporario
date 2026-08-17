from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.discovery.budget import DiscoveryBudget
from src.discovery.subagent import DISCOVERY_INSTRUCTIONS, DiscoveryAgent
from src.discovery.tools import (
    CODE_MANUAL,
    build_code_spec,
    build_discovery_registry,
)
from src.tools.base import ToolSpec

#: Concrete read-only actions the discovery tool accepts, matching the natural
#: vocabulary small models use (they rarely drive an abstract ``run`` subagent
#: on the first try). Each maps to a base read-only tool + action.
ACTION_ALIASES: dict[str, tuple[str, str]] = {
    "list_dir": ("list_dir", "list"),
    "list_files": ("list_dir", "list"),
    "ls": ("list_dir", "list"),
    "list": ("list_dir", "list"),
    "search": ("search_files", "search"),
    "search_files": ("search_files", "search"),
    "grep": ("grep_files", "search"),
    "grep_files": ("grep_files", "search"),
    "read": ("read_file", "read"),
    "read_file": ("read_file", "read"),
    "cat": ("read_file", "read"),
    "inspect": ("code", "inspect"),
    "inspect_file": ("code", "inspect"),
    "find_symbol": ("code", "find_symbol"),
    "find_definition": ("code", "find_definition"),
    "find_references": ("code", "find_references"),
    "find_imports": ("code", "find_imports"),
    "find_importers": ("code", "find_importers"),
}

_DISCOVERY_BUDGET_PARAMS = {
    "max_iterations",
    "max_tool_calls",
    "max_context_tokens",
    "max_files_read",
    "max_lines_read",
}

#: Appended to concrete (read-only) action results so the agent loop nudges the
#: model toward the full ``run`` subagent instead of stopping after a single
#: quick lookup.
_LOOKUP_HINT = (
    "Quick lookup only. If the task is to investigate what changes are needed "
    "(e.g. applying DESIGN.md to the frontend, which files need adjustments), "
    "continue with discovery run using the full request."
)

DISCOVERY_MANUAL = (
    "discovery: read-only investigation over the workspace (bounded budget, "
    "never modifies files, never runs commands).\n"
    "USE run FOR INVESTIGATION TASKS. Investigation means analyzing what is "
    "needed to accomplish something or which files need changes (e.g. 'o que "
    "é preciso para aplicar o DESIGN.md no front e quais arquivos devem "
    "receber ajustes'). run hands the full natural-language request to the "
    "discovery subagent, which reads the relevant files and returns a "
    "structured DISCOVERY RESULT report (task, summary, relevant files, "
    "relationships, hypotheses, evidence, discarded, risks). Example:\n"
    '```json\n{"tool": "discovery", "action": "run", "params": {"request": '
    '"o que é preciso para aplicar o DESIGN.md no front e quais arquivos '
    'devem receber ajustes"}}\n```\n'
    "Actions (paths are relative to the workspace root):\n"
    "  - run\n"
    "    Params: request (str, required; synonyms query/prompt/task/question; "
    "optional path to focus on).\n"
    '    Returns: {"status": "success"|"partial"|"error", "task", "workspace", '
    '"report", "metrics"}. The report is a structured plain-text DISCOVERY '
    "RESULT meant to be passed to the executor so it does not repeat the "
    "investigation.\n"
    "The concrete actions below are QUICK LOOKUPS ONLY - a single "
    "search/list/grep/read is NOT a complete investigation. If the task "
    "requires analyzing what changes are needed, escalate to run instead:\n"
    "  - list_files (also list_dir, ls, list)\n"
    '    Params: path (str, default "."); recursive (bool, default false); '
    "max_depth (int, default 3); extension (str, optional).\n"
    "    Returns: paginated items (name, path, type, size).\n"
    "  - search (also search_files)\n"
    '    Params: pattern (str, required); path (str, default ".").\n'
    "    Returns: files whose names match the glob pattern.\n"
    "  - grep (also grep_files)\n"
    '    Params: pattern (str, required); path (str, default "."); include '
    "(str, optional); case_sensitive (bool, default false).\n"
    "    Returns: files whose content matches the regex, with line numbers.\n"
    "  - read (also read_file, cat)\n"
    "    Params: file_path (str, required); start_line (int, optional); "
    "end_line (int, optional); max_lines (int, default 2000).\n"
    "    Returns: file content or a line range.\n"
    "  - find_symbol / find_definition / find_references / find_imports / "
    "find_importers / inspect (also inspect_file)\n"
    "    Params: name (str) or path (str) per action.\n"
    "    Returns: compact symbol-level analysis of the workspace."
)


@contextmanager
def _workspace_dir(path: str) -> Iterator[None]:
    """chdir into ``path`` so base tools resolve relative paths correctly."""
    original = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _readonly_module(tool_name: str) -> Any:
    from src.tools import grep_files, list_dir, read_file, search_files

    return {
        "list_dir": list_dir,
        "search_files": search_files,
        "grep_files": grep_files,
        "read_file": read_file,
    }[tool_name]


def create_discovery_tool(
    provider: Any,
    *,
    root: str = ".",
    name: str = "discovery",
    budget: DiscoveryBudget | None = None,
) -> ToolSpec:
    """Build the ``discovery`` tool bound to a workspace root.

    Registering it in the main registry lets the agent decide when to run a
    discovery pass. The tool exposes the ``run`` subagent action plus concrete
    read-only aliases (list_files/search/grep/read/inspect) that delegate to
    the base tools, so small models can drive it with their natural vocabulary.
    """
    root_str = str(Path(root).resolve())
    agent = DiscoveryAgent(provider, root=root_str, budget=budget)
    code_spec = build_code_spec(root_str)

    def run_handler(**params: Any) -> dict:
        request = params.pop("request", "")
        if not request:
            request = (
                params.pop("query", "")
                or params.pop("prompt", "")
                or params.pop("task", "")
                or params.pop("question", "")
            )
        request = str(request or "").strip()
        if not request:
            return {
                "status": "error",
                "error": {
                    "type": "invalid_arguments",
                    "message": "request is required",
                    "recoverable": True,
                },
            }
        path = params.pop("path", None)
        if path:
            request = f"{request} (focus: {path})" if request else f"Investigate {path}"
        budget_params = {
            k: v for k, v in params.items() if k in _DISCOVERY_BUDGET_PARAMS
        }
        try:
            return agent.run(request, **budget_params)
        except Exception as exc:  # noqa: BLE001 - surface as a clean error
            return {
                "status": "error",
                "error": {
                    "type": "discovery_error",
                    "message": str(exc),
                    "recoverable": True,
                },
            }

    def readonly_handler(tool_name: str, action: str):
        def handler(**params: Any) -> dict:
            with _workspace_dir(root_str):
                if tool_name == "code":
                    result = code_spec.dispatch(action, **params)
                else:
                    result = _readonly_module(tool_name).dispatch(action, **params)
            if isinstance(result, dict):
                result.setdefault("hint", _LOOKUP_HINT)
            return result

        return handler

    handlers: dict[str, Any] = {"run": run_handler}
    for alias, (tool_name, action) in ACTION_ALIASES.items():
        handlers[alias] = readonly_handler(tool_name, action)

    return ToolSpec(
        name=name,
        handlers=handlers,
        manual=DISCOVERY_MANUAL,
    )


__all__ = [
    "ACTION_ALIASES",
    "CODE_MANUAL",
    "DISCOVERY_INSTRUCTIONS",
    "DISCOVERY_MANUAL",
    "DiscoveryAgent",
    "DiscoveryBudget",
    "build_code_spec",
    "build_discovery_registry",
    "create_discovery_tool",
]
