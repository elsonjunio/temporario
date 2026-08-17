from __future__ import annotations

from pathlib import Path
from typing import Any

from src.discovery import symbols
from src.tools.base import ToolSpec
from src.tools.registry import ToolRegistry

CODE_MANUAL = (
    "code: symbol-level queries over the workspace source. Cheap structural "
    "analysis (stdlib, no provider); each action returns compact, capped "
    "results and never dumps full file contents.\n"
    "Actions (path is relative to the workspace or absolute; when omitted the "
    "workspace root is used):\n"
    "  - find_definition\n"
    "    Params: name (str, required); path (str, optional); include (str, "
    "    optional comma-separated globs).\n"
    "    Returns: where `name` is defined ({file, line, kind}), Python via "
    "    AST, other languages via definition-pattern heuristics.\n"
    "  - find_references\n"
    "    Params: name (str, required); path (str, optional); include (str, "
    "    optional); case_sensitive (bool, default false).\n"
    "    Returns: files using `name` with per-file match counts and sample "
    "    line numbers.\n"
    "  - find_symbol\n"
    "    Params: name (str, required); path (str, optional); include (str, "
    "    optional).\n"
    "    Returns: definitions plus usage files (compact, capped).\n"
    "  - find_imports\n"
    "    Params: path (str, required).\n"
    "    Returns: imports/dependencies of one file with line numbers, marking "
    "    local vs external modules.\n"
    "  - find_importers\n"
    "    Params: path (str, required); include (str, optional).\n"
    "    Returns: which files import or reference the module (reverse "
    "    dependencies).\n"
    "  - inspect\n"
    "    Params: path (str, required).\n"
    "    Returns: language, lines, size, one-line purpose, top-level symbols "
    "    and imports of a file without returning its body."
)


def build_code_spec(root: str | Path = ".") -> ToolSpec:
    """Build the ``code`` ToolSpec bound to a workspace root."""
    root_str = str(Path(root).resolve())

    def _kw(params: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in params.items() if v is not None}

    def handle_find_definition(**params: Any) -> dict:
        return symbols.find_definition(root_str, **_kw(params))

    def handle_find_references(**params: Any) -> dict:
        return symbols.find_references(root_str, **_kw(params))

    def handle_find_symbol(**params: Any) -> dict:
        return symbols.find_symbol(root_str, **_kw(params))

    def handle_find_imports(**params: Any) -> dict:
        return symbols.find_imports(root_str, **params)

    def handle_find_importers(**params: Any) -> dict:
        return symbols.find_importers(root_str, **params)

    def handle_inspect(**params: Any) -> dict:
        return symbols.inspect_file(root_str, **params)

    return ToolSpec(
        name="code",
        handlers={
            "find_definition": handle_find_definition,
            "find_references": handle_find_references,
            "find_symbol": handle_find_symbol,
            "find_imports": handle_find_imports,
            "find_importers": handle_find_importers,
            "inspect": handle_inspect,
        },
        manual=CODE_MANUAL,
    )


#: Alias tool names the discovery agent accepts, delegating to the base
#: read-only tools or the ``code`` symbol toolset. Small models invent these
#: names instead of the canonical registry names; registering them avoids
#: ``unknown_tool`` results that would burn the investigation budget.
_ALIASES: dict[str, tuple[str, str]] = {
    "list_files": ("list_dir", "list"),
    "ls": ("list_dir", "list"),
    "list": ("list_dir", "list"),
    "read": ("read_file", "read"),
    "cat": ("read_file", "read"),
    "search": ("search_files", "search"),
    "grep": ("grep_files", "search"),
    "grep_file": ("grep_files", "search"),
    "find_symbol": ("code", "find_symbol"),
    "find_definition": ("code", "find_definition"),
    "find_references": ("code", "find_references"),
    "find_imports": ("code", "find_imports"),
    "find_importers": ("code", "find_importers"),
    "inspect": ("code", "inspect"),
    "inspect_file": ("code", "inspect"),
}

_ALIAS_MANUAL = (
    "Alias for a read-only discovery tool. Accepts the same params as its "
    "target. Never modifies files or runs commands."
)


def _base_module(name: str) -> Any:
    from src.tools import grep_files, list_dir, read_file, search_files

    return {
        "list_dir": list_dir,
        "search_files": search_files,
        "grep_files": grep_files,
        "read_file": read_file,
    }[name]


def _alias_spec(name: str, target: str, action: str, code_spec: ToolSpec) -> ToolSpec:
    """Wrap a delegation to a base tool or the ``code`` toolset as a ToolSpec,
    so the alias works under several action spellings the model may emit."""

    def delegate(**params: Any) -> dict:
        if target == "code":
            return code_spec.dispatch(action, **params)
        return _base_module(target).dispatch(action, **params)

    return ToolSpec(
        name=name,
        handlers={action: delegate, "run": delegate, name: delegate},
        manual=_ALIAS_MANUAL,
    )


def build_discovery_registry(root: str | Path = ".") -> ToolRegistry:
    """Build the private, read-only registry used by the Discovery Agent.

    Reuses the existing read-only tool modules unchanged and adds the ``code``
    symbol toolset plus natural-language aliases. Mutating tools and command
    execution are never registered, so the agent cannot modify the project or
    run scripts.
    """
    from src.tools import (
        grep_files,
        list_dir,
        read_file,
        search_files,
    )

    registry = ToolRegistry()
    registry.register("list_dir", list_dir)
    registry.register("search_files", search_files)
    registry.register("grep_files", grep_files)
    registry.register("read_file", read_file)
    code_spec = build_code_spec(root)
    registry.register("code", code_spec)
    for alias, (target, action) in _ALIASES.items():
        registry.register(alias, _alias_spec(alias, target, action, code_spec))
    return registry


__all__ = ["CODE_MANUAL", "build_code_spec", "build_discovery_registry"]
