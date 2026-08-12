"""Backward-compatible facade over the default tool registry.

Exposes module-level ``dispatch``/``get_manual``/``register_tool``/... bound to
the ``default_registry`` singleton. Prefer using a ``ToolRegistry`` instance
directly for new code; this shim exists only for legacy callers."""

from __future__ import annotations

from typing import Any

from src.tools.registry import default_registry


def register_tool(name: str, module: Any) -> None:
    """Register an extra tool (e.g. the agentic orchestrator) in the default
    registry. Tools added here are removable via ``unregister_tool``."""
    default_registry.register(name, module)


def unregister_tool(name: str) -> bool:
    """Remove a tool from the default registry. Returns True if removed."""
    return default_registry.unregister(name)


def dispatch(tool: str, action: str, **params: object) -> dict:
    return default_registry.dispatch(tool, action, **params)


def get_manual() -> str:
    return default_registry.get_manual()


def list_tools() -> list[str]:
    return default_registry.list_tools()
