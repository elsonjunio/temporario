from __future__ import annotations

from typing import Any

from src.tools import (
    delete_file,
    grep_files,
    list_dir,
    move_file,
    patch_file,
    read_file,
    run_command,
    search_files,
    write_file,
)

BASE_TOOLS = [
    "read_file",
    "list_dir",
    "search_files",
    "grep_files",
    "write_file",
    "patch_file",
    "delete_file",
    "move_file",
    "run_command",
]


class ToolRegistry:
    """Registry of tool modules, each exposing ``get_manual()`` and
    ``dispatch(action, **params)``. The orchestrator is just another entry here,
    so removing it keeps the agent working with the base tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, name: str, module: Any) -> None:
        self._tools[name] = module

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Any | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def dispatch(self, tool: str, action: str, **params: Any) -> dict:
        module = self._tools.get(tool)
        if module is None:
            return {"error": "unknown_tool", "tool": tool}
        return module.dispatch(action, **params)

    def get_manual(self) -> str:
        sections = [module.get_manual() for module in self._tools.values()]
        return "\n\n".join(sections)


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("read_file", read_file)
    registry.register("list_dir", list_dir)
    registry.register("search_files", search_files)
    registry.register("grep_files", grep_files)
    registry.register("write_file", write_file)
    registry.register("patch_file", patch_file)
    registry.register("delete_file", delete_file)
    registry.register("move_file", move_file)
    registry.register("run_command", run_command)
    return registry


default_registry = build_default_registry()
