from __future__ import annotations

from typing import Any

from src.memory import Memory
from src.orchestrator.undo import MUTATING_TOOLS, UndoLog
from src.tools._fs import resolve_path
from src.tools.registry import ToolRegistry

READ_PAGE_SIZE = 100_000


class Executor:
    """Runs a plan step by step through the registry, validating each mutation
    and rolling everything back on failure."""

    def __init__(
        self,
        registry: ToolRegistry,
        undo_log: UndoLog,
        memory: Memory | None = None,
    ) -> None:
        self.registry = registry
        self.undo_log = undo_log
        self.memory = memory

    def _record(self, tool: str, action: str, params: dict, result: dict) -> None:
        if self.memory is not None:
            self.memory.add_tool(tool, action, params, result)

    def _target_path(self, tool: str, action: str, params: dict) -> str | None:
        if tool in ("write_file", "patch_file"):
            return params.get("file_path")
        if tool == "move_file":
            return params.get("destination")
        if tool == "delete_file":
            return params.get("path")
        return None

    def _validate(self, tool: str, action: str, params: dict, step: dict) -> dict:
        target = self._target_path(tool, action, params)
        expect = step.get("expect")
        if tool == "delete_file" and target is not None:
            exists = resolve_path(target).exists()
            return {"ok": not exists, "check": "path_removed", "path_exists": exists}

        if target is None:
            return {"ok": True, "check": "no_target"}

        read = self.registry.dispatch(
            "read_file",
            "read",
            file_path=target,
            page_size=READ_PAGE_SIZE,
        )
        content = read.get("current_page_content") or read.get("content") or ""
        ok = read.get("status") == "success"
        if ok and expect:
            ok = expect in content
        return {
            "ok": ok,
            "check": "readable_text",
            "expect": expect,
            "content_length": len(content),
        }

    def execute(self, steps: list[dict]) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        for step in steps:
            tool = step["tool"]
            action = step["action"]
            params = step.get("params", {})
            description = step.get("description", "")

            self.undo_log.snapshot(tool, action, params)
            result = self.registry.dispatch(tool, action, **params)
            self._record(tool, action, params, result)

            ok = result.get("status") == "success" and "error" not in result

            validated = None
            if ok and (tool in MUTATING_TOOLS or step.get("validate_after")):
                validated = self._validate(tool, action, params, step)
                ok = validated["ok"]

            trace.append(
                {
                    "tool": tool,
                    "action": action,
                    "description": description,
                    "params": params,
                    "result": result,
                    "ok": ok,
                    "validated": validated,
                }
            )

            if not ok:
                rollback = self.undo_log.rollback()
                return {
                    "status": "failed",
                    "failed_step": step,
                    "trace": trace,
                    "rollback": rollback,
                }

        return {"status": "success", "trace": trace}
