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
        impact: Any | None = None,
    ) -> None:
        self.registry = registry
        self.undo_log = undo_log
        self.memory = memory
        self.impact = impact

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

    def _deep_validate(self, target: str) -> dict[str, Any] | None:
        """Re-import changed Python files that back a registered tool module
        and verify the module contract survived the mutation."""
        if self.impact is None:
            return None
        profile = self.impact.profile(target)
        if not (profile.get("exists") and profile.get("is_registered_tool")):
            return None
        contract = profile.get("contract", {})
        if contract.get("import_error"):
            return {"ok": False, "error": contract["import_error"]}
        missing = profile.get("contract_missing", [])
        if missing:
            return {"ok": False, "missing": missing}
        return {"ok": True, "missing": []}

    def _validate(
        self,
        tool: str,
        action: str,
        params: dict,
        step: dict,
        result: dict,
    ) -> dict:
        target = self._target_path(tool, action, params)
        expect = step.get("expect")
        if tool == "delete_file" and target is not None:
            exists = resolve_path(target).exists()
            return {"ok": not exists, "check": "path_removed", "path_exists": exists}

        if tool == "run_command":
            output = "\n".join(
                str(result.get(key) or "") for key in ("stdout", "stderr")
            )
            ok = result.get("status") == "success"
            if ok and expect:
                ok = expect in output
            return {"ok": ok, "check": "command_output", "expect": expect}

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
            mode = result.get("mode")
            ok = expect in content or (mode is not None and expect == mode)

        deep = self._deep_validate(target)
        if deep is not None:
            ok = ok and deep["ok"]
            return {
                "ok": ok,
                "check": "module_contract",
                "expect": expect,
                "contract": deep,
                "content_length": len(content),
            }
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
            params = dict(step.get("params", {}))
            description = step.get("description", "")

            if (
                tool == "run_command"
                and not params.get("cwd")
                and self.impact is not None
            ):
                params["cwd"] = self.impact.root

            self.undo_log.snapshot(tool, action, params)
            result = self.registry.dispatch(tool, action, **params)
            self._record(tool, action, params, result)

            ok = result.get("status") == "success" and "error" not in result

            validated = None
            if ok and (tool in MUTATING_TOOLS or step.get("validate_after")):
                validated = self._validate(tool, action, params, step, result)
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
