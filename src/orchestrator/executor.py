from __future__ import annotations

from typing import Any

from src.memory import Memory
from src.orchestrator.undo import MUTATING_TOOLS, UndoLog
from src.tools._fs import resolve_path
from src.tools.registry import ToolRegistry

READ_PAGE_SIZE = 100_000


class Executor:
    """Runs a plan step by step through the registry, validating each mutation
    and rolling everything back on failure.

    Every step is atomic: it is dispatched purely from its own params. When a
    ``patch_file`` step carries an ``instruction`` instead of hand-written
    anchors, the concrete payload is generated right before dispatch by the
    injected ``patch_generator`` from the CURRENT file content on disk — never
    from context accumulated in earlier steps.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        undo_log: UndoLog,
        memory: Memory | None = None,
        impact: Any | None = None,
        patch_generator: Any | None = None,
    ) -> None:
        self.registry = registry
        self.undo_log = undo_log
        # Local (orchestrator-owned) memory only; never the agent's shared one.
        self.memory = memory
        self.impact = impact
        self.patch_generator = patch_generator

    def _record(self, tool: str, action: str, params: dict, result: dict) -> None:
        if self.memory is not None:
            self.memory.add_tool(tool, action, params, result)

    def _read_target(self, file_path: str) -> str | None:
        read = self.registry.dispatch(
            "read_file",
            "read",
            file_path=file_path,
            page_size=READ_PAGE_SIZE,
        )
        if read.get("status") != "success":
            return None
        return read.get("current_page_content") or read.get("content") or ""

    def _generate_patch_params(
        self, params: dict, description: str
    ) -> dict[str, Any] | None:
        """Fill a patch_file step's payload via the dedicated generator.

        Returns ``None`` when the step does not need generation (explicit
        anchors already present) or when no generator is wired.
        """
        if self.patch_generator is None:
            return None
        if "old" in params or "diff" in params or not params.get("instruction"):
            return None
        file_path = str(params.get("file_path", ""))
        content = self._read_target(file_path)
        if content is None:
            return {
                "status": "error",
                "message": (
                    f"could not read {file_path} to generate the patch "
                    "(file missing or unreadable)"
                ),
            }
        instruction = str(params.get("instruction")) or description
        generated = self.patch_generator.generate(file_path, content, instruction)
        if generated.get("status") != "ok":
            return {
                "status": "error",
                "message": generated.get("message", "patch generation failed"),
            }
        return {"status": "ok", "params": dict(generated.get("params", {}))}

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
            warn = None
            if ok and expect:
                ok = expect.lower() in output.lower()
                if not ok:
                    ok = True
                    warn = f"expect {expect!r} not found in command output"
            return {
                "ok": ok,
                "check": "command_output",
                "expect": expect,
                "expect_warn": warn,
            }

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
        warn = None
        if ok and expect:
            mode = result.get("mode")
            ok = expect in content or (mode is not None and expect == mode)
            if not ok and tool == "write_file" and mode in ("created", "overwritten"):
                ok = True
                warn = f"expect {expect!r} not found verbatim in written content"

        deep = self._deep_validate(target)
        if deep is not None:
            ok = ok and deep["ok"]
            return {
                "ok": ok,
                "check": "module_contract",
                "expect": expect,
                "expect_warn": warn,
                "contract": deep,
                "content_length": len(content),
            }
        return {
            "ok": ok,
            "check": "readable_text",
            "expect": expect,
            "expect_warn": warn,
            "content_length": len(content),
        }

    def execute(
        self, steps: list[dict], rollback_on_failure: bool = True
    ) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        soft_failures: list[dict[str, Any]] = []
        for step in steps:
            tool = step["tool"]
            action = step["action"]
            params = dict(step.get("params", {}))
            description = step.get("description", "")
            soft = bool(step.get("soft"))

            if (
                tool == "run_command"
                and not params.get("cwd")
                and self.impact is not None
            ):
                params["cwd"] = self.impact.root

            generated: dict[str, Any] | None = None
            if tool == "patch_file":
                generated = self._generate_patch_params(params, description)
                if generated is not None and generated.get("status") != "ok":
                    result = {
                        "status": "error",
                        "error": {
                            "message": generated.get(
                                "message", "patch generation failed"
                            )
                        },
                    }
                    self.undo_log.snapshot(tool, action, params)
                    self._record(tool, action, params, result)
                    trace.append(
                        {
                            "tool": tool,
                            "action": action,
                            "description": description,
                            "params": params,
                            "result": result,
                            "ok": False,
                            "validated": None,
                            "read_warn": None,
                            "patch_generated": False,
                        }
                    )
                    if rollback_on_failure:
                        rollback = self.undo_log.rollback()
                    else:
                        rollback = {
                            "status": "kept",
                            "reason": "rollback_on_failure=False",
                        }
                    return {
                        "status": "failed",
                        "failed_step": step,
                        "trace": trace,
                        "rollback": rollback,
                    }
                if generated is not None:
                    params.update(generated["params"])

            self.undo_log.snapshot(tool, action, params)
            # ``instruction`` is an orchestration-only hint (drives the patch
            # generator); the underlying tool does not know about it.
            dispatch_params = {
                key: value
                for key, value in params.items()
                if not (tool == "patch_file" and key == "instruction")
            }
            result = self.registry.dispatch(tool, action, **dispatch_params)
            self._record(tool, action, dispatch_params, result)

            ok = result.get("status") == "success" and "error" not in result

            read_warn = None
            if not ok and tool == "read_file":
                ok = True
                read_warn = (
                    "read_file did not succeed (file may not exist); "
                    "proceeding so the plan can recover by listing the directory"
                )

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
                    "read_warn": read_warn,
                    "patch_generated": bool(generated is not None),
                }
            )

            if not ok and soft:
                error = result.get("error")
                message = (
                    error.get("message")
                    if isinstance(error, dict)
                    else result.get("message")
                )
                trace[-1]["soft"] = True
                trace[-1]["soft_failure"] = True
                soft_failures.append(
                    {
                        "step": len(trace),
                        "tool": tool,
                        "action": action,
                        "description": description,
                        "message": message,
                    }
                )
                continue

            if not ok:
                if rollback_on_failure:
                    rollback = self.undo_log.rollback()
                else:
                    rollback = {"status": "kept", "reason": "rollback_on_failure=False"}
                return {
                    "status": "failed",
                    "failed_step": step,
                    "trace": trace,
                    "rollback": rollback,
                }

        return {
            "status": "success",
            "trace": trace,
            "soft_failures": soft_failures,
        }
