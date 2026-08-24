from __future__ import annotations

from pathlib import Path
from typing import Any

from src.patcher.agent import (
    BUDGET_PARAMS,
    LOCATE_INSTRUCTIONS,
    PatcherAgent,
    sandbox_path,
)
from src.patcher.budget import PatcherBudget
from src.patcher.generator import PATCH_SYSTEM, PatchGenerator
from src.tools.base import ToolSpec

PATCHER_MANUAL = (
    "patcher: turn ONE change instruction into VALIDATED patch payloads for "
    "existing files. It reads the target files itself, finds the right "
    "place, generates the patch and checks it deterministically against the "
    "current disk content (anchors verbatim and unique, hunks applied in an "
    "in-memory dry run, no-op patches rejected). It NEVER modifies files: "
    "apply the returned params yourself with patch_file.\n"
    "Actions:\n"
    "  - run\n"
    "    Params:\n"
    "      instruction (str, required; synonyms request/task/prompt): what\n"
    "        must change, self-contained per file.\n"
    "      paths (list[str], optional; synonyms path/file/files): explicit\n"
    "        target files; when omitted a bounded read-only locate phase\n"
    "        picks them (paths are relative to the workspace root).\n"
    "      context (str, optional): extra constraints, capped in size.\n"
    "    Returns: status ok/partial/error, located, patches (each with\n"
    "      file_path/mode/params ready for patch_file dispatch), failures,\n"
    "      metrics.\n"
    "  - validate\n"
    "    Deterministic check of an explicit payload against the current\n"
    "    file content (no LLM involved).\n"
    "    Params: file_path (str, required) plus either old/new/replace_all\n"
    "    or diff.\n"
    "    Returns: status ok/rejected/error and the sanitized params when ok.\n"
)

_INSTRUCTION_KEYS = ("instruction", "request", "task", "prompt", "job", "query")
_PATH_KEYS = ("paths", "path", "file", "files")


def create_patcher_tool(
    provider: Any,
    *,
    root: str = ".",
    name: str = "patcher",
    budget: PatcherBudget | None = None,
) -> ToolSpec:
    """Build the ``patcher`` tool bound to a workspace root.

    Registering it gives any agent (with or without the orchestrator) a safe
    way to produce concrete ``patch_file`` payloads: generation is isolated
    from conversation memory and every returned patch already passed the
    deterministic validation gate.
    """
    root_str = str(Path(root).resolve())
    agent = PatcherAgent(provider, root=root_str, budget=budget)

    def run_handler(**params: Any) -> dict:
        instruction = ""
        for key in _INSTRUCTION_KEYS:
            value = params.pop(key, None)
            if isinstance(value, str) and value.strip():
                instruction = value.strip()
                break
        if not instruction:
            return {
                "status": "error",
                "error": {
                    "type": "invalid_arguments",
                    "message": "instruction is required",
                    "recoverable": True,
                },
            }

        raw_paths: Any = None
        for key in _PATH_KEYS:
            value = params.pop(key, None)
            if value:
                raw_paths = value
                break
        paths: list[str] | None = None
        if isinstance(raw_paths, str):
            paths = [raw_paths]
        elif isinstance(raw_paths, (list, tuple)):
            paths = [str(item) for item in raw_paths]

        context = params.pop("context", None)
        if not isinstance(context, str):
            context = None

        budget_params = {k: v for k, v in params.items() if k in BUDGET_PARAMS}
        try:
            return agent.run(instruction, paths=paths, context=context, **budget_params)
        except Exception as exc:  # noqa: BLE001 - surface as a clean error
            return {
                "status": "error",
                "error": {
                    "type": "patcher_error",
                    "message": str(exc),
                    "recoverable": True,
                },
            }

    def validate_handler(**params: Any) -> dict:
        from src.tools._fs import is_binary

        resolved, error = sandbox_path(root_str, params.get("file_path"))
        if error or resolved is None:
            return {"status": "error", "message": error or "invalid path"}
        path = Path(resolved)
        if is_binary(path):
            return {"status": "error", "message": "binary file, patches not supported"}
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return {"status": "error", "message": f"could not read file: {exc}"}

        payload: dict[str, Any] = {}
        if params.get("diff") is not None:
            payload["mode"] = "apply"
            payload["diff"] = str(params["diff"])
        elif params.get("old") is not None:
            payload["mode"] = "replace"
            payload["old"] = str(params["old"])
            payload["new"] = str(params.get("new", ""))
            if params.get("replace_all"):
                payload["replace_all"] = True
        else:
            return {
                "status": "error",
                "message": "provide either old/new (replace) or diff (apply)",
            }

        problem, clean_params = PatchGenerator._validate(payload, content)
        if problem is not None:
            return {"status": "rejected", "message": problem}
        final_params: dict[str, Any] = {"file_path": resolved}
        final_params.update(clean_params)
        mode = payload.get("mode")
        return {"status": "ok", "mode": mode, "params": final_params}

    return ToolSpec(
        name=name,
        handlers={
            "run": run_handler,
            "generate": run_handler,
            "patch": run_handler,
            "validate": validate_handler,
            "check": validate_handler,
        },
        manual=PATCHER_MANUAL,
    )


__all__ = [
    "BUDGET_PARAMS",
    "LOCATE_INSTRUCTIONS",
    "PATCH_SYSTEM",
    "PatcherAgent",
    "PatcherBudget",
    "PatchGenerator",
    "PATCHER_MANUAL",
    "create_patcher_tool",
    "sandbox_path",
]
