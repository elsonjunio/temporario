from __future__ import annotations

import json
from typing import Any

from src.tools.registry import ToolRegistry
from src.utils import extract_json_object

PLANNING_SYSTEM = (
    "You are a planning engine for a filesystem agent. You only produce JSON "
    "plans that use the provided tools; you never execute anything."
)


class Planner:
    """Turns a request plus discovery evidence into a validated JSON plan of
    tool steps, using the injected provider."""

    def __init__(
        self,
        provider: Any,
        registry: ToolRegistry,
        max_plan_steps: int = 8,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.max_plan_steps = max_plan_steps

    def build_prompt(self, request: str, evidence: dict[str, Any]) -> str:
        manual = self.registry.get_manual()
        evidence_block = json.dumps(evidence, ensure_ascii=False, indent=2)
        return (
            "Plan a filesystem modification.\n\n"
            f"Request:\n{request}\n\n"
            f"Discovery evidence (candidate targets, including impact):\n{evidence_block}\n\n"
            f"Available tools:\n{manual}\n\n"
            'Return ONLY a JSON object with a "steps" list. Each step is:\n'
            '{"tool": "<tool>", "action": "<action>", "params": {...}, '
            '"validate_after": true, "expect": "<optional substring to verify", '
            '"description": "..."}\n\n'
            "expect semantics: for write_file/patch_file it is a substring "
            "expected in the written file content, or a result keyword "
            '("created"/"overwritten"/"unchanged"/"applied"). For run_command '
            "it is a substring of stdout/stderr. Set validate_after=true only "
            "when a validation matters; the step fails and rolls back if "
            "expect is set but not found.\n"
            "For write_file/patch_file, set expect to a SHORT string you are "
            "certain appears VERBATIM in the file (an import line, a class or "
            "function name, a router prefix literal). NEVER set expect to a "
            "value assembled from separate literals, e.g. a full endpoint "
            "path like /api/auth/register when the file only holds prefix "
            "=/api/auth and route /register separately; that check will not "
            "match and the step rolls back.\n"
            "Reading a file that does not exist no longer fails the plan: "
            "the step logs a read_warn and execution continues. Before "
            "reading a file, list its directory first (list_dir) and use "
            "exact paths from that listing; never invent a filename.\n"
            "Rules:\n"
            "- Use only the available tools listed above.\n"
            "- Use concrete file paths from the discovery evidence.\n"
            "- If the workspace has a docs/SPEC.md (or docs/PLAYBOOK.md), add a "
            "read_file step for it first and follow its contract for the files "
            "you create or modify.\n"
            "- Before writing a module that depends on existing project files "
            "(models, schemas, services, routes), add read_file steps for those "
            "files and REUSE their exact class/enum/field/endpoint names and "
            "values; never invent new ones. When the SPEC conflicts with "
            "existing code, follow the SPEC and call out the deviation.\n"
            "- EXISTING targets (change_mode 'modify') must be edited with "
            "patch_file (replace or apply). Use write_file ONLY to create NEW "
            "files (change_mode 'new'), or for an intentional full rewrite in "
            'which case set "rewrite": true on the step.\n'
            "- GREENFIELD: targets seeded with type 'new_file' do not exist yet "
            "and must be created with write_file using the exact path from the "
            "evidence. Create parent directories implicitly via write_file "
            "(create_dirs defaults to true); there is no separate mkdir tool.\n"
            "- run_command steps that scaffold or install (pip install, npm "
            "install, ng new, git init) must set cwd to the project root and a "
            "generous timeout (600+) so they do not time out; set "
            "validate_after=true and expect a success marker when possible.\n"
            "- Registered tool modules (is_registered_tool=true) must keep "
            "their module contract: MANUAL, SPEC (name + handlers), "
            "get_manual() and dispatch(action, **params). A rewrite that drops "
            "these attributes fails validation and is rolled back.\n"
            "- Targets work in ANY language (python, typescript, javascript, "
            "java, kotlin, c, cpp, go, rust, ruby, php, csharp, swift, ...). "
            "The impact block reports the language of each target. Follow the "
            "language's idioms; never assume Python conventions.\n"
            "- Tests for changed modules:\n"
            "  * If the impact block lists test_files with a test_command for a "
            "changed module, add a final run_command step running that "
            "test_command verbatim, with validate_after=true and "
            "expect=<the impact block's success marker>.\n"
            "  * If test_files is empty but test_framework_configured=true and "
            "a test_suggestion exists, add final steps that (1) write_file the "
            "suggested new_test_path with a unit test covering the change "
            "(change_mode 'new'), then (2) run_command the suggested command "
            "with validate_after=true and expect=<success marker>.\n"
            "  * If test_framework_configured=false, the impact block carries a "
            "test_recommendation. Do NOT auto-create a test harness or run any "
            "test command. Instead, mention the recommendation to the user and "
            "implement a test only if the user explicitly asks for one or "
            "suggests a specific test to write.\n"
            "- Prefer patch_file/write_file for edits; delete_file and "
            "move_file only when required.\n"
            f"- At most {self.max_plan_steps} steps.\n"
            "- No markdown, no prose: pure JSON only."
        )

    def _validate_steps(self, steps: Any) -> str | None:
        if not isinstance(steps, list) or not steps:
            return "plan must contain a non-empty 'steps' list"
        if len(steps) > self.max_plan_steps:
            return f"plan exceeds the {self.max_plan_steps} step limit"
        available = set(self.registry.list_tools())
        for i, step in enumerate(steps, 1):
            if not isinstance(step, dict):
                return f"step {i} is not an object"
            tool = step.get("tool")
            action = step.get("action")
            params = step.get("params", {})
            if not isinstance(tool, str) or tool not in available:
                return f"step {i}: unknown tool {tool!r}"
            if not isinstance(action, str) or not action:
                return f"step {i}: missing action"
            if not isinstance(params, dict):
                return f"step {i}: params must be an object"
        return None

    def plan(self, request: str, evidence: dict[str, Any]) -> dict[str, Any]:
        prompt = self.build_prompt(request, evidence)
        raw = self.provider.infer(prompt, PLANNING_SYSTEM)

        data = extract_json_object(raw)
        if not isinstance(data, dict) or "steps" not in data:
            return {
                "status": "error",
                "message": "provider did not return a JSON plan with 'steps'",
                "raw": raw[:500],
            }

        steps = data["steps"]
        problem = self._validate_steps(steps)
        if problem is not None:
            return {"status": "error", "message": problem, "raw": raw[:500]}

        return {"status": "ok", "steps": steps}
