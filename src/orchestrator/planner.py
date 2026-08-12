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
            f"Discovery evidence (candidate targets):\n{evidence_block}\n\n"
            f"Available tools:\n{manual}\n\n"
            'Return ONLY a JSON object with a "steps" list. Each step is:\n'
            '{"tool": "<tool>", "action": "<action>", "params": {...}, '
            '"validate_after": true, "expect": "<optional substring to verify", '
            '"description": "..."}\n\n'
            "Rules:\n"
            "- Use only the available tools listed above.\n"
            "- Use concrete file paths from the discovery evidence.\n"
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
