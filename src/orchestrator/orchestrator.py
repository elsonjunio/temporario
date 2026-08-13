from __future__ import annotations

from typing import Any

from src.memory import Memory
from src.orchestrator.discovery import Discovery
from src.orchestrator.executor import Executor
from src.orchestrator.impact import ImpactAssessor
from src.orchestrator.planner import Planner
from src.orchestrator.undo import UndoLog
from src.tools.registry import ToolRegistry

MANUAL = (
    "orchestrator: plan and execute multi-step filesystem changes end to end. "
    "Use it for non-trivial modifications (edits, refactors, multi-file "
    "changes). For simple single reads or listings, use the direct tools "
    "instead.\n"
    "Actions:\n"
    "  - run\n"
    "    Params:\n"
    "      request (str, required): what to change, in natural language.\n"
    "      terms (list[str], optional): explicit search terms for discovery.\n"
    "      confirm (bool, default false): when false (default) the tool stops\n"
    "        after planning and returns an 'awaiting_confirmation' preview with\n"
    "        the steps and impact analysis to be executed. Re-run with\n"
    "        confirm=True only after the user approves; the pending plan is\n"
    "        reused, not re-planned.\n"
    "    Returns: the plan preview awaiting confirmation, the executed trace,\n"
    "      or a rollback/abort notice.\n"
    "    Strategy: existing files are edited with patch_file; write_file is\n"
    "      reserved for new files or explicit rewrites (rewrite:true).\n"
    "      Registered tool modules must keep MANUAL/SPEC/get_manual/dispatch.\n"
    "      Any language is supported (python, typescript, javascript, java,\n"
    "      kotlin, c, cpp, go, rust, ruby, php, csharp, swift, ...): the impact\n"
    "      analysis detects the target language, maps existing unit tests by the\n"
    "      language's conventions and detects the project's test runner. When a\n"
    "      changed module is covered by tests, a final step runs them; when a\n"
    "      test framework is configured but nothing covers the change, the plan\n"
    "      adds a suggested test file and runs it; when no test setup exists, a\n"
    "      recommendation is shown instead of auto-creating a harness.\n"
    "  - discover\n"
    "    Params:\n"
    "      request (str, required): what to find or change.\n"
    "      terms (list[str], optional): explicit search terms.\n"
    "    Returns: candidate targets with evidence snippets. status 'poor'\n"
    "      means discovery failed and the flow should abort.\n"
    "  - assess\n"
    "    Params:\n"
    "      request (str, required): what to find or change.\n"
    "      terms (list[str], optional): explicit search terms.\n"
    "    Returns: per-candidate impact analysis (change mode, recommended tool,\n"
    "      contract status, risks, mapped unit tests) without planning.\n"
    "  - plan\n"
    "    Params:\n"
    "      request (str, required): what to change.\n"
    "      terms (list[str], optional): explicit search terms.\n"
    "    Returns: a validated JSON plan of steps, 'aborted' when discovery\n"
    "      was poor, or 'error' when the plan would clobber a registered tool\n"
    "      module without an explicit rewrite.\n"
    "  - execute\n"
    "    Params:\n"
    "      steps (list, optional): custom steps; defaults to the pending plan.\n"
    "      confirm (bool, default false): execution requires confirm=True;\n"
    "        without it returns an 'awaiting_confirmation' preview. To resume\n"
    "        the pending plan after user approval, call execute with\n"
    "        confirm=True and NO steps.\n"
    "    Returns: per-step trace with validation results (deep module-contract\n"
    "      checks on registered tool modules). On failure all changes made so\n"
    "      far are rolled back automatically.\n"
    "  - pending\n"
    "    Params: (none)\n"
    "    Returns: the pending plan (request and steps) awaiting confirmation,\n"
    "      or 'no_pending_plan' when there is nothing to resume.\n"
    "  - validate\n"
    "    Params:\n"
    "      path (str, required): file to check.\n"
    "    Returns: whether the file is readable text plus its first lines.\n"
    "  - undo\n"
    "    Params: (none)\n"
    "    Returns: restores every snapshotted file to its previous state.\n"
    "  - abort\n"
    "    Params: (none)\n"
    "    Returns: discards the current plan and any pending snapshots.\n"
    "Safety: every mutating step (write/patch/move/delete) is snapshotted "
    "before execution and rolled back automatically if a step fails. The tool "
    "never executes before returning an awaiting_confirmation preview: always "
    "relay the summary to the user and only proceed after approval. Use undo "
    "to revert manually after a successful run."
)


class Orchestrator:
    """Special tool that performs discovery, planning, step-by-step execution
    with validation, and rollback — all through the injected tool registry."""

    def __init__(
        self,
        registry: ToolRegistry,
        provider: Any,
        memory: Memory | None = None,
        *,
        backup_root: str | None = None,
        max_plan_steps: int = 8,
        root: str = ".",
    ) -> None:
        self.registry = registry
        self.provider = provider
        self.memory = memory
        self.discovery = Discovery(registry, root=root)
        self.impact = ImpactAssessor(registry, root=root)
        self.planner = Planner(provider, registry, max_plan_steps=max_plan_steps)
        self.undo_log = UndoLog(backup_root=backup_root)
        self.executor = Executor(registry, self.undo_log, memory, impact=self.impact)
        self.last_evidence: dict[str, Any] | None = None
        self.last_impact: dict[str, Any] | None = None
        self.last_plan: dict[str, Any] | None = None
        self.last_request: str | None = None

    def discover(self, request: str, terms: list[str] | None = None) -> dict[str, Any]:
        evidence = self.discovery.discover(request, terms)
        self.last_evidence = evidence
        return evidence

    def plan(self, request: str, terms: list[str] | None = None) -> dict[str, Any]:
        if self.last_evidence is None or self.last_evidence.get("request") != request:
            self.last_evidence = self.discovery.discover(request, terms)
        evidence = self.last_evidence

        if evidence.get("status") != "ok":
            return {
                "status": "aborted",
                "reason": evidence.get("reason"),
                "evidence": evidence,
            }

        impact = self.impact.assess(request, evidence.get("candidates", []))
        self.last_impact = impact

        plan = self.planner.plan(request, {**evidence, "impact": impact})
        if plan.get("status") == "ok":
            issues = self.impact.check_plan_steps(plan.get("steps", []))
            if issues:
                return {
                    "status": "error",
                    "message": "\n".join(issues),
                    "plan": plan,
                }
        self.last_plan = plan
        self.last_request = request
        return plan

    def assess(self, request: str, terms: list[str] | None = None) -> dict[str, Any]:
        if self.last_evidence is None or self.last_evidence.get("request") != request:
            self.last_evidence = self.discovery.discover(request, terms)
        evidence = self.last_evidence
        impact = self.impact.assess(request, evidence.get("candidates", []))
        self.last_impact = impact
        return {"status": "ok", "request": request, **impact}

    def _awaiting_confirmation(
        self,
        steps: list[dict],
        impact: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        summary = [
            {
                "step": i + 1,
                "tool": step.get("tool"),
                "action": step.get("action"),
                "description": step.get("description", ""),
                "params": step.get("params", {}),
            }
            for i, step in enumerate(steps)
        ]
        result: dict[str, Any] = {
            "status": "awaiting_confirmation",
            "message": (
                "The plan below is ready to execute. Ask the user for "
                "approval first. When the user approves, resume by calling "
                "orchestrator execute with confirm=true and no steps (the "
                "pending plan is reused, not re-planned). If the user wants "
                "changes, call orchestrator run with the modified request. "
                "To discard, call orchestrator abort."
            ),
            "summary": summary,
            "steps": steps,
        }
        if impact is not None:
            result["impact"] = self._format_impact(impact)
        return result

    @staticmethod
    def _format_impact(impact: dict[str, Any]) -> str:
        lines = ["Impact analysis:"]
        for target in impact.get("targets", []):
            lines.append(
                f"- {target.get('path')}: mode={target.get('change_mode')} "
                f"(recommended {target.get('recommended_tool')})"
            )
            language = target.get("language")
            if language:
                lines.append(f"    language: {language}")
            if target.get("is_registered_tool"):
                preserved = target.get("contract_preserved")
                lines.append(
                    f"    registered tool module; contract preserved: {preserved}"
                )
            for risk in target.get("risks", []):
                lines.append(f"    risk: {risk}")
            if target.get("test_command"):
                lines.append(f"    tests: {target.get('test_command')}")
            suggestion = target.get("test_suggestion")
            if suggestion:
                lines.append(
                    "    test setup found but nothing covers this target; plan "
                    "will add a test and run: "
                    f"{suggestion.get('new_test_path')} -> "
                    f"{suggestion.get('command')}"
                )
            recommendation = target.get("test_recommendation")
            if recommendation:
                lines.append(f"    no test setup; {recommendation}")
        return "\n".join(lines)

    def pending(self) -> dict[str, Any]:
        steps = (self.last_plan or {}).get("steps")
        if self.last_plan is None or not steps:
            return {"status": "no_pending_plan"}
        return {
            "status": "pending_plan",
            "request": self.last_request,
            "steps": steps,
        }

    def execute(
        self,
        steps: list[dict] | None = None,
        confirm: bool = False,
    ) -> dict[str, Any]:
        plan = steps or (self.last_plan or {}).get("steps")
        if not plan:
            return {
                "status": "error",
                "message": "no plan to execute; call plan first or pass steps.",
            }
        if not confirm:
            return self._awaiting_confirmation(plan)
        return self.executor.execute(plan)

    def validate(self, path: str) -> dict[str, Any]:
        result = self.registry.dispatch("read_file", "read", file_path=path)
        ok = result.get("status") == "success"
        return {
            "status": "success" if ok else "error",
            "path": path,
            "result": result,
        }

    def run(
        self,
        request: str,
        terms: list[str] | None = None,
        confirm: bool = False,
    ) -> dict[str, Any]:
        if self.last_request != request or self.last_plan is None:
            plan = self.plan(request, terms)
        else:
            plan = self.last_plan
        if plan.get("status") == "error":
            return {"status": "error", "plan": plan}
        if plan.get("status") != "ok":
            return {"status": "aborted", "plan": plan}
        if not confirm:
            return self._awaiting_confirmation(
                plan.get("steps", []), impact=self.last_impact
            )
        return self.executor.execute(plan.get("steps", []))

    def undo(self) -> dict[str, Any]:
        return self.undo_log.rollback()

    def abort(self) -> dict[str, Any]:
        discarded = self.undo_log.discard()
        self.last_evidence = None
        self.last_impact = None
        self.last_plan = None
        self.last_request = None
        return {"status": "aborted", "discarded_snapshots": discarded}
