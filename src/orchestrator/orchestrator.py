from __future__ import annotations

from typing import Any, Callable
from pathlib import Path

from src.memory import Memory
from src.orchestrator.discovery import Discovery
from src.orchestrator.executor import Executor
from src.orchestrator.impact import ImpactAssessor
from src.orchestrator.planner import Planner
from src.orchestrator.undo import UndoLog
from src.tools.registry import ToolRegistry

_DOC_EXTENSIONS = {".md", ".txt", ".rst", ".doc", ".docx", ".markdown"}
_CODE_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".java",
    ".kt",
    ".kts",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".cs",
    ".swift",
    ".scss",
    ".css",
    ".html",
    ".vue",
    ".svelte",
}
_UI_SIGNALS = {
    "estilo",
    "style",
    "design",
    "catálogo",
    "catalogo",
    "catalog",
    "tela",
    "page",
    "componente",
    "component",
    "cor",
    "color",
    "colour",
    "tema",
    "theme",
    "ui",
    "css",
    "layout",
    "visual",
}


def _has_ui_signal(request: str) -> bool:
    lowered = request.lower()
    return any(signal in lowered for signal in _UI_SIGNALS)


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
    "      paths (list[str], optional): explicit target file/dir paths to\n"
    "        create or modify. When given, keyword discovery is SKIPPED and\n"
    "        each path is treated as a candidate (new_file when missing). Use\n"
    "        this for greenfield work (new project, empty directory) where the\n"
    "        request names the exact files to create.\n"
    "      confirm (bool, default false): when false (default) the tool stops\n"
    "        after planning and returns an 'awaiting_confirmation' preview with\n"
    "        the steps and impact analysis to be executed.\n"
    "        When called again with confirm=true (and no request) the pending\n"
    "        plan is reused, not re-planned. To resume after user approval,\n"
    "        prefer orchestrator execute confirm=true.\n"
    "    Returns: the plan preview awaiting confirmation, the executed trace,\n"
    "      or a rollback/abort notice.\n"
    "    Strategy: existing files are edited with patch_file; write_file is\n"
    "      reserved for new files or explicit rewrites (rewrite:true).\n"
    "      Registered tool modules must keep MANUAL/SPEC/get_manual/dispatch.\n"
    "      Any language is supported (python, typescript, javascript, java,\n"
    "      kotlin, c, cpp, go, rust, ruby, php, csharp, swift, ...): the impact\n"
    "      analysis detects the target language, maps existing unit tests by the\n"
    "      language's conventions and detects the project's test runner. Tests are\n"
    "      BEST EFFORT and never mandatory: when a changed module is covered by tests\n"
    "      a final soft step runs them (a failure keeps the changes and reports);\n"
    "      when a test framework is configured but nothing covers the change, the\n"
    "      plan adds a suggested test file and runs it softly; when no test setup\n"
    "      exists the result warns the user that the change could not be tested and\n"
    "      asks whether to keep the changes or apply a custom test.\n"
    "  - discover\n"
    "    Params:\n"
    "      request (str, required): what to find or change.\n"
    "      terms (list[str], optional): explicit search terms.\n"
    "      paths (list[str], optional): explicit target paths to seed as\n"
    "        candidates (skips keyword search).\n"
    "    Returns: candidate targets with evidence snippets. status 'poor'\n"
    "      means discovery failed and the flow should abort.\n"
    "  - assess\n"
    "    Params:\n"
    "      request (str, required): what to find or change.\n"
    "      terms (list[str], optional): explicit search terms.\n"
    "      paths (list[str], optional): explicit target paths to seed as\n"
    "        candidates (skips keyword search).\n"
    "    Returns: per-candidate impact analysis (change mode, recommended tool,\n"
    "      contract status, risks, mapped unit tests) without planning.\n"
    "  - plan\n"
    "    Params:\n"
    "      request (str, required): what to change.\n"
    "      terms (list[str], optional): explicit search terms.\n"
    "      paths (list[str], optional): explicit target paths to seed as\n"
    "        candidates (skips keyword search).\n"
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
    "relay the summary to the user and only proceed after approval. Patch "
    "anchors are validated against the file at plan time; if a patch_file "
    "still fails at runtime, the plan is automatically re-planned (up to "
    "exec_retries) and re-executed instead of stopping. Use undo to revert "
    "manually after a successful run."
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
        max_plan_steps: int = 16,
        exec_retries: int = 1,
        root: str = ".",
        discovery_fallback: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.registry = registry
        self.provider = provider
        self.memory = memory
        self.exec_retries = max(0, exec_retries)
        self.discovery = Discovery(registry, root=root)
        self.impact = ImpactAssessor(registry, root=root)
        self.planner = Planner(provider, registry, max_plan_steps=max_plan_steps)
        self.undo_log = UndoLog(backup_root=backup_root)
        self.executor = Executor(registry, self.undo_log, memory, impact=self.impact)
        self.discovery_fallback = discovery_fallback
        self.last_evidence: dict[str, Any] | None = None
        self.last_impact: dict[str, Any] | None = None
        self.last_plan: dict[str, Any] | None = None
        self.last_request: str | None = None
        self.last_paths: tuple[str, ...] | None = None

    def _obtain_evidence(
        self,
        request: str,
        terms: list[str] | None,
        paths: list[str] | None,
    ) -> dict[str, Any]:
        """Run the deterministic internal discovery. The discovery subagent is
        only used as a fallback when its evidence is weak AND the subagent is
        actually registered (balanced mode). In fast mode the subagent is not
        registered, so the orchestrator relies solely on its internal
        discovery and never references the missing subagent."""
        evidence = self.discovery.discover(request, terms, seed_paths=paths)
        if (
            evidence.get("status") != "ok"
            and self.discovery_fallback is not None
            and "discovery" in self.registry.list_tools()
        ):
            try:
                fallback = self.discovery_fallback(request, terms, paths)
            except Exception:
                fallback = None
            if fallback:
                return fallback
        return evidence

    def discover(
        self,
        request: str,
        terms: list[str] | None = None,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        evidence = self._obtain_evidence(request, terms, paths)
        self.last_evidence = evidence
        return evidence

    @staticmethod
    def _seeded_targets_look_wrong(request: str, evidence: dict[str, Any]) -> bool:
        """Detect when seeded ``paths`` point only at documentation while the
        request is clearly about a UI/style change. In that case the seeded
        doc is almost certainly NOT the implementation artifact (e.g. a spec
        markdown mistaken for 'the catalog'), and planning would rewrite docs
        instead of the real component/style files."""
        candidates = evidence.get("candidates", [])
        if not candidates or not _has_ui_signal(request):
            return False
        has_code = False
        all_docs = True
        for candidate in candidates:
            path = str(candidate.get("path", ""))
            suffix = Path(path).suffix.lower()
            if suffix in _DOC_EXTENSIONS:
                continue
            all_docs = False
            if suffix in _CODE_EXTENSIONS or candidate.get("type") == "new_file":
                has_code = True
        # Only flag when every seeded candidate is a doc and none is code.
        return all_docs and not has_code

    def plan(
        self,
        request: str,
        terms: list[str] | None = None,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        paths_key = tuple(paths) if paths else None
        if (
            self.last_evidence is None
            or self.last_evidence.get("request") != request
            or self.last_paths != paths_key
        ):
            self.last_evidence = self._obtain_evidence(request, terms, paths)
        self.last_paths = paths_key
        evidence = self.last_evidence

        if evidence.get("status") != "ok":
            return {
                "status": "aborted",
                "reason": evidence.get("reason"),
                "evidence": evidence,
            }

        if paths and self._seeded_targets_look_wrong(request, evidence):
            alt = self.discovery.discover(request)
            suggestion = ""
            if alt.get("status") == "ok":
                found = [
                    c.get("path")
                    for c in alt.get("candidates", [])
                    if Path(str(c.get("path", ""))).suffix.lower()
                    not in _DOC_EXTENSIONS
                ]
                if found:
                    suggestion = (
                        " Candidates found by keyword discovery (prefer one of "
                        f"these for a UI/style change): {', '.join(found[:5])}."
                    )
            return {
                "status": "clarify",
                "message": (
                    "The seeded target(s) are documentation files, but the "
                    "request looks like a UI/style change. Confirm the real "
                    "implementation file (e.g. a component/.css) before "
                    f"editing; do not rewrite the doc.{suggestion}"
                ),
                "evidence": evidence,
            }

        impact = self.impact.assess(request, evidence.get("candidates", []))
        self.last_impact = impact

        mismatches = [
            c.get("path")
            for c in evidence.get("candidates", [])
            if c.get("root_mismatch")
        ]
        if mismatches:
            impact.setdefault("warnings", []).append(
                "Seeded path(s) did not resolve under the workspace root; "
                "corrected to the existing file(s): " + ", ".join(mismatches)
            )

        feedback: str | None = None
        attempts = 0
        while True:
            attempts += 1
            plan = self.planner.plan(
                request, {**evidence, "impact": impact}, feedback=feedback
            )
            if plan.get("status") != "ok":
                return plan
            issues = self.impact.check_plan_steps(plan.get("steps", []))
            if not issues:
                break
            if attempts >= self.planner.max_plan_retries:
                return {
                    "status": "error",
                    "message": "\n".join(issues),
                    "plan": plan,
                }
            feedback = "; ".join(issues)
        self.last_plan = plan
        self.last_request = request
        return plan

    def assess(
        self,
        request: str,
        terms: list[str] | None = None,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        paths_key = tuple(paths) if paths else None
        if (
            self.last_evidence is None
            or self.last_evidence.get("request") != request
            or self.last_paths != paths_key
        ):
            self.last_evidence = self._obtain_evidence(request, terms, paths)
        self.last_paths = paths_key
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
        for warning in impact.get("warnings", []):
            lines.append(f"    warning: {warning}")
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
        rollback_on_failure: bool = True,
    ) -> dict[str, Any]:
        plan = steps or (self.last_plan or {}).get("steps")
        if not plan:
            return {
                "status": "error",
                "message": "no plan to execute; call plan first or pass steps.",
            }
        if not confirm:
            return self._awaiting_confirmation(plan)
        return self._execute(plan, rollback_on_failure=rollback_on_failure)

    def _execution_feedback(self, result: dict[str, Any]) -> str:
        """Build the planner rejection feedback from a failed execution."""
        failed = result.get("failed_step") or {}
        parts = [
            "EXECUTION FAILURE: a step failed at runtime and all changes were "
            "rolled back automatically. Re-plan the affected steps.",
            f"failed step: tool={failed.get('tool')} "
            f"action={failed.get('action')} "
            f"description={failed.get('description', '')}",
        ]
        trace = result.get("trace") or []
        for entry in trace:
            if entry.get("tool") == failed.get("tool") and entry.get(
                "action"
            ) == failed.get("action"):
                step_result = entry.get("result") or {}
                error = step_result.get("error")
                if isinstance(error, dict):
                    message = error.get("message")
                else:
                    message = step_result.get("message")
                if message:
                    parts.append(f"error: {message}")
                validated = entry.get("validated")
                if isinstance(validated, dict) and validated.get("check"):
                    parts.append(
                        f"validation: {validated.get('check')} ok={validated.get('ok')}"
                    )
                break
        parts.append(
            "If the patch anchors were rejected: add a read_file step for the "
            "exact target and copy the old/context lines VERBATIM from that "
            "read; never paraphrase or guess them."
        )
        return "\n".join(parts)

    def _replan_after_failure(self, feedback: str) -> dict[str, Any]:
        """Re-plan from the last request/evidence with execution feedback. Only
        valid when rollback restored the files (the evidence still describes
        the current state)."""
        if self.last_evidence is None or self.last_request is None:
            return {"status": "error", "message": "no request/evidence to re-plan"}
        plan = self.planner.plan(
            self.last_request,
            {**self.last_evidence, "impact": self.last_impact},
            feedback=feedback,
        )
        if plan.get("status") != "ok":
            return plan
        issues = self.impact.check_plan_steps(plan.get("steps", []))
        if issues:
            return {"status": "error", "message": "; ".join(issues), "plan": plan}
        self.last_plan = plan
        return plan

    def _execute(
        self,
        steps: list[dict],
        rollback_on_failure: bool = True,
    ) -> dict[str, Any]:
        """Run the executor, automatically re-planning up to ``exec_retries``
        times when a patch_file step fails at runtime (typically a hallucinated
        anchor; the rollback restores the files so re-planning is safe)."""
        attempts = 0
        while True:
            result = self.executor.execute(
                steps, rollback_on_failure=rollback_on_failure
            )
            if result.get("status") == "success" or attempts >= self.exec_retries:
                if result.get("status") == "success":
                    result["testing"] = self._testing_summary(result, steps)
                return result
            failed = result.get("failed_step")
            if not isinstance(failed, dict) or failed.get("tool") != "patch_file":
                return result
            attempts += 1
            replan = self._replan_after_failure(self._execution_feedback(result))
            if replan.get("status") != "ok":
                result["replan_error"] = replan.get("message")
                return result
            steps = replan.get("steps", [])

    def _testing_summary(
        self, result: dict[str, Any], steps: list[dict]
    ) -> dict[str, Any]:
        """Best-effort testing report: tests are never mandatory. When a plan
        ran soft test steps they either passed or are reported as failures that
        kept the changes; when the project has no test framework/runner, the
        change could not be tested and the user is asked whether to keep it or
        apply a custom test."""
        if any(step.get("soft") for step in steps):
            soft_failures = result.get("soft_failures") or []
            if soft_failures:
                return {
                    "tested": False,
                    "status": "tests_failed_after_changes",
                    "failures": soft_failures,
                    "message": (
                        "Tests failed after applying the changes; the changes "
                        "were kept because testing is best effort. Ask the user "
                        "whether to keep the changes or apply a custom test."
                    ),
                }
            return {"tested": True, "status": "tests_passed"}
        return {
            "tested": False,
            "status": "no_test_framework",
            "message": (
                "Could not test the change: no test framework or runner is "
                "available for it in this project. Ask the user whether to "
                "keep the changes or apply a custom test."
            ),
        }

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
        request: str | None = None,
        terms: list[str] | None = None,
        confirm: bool = False,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        # When resuming a pending plan, reuse the last request/paths if none
        # are supplied (the model often drops them on re-run).
        if request is None:
            if self.last_request is not None:
                request = self.last_request
                if paths is None and self.last_paths is not None:
                    paths = list(self.last_paths)
            else:
                return {
                    "status": "error",
                    "error": "invalid_arguments",
                    "message": (
                        "orchestrator run requires a 'request' parameter. "
                        "To resume a pending plan, call orchestrator execute "
                        "with confirm=true instead."
                    ),
                }
        paths_key = tuple(paths) if paths else None
        if (
            self.last_request != request
            or self.last_plan is None
            or self.last_paths != paths_key
        ):
            plan = self.plan(request, terms, paths=paths)
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
        return self._execute(plan.get("steps", []))

    def undo(self) -> dict[str, Any]:
        return self.undo_log.rollback()

    def abort(self) -> dict[str, Any]:
        discarded = self.undo_log.discard()
        self.last_evidence = None
        self.last_impact = None
        self.last_plan = None
        self.last_request = None
        self.last_paths = None
        return {"status": "aborted", "discarded_snapshots": discarded}
