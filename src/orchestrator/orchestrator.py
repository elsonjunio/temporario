from __future__ import annotations

import os
import sys
from typing import Any, Callable
from pathlib import Path

from src.memory import Memory
from src.orchestrator.discovery import Discovery
from src.orchestrator.executor import Executor
from src.orchestrator.impact import ImpactAssessor
from src.orchestrator.patchgen import PatchGenerator
from src.orchestrator.planner import Planner
from src.orchestrator.preassessment import Decomposer, PreAssessor
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


_CONFIRM_MODE_ENV = "ORCH_CONFIRM_MODE"
_APPROVAL_WORDS = {"s", "si", "sim", "y", "yes", "ok", "aprovar", "approve"}


def _confirm_mode() -> str:
    """``auto`` (default): ask the user directly when stdin is a terminal and
    fall back to the legacy ``awaiting_confirmation`` payload otherwise.
    ``agent``: always use the legacy payload (scripts/pipelines)."""
    return (os.getenv(_CONFIRM_MODE_ENV) or "auto").strip().lower() or "auto"


def _interactive_stdin() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (OSError, ValueError, AttributeError):
        return False


def _ask_user_approval(question: str = "Aprovar execução? [s/N]: ") -> bool:
    """Ask the human directly at the terminal. Anything unparsable/EOF counts
    as a rejection (fail safe)."""
    try:
        answer = input(question)
    except (EOFError, OSError, ValueError):
        return False
    return answer.strip().lower() in _APPROVAL_WORDS


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
    "      confirm (bool, default false): when false (default) the tool asks\n"
    "        the human directly at the terminal (input) and executes or\n"
    "        cancels in the SAME call — the main agent never relays the plan.\n"
    "        In non-interactive sessions (or ORCH_CONFIRM_MODE=agent) it\n"
    "        instead returns an 'awaiting_confirmation' preview for the agent\n"
    "        to relay; resume that with execute confirm=true. When called with\n"
    "        confirm=true (and no request) the pending plan is reused, not\n"
    "        re-planned; approval is assumed.\n"
    "    Returns: the executed trace, a cancelled notice, the legacy plan\n"
    "      preview awaiting confirmation (non-interactive only), or a\n"
    "      rollback/abort notice.\n"
    "    Pre-assessment: every run first scores the request deterministically\n"
    "      (enumerations, additive connectors, distinct targets/verbs). When it\n"
    "      holds several adjustment points (>= ORCH_SPLIT_THRESHOLD, default 3;\n"
    "      explicit terms/paths disable splitting), the request is AUTOMATICALLY\n"
    "      decomposed into ordered self-contained parts and each part is then\n"
    "      discovered, planned, approved individually and executed in sequence.\n"
    "      A part failure rolls back only that part; earlier approved parts stay\n"
    "      applied and are reported (undo reverts them). The result carries\n"
    "      parts_overview / completed_parts / remaining_parts so progress is\n"
    "      visible. In agent mode each execute confirm=true completes ONE part\n"
    "      and queues the next preview.\n"
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
    "      confirm (bool, default false): execution requires approval; with\n"
    "        confirm=false the tool asks the human directly at the terminal\n"
    "        and executes/cancels in the same call when stdin is a TTY\n"
    "        (non-interactive sessions return an 'awaiting_confirmation'\n"
    "        preview instead). To resume the pending plan after user approval\n"
    "        in that legacy flow, call execute with confirm=True and NO steps.\n"
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
    "before execution and rolled back automatically if a step fails. In an "
    "interactive terminal the tool asks the human for approval itself "
    "(input) before executing and never returns an awaiting_confirmation "
    "payload to the agent; set ORCH_CONFIRM_MODE=agent to restore the legacy "
    "relay flow. Patch payloads are generated by a dedicated patch generator "
    "that sees ONLY the change instruction plus the file content read from "
    "disk — plan steps carry the intent, not hand-written anchors. If a "
    "patch_file still fails at runtime, the plan is automatically re-planned "
    "(up to exec_retries) and re-executed instead of stopping. Use undo to "
    "revert manually after a successful run."
)


class Orchestrator:
    """Special tool that performs discovery, planning, step-by-step execution
    with validation, and rollback — all through the injected tool registry.

    The orchestrator keeps its OWN memory (a plain ``Memory`` without a
    compressor). Nothing it records ever reaches the agent's conversation
    memory, so a long run can never trigger context compression mid-flight,
    and every executed step is atomic: its params fully describe the change
    (patch anchors are generated fresh from the file on disk), never leaning
    on results from previous steps.
    """

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
        split_threshold: int | None = None,
        max_parts: int | None = None,
    ) -> None:
        self.registry = registry
        self.provider = provider
        # ``memory`` is accepted for signature compatibility but deliberately
        # IGNORED: the orchestrator never shares the agent's memory. It uses
        # an independent, compressor-free instance so discovery/planning/
        # execution can never be disturbed by (or trigger) context
        # compression, and per-step results stay out of the main dialogue.
        self.memory = Memory()
        self.exec_retries = max(0, exec_retries)
        self.discovery = Discovery(registry, root=root)
        self.impact = ImpactAssessor(registry, root=root)
        self.planner = Planner(provider, registry, max_plan_steps=max_plan_steps)
        self.patch_generator = PatchGenerator(provider)
        self.undo_log = UndoLog(backup_root=backup_root)
        self.executor = Executor(
            registry,
            self.undo_log,
            self.memory,
            impact=self.impact,
            patch_generator=self.patch_generator,
        )
        self.discovery_fallback = discovery_fallback
        # Pre-assessment / decomposition of compound requests.
        self.preassessor = PreAssessor(threshold=split_threshold)
        self.decomposer = Decomposer(
            provider, max_parts=max_parts, max_steps_per_part=max_plan_steps
        )
        self.last_evidence: dict[str, Any] | None = None
        self.last_impact: dict[str, Any] | None = None
        self.last_plan: dict[str, Any] | None = None
        self.last_request: str | None = None
        self.last_paths: tuple[str, ...] | None = None
        # Multi-part run state (only set when a request was decomposed).
        self.split_active: bool = False
        self.last_parts: list[dict[str, Any]] | None = None
        self.part_cursor: int = 0
        self.part_results: list[dict[str, Any]] = []
        self.last_assessment: dict[str, Any] | None = None
        # Index of the part whose plan is currently stored in ``last_plan``
        # (-1 when none); lets a resumed execute(confirm=true) run exactly
        # the plan the user saw, instead of silently re-planning.
        self.planned_part_index: int = -1
        # Set when the current part's queued plan FAILED at execution: the
        # next advance must re-plan (with the error as feedback) instead of
        # replaying a proven-broken plan.
        self.part_failed: bool = False
        self.last_part_feedback: str | None = None

    def _reset_memory(self) -> None:
        """Drop step/tool records from previous invocations so every public
        action starts from an independent context (the flow-state fields
        ``last_*`` are kept — they back the pending/resume contract)."""
        self.memory.clear()

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
        self._reset_memory()
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
        self._reset_memory()
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

    # ------------------------------------------------------------------
    # Pre-assessment and automatic decomposition into parts

    def _clear_parts(self) -> None:
        """Drop the multi-part run state (files already changed stay as-is;
        use ``undo`` to revert them)."""
        self.split_active = False
        self.last_parts = None
        self.part_cursor = 0
        self.part_results = []
        self.last_assessment = None
        self.planned_part_index = -1
        self.part_failed = False
        self.last_part_feedback = None

    def _begin_decomposition(
        self, request: str, auto: bool = False
    ) -> dict[str, Any] | None:
        """Pre-assess the request and, above the split threshold, decompose
        it into ordered parts. Returns ``None`` when no split applies so the
        caller proceeds with the monolithic flow unchanged."""
        assessment = self.preassessor.assess(request)
        if not assessment.get("needs_split"):
            return None
        decomposition = self.decomposer.decompose(request, assessment)
        parts = decomposition.get("parts") or []
        if decomposition.get("status") != "ok" or len(parts) < 2:
            return None
        self.split_active = True
        self.last_parts = parts
        self.part_cursor = 0
        self.part_results = []
        self.last_assessment = {
            **assessment,
            "strategy": decomposition.get("strategy"),
        }
        self.planned_part_index = -1
        self.last_request = request
        self.last_paths = None
        self.last_plan = None
        return self._advance_parts(auto=auto)

    def _plan_part(
        self, part: dict[str, Any], feedback: str | None = None
    ) -> dict[str, Any]:
        """Fresh discovery + impact + planner validation loop for ONE part,
        run lazily right before its approval so the evidence reflects the
        changes made by previously executed parts. Optional ``paths`` emitted
        by the decomposer seed the discovery (greenfield parts that create
        files whose names appear nowhere on disk would otherwise search
        poorly). ``feedback`` carries a previous execution failure of this
        same part so the planner avoids repeating it."""
        request = str(part.get("request", ""))
        paths = part.get("paths") or None
        evidence = self._obtain_evidence(request, None, paths)
        self.last_evidence = evidence
        if evidence.get("status") != "ok":
            return {
                "status": "aborted",
                "reason": evidence.get("reason"),
                "evidence": evidence,
            }
        impact = self.impact.assess(request, evidence.get("candidates", []))
        self.last_impact = impact
        planner_feedback = feedback
        attempts = 0
        while True:
            attempts += 1
            plan = self.planner.plan(
                request,
                {**evidence, "impact": impact},
                feedback=planner_feedback,
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
            planner_feedback = "; ".join(issues)
        return plan

    def _part_meta(self, index: int) -> dict[str, Any]:
        parts = self.last_parts or []
        part = parts[min(index, len(parts) - 1)]
        return {
            "index": min(index, len(parts) - 1) + 1,
            "total": len(parts),
            "id": part.get("id"),
            "request": part.get("request"),
            "rationale": part.get("rationale"),
        }

    def _parts_states(self) -> list[dict[str, Any]]:
        return [
            {
                "id": part.get("id"),
                "request": part.get("request"),
                "state": (
                    "completed"
                    if i < self.part_cursor
                    else "current" if i == self.part_cursor else "pending"
                ),
            }
            for i, part in enumerate(self.last_parts or [])
        ]

    def _execute_isolated(self, steps: list[dict]) -> dict[str, Any]:
        """Execute one part's steps with checkpoint isolation: on failure only
        THIS part's mutations roll back; snapshots from previously completed
        parts stay in the log for a later manual ``undo``."""
        mark = self.undo_log.checkpoint()
        return self._execute([dict(step) for step in steps], undo_mark=mark)

    def _stopped_result(self, reason: str, **extra: Any) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "decomposition_stopped",
            "reason": reason,
            "completed_parts": list(self.part_results),
            "remaining_parts": [
                {"id": part.get("id"), "request": part.get("request")}
                for part in (self.last_parts or [])[self.part_cursor :]
            ],
            "message": (
                "The decomposed run stopped. Previously approved parts "
                "remain applied on disk: call orchestrator undo to revert "
                "them, re-run the original request to retry the remaining "
                "part(s), or abort to discard the sequence."
            ),
        }
        result.update(extra)
        return result

    def _finalize_parts(self) -> dict[str, Any]:
        aggregate = {
            "status": "success",
            "decomposed": True,
            "parts_executed": list(self.part_results),
            "message": (
                "All parts were planned, approved and executed sequentially. "
                "Call orchestrator undo to revert the entire run."
            ),
        }
        self._clear_parts()
        self.last_evidence = None
        self.last_impact = None
        self.last_plan = None
        return aggregate

    def _advance_parts(
        self, *, confirm: bool = False, auto: bool = False
    ) -> dict[str, Any]:
        """Drive the decomposed sequence.

        Each part is planned lazily, approved individually and executed in
        isolation (a failure rolls back only that part; previously completed
        parts are kept and reported).

        Modes:
        - ``auto`` (run with confirm=true): execute every part sequentially
          without stopping at previews (scripts already opted in).
        - interactive terminal: ask the human directly at the prompt for
          EACH part inside this call.
        - legacy agent relay: queue ONE ``awaiting_confirmation`` preview
          per call; each ``execute(confirm=true)`` completes the current
          part and queues the following one (per-part approval contract).
        """
        parts = self.last_parts or []
        total = len(parts)
        interactive = self._should_ask_directly()
        while self.part_cursor < total:
            index = self.part_cursor
            part = parts[index]
            if self.planned_part_index == index and not self.part_failed:
                # Resume: run exactly the plan that was queued/approved for
                # this part; re-planning would betray the shown preview.
                steps = list((self.last_plan or {}).get("steps", []))
                plan = (
                    {"status": "ok"}
                    if steps
                    else {"status": "error", "message": "queued part plan is empty"}
                )
            else:
                feedback = (
                    self.last_part_feedback
                    if self.planned_part_index == index and self.part_failed
                    else None
                )
                plan = self._plan_part(part, feedback=feedback)
                steps = list(plan.get("steps", []))
            if plan.get("status") != "ok":
                detail = {
                    key: value
                    for key, value in plan.items()
                    if key not in ("evidence",)
                }
                return self._stopped_result(
                    reason=(
                        f"part {index + 1}/{total} could not be planned "
                        f"(status={plan.get('status')})"
                    ),
                    failed_part=self._part_meta(index),
                    detail=detail,
                )
            self.last_plan = {"status": "ok", "steps": steps}
            self.planned_part_index = index
            self.part_failed = False
            self.last_part_feedback = None

            if auto:
                execution = self._execute_isolated(steps)
            elif interactive:
                print(
                    f"\nParte {index + 1}/{total} ({part.get('id')}): "
                    f"{part.get('request')}"
                )
                print(self._render_plan_for_user(steps))
                if not _ask_user_approval("Aprovar esta parte? [s/N]: "):
                    return self._stopped_result(
                        reason=(
                            f"part {index + 1}/{total} rejected at the "
                            "approval prompt"
                        ),
                        failed_part=self._part_meta(index),
                    )
                execution = self._execute_isolated(steps)
            else:
                if not confirm:
                    payload = self._awaiting_confirmation(steps)
                    payload["part"] = self._part_meta(index)
                    payload["parts_total"] = total
                    payload["parts_overview"] = self._parts_states()
                    payload["message"] = (
                        f"The request was decomposed into {total} parts; "
                        f"this is part {index + 1} ({part.get('id')}). Ask "
                        "the user for approval of THIS part only. When "
                        "approved, resume with orchestrator execute "
                        "confirm=true (no steps) — the next part will be "
                        "queued afterwards. To reject, call abort."
                    )
                    return payload
                execution = self._execute_isolated(steps)

            summary = {
                "index": index + 1,
                "id": part.get("id"),
                "request": part.get("request"),
                "status": execution.get("status"),
                "steps": len(steps),
            }
            if execution.get("status") != "success":
                summary["rollback"] = execution.get("rollback")
                self.part_failed = True
                self.last_part_feedback = self._execution_feedback(execution)
                stopped = self._stopped_result(
                    reason=(
                        f"execution failed at part {index + 1}/{total}; the "
                        "part's own changes were rolled back and previously "
                        "completed parts were kept"
                    ),
                    failed_part=summary,
                )
                stopped["execution"] = execution
                return stopped
            summary["testing"] = execution.get("testing")
            self.part_results.append(summary)
            self.part_cursor += 1
            self.part_failed = False
            self.last_part_feedback = None
            # Every subsequent part needs its own approval round trip
            # (unless the whole sequence was launched with auto=true).
            confirm = False

        return self._finalize_parts()

    def assess(
        self,
        request: str,
        terms: list[str] | None = None,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        self._reset_memory()
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

    def _should_ask_directly(self) -> bool:
        """Interactive confirmation is used when the mode is ``auto`` (default)
        AND stdin is a terminal; scripts/pipelines (or ``ORCH_CONFIRM_MODE=
        agent``) keep the legacy awaiting_confirmation relay."""
        return _confirm_mode() != "agent" and _interactive_stdin()

    def _render_plan_for_user(
        self,
        steps: list[dict],
        impact: dict[str, Any] | None = None,
    ) -> str:
        lines = ["Plano pronto para execução:"]
        for i, step in enumerate(steps, 1):
            description = step.get("description") or (
                f"{step.get('tool')} {step.get('action')}".strip()
            )
            lines.append(f"{i}. {description}")
            params = step.get("params")
            if params:
                lines.append(f"   params: {params}")
        if impact is not None:
            lines.append("")
            lines.append(self._format_impact(impact))
        return "\n".join(lines)

    def _confirm_and_execute(
        self,
        steps: list[dict],
        impact: dict[str, Any] | None = None,
        rollback_on_failure: bool = True,
    ) -> dict[str, Any]:
        """Ask the human directly at the terminal and execute or cancel in the
        SAME call — the plan never round-trips through the main agent, whose
        memory/conversation stays untouched by the confirmation dialogue."""
        print(self._render_plan_for_user(steps, impact))
        if not _ask_user_approval():
            discarded = self.undo_log.discard()
            self.last_evidence = None
            self.last_impact = None
            self.last_plan = None
            self.last_request = None
            self.last_paths = None
            return {
                "status": "cancelled",
                "message": (
                    "Execution rejected at the confirmation prompt; nothing "
                    "was executed."
                ),
                "discarded_snapshots": discarded,
            }
        return self._execute(list(steps), rollback_on_failure=rollback_on_failure)

    def pending(self) -> dict[str, Any]:
        steps = (self.last_plan or {}).get("steps")
        if self.last_plan is None or not steps:
            return {"status": "no_pending_plan"}
        result = {
            "status": "pending_plan",
            "request": self.last_request,
            "steps": steps,
        }
        if self.split_active and self.last_parts:
            result["part"] = self._part_meta(self.part_cursor)
            result["parts_overview"] = self._parts_states()
        return result

    def execute(
        self,
        steps: list[dict] | None = None,
        confirm: bool = False,
        rollback_on_failure: bool = True,
    ) -> dict[str, Any]:
        if self.split_active and not steps:
            # Decomposed sequence in flight: confirm executes the CURRENT
            # part only; without confirm, re-queue its preview.
            if not confirm:
                return self._requeue_current_part()
            return self._advance_parts(confirm=True)
        plan = steps or (self.last_plan or {}).get("steps")
        if not plan:
            return {
                "status": "error",
                "message": "no plan to execute; call plan first or pass steps.",
            }
        if confirm:
            return self._execute(plan, rollback_on_failure=rollback_on_failure)
        if self._should_ask_directly():
            return self._confirm_and_execute(
                list(plan), rollback_on_failure=rollback_on_failure
            )
        return self._awaiting_confirmation(plan)

    def _requeue_current_part(self) -> dict[str, Any]:
        """Re-render the queued preview of the current part (no execution)."""
        parts = self.last_parts or []
        index = min(self.part_cursor, len(parts) - 1)
        steps = (self.last_plan or {}).get("steps") or []
        payload = self._awaiting_confirmation(list(steps))
        payload["part"] = self._part_meta(index)
        payload["parts_total"] = len(parts)
        payload["parts_overview"] = self._parts_states()
        return payload

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
            "If a patch_file step failed: emit it with an 'instruction' param "
            "describing the precise change (the runtime patch generator "
            "derives old/new from the CURRENT file content); never hand-write "
            "'old'/'new'/'diff' values, and never paraphrase anchors."
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
        undo_mark: int | None = None,
    ) -> dict[str, Any]:
        """Run the executor, automatically re-planning up to ``exec_retries``
        times when a patch_file step fails at runtime (typically a hallucinated
        anchor; the rollback restores the files so re-planning is safe).

        With ``undo_mark`` (multi-part runs) the executor never rolls back
        internally: every failed attempt is followed by a checkpointed
        rollback that reverts ONLY the mutations appended after the mark, so
        previously completed parts stay applied."""
        attempts = 0
        while True:
            result = self.executor.execute(
                steps,
                rollback_on_failure=(rollback_on_failure and undo_mark is None),
            )
            if result.get("status") != "success" and undo_mark is not None:
                # Revert only this part's mutations; earlier parts' snapshots
                # before the mark are preserved for a later manual undo.
                result["rollback"] = self.undo_log.rollback_to(undo_mark)
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
        self._reset_memory()
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

        # Decomposed sequence in flight: same request resumes it (the agent
        # often drops params on relay); a different request abandons the old
        # sequence (applied parts stay on disk) and assesses the new one.
        if self.split_active and self.last_parts:
            if request == self.last_request:
                return self._advance_parts(confirm=bool(confirm), auto=bool(confirm))
            self._clear_parts()

        if (
            self.last_request != request
            or self.last_plan is None
            or self.last_paths != paths_key
        ):
            # Pre-assessment gate: explicit terms/paths mean the caller
            # already knows the scope — never split those. Above the split
            # threshold the request is decomposed and planned part by part.
            if paths is None and terms is None:
                started = self._begin_decomposition(request, auto=bool(confirm))
                if started is not None:
                    return started
            plan = self.plan(request, terms, paths=paths)
        else:
            plan = self.last_plan
        if plan.get("status") == "error":
            return {"status": "error", "plan": plan}
        if plan.get("status") != "ok":
            return {"status": "aborted", "plan": plan}
        if confirm:
            return self._execute(plan.get("steps", []))
        if self._should_ask_directly():
            return self._confirm_and_execute(
                list(plan.get("steps", [])), impact=self.last_impact
            )
        return self._awaiting_confirmation(
            plan.get("steps", []), impact=self.last_impact
        )

    def undo(self) -> dict[str, Any]:
        return self.undo_log.rollback()

    def abort(self) -> dict[str, Any]:
        discarded = self.undo_log.discard()
        self._clear_parts()
        self.last_evidence = None
        self.last_impact = None
        self.last_plan = None
        self.last_request = None
        self.last_paths = None
        result = {"status": "aborted", "discarded_snapshots": discarded}
        if discarded:
            result["message"] = (
                "Snapshots were discarded; files already changed (e.g. by "
                "approved parts) stay as-is."
            )
        return result
