from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from src.context import estimate_tokens
from src.executor.budget import ExecutorBudget
from src.executor.interact import ExecutorInteractor
from src.executor.presenter import format_confirmation, format_question
from src.executor.result import TaskResult, build_run_result, now_iso
from src.executor.risk import (
    RiskLevel,
    classify_action,
    classify_task,
    outside_workspace,
)
from src.executor.state import TaskState, initial_state_from_plan_status
from src.executor.repair import (
    build_replanning_request,
    build_repair_prompt,
    classify_failure,
)
from src.executor.validate import ValidationManager
from src.memory import Memory
from src.planning.plan import as_list, normalize_task, validate_plan
from src.tools.registry import ToolRegistry
from src.utils import build_config_prompt, build_environment_info, parse_tool_call

EXECUTOR_INSTRUCTIONS = (
    "You are an Executor Agent. You execute ONE task from a plan at a time, "
    "using the filesystem and command tools. Your context contains ONLY the "
    "current task: its objective, target files, context, expected changes and "
    "acceptance criteria, plus anything discovery found and decisions the user "
    "made FOR THIS TASK. Work exclusively on that task; never touch other "
    "tasks' files or invent targets that are not listed.\n"
    "Available tools: read_file, list_dir, search_files, grep_files, "
    "write_file, patch_file, delete_file, move_file, run_command, and "
    "discovery (read-only).\n"
    "- Create or edit files with write_file/patch_file; use patch_file for "
    "targeted edits of existing files.\n"
    "- Before editing an existing file, read it and copy exact anchors; never "
    "guess content.\n"
    "- When you need shell commands (installs, scaffolding, build/test), use "
    "run_command with cwd set to the workspace root and a generous timeout.\n"
    "- Use discovery:run to resolve an UNCERTAINTY about THIS task with a "
    "SPECIFIC question (e.g. 'Quais arquivos consomem AuthenticationService?'). "
    "Never ask discovery to analyze the whole project; keep the question "
    "focused and limited to the current task's scope.\n"
    "- If a decision requires the user's preference (e.g. which provider to "
    "use), call executor ask with the question and optional options; do not "
    "choose arbitrarily.\n"
    "- If discovery reveals your task now requires changing files outside the "
    "declared task files, or adding new architectural dependencies, STOP and "
    "emit REPLAN_REQUIRED with the reason instead of silently continuing "
    "(start your plain-text reply with 'REPLAN_REQUIRED:' or call executor "
    "replan_required).\n"
    "- HIGH/CRITICAL actions may pause for explicit user confirmation; that is "
    "expected, answer truthfully when asked.\n"
    "- Meet the acceptance criteria; never claim a file changed unless a tool "
    "actually reported success for it.\n"
    "- After your final answer the task is VALIDATED automatically: acceptance "
    "criteria are checked (existence, colors, build/test/lint markers, and an "
    "LLM judge for anything unresolved). A validation failure does not end the "
    "task: you get REPAIR guidance and a chance to fix the work and answer "
    "again (up to EXECUTOR_MAX_RETRIES repair attempts). Fix the failing "
    "criteria rather than re-explaining; when the failure is structural (the "
    "declared target cannot be created), emit REPLAN_REQUIRED instead.\n"
    "Each tool call must be a single JSON block:\n"
    '{"tool": "<tool>", "action": "<action>", "params": {<arguments>}}\n'
    "When you believe the task is complete, STOP and reply in PLAIN TEXT (no "
    "JSON block) summarizing what you changed, which files, which commands, "
    "and any deviations or problems."
)

_START_GUIDANCE = (
    "Execute this task. Inspect what you need, make the required changes with "
    "the tools, and verify your own work. When the task is done, reply with a "
    "plain-text summary of what you changed (files, commands, deviations) - do "
    "NOT emit a JSON tool block in the final answer."
)

_CONTINUE_GUIDANCE = (
    "Tool result above. Continue working on the current task, or if it is "
    "complete give your final plain-text summary (no JSON block)."
)

_BUDGET_KEYS = {
    "max_tasks",
    "max_tool_calls",
    "max_retries",
    "max_context_tokens",
    "max_discovery_calls",
    "max_discovery_context_chars",
    "max_validation_commands",
    "validation_timeout",
    "max_browser_validation_steps",
}

_REPLAN_MARKER = re.compile(r"^\s*REPLAN_REQUIRED\b\s*[:：]?\s*(.*)$", re.MULTILINE)

_PATH_RE = re.compile(
    r"(?:[A-Za-z0-9_./-]+\.(?:py|js|ts|jsx|tsx|css|scss|html|json|md|go|rs|java|kt|c|cpp|rb|php|swift|sh|yml|yaml|sql|tf|vue))"
)

#: Model-side discovery actions the executor intercepts before generic dispatch.
_ASK_ACTIONS = {"ask", "ask_user", "question", "request_input"}
_REPLAN_ACTIONS = {"replan_required", "replan", "request_replan"}


@contextmanager
def _working_dir(path: str) -> Iterator[None]:
    """chdir into ``path`` so base tools resolve relative paths, restoring
    cwd afterwards."""
    original = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


@dataclass
class _TaskSession:
    """Resumable state of a single task's execution loop.

    Created at task start and preserved across NEED_* pauses so a task can
    stop for confirmation/input/discovery and later continue without losing
    its memory (isolated per task) or its position in the plan.
    """

    task: dict[str, Any]
    result: TaskResult
    memory: Memory
    current: str
    changed: set[str] = field(default_factory=set)
    started: float = field(default_factory=time.monotonic)
    tool_calls: int = 0
    discovery_calls: int = 0
    confirmed: bool = False
    scope_growth: set[str] = field(default_factory=set)
    pause: dict[str, Any] | None = None
    validation_attempts: int = 0
    repairs: int = 0
    tokens_used: int = 0
    hard_fail: str | None = None


class ExecutorAgent:
    """Consumes a plan produced by the Planning Agent and executes its atomic
    tasks one at a time through the injected tool registry.

    Stage 2 (interaction & control): before any workspace change the plan is
    presented for confirmation; tasks classified HIGH/CRITICAL and individual
    destructive actions pause for explicit confirmation; the model can pause a
    task to ask the user a question or run a FOCUSED discovery; a task whose
    scope grows is stopped as REPLAN_REQUIRED with a structured handoff for the
    planner. Pauses use the NEED_* states and resume without losing the task
    session (memory + position preserved).

    Each task still runs as an isolated, model-driven loop (fresh ``Memory``
    with only the task's context, its own tool results, discovery findings and
    user decisions - never the whole plan or previous tasks). Dependencies,
    preservation, validation and budget limits behave as in stage 1.

    Stage 3 (validation & recovery): every task passes an explicit VALIDATING
    stage driven by its acceptance criteria (heuristics, targeted commands,
    functional browser pass and an LLM judge). A validation failure triggers a
    bounded repair loop (REPAIRING -> EXECUTING -> VALIDATING, up to
    ``max_retries``) with the failure fed back as guidance; after that the
    failure is classified FAILED or REPLAN_REQUIRED (``failure_classifier``).
    """

    def __init__(
        self,
        provider: Any,
        registry: ToolRegistry,
        root: str = ".",
        *,
        budget: ExecutorBudget | None = None,
        environment: str | None = None,
        validator: Any = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        interactor: ExecutorInteractor | None = None,
        confirm_level: str | None = None,
        discovery_provider: Callable[[str], dict[str, Any]] | None = None,
        replanner: Callable[[dict[str, Any]], Any] | None = None,
        replan_extra_files_threshold: int = 1,
        failure_classifier: (
            Callable[[dict[str, Any], TaskResult, dict[str, Any]], str] | None
        ) = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.root = str(Path(root).resolve())
        self.budget = (budget or ExecutorBudget()).with_env()
        self.environment = environment or build_environment_info(workspace=self.root)
        self.validator = validator or ValidationManager(
            root=self.root, registry=registry, provider=provider
        )
        self.on_event = on_event
        self.interactor: ExecutorInteractor | None = interactor
        self.confirm_level = str(
            confirm_level or os.getenv("EXECUTOR_CONFIRM_LEVEL") or RiskLevel.HIGH
        ).upper()
        self.discovery_provider = discovery_provider
        self.replanner = replanner
        self.replan_extra_files_threshold = max(1, replan_extra_files_threshold)
        self.failure_classifier = failure_classifier or classify_failure
        self.last_run: dict[str, Any] | None = None
        self._pending_plan: dict[str, Any] | None = None
        self._active_run: dict[str, Any] | None = None
        self._active_budget: ExecutorBudget | None = None
        self._paused: dict[str, _TaskSession] = {}

    # -- events ---------------------------------------------------------------

    def _emit_event(self, event: str, **payload: Any) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event({"event": event, **payload})
        except Exception:
            return

    # -- plan preparation -----------------------------------------------------

    def _init_plan(
        self, plan: Any
    ) -> tuple[dict[str, Any] | None, list[str], list[str]]:
        """Normalize and validate a plan into the executor's working state.

        Tasks are normalized into the canonical schema (reusing the Planning
        Agent's contract) and structurally validated; the caller's dict is
        never mutated. An explicit ``risk`` on a raw task is carried over so
        the risk classifier can honor it. Returns ``(run, errors, warnings)``
        where ``run`` is None when the plan is structurally invalid.
        """
        errors: list[str] = []
        warnings: list[str] = []
        if not isinstance(plan, dict):
            return None, ["plan must be a dict"], warnings
        raw_tasks = plan.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            return None, ["plan must contain a non-empty 'tasks' list"], warnings

        tasks: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_tasks):
            try:
                task = normalize_task(raw, index + 1)
            except ValueError as exc:
                errors.append(f"task {index + 1}: {exc}")
                continue
            if raw.get("risk") is not None and task.get("risk") is None:
                task["risk"] = raw["risk"]
            tasks.append(task)
        if errors:
            return None, errors, warnings

        copy: dict[str, Any] = {
            "goal": str(plan.get("goal") or ""),
            "summary": str(plan.get("summary") or ""),
            "tasks": tasks,
        }
        issues, warns = validate_plan(copy)
        if issues:
            return None, issues, warnings
        warnings.extend(warns)

        statuses: dict[str, str] = {}
        reasons: dict[str, str] = {}
        for task in tasks:
            task_id = task["id"]
            statuses[task_id] = initial_state_from_plan_status(task["status"])
            reasons[task_id] = ""
        return (
            {
                "plan": copy,
                "tasks": tasks,
                "task_by_id": {task["id"]: task for task in tasks},
                "statuses": statuses,
                "reasons": reasons,
                "results": {},
                "task_order": [task["id"] for task in tasks],
                "_executed_count": 0,
            },
            errors,
            warnings,
        )

    # -- task selection -------------------------------------------------------

    def _select_next(self, run: dict[str, Any]) -> dict[str, Any] | None:
        """Pick the next executable task in plan order.

        Skips preserved states (COMPLETED/FAILED/CANCELLED/REPLAN_REQUIRED),
        records NEED_DISCOVERY tasks, and blocks tasks whose dependencies are
        not all COMPLETED. A plan-level BLOCKED task whose dependencies are
        satisfied is unblocked and returned.
        """
        for task in run["tasks"]:
            task_id = task["id"]
            state = run["statuses"][task_id]

            if state == TaskState.COMPLETED:
                continue
            if state in (
                TaskState.FAILED,
                TaskState.CANCELLED,
                TaskState.REPLAN_REQUIRED,
            ):
                continue
            if state == TaskState.NEED_DISCOVERY:
                if not run["reasons"][task_id]:
                    run["reasons"][task_id] = "requires discovery before execution"
                continue

            blockers = [
                dep
                for dep in (task.get("dependencies") or [])
                if not TaskState.satisfies_dependency(
                    run["statuses"].get(dep, TaskState.READY)
                )
            ]
            if blockers:
                if state == TaskState.READY:
                    run["statuses"][task_id] = TaskState.transition(
                        state, TaskState.BLOCKED
                    )
                if not run["reasons"][task_id]:
                    detail = "; ".join(
                        f"{dep} ({run['statuses'].get(dep, '?')})" for dep in blockers
                    )
                    run["reasons"][task_id] = f"blocked by dependency: {detail}"
                continue

            if state == TaskState.BLOCKED:
                run["statuses"][task_id] = TaskState.transition(state, TaskState.READY)
                run["reasons"][task_id] = ""
                return task
            if state == TaskState.READY:
                return task
        return None

    # -- task context (isolated) ----------------------------------------------

    def _task_context(self, task: dict[str, Any]) -> str:
        lines = [
            "CURRENT TASK",
            f"id: {task.get('id')}",
            f"title: {task.get('title')}",
            f"objective: {task.get('objective')}",
        ]
        dependencies = as_list(task.get("dependencies"))
        if dependencies:
            lines.append(f"dependencies: {', '.join(dependencies)}")
        files = as_list(task.get("files"))
        if files:
            lines.append(f"files: {', '.join(files)}")
        context = str(task.get("context") or "").strip()
        if context:
            lines.append(f"context:\n{context}")
        expected = as_list(task.get("expected_changes"))
        if expected:
            lines.append("expected_changes:")
            lines.extend(f"- {entry}" for entry in expected)
        criteria = as_list(task.get("acceptance_criteria"))
        if criteria:
            lines.append("acceptance_criteria:")
            lines.extend(f"- {entry}" for entry in criteria)
        evidence = str(task.get("evidence") or "").strip()
        if evidence:
            lines.append(f"evidence: {evidence}")
        for field in ("risks", "constraints"):
            value = as_list(task.get(field))
            if value:
                lines.append(f"{field}:")
                lines.extend(f"- {entry}" for entry in value)
        return "\n".join(lines)

    def _config(self, memory: Memory, budget: ExecutorBudget) -> str:
        extra = (
            f"\nBudget (stop before reaching these): {budget.to_dict()}\n"
            "You are executing real changes in the workspace."
        )
        return build_config_prompt(
            tool_manuals=self.registry.get_manual(),
            memory=memory,
            instructions=EXECUTOR_INSTRUCTIONS,
            max_history_entries=30,
            environment=self.environment + extra,
        )

    # -- risk helpers ---------------------------------------------------------

    def _task_risk_message(self, task: dict[str, Any], level: str) -> str:
        parts = [f"Esta tarefa é classificada como {level}."]
        objective = str(task.get("objective") or "").strip()
        if objective:
            parts.append(f"\n{objective}")
        expected = as_list(task.get("expected_changes"))
        if expected:
            parts.append("\nAlterações esperadas:")
            parts.extend(f"- {entry}" for entry in expected)
        risks = as_list(task.get("risks"))
        if risks:
            parts.append("\nRiscos declarados:")
            parts.extend(f"- {entry}" for entry in risks)
        return "\n".join(parts)

    @staticmethod
    def _action_risk_message(
        tool: str, action: str, params: dict[str, Any], level: str
    ) -> str:
        if tool == "delete_file":
            path = params.get("path") or params.get("file_path") or "?"
            return f"Esta operação irá REMOVER:\n- {path}\nA exclusão pode ser irreversível."
        if tool == "run_command":
            command = params.get("command") or params.get("script") or "?"
            return f"Este comando é classificado como {level}:\n- {command}"
        if tool in ("write_file", "patch_file"):
            path = params.get("file_path") or params.get("path") or "?"
            return f"Este arquivo será alterado:\n- {path}"
        return f"Operação de risco {level}: {tool}:{action}"

    @staticmethod
    def _action_affected(tool: str, params: dict[str, Any]) -> list[str]:
        for key in ("path", "file_path", "destination"):
            value = params.get(key)
            if value:
                return [str(value)]
        return []

    # -- single task execution (resumable) ------------------------------------

    def _new_session(self, task: dict[str, Any]) -> _TaskSession:
        context = self._task_context(task)
        memory = Memory()
        memory.add_user(context)
        result = TaskResult(task_id=task["id"])
        result.attempts = 1
        return _TaskSession(
            task=task,
            result=result,
            memory=memory,
            current=f"{context}\n\n{_START_GUIDANCE}",
        )

    def _start_task(
        self, run: dict[str, Any], task: dict[str, Any], budget: ExecutorBudget
    ) -> tuple[str, _TaskSession]:
        """Transition a task into EXECUTING, create its session and apply the
        task-level risk confirmation gate. Returns ``(control, session)`` with
        control in {"ok", "paused", "cancelled"}."""
        task_id = task["id"]
        state = run["statuses"][task_id]
        if state == TaskState.BLOCKED:
            run["statuses"][task_id] = TaskState.transition(state, TaskState.READY)
            state = TaskState.READY
        run["statuses"][task_id] = TaskState.transition(state, TaskState.EXECUTING)

        session = self._new_session(task)
        self._emit_event("task_started", task_id=task_id, title=task.get("title"))

        level = classify_task(task)
        if RiskLevel.at_least(level, self.confirm_level):
            message = self._task_risk_message(task, level)
            affected = [str(f) for f in as_list(task.get("files"))]
            if self.interactor is not None:
                approved = self.interactor.confirm_task(
                    task_id, level, message, affected
                )
                if approved:
                    self._record_confirmation(session, level, "task", task_id)
                else:
                    return "cancelled", session
            else:
                return "paused", self._pause(
                    run,
                    session,
                    "confirmation",
                    message,
                    level=level,
                    files=affected,
                )
        return "ok", session

    def _task_loop(
        self, run: dict[str, Any], session: _TaskSession, budget: ExecutorBudget
    ) -> str:
        """Run the model/tool loop until a final answer, a pause, a replan or
        the tool-call budget.

        A final answer enters VALIDATING; a validation failure triggers a
        bounded REPAIR loop (the guidance becomes the next prompt and the loop
        continues) until the task is COMPLETED, FAILED, REPLAN_REQUIRED or
        cancelled. Returns the control value for the caller."""
        with _working_dir(self.root):
            while session.tool_calls < budget.max_tool_calls:
                control = self._step(run, session, budget)
                if control == "continue":
                    continue
                if control in ("paused", "replan", "cancelled"):
                    return control
                if control == "final":
                    control = self._finalize_task(run, session, budget)
                    if control == "repair":
                        continue
                    return control
            session.result.reason = (
                f"max tool calls reached ({budget.max_tool_calls}) without "
                "a final answer"
            )
            return self._finalize_task(
                run, session, budget, hard_fail=session.result.reason
            )
        return "done"

    def _step(
        self, run: dict[str, Any], session: _TaskSession, budget: ExecutorBudget
    ) -> str:
        session.tool_calls += 1
        config = self._config(session.memory, budget)
        config_tokens = estimate_tokens(config)
        if budget.max_context_tokens and config_tokens > budget.max_context_tokens:
            reason = (
                f"context exceeded {budget.max_context_tokens} tokens "
                f"({config_tokens} estimated)"
            )
            session.hard_fail = reason
            return "final"

        response = self.provider.infer(session.current, config)
        session.tokens_used += config_tokens + estimate_tokens(response)
        call = parse_tool_call(response)
        if call is None:
            marker = _REPLAN_MARKER.search(response)
            if marker:
                reason = (marker.group(1) or "").strip()
                return self._replan_task(
                    run, session, reason or "scope grew beyond the task"
                )
            session.result.output = response.strip()
            return "final"

        return self._dispatch_step(run, session, call, response, budget)

    def _dispatch_step(
        self,
        run: dict[str, Any],
        session: _TaskSession,
        call: dict[str, Any],
        response: str,
        budget: ExecutorBudget,
    ) -> str:
        tool = call["tool"]
        action = call["action"]
        params = dict(call.get("params") or {})

        if tool == "executor":
            if action in _ASK_ACTIONS:
                return self._handle_ask(run, session, params)
            if action in _REPLAN_ACTIONS:
                reason = str(params.get("reason") or params.get("problem") or "")
                return self._replan_task(
                    run, session, reason or "scope grew beyond the task"
                )
            session.current = (
                f"The executor cannot call itself; tool 'executor' action "
                f"'{action}' was ignored.\n\n{_CONTINUE_GUIDANCE}"
            )
            return "continue"

        if tool == "discovery" and action in ("run", "discover", "explore", ""):
            return self._handle_discovery(run, session, budget, params)

        level = classify_action(tool, action, params)
        if not session.confirmed and RiskLevel.at_least(level, self.confirm_level):
            message = self._action_risk_message(tool, action, params, level)
            affected = self._action_affected(tool, params)
            if self.interactor is not None:
                approved = self.interactor.confirm_task(
                    session.task["id"], level, message, affected
                )
                if approved:
                    self._record_confirmation(
                        session, level, "action", f"{tool}:{action}"
                    )
                else:
                    return self._cancel_task(
                        run, session, "user declined critical action confirmation"
                    )
            else:
                self._pause(
                    run,
                    session,
                    "confirmation",
                    message,
                    level=level,
                    files=affected,
                    pending={"tool": tool, "action": action, "params": params},
                )
                return "paused"

        self._dispatch(run, session, tool, action, params)
        return "continue"

    def _dispatch(
        self,
        run: dict[str, Any],
        session: _TaskSession,
        tool: str,
        action: str,
        params: dict[str, Any],
        call: dict[str, Any] | None = None,
    ) -> None:
        call = call or {"tool": tool, "action": action, "params": params}
        call_result = self.registry.dispatch(tool, action, **params)
        self._record_call(
            session.result, call, call_result, session.memory, f"tool {tool}:{action}"
        )
        self._track_changed(session.changed, tool, params, call_result)
        session.current = f"Tool result:\n{call_result}\n\n{_CONTINUE_GUIDANCE}"

    # -- interaction ----------------------------------------------------------

    def _record_confirmation(
        self, session: _TaskSession, level: str, scope: str, target: str
    ) -> None:
        session.confirmed = True
        session.result.confirmations.append(
            {"level": level, "scope": scope, "target": target, "approved": True}
        )

    def _record_user_input(
        self,
        session: _TaskSession,
        question: str,
        answer: str,
        options: list[str],
    ) -> None:
        session.result.user_inputs.append(
            {"question": question, "answer": answer, "options": options}
        )

    def _handle_ask(
        self, run: dict[str, Any], session: _TaskSession, params: dict[str, Any]
    ) -> str:
        question = str(params.get("question") or "").strip()
        if not question:
            session.current = (
                "executor ask requires a 'question' param.\n\n" + _CONTINUE_GUIDANCE
            )
            return "continue"
        options = [str(option) for option in (params.get("options") or [])]
        if self.interactor is not None:
            answer = self.interactor.ask(session.task["id"], question, options or None)
            self._record_user_input(session, question, answer, options)
            session.memory.add_user(f"User answered: {answer}")
            session.current = f"User decision:\n{answer}\n\n{_CONTINUE_GUIDANCE}"
            return "continue"
        self._pause(run, session, "question", question, options=options)
        return "paused"

    # -- discovery ------------------------------------------------------------

    def _run_discovery(self, request: str) -> dict[str, Any]:
        if self.discovery_provider is not None:
            try:
                return self.discovery_provider(request)
            except Exception as exc:  # noqa: BLE001
                return {"status": "error", "error": {"message": str(exc)}}
        if "discovery" not in self.registry.list_tools():
            return {
                "status": "error",
                "error": {
                    "message": "discovery tool is not available in this registry"
                },
            }
        try:
            return self.registry.dispatch("discovery", "run", request=request)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": {"message": str(exc)}}

    def _compact_discovery(self, result: dict[str, Any], cap: int) -> str:
        status = result.get("status")
        if status == "error":
            error = result.get("error")
            detail = error.get("message") if isinstance(error, dict) else str(error)
            return f"status: error\n{detail}"
        report = result.get("report")
        if isinstance(report, dict):
            parts = []
            for key in (
                "summary",
                "relevant_files",
                "relationships",
                "hypotheses",
                "evidence",
                "risks",
                "confidence",
            ):
                if key not in report:
                    continue
                value = report[key]
                rendered = (
                    "\n".join(f"- {item}" for item in value)
                    if isinstance(value, list)
                    else str(value)
                )
                parts.append(f"{key}:\n{rendered}")
            body = "\n\n".join(parts) if parts else str(report)
        else:
            body = str(report or "")
        prefix = f"status: {status or 'unknown'}\n"
        body = prefix + body
        if len(body) > cap:
            body = body[:cap] + f"\n[... truncated at {cap} chars]"
        return body

    @staticmethod
    def _extract_paths(result: dict[str, Any]) -> list[str]:
        texts: list[str] = []
        for key in ("report", "task"):
            value = result.get(key)
            if value:
                texts.append(str(value))
        found: list[str] = []
        for text in texts:
            for match in _PATH_RE.findall(text):
                path = match.strip(" ./")
                if path not in found:
                    found.append(path)
        return found

    def _handle_discovery(
        self,
        run: dict[str, Any],
        session: _TaskSession,
        budget: ExecutorBudget,
        params: dict[str, Any],
    ) -> str:
        request = str(
            params.get("request")
            or params.get("query")
            or params.get("prompt")
            or params.get("task")
            or ""
        ).strip()
        if not request:
            session.current = (
                "discovery run requires a focused 'request'.\n\n" + _CONTINUE_GUIDANCE
            )
            return "continue"
        if session.discovery_calls >= budget.max_discovery_calls:
            session.current = (
                f"Discovery budget exhausted ({budget.max_discovery_calls} "
                "per task); continue with what you know.\n\n" + _CONTINUE_GUIDANCE
            )
            return "continue"

        result = self._run_discovery(request)
        session.discovery_calls += 1
        finding = self._compact_discovery(result, budget.max_discovery_context_chars)
        found = self._extract_paths(result)
        declared = {str(f) for f in as_list(session.task.get("files"))}
        extra = [p for p in found if p not in declared]
        if extra:
            session.scope_growth.update(extra)
        session.result.discovery_calls.append(
            {
                "question": request,
                "status": result.get("status"),
                "files_found": found,
                "context_chars": len(finding),
            }
        )
        self._emit_event(
            "discovery",
            task_id=session.task["id"],
            question=request,
            status=result.get("status"),
            extra_files=extra,
        )
        session.memory.add_tool(
            "discovery",
            "run",
            {"request": request},
            {"finding": finding, "status": result.get("status")},
        )
        if extra and len(extra) >= self.replan_extra_files_threshold:
            session.current = (
                f"DISCOVERY FINDING:\n{finding}\n\n"
                "Note: discovery mentions files outside this task's declared "
                f"scope: {extra}. If your task now requires modifying them or "
                "adding new architectural dependencies, STOP and emit "
                "REPLAN_REQUIRED with the reason. Otherwise continue.\n\n"
                + _CONTINUE_GUIDANCE
            )
        else:
            session.current = f"DISCOVERY FINDING:\n{finding}\n\n{_CONTINUE_GUIDANCE}"
        return "continue"

    # -- pause / resume -------------------------------------------------------

    def _pause(
        self,
        run: dict[str, Any],
        session: _TaskSession,
        kind: str,
        message: str,
        *,
        level: str | None = None,
        files: list[str] | None = None,
        options: list[str] | None = None,
        pending: dict[str, Any] | None = None,
    ) -> _TaskSession:
        task_id = session.task["id"]
        target = {
            "confirmation": TaskState.NEED_CONFIRMATION,
            "question": TaskState.NEED_USER_INPUT,
            "discovery": TaskState.NEED_DISCOVERY,
        }[kind]
        run["statuses"][task_id] = TaskState.transition(
            run["statuses"][task_id], target
        )
        run["reasons"][task_id] = message
        pause: dict[str, Any] = {
            "kind": kind,
            "task_id": task_id,
            "message": message,
            "level": level,
            "files": files or [],
            "options": options or [],
            "pending": pending,
        }
        session.pause = pause
        self._paused[task_id] = session
        self._active_run = run
        self._emit_event("task_paused", task_id=task_id, kind=kind, message=message)
        return session

    def _paused_result(self) -> dict[str, Any]:
        task_id = next(iter(self._paused))
        session = self._paused[task_id]
        pause = session.pause or {}
        if pause.get("kind") == "confirmation":
            body = format_confirmation(
                task_id,
                pause.get("level") or RiskLevel.HIGH,
                pause.get("message") or "",
                pause.get("files"),
            )
        else:
            body = format_question(
                task_id,
                pause.get("message") or "",
                pause.get("options") or None,
            )
        return {
            "status": "awaiting_confirmation",
            "pause": pause,
            "task_id": task_id,
            "kind": pause.get("kind", "confirmation"),
            "summary": body,
            "message": (
                "A task is paused and needs your input. Approve to continue "
                "(execute with confirm=true or confirm_task), answer the "
                "question (answer), or cancel (abort)."
            ),
        }

    # -- task finalization ----------------------------------------------------

    def _finalize_task(
        self,
        run: dict[str, Any],
        session: _TaskSession,
        budget: ExecutorBudget,
        hard_fail: str | None = None,
    ) -> str:
        """Explicit VALIDATING stage + COMPLETED/repair/failure decision.

        Transitions the task into VALIDATING, runs the validator, then:
        COMPLETED on PASS; REPAIRING with the failure fed back when retries
        remain (``max_retries``); otherwise FAILED or REPLAN_REQUIRED via the
        ``failure_classifier``. A ``hard_fail`` (tool/context budget) always
        ends FAILED, still running validation so the evidence is recorded.
        Returns "done", "repair" or "replan".
        """
        task_id = session.task["id"]
        session.result.changed_files = sorted(session.changed)

        state = run["statuses"][task_id]
        if state != TaskState.VALIDATING:
            run["statuses"][task_id] = TaskState.transition(state, TaskState.VALIDATING)
        session.result.status = TaskState.VALIDATING
        if hard_fail:
            session.hard_fail = hard_fail
        self._emit_event("task_validating", task_id=task_id)

        validation = self._validate_task(run, session, budget)

        if session.hard_fail:
            return self._finish_failed(run, session, reason=session.hard_fail)
        if validation["ok"]:
            return self._finish_completed(run, session)
        if session.result.attempts <= budget.max_retries:
            self._start_repair(run, session, validation, budget)
            return "repair"
        decision = self.failure_classifier(session.task, session.result, validation)
        if decision == TaskState.REPLAN_REQUIRED:
            problem = str(
                validation.get("summary")
                or "acceptance validation failed after retries"
            )
            return self._replan_task(run, session, problem)
        return self._finish_failed(run, session)

    def _validate_task(
        self, run: dict[str, Any], session: _TaskSession, budget: ExecutorBudget
    ) -> dict[str, Any]:
        task_id = session.task["id"]
        session.result.validation_attempts += 1
        session.validation_attempts += 1
        validation = self.validator.validate(
            session.task, session.result, budget=budget
        )
        session.result.validation_result = validation
        session.result.validation_history.append(validation)
        session.tokens_used += int(validation.get("tokens_used") or 0)
        self._emit_event(
            "task_validated",
            task_id=task_id,
            status=validation.get("status"),
            attempt=session.result.validation_attempts,
        )
        return validation

    def _start_repair(
        self,
        run: dict[str, Any],
        session: _TaskSession,
        validation: dict[str, Any],
        budget: ExecutorBudget,
    ) -> None:
        task_id = session.task["id"]
        run["statuses"][task_id] = TaskState.transition(
            TaskState.VALIDATING, TaskState.FAILED
        )
        run["statuses"][task_id] = TaskState.transition(
            TaskState.FAILED, TaskState.REPAIRING
        )
        session.result.status = TaskState.REPAIRING
        session.result.attempts += 1
        session.result.repairs += 1
        session.repairs += 1
        guidance = build_repair_prompt(
            session.task,
            validation,
            session.result.repairs,
            budget.max_retries,
        )
        session.current = f"{guidance}\n\n{_CONTINUE_GUIDANCE}"
        self._emit_event(
            "task_repairing",
            task_id=task_id,
            attempt=session.result.repairs,
            max_retries=budget.max_retries,
        )

    def _finish_completed(self, run: dict[str, Any], session: _TaskSession) -> str:
        task_id = session.task["id"]
        session.result.status = TaskState.COMPLETED
        session.result.reason = ""
        session.result.tokens_used = session.tokens_used
        session.result.finished_at = now_iso()
        session.result.duration = round(time.monotonic() - session.started, 3)
        run["statuses"][task_id] = TaskState.transition(
            TaskState.VALIDATING, TaskState.COMPLETED
        )
        run["results"][task_id] = session.result
        run["reasons"][task_id] = ""
        self._emit_event(
            "task_finished",
            task_id=task_id,
            status=TaskState.COMPLETED,
            reason="",
        )
        return "done"

    def _finish_failed(
        self,
        run: dict[str, Any],
        session: _TaskSession,
        reason: str | None = None,
    ) -> str:
        task_id = session.task["id"]
        if reason is None:
            validation = session.result.validation_result or {}
            reason = str(validation.get("summary") or "acceptance validation failed")
        session.result.status = TaskState.FAILED
        session.result.reason = reason
        session.result.tokens_used = session.tokens_used
        session.result.finished_at = now_iso()
        session.result.duration = round(time.monotonic() - session.started, 3)
        run["statuses"][task_id] = TaskState.transition(
            TaskState.VALIDATING, TaskState.FAILED
        )
        run["results"][task_id] = session.result
        run["reasons"][task_id] = reason
        self._emit_event(
            "task_finished",
            task_id=task_id,
            status=TaskState.FAILED,
            reason=reason,
        )
        return "done"

    def _cancel_task(
        self, run: dict[str, Any], session: _TaskSession, reason: str
    ) -> str:
        task_id = session.task["id"]
        run["statuses"][task_id] = TaskState.transition(
            run["statuses"][task_id], TaskState.CANCELLED
        )
        session.result.status = TaskState.CANCELLED
        session.result.reason = reason
        session.result.tokens_used = session.tokens_used
        session.result.finished_at = now_iso()
        session.result.duration = round(time.monotonic() - session.started, 3)
        run["results"][task_id] = session.result
        run["reasons"][task_id] = reason
        self._emit_event(
            "task_finished",
            task_id=task_id,
            status=TaskState.CANCELLED,
            reason=reason,
        )
        return "cancelled"

    def _replan_task(
        self, run: dict[str, Any], session: _TaskSession, reason: str
    ) -> str:
        task_id = session.task["id"]
        run["statuses"][task_id] = TaskState.transition(
            run["statuses"][task_id], TaskState.BLOCKED
        )
        run["statuses"][task_id] = TaskState.transition(
            TaskState.BLOCKED, TaskState.REPLAN_REQUIRED
        )
        extra_evidence = [f"scope growth: {p}" for p in sorted(session.scope_growth)]
        replan = build_replanning_request(
            session.task,
            session.result,
            session.result.validation_result or {},
            reason,
            extra_evidence=extra_evidence,
            extra_files=sorted(session.scope_growth),
            for_planner=(
                f"Task {task_id} ({session.task.get('title') or ''}) was stopped "
                f"during execution: {reason}. Files already touched: "
                f"{sorted(session.changed) or 'none'}. Scope growth reported: "
                f"{sorted(session.scope_growth) or 'none'}. Planner should "
                "revise the plan for this task and its dependents."
            ),
        )
        session.result.replan = replan
        session.result.status = TaskState.REPLAN_REQUIRED
        session.result.reason = f"REPLAN_REQUIRED: {reason}"
        session.result.tokens_used = session.tokens_used
        session.result.finished_at = now_iso()
        session.result.duration = round(time.monotonic() - session.started, 3)
        run["results"][task_id] = session.result
        run["reasons"][task_id] = session.result.reason
        self._emit_event(
            "task_finished",
            task_id=task_id,
            status=TaskState.REPLAN_REQUIRED,
            reason=reason,
        )
        if self.replanner is not None:
            try:
                self.replanner(replan)
            except Exception:  # noqa: BLE001
                pass
        return "replan"

    # -- tool call recording --------------------------------------------------

    def _relativize(self, path: str) -> str:
        """Prefer paths relative to the workspace root in results."""
        try:
            resolved = str(Path(path).expanduser().resolve())
        except OSError:
            return path
        if resolved.startswith(self.root):
            return resolved[len(self.root) :].lstrip("/")
        return path

    def _track_changed(
        self, changed: set[str], tool: str, params: dict, call_result: Any
    ) -> None:
        if not isinstance(call_result, dict) or call_result.get("status") != "success":
            return
        path: Any = None
        if tool in ("write_file", "patch_file"):
            path = call_result.get("path") or params.get("file_path")
        elif tool == "move_file":
            path = (
                call_result.get("destination")
                or call_result.get("path")
                or params.get("destination")
            )
        elif tool == "delete_file":
            path = call_result.get("path") or params.get("path")
        if path:
            changed.add(self._relativize(str(path)))

    @staticmethod
    def _error_message(call_result: dict[str, Any]) -> str:
        error = call_result.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
        message = call_result.get("message")
        if message:
            return str(message)
        return f"tool status {call_result.get('status', 'unknown')}"

    def _record_call(
        self,
        result: TaskResult,
        call: dict[str, Any],
        call_result: Any,
        memory: Memory,
        response: str,
    ) -> None:
        params = dict(call.get("params") or {})
        result.record_tool_call(call["tool"], call["action"], params, call_result)
        memory.add_tool(call["tool"], call["action"], params, call_result)
        memory.add_assistant(response)
        self._emit_event(
            "tool_call",
            task_id=result.task_id,
            tool=call["tool"],
            action=call["action"],
        )
        if isinstance(call_result, dict):
            status = call_result.get("status")
            if status == "awaiting_confirmation":
                result.warnings.append(
                    "a tool requested confirmation that was not relayed"
                )
            elif status in ("failed", "error", "timeout") or "error" in call_result:
                result.record_error(
                    call["tool"], call["action"], self._error_message(call_result)
                )

    # -- plan preview ---------------------------------------------------------

    def _plan_preview(self, run: dict[str, Any]) -> dict[str, Any]:
        summary = []
        for index, task in enumerate(run["tasks"], 1):
            summary.append(
                {
                    "step": index,
                    "task_id": task["id"],
                    "title": task.get("title"),
                    "description": (f"{task.get('title')} — {task.get('objective')}"),
                    "risk": classify_task(task),
                    "files": as_list(task.get("files")),
                    "dependencies": as_list(task.get("dependencies")),
                }
            )
        counts: dict[str, int] = {}
        for item in summary:
            counts[item["risk"]] = counts.get(item["risk"], 0) + 1
        files = sorted({f for item in summary for f in item["files"]})
        impact_lines = [
            "Riscos relevantes: " + ", ".join(f"{k}={v}" for k, v in counts.items()),
            "Arquivos potencialmente afetados:",
        ]
        impact_lines.extend(f"- {f}" for f in files or ["(nenhum)"])
        return {
            "status": "awaiting_confirmation",
            "message": (
                "Plan ready to execute. Ask the user for approval first. On "
                "approval resume with executor execute confirm=true."
            ),
            "summary": summary,
            "impact": "\n".join(impact_lines),
            "details": run["plan"],
            "task_count": len(run["tasks"]),
        }

    # -- run orchestration ----------------------------------------------------

    def _run_loop(
        self,
        run: dict[str, Any],
        budget: ExecutorBudget,
        start: float,
        errors: list[str],
        warnings: list[str],
        resume: tuple[str, _TaskSession] | None = None,
        emit_start: bool = True,
    ) -> dict[str, Any]:
        if emit_start:
            self._emit_event(
                "run_started",
                goal=run["plan"].get("goal"),
                task_count=len(run["tasks"]),
                budget=budget.to_dict(),
            )

        self._active_budget = budget
        executed = int(run.get("_executed_count", 0))
        stop_reason = ""
        resumed_tid, resumed_session = resume if resume else (None, None)
        while executed < budget.max_tasks:
            task = None
            if resumed_tid is not None:
                task = run["task_by_id"].get(resumed_tid)
                resumed_tid = None
            if task is None:
                task = self._select_next(run)
            if task is None:
                break

            if resumed_session is not None:
                control = self._task_loop(run, resumed_session, budget)
                resumed_session = None
            else:
                control, session = self._start_task(run, task, budget)
                if control == "paused":
                    return self._paused_result()
                if control == "cancelled":
                    self._cancel_task(
                        run, session, "user declined critical task confirmation"
                    )
                    executed += 1
                    run["_executed_count"] = executed
                    continue
                control = self._task_loop(run, session, budget)
            if control == "paused":
                return self._paused_result()
            executed += 1
            run["_executed_count"] = executed
        else:
            stop_reason = f"max_tasks reached ({budget.max_tasks})"
            for task in run["tasks"]:
                task_id = task["id"]
                if run["statuses"][task_id] == TaskState.READY:
                    run["reasons"][task_id] = f"not executed: {stop_reason}"

        if not stop_reason and executed == 0:
            stop_reason = "no executable task found"

        self._fill_results(run)

        results = list(run["results"].values())
        tools_total = sum(len(res.tool_calls) for res in results)
        first_attempt = lambda res: res.status == TaskState.COMPLETED and (
            res.validation_attempts <= 1
        )
        after_validation = (
            lambda res: res.status == TaskState.COMPLETED and res.repairs > 0
        )
        caught_by_validation = (
            lambda res: res.status
            in (
                TaskState.FAILED,
                TaskState.REPLAN_REQUIRED,
            )
            and res.validation_attempts > 0
        )
        duration_per_task = {
            res.task_id: res.duration for res in results if res.duration > 0
        }
        tool_calls_per_task = {res.task_id: len(res.tool_calls) for res in results}
        validations_per_task = {res.task_id: res.validation_attempts for res in results}
        metrics = {
            "tasks_total": len(run["tasks"]),
            "tasks_executed": executed,
            "tool_calls": tools_total,
            "elapsed": round(time.monotonic() - start, 3),
            "tasks_completed_first_attempt": sum(
                1 for res in results if first_attempt(res)
            ),
            "tasks_completed_after_validation": sum(
                1 for res in results if after_validation(res)
            ),
            "failures_caught_by_validation": sum(
                1 for res in results if caught_by_validation(res)
            ),
            "retries_total": sum(res.repairs for res in results),
            "validation_attempts_total": sum(
                res.validation_attempts for res in results
            ),
            "discovery_calls": sum(len(res.discovery_calls) for res in results),
            "executor_discovery_total": sum(
                len(res.discovery_calls) for res in results
            ),
            "user_questions": sum(len(res.user_inputs) for res in results),
            "confirmations": sum(len(res.confirmations) for res in results),
            "replanning_requests": sum(1 for res in results if res.replan),
            "tokens_used": sum(res.tokens_used for res in results),
            "duration_per_task": duration_per_task,
            "tool_calls_per_task": tool_calls_per_task,
            "validations_per_task": validations_per_task,
        }
        result = build_run_result(
            plan=run["plan"],
            task_order=run["task_order"],
            final_states=run["statuses"],
            reasons=run["reasons"],
            results=run["results"],
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            reason=stop_reason,
        )
        self.last_run = result
        self._active_run = None
        self._active_budget = None
        self._paused = {}
        self._emit_event("run_finished", status=result["status"], metrics=metrics)
        return result

    def _fill_results(self, run: dict[str, Any]) -> None:
        """Ensure every task has a TaskResult in ``results``.

        Executed tasks already have theirs; blocked, preserved, skipped and
        needs-discovery tasks get lightweight placeholders (no tool calls) so
        the structured result is uniform for presentation layers.
        """
        defaults = {
            TaskState.COMPLETED: "preserved from plan (already completed)",
            TaskState.FAILED: "preserved from plan (already failed)",
            TaskState.CANCELLED: "preserved from plan (already cancelled)",
            TaskState.BLOCKED: "blocked",
            TaskState.NEED_DISCOVERY: "requires discovery before execution",
            TaskState.REPLAN_REQUIRED: "requires replanning",
            TaskState.READY: "not executed",
        }
        for task in run["tasks"]:
            task_id = task["id"]
            if task_id in run["results"]:
                continue
            state = run["statuses"][task_id]
            reason = run["reasons"][task_id] or defaults.get(state, "")
            run["results"][task_id] = TaskResult(
                task_id=task_id,
                status=state,
                reason=reason,
                finished_at=now_iso(),
            )

    # -- public API -----------------------------------------------------------

    @staticmethod
    def _int_overrides(overrides: dict[str, Any]) -> dict[str, int]:
        cleaned: dict[str, int] = {}
        for key, value in overrides.items():
            if key not in _BUDGET_KEYS or value is None:
                continue
            try:
                cleaned[key] = int(value)
            except (TypeError, ValueError):
                continue
        return cleaned

    def _error_result(
        self, errors: list[str], start: float, plan: Any = None
    ) -> dict[str, Any]:
        result = {
            "status": "error",
            "goal": (
                str((plan or {}).get("goal") or "") if isinstance(plan, dict) else ""
            ),
            "summary": "",
            "tasks": [],
            "results": {},
            "completed": [],
            "failed": [],
            "blocked": [],
            "needs_discovery": [],
            "replans": [],
            "cancelled": [],
            "skipped": [],
            "plan_status": "error",
            "completed_tasks": [],
            "failed_tasks": [],
            "blocked_tasks": [],
            "cancelled_tasks": [],
            "changed_files": [],
            "validations": {},
            "user_decisions": [],
            "discovery_calls": [],
            "replanning_requests": [],
            "metrics": {
                "tasks_total": 0,
                "tasks_executed": 0,
                "tool_calls": 0,
                "elapsed": round(time.monotonic() - start, 3),
            },
            "errors": errors,
            "warnings": [],
            "reason": "; ".join(errors),
        }
        self.last_run = result
        return result

    def execute(
        self, plan: Any = None, *, confirm: bool = False, **overrides: Any
    ) -> dict[str, Any]:
        budget = self.budget.apply_overrides(**self._int_overrides(overrides))
        confirm_level = str(
            overrides.get("confirm_level") or self.confirm_level
        ).upper()
        self.confirm_level = confirm_level
        start = time.monotonic()

        # Resume a paused run: execute(confirm=True) approves a pending
        # confirmation pause and continues (agent-loop relay path).
        if self._active_run is not None and self._paused:
            if not confirm:
                return self._paused_result()
            task_id = next(iter(self._paused))
            kind = (self._paused[task_id].pause or {}).get("kind")
            if kind != "confirmation":
                return self._error_result(
                    [
                        f"task {task_id} is awaiting user input, not "
                        "confirmation; use executor answer(task_id, ...) "
                        "instead of execute(confirm=true)"
                    ],
                    start,
                    plan=self._active_run["plan"],
                )
            return self._resume_confirmation(task_id, approve=True, budget=budget)

        if plan is None:
            plan = self._pending_plan
            if plan is None:
                return self._error_result(
                    [
                        "no plan to execute; call execute(plan) first or "
                        "approve a pending preview"
                    ],
                    start,
                )

        run, errors, warnings = self._init_plan(plan)
        if run is None:
            return self._error_result(errors, start, plan=plan)

        if not confirm:
            preview = self._plan_preview(run)
            if self.interactor is not None:
                approved = self.interactor.confirm_plan(preview)
                if not approved:
                    return {
                        "status": "cancelled",
                        "message": "plan cancelled by user; nothing was executed.",
                        "goal": str(plan.get("goal") or ""),
                        "summary": "",
                        "tasks": [],
                        "results": {},
                        "completed": [],
                        "failed": [],
                        "blocked": [],
                        "needs_discovery": [],
                        "replans": [],
                        "cancelled": [],
                        "skipped": [],
                        "errors": errors,
                        "warnings": warnings,
                    }
            else:
                self._pending_plan = plan
                return preview

        self._pending_plan = None
        self._active_run = run
        self._active_budget = budget
        return self._run_loop(run, budget, start, errors, warnings)

    def _resume_confirmation(
        self, task_id: str, approve: bool, budget: ExecutorBudget
    ) -> dict[str, Any]:
        run = self._active_run
        assert run is not None
        session = self._paused.pop(task_id)
        pause = session.pause or {}
        self._emit_event("task_resumed", task_id=task_id, kind=pause.get("kind"))

        if pause.get("kind") != "confirmation" or not approve:
            return self._cancel_task_result(
                run, session, "user declined critical action confirmation", budget
            )

        run["statuses"][task_id] = TaskState.transition(
            run["statuses"][task_id], TaskState.EXECUTING
        )
        pending = pause.get("pending")
        if pending:
            self._record_confirmation(
                session,
                pause.get("level") or RiskLevel.HIGH,
                "action",
                f"{pending['tool']}:{pending['action']}",
            )
            with _working_dir(self.root):
                self._dispatch(
                    run,
                    session,
                    pending["tool"],
                    pending["action"],
                    pending["params"],
                )
        else:
            self._record_confirmation(
                session,
                pause.get("level") or RiskLevel.HIGH,
                "task",
                task_id,
            )
        session.pause = None
        return self._run_loop(
            run,
            budget,
            time.monotonic(),
            [],
            [],
            resume=(task_id, session),
            emit_start=False,
        )

    def _cancel_task_result(
        self,
        run: dict[str, Any],
        session: _TaskSession,
        reason: str,
        budget: ExecutorBudget,
    ) -> dict[str, Any]:
        self._cancel_task(run, session, reason)
        return self._run_loop(
            run,
            budget,
            time.monotonic(),
            [],
            [],
            emit_start=False,
        )

    def confirm_task(
        self, task_id: str, confirm: bool = True, **overrides: Any
    ) -> dict[str, Any]:
        if self._active_run is None or task_id not in self._paused:
            return {
                "status": "error",
                "error": f"no paused task awaiting confirmation: {task_id}",
            }
        session = self._paused[task_id]
        if (session.pause or {}).get("kind") != "confirmation":
            return {
                "status": "error",
                "error": f"task {task_id} is not awaiting confirmation",
            }
        budget = self._active_budget or self.budget.apply_overrides(
            **self._int_overrides(overrides)
        )
        if not confirm:
            self._paused.pop(task_id)
            return self._cancel_task_result(
                self._active_run,
                session,
                "user declined critical action confirmation",
                budget,
            )
        return self._resume_confirmation(task_id, True, budget)

    def answer(self, task_id: str, answer: str, **overrides: Any) -> dict[str, Any]:
        if self._active_run is None or task_id not in self._paused:
            return {
                "status": "error",
                "error": f"no paused task awaiting input: {task_id}",
            }
        session = self._paused[task_id]
        pause = session.pause or {}
        if pause.get("kind") != "question":
            return {
                "status": "error",
                "error": f"task {task_id} is not awaiting user input",
            }
        self._paused.pop(task_id)
        run = self._active_run
        run["statuses"][task_id] = TaskState.transition(
            run["statuses"][task_id], TaskState.EXECUTING
        )
        self._record_user_input(
            session, pause.get("message") or "", str(answer), pause.get("options") or []
        )
        session.memory.add_user(f"User answered: {answer}")
        session.current = f"User decision:\n{answer}\n\n{_CONTINUE_GUIDANCE}"
        session.pause = None
        self._emit_event("task_resumed", task_id=task_id, kind="question")
        budget = self._active_budget or self.budget.apply_overrides(
            **self._int_overrides(overrides)
        )
        return self._run_loop(
            run,
            budget,
            time.monotonic(),
            [],
            [],
            resume=(task_id, session),
            emit_start=False,
        )

    def status(self) -> dict[str, Any]:
        """Return the last run's structured result, a paused run, or no_run."""
        if self._paused:
            return {"status": "paused", "pause": self._paused_result().get("pause")}
        if self.last_run is not None:
            return {"status": "last_run", "result": self.last_run}
        return {"status": "no_run"}

    def pending(self) -> dict[str, Any]:
        """Return what is awaiting action: a paused task, a pending plan
        preview, the last run, or nothing."""
        if self._paused:
            task_id = next(iter(self._paused))
            session = self._paused[task_id]
            return {
                "status": "paused",
                "task_id": task_id,
                "kind": (session.pause or {}).get("kind"),
                "pause": session.pause,
            }
        if self._pending_plan is not None:
            run, _, _ = self._init_plan(self._pending_plan)
            if run is not None:
                preview = self._plan_preview(run)
                return {
                    "status": "pending_plan",
                    "preview": preview,
                    "details": self._pending_plan,
                }
        if self.last_run is not None:
            return {"status": "last_run", "result": self.last_run}
        return {"status": "no_pending"}

    def abort(self) -> dict[str, Any]:
        """Cancel the pending plan, or abort a paused run.

        Nothing is executed when only a preview is pending. When a run is
        paused, the paused task and every still-pending task are marked
        CANCELLED and the run result is finalized.
        """
        if self._active_run is not None and self._paused:
            run = self._active_run
            budget = self._active_budget or self.budget
            for task_id in list(self._paused):
                self._cancel_task(run, self._paused.pop(task_id), "aborted by user")
            for task in run["tasks"]:
                task_id = task["id"]
                if run["statuses"][task_id] == TaskState.READY:
                    run["statuses"][task_id] = TaskState.transition(
                        TaskState.READY, TaskState.CANCELLED
                    )
                    run["reasons"][task_id] = "aborted by user"
            self._fill_results(run)
            result = build_run_result(
                plan=run["plan"],
                task_order=run["task_order"],
                final_states=run["statuses"],
                reasons=run["reasons"],
                results=run["results"],
                metrics={
                    "tasks_total": len(run["tasks"]),
                    "tasks_executed": 0,
                    "tool_calls": 0,
                    "elapsed": 0.0,
                },
                errors=[],
                warnings=[],
                reason="aborted by user",
            )
            self.last_run = result
            self._active_run = None
            self._active_budget = None
            return {
                "status": "aborted",
                "message": "paused run aborted",
                "result": result,
            }
        self._pending_plan = None
        return {
            "status": "aborted",
            "message": "nothing pending; no changes were made.",
        }


__all__ = ["EXECUTOR_INSTRUCTIONS", "ExecutorAgent"]
