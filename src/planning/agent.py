from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.context import estimate_tokens
from src.memory import Memory
from src.planning.budget import PlanningBudget
from src.planning.plan import (
    as_list,
    normalize_task,
    plan_all_ready,
    task_readiness_issues,
    validate_plan,
)
from src.tools.registry import ToolRegistry
from src.utils import (
    build_config_prompt,
    build_environment_info,
    extract_json_object,
    parse_tool_call,
)

PLANNING_INSTRUCTIONS = (
    "You are a Planning Agent. You transform a user request into a set of "
    "atomic, executable tasks for a separate Executor Agent. You NEVER modify "
    "files and NEVER run commands: everything you can call is read-only or a "
    "planning/discovery meta-tool.\n"
    "Workflow:\n"
    "1. Understand the request.\n"
    "2. Determine what changes are needed and how they depend on each other.\n"
    "3. When you lack information (where code lives, who uses a symbol, which "
    "files are affected), run ONE FOCUSED discovery call: a specific question "
    "with a scope when known. Never ask discovery to 'analyze the whole "
    "project'. You may run several focused discoveries, one question at a "
    "time, as the plan needs them.\n"
    "4. Build atomic tasks. Each task is small: a single objective, known "
    "files, specific context, expected changes and verifiable acceptance "
    "criteria. Never mix independent changes in one task.\n"
    "5. Set dependencies between tasks (by task id). Tasks with no "
    "dependencies may later run in parallel. Only create a dependency when it "
    "reflects a real need (an output, an information flow or a shared file "
    "another task will change); never add artificial dependencies between "
    "independent tasks.\n"
    "6. Emit the final plan.\n"
    "A task is READY_FOR_EXECUTION only when ALL of these are known:\n"
    "- objective: what to change\n"
    "- files: concrete target paths (e.g. 'src/styles.css'), not 'the "
    "frontend' or 'the styles'\n"
    "- context: what exists today and what is expected, limited to what THIS "
    "task's executor needs (never the whole discovery log or conversation)\n"
    "- expected_changes: what will change in those files\n"
    "- acceptance_criteria: objective, verifiable checks (e.g. 'X uses the "
    "value defined in DESIGN.md', 'no occurrences of #D4AF37 remain outside "
    "the global tokens', 'existing auth tests still pass')\n"
    "- evidence: the discovery facts/hypotheses that justify this task\n"
    "If any of these is missing, the task is NOT ready: run discovery to get "
    "the missing information, or leave it with status DISCOVERY_REQUIRED. "
    "Never invent paths, symbols or values to fake readiness. Distinguish "
    "FACT, HYPOTHESIS and UNKNOWN in evidence; never present a hypothesis as "
    "a fact.\n"
    "Statuses:\n"
    "- DISCOVERY_REQUIRED: needs investigation before execution.\n"
    "- READY_FOR_EXECUTION: fully specified; an executor can start without "
    "re-running discovery.\n"
    "- BLOCKED: fully specified but waits on another task.\n"
    "- COMPLETED: already executed (used when replanning).\n"
    "Final output: when the plan is complete, emit ONLY the final plan as a "
    "JSON object (no markdown fences) with this structure:\n"
    '{"goal": "<objetivo>", "summary": "<resumo>", "tasks": ['
    "{task objects with id, title, objective, status, dependencies, files, "
    "context, expected_changes, acceptance_criteria, evidence}]}\n"
    "Prefer a focused discovery call over guessing. A small plan of three "
    "well-contextualized tasks beats twenty vague ones. Stop as soon as every "
    "task is READY_FOR_EXECUTION (or is marked DISCOVERY_REQUIRED with a "
    "justification)."
)

_PLANNING_TOOLS_MANUAL = (
    "Planning tools (you decide when each is needed):\n"
    "  - discovery\n"
    "    Focused, read-only investigation. Ask ONE specific question.\n"
    '    {"tool": "discovery", "action": "run", "params": {"question": '
    '"Onde é implementado o fluxo de autenticação?", "scope": "src"}}\n'
    "    Params: question (str, required; synonyms request/query/task); scope "
    "(str, optional path); optional budget: max_iterations, max_tool_calls, "
    "max_files_read, max_lines_read.\n"
    "    Returns a DISCOVERY RESULT report with evidence. Use it to build "
    "tasks; do not re-ask the same question.\n"
    "  - planner\n"
    "    Manages the plan under construction.\n"
    "    add_task: submit one task (params = the task object). Example:\n"
    '    {"tool": "planner", "action": "add_task", "params": {"title": "...", '
    '"objective": "...", "status": "READY_FOR_EXECUTION", "dependencies": [], '
    '"files": ["src/styles.css"], "context": "...", "expected_changes": [...], '
    '"acceptance_criteria": [...], "evidence": "..."}}\n'
    "    finalize: submit the whole plan at once (params = the plan object "
    '{"goal": ..., "summary": ..., "tasks": [...]}).\n'
    "  - analysis (read-only, cheap verification only)\n"
    "    list_dir, search_files, grep_files, read_file, and code "
    "(find_symbol/find_definition/find_references/find_imports/"
    "find_importers/inspect). Never modify anything."
)

_BUDGET_KEYS = {
    "max_iterations",
    "max_discovery_calls",
    "max_tasks",
    "max_context_tokens",
    "max_context_chars",
}

_DISCOVERY_BUDGET_KEYS = {
    "max_iterations",
    "max_tool_calls",
    "max_context_tokens",
    "max_files_read",
    "max_lines_read",
}

_COMPLETED_STATUSES = {
    "success",
    "passed",
    "completed",
    "done",
    "ok",
    "approved",
    "aprovado",
    "successo",
}

#: Discovery questions that never yield usable evidence: they ask for a whole
#: project overview instead of a specific fact about a concrete artifact.
_GENERIC_DISCOVERY_PATTERNS = (
    "analise o projeto",
    "analise o código",
    "analise o codigo",
    "analyze the project",
    "analyze the code",
    "entenda o código",
    "entenda o codigo",
    "entender o código",
    "entender o codigo",
    "understand the code",
    "o que precisa ser feito",
    "o que devo fazer",
    "what needs to be done",
    "what should i do",
    "resumo geral",
    "visão geral",
    "visao geral",
    "general overview",
    "overview do projeto",
    "contexto geral",
    "general context",
    "todo o projeto",
    "o projeto inteiro",
    "whole project",
    "entire project",
    "explore o projeto",
    "explore o codebase",
    "explore the project",
    "explore the codebase",
    "get familiar",
    "o que você sabe",
    "what do you know",
)
_GENERIC_DISCOVERY_RE = re.compile(
    "|".join(re.escape(pattern) for pattern in _GENERIC_DISCOVERY_PATTERNS),
    re.IGNORECASE,
)

#: Generic one/two-word commands that carry no concrete artifact to investigate.
_GENERIC_DISCOVERY_VERBS = frozenset(
    {
        "descubra",
        "investigue",
        "procure",
        "explore",
        "analise",
        "explain",
        "find",
        "investigate",
        "inspecione",
    }
)


def _is_generic_discovery_question(question: str) -> bool:
    """True when the question asks for an overview instead of a specific fact."""
    text = question.strip()
    if not text:
        return False
    if _GENERIC_DISCOVERY_RE.search(text):
        return True
    words = text.lower().split()
    if len(words) <= 2 and words and words[0] in _GENERIC_DISCOVERY_VERBS:
        return True
    return False


@contextmanager
def _working_dir(path: str) -> Iterator[None]:
    """chdir into ``path`` so the read-only analysis tools resolve relative
    paths, restoring cwd afterwards."""
    original = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


class PlanningAgent:
    """Ephemeral subagent that turns a user request into a plan of atomic
    tasks, calling a focused Discovery investigation on demand.

    Modeled after ``DiscoveryAgent``: it owns a private read-only registry and
    its own ``Memory``, runs a config-prompt -> infer -> parse -> dispatch
    loop, and stops when the model emits a valid final plan or a budget limit
    is reached. It never registers mutating tools, so it cannot modify the
    workspace.
    """

    def __init__(
        self,
        provider: Any,
        root: str = ".",
        *,
        budget: PlanningBudget | None = None,
        discovery: Any = None,
        outer_registry: ToolRegistry | None = None,
        environment: str | None = None,
        max_finalize_retries: int = 2,
    ) -> None:
        self.provider = provider
        self.root = str(Path(root).resolve())
        self.budget = (budget or PlanningBudget()).with_env()
        self.environment = environment or build_environment_info(workspace=self.root)
        self.max_finalize_retries = max(0, max_finalize_retries)
        self.readonly_registry = build_discovery_registry(self.root)
        self._discovery_source = self._resolve_discovery(discovery, outer_registry)
        self.memory = Memory()
        self.tasks: list[dict[str, Any]] = []
        self.last_plan: dict[str, Any] | None = None
        self.last_request: str | None = None
        self._request = ""
        self._mode = "plan"
        self._replan_context: str | None = None

    # -- discovery wiring ----------------------------------------------------

    def _resolve_discovery(
        self, discovery: Any, outer_registry: ToolRegistry | None
    ) -> Any:
        if discovery is not None:
            return discovery
        if outer_registry is not None:
            spec = outer_registry.get("discovery")
            if spec is not None:
                return spec
        from src.discovery import create_discovery_tool

        return create_discovery_tool(self.provider, root=self.root)

    def _discovery_dispatch(
        self,
        question: str,
        scope: str | None,
        budget_params: dict[str, int],
    ) -> dict[str, Any]:
        spec = self._discovery_source
        if callable(spec):
            return spec(question=question, scope=scope, **budget_params)
        params: dict[str, Any] = {"request": question}
        if scope:
            params["path"] = scope
        params.update(budget_params)
        result = spec.dispatch("run", **params)
        return result if isinstance(result, dict) else {"result": result}

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _empty_metrics() -> dict[str, Any]:
        return {
            "planning_iterations": 0,
            "discovery_calls": 0,
            "discovery_revisits": 0,
            "tasks_created": 0,
            "tasks_ready": 0,
            "tasks_blocked": 0,
            "tasks_discovery_required": 0,
            "tasks_completed": 0,
            "estimated_input_tokens": 0,
            "estimated_output_tokens": 0,
            "tokens_used": 0,
            "final_plan_tokens": 0,
            "planning_time": 0.0,
        }

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

    def _storage(self, result: Any) -> Any:
        text = str(result)
        if len(text) <= self.budget.max_context_chars:
            return result
        return {
            "truncated_for_history": True,
            "size": len(text),
            "preview": text[: self.budget.max_context_chars],
        }

    def _tasks_summary(self, tasks: list[dict[str, Any]]) -> str:
        if not tasks:
            return "(none)"
        lines = []
        for task in tasks:
            lines.append(
                f"- {task.get('id')} [{task.get('status')}] {task.get('title')}"
            )
            dependencies = task.get("dependencies") or []
            if dependencies:
                lines.append(f"    depends on: {', '.join(dependencies)}")
        return "\n".join(lines)

    def _config(self, request: str, budget: PlanningBudget) -> str:
        extra = (
            f"\nBudget (stop before reaching these): {budget.to_dict()}\n"
            "You are READ-ONLY: never modify files or run commands."
        )
        environment = self.environment + extra
        if self._replan_context:
            environment += "\n\nREPLAN CONTEXT\n" + self._replan_context
        if self.tasks:
            environment += "\n\nTASKS SO FAR\n" + self._tasks_summary(self.tasks)
        return build_config_prompt(
            tool_manuals=_PLANNING_TOOLS_MANUAL,
            memory=self.memory,
            instructions=PLANNING_INSTRUCTIONS,
            max_history_entries=40,
            environment=environment,
        )

    # -- response interpretation ---------------------------------------------

    @staticmethod
    def _interpret(response: str) -> tuple[str, Any]:
        """Classify a model response as one of:
        ``plan`` (final plan JSON), ``call`` (tool call), ``fragment`` (single
        implicit task), ``plan_bad`` (incomplete plan) or ``text``.
        """
        data = extract_json_object(response)
        if isinstance(data, dict):
            inner: Any = data.get("plan")
            if not isinstance(inner, dict):
                inner = data
            if isinstance(inner.get("tasks"), (list, dict)):
                return "plan", inner
        call = parse_tool_call(response)
        if call is not None:
            return "call", call
        if isinstance(data, dict):
            if any(key in data for key in ("tasks", "goal", "summary")):
                return "plan_bad", data
            return "fragment", data
        return "text", response

    # -- internal actions ----------------------------------------------------

    def _run_discovery(
        self,
        question: str,
        scope: str | None,
        budget_params: dict[str, int],
        budget: PlanningBudget,
        metrics: dict[str, Any],
        asked: set[str],
    ) -> dict[str, Any]:
        if _is_generic_discovery_question(question):
            return {
                "status": "error",
                "message": (
                    "discovery question is too generic; ask ONE specific "
                    "question about a concrete artifact (e.g. 'Onde está "
                    "implementado o fluxo de autenticação?' or 'Quais arquivos "
                    "importam AuthenticationService?'), optionally with a scope "
                    "path."
                ),
                "error": {"type": "invalid_arguments", "recoverable": True},
                "question": question,
            }
        if metrics["discovery_calls"] >= budget.max_discovery_calls:
            return {
                "status": "error",
                "message": (
                    f"discovery budget exhausted ({budget.max_discovery_calls} "
                    "calls); plan with the evidence you have and mark remaining "
                    "tasks DISCOVERY_REQUIRED."
                ),
            }
        if question in asked:
            return {
                "status": "error",
                "message": (
                    "you already asked that discovery question; reuse its "
                    "evidence instead of re-asking."
                ),
                "question": question,
            }
        asked.add(question)
        metrics["discovery_calls"] += 1
        metrics["discovery_revisits"] = max(0, metrics["discovery_calls"] - 1)
        result = self._discovery_dispatch(question, scope, budget_params)
        ok = result.get("status") in ("success", "partial")
        report = str(result.get("report") or "")
        if report and len(report) > budget.max_context_chars:
            result = {**result, "report": report[: budget.max_context_chars] + "..."}
        return {
            "status": "discovery_done" if ok else "discovery_error",
            "question": question,
            "scope": scope or "",
            "result": result,
        }

    def _add_task(
        self, raw: Any, tasks: list[dict[str, Any]], budget: PlanningBudget
    ) -> dict[str, Any]:
        if len(tasks) >= budget.max_tasks:
            return {
                "status": "error",
                "message": (
                    f"max tasks reached ({budget.max_tasks}); finalize the "
                    "plan or revise existing tasks."
                ),
            }
        try:
            task = normalize_task(raw, len(tasks) + 1)
        except ValueError as exc:
            return {
                "status": "task_rejected",
                "message": str(exc),
                "error": {"type": "invalid_task", "recoverable": True},
            }
        replaced = False
        for index, existing in enumerate(tasks):
            if existing["id"] == task["id"]:
                tasks[index] = task
                replaced = True
                break
        if not replaced:
            tasks.append(task)
        missing = (
            task_readiness_issues(task)
            if task["status"] in ("READY_FOR_EXECUTION", "BLOCKED")
            else []
        )
        if missing:
            task["status"] = "DISCOVERY_REQUIRED"
        return {
            "status": "task_added",
            "task_id": task["id"],
            "task": task,
            "missing": missing,
            "replaced": replaced,
        }

    def _finalize(self, payload: Any) -> tuple[dict[str, Any], bool, str]:
        """Normalize + validate a final plan object. Returns (plan, accepted,
        status). Readiness gaps downgrade tasks to DISCOVERY_REQUIRED; only
        structural problems reject the plan."""
        plan = self._normalize_plan(payload)
        raw_tasks = plan.get("tasks") or []
        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_tasks):
            try:
                normalized.append(normalize_task(raw, index + 1))
            except ValueError as exc:
                plan["tasks"] = []
                plan["issues"] = [str(exc)]
                plan["warnings"] = []
                plan["valid"] = False
                return plan, False, "partial"
        if self._mode == "replan":
            self._assign_replan_ids(normalized, raw_tasks, self.tasks)
            plan["tasks"] = self._merge_tasks(self.tasks, normalized)
        else:
            plan["tasks"] = normalized if normalized else list(self.tasks)
        if not str(plan.get("goal") or "").strip():
            plan["goal"] = self._request
        issues, warnings = validate_plan(plan)
        plan["issues"] = issues
        plan["warnings"] = warnings
        plan["valid"] = not issues
        status = (
            "success" if plan["valid"] and plan_all_ready(plan["tasks"]) else "partial"
        )
        return plan, not issues, status

    @staticmethod
    def _normalize_plan(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        tasks = payload.get("tasks")
        if isinstance(tasks, dict):
            tasks = [dict(task) for task in tasks.values()]
        elif not isinstance(tasks, list):
            tasks = []
        return {
            "goal": str(payload.get("goal") or "").strip(),
            "summary": str(payload.get("summary") or "").strip(),
            "tasks": tasks,
        }

    @staticmethod
    def _assign_replan_ids(
        normalized: list[dict[str, Any]],
        raw_tasks: list[Any],
        existing: list[dict[str, Any]],
    ) -> None:
        """Resolve emitted task ids during replan so they do not clobber the
        seeded tasks. Emitted tasks that carry an explicit id keep it (they
        update the matching seeded task, if any). Positional ids (renumbered
        by list position) collide with seeded ids, so a positional task is
        either matched to the single non-completed seeded task with the same
        title (an update) or assigned the next free TASK-{n} id (a new task).
        """
        used = {str(t.get("id") or "") for t in existing}
        next_num = 1
        for task_id in used:
            match = re.match(r"^TASK-(\d+)$", task_id)
            if match:
                next_num = max(next_num, int(match.group(1)) + 1)
        for raw, task in zip(raw_tasks, normalized):
            if isinstance(raw, dict) and str(raw.get("id") or "").strip():
                final_id = str(raw["id"]).strip()
                task["id"] = final_id
                used.add(final_id)
                continue
            positional = str(task.get("id") or "")
            if positional in used:
                matches = [
                    t
                    for t in existing
                    if t.get("status") != "COMPLETED"
                    and t.get("title") == task.get("title")
                ]
                if len(matches) == 1:
                    task["id"] = matches[0]["id"]
                    continue
            while f"TASK-{next_num:03d}" in used:
                next_num += 1
            task["id"] = f"TASK-{next_num:03d}"
            used.add(task["id"])
            next_num += 1

    @staticmethod
    def _merge_tasks(
        accumulated: list[dict[str, Any]], emitted: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Union tasks by id: emitted wins, unmentioned accumulated tasks are
        kept (used by replan to preserve completed tasks the model did not
        repeat). Failure context from the accumulated task survives a revision:
        notes are merged rather than silently dropped."""
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for task in emitted:
            task_id = str(task.get("id") or "")
            if task_id in seen:
                continue
            seen.add(task_id)
            prev = next((t for t in accumulated if t.get("id") == task_id), None)
            if prev is not None:
                old_notes = str(prev.get("notes") or "")
                new_notes = str(task.get("notes") or "")
                if old_notes and new_notes and old_notes not in new_notes:
                    task["notes"] = (old_notes + " | " + new_notes).strip(" |")
                elif old_notes and not new_notes:
                    task["notes"] = old_notes
            merged.append(task)
        for task in accumulated:
            if task.get("id") not in seen:
                merged.append(task)
        return merged

    def _dispatch_call(
        self,
        call: dict[str, Any],
        budget: PlanningBudget,
        metrics: dict[str, Any],
        asked: set[str],
    ) -> dict[str, Any]:
        tool = call.get("tool")
        action = call.get("action") or ""
        params = call.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        if tool == "discovery" or (
            tool in ("planner", "planning") and action in ("discover", "ask")
        ):
            question = str(
                params.get("question")
                or params.get("request")
                or params.get("query")
                or params.get("task")
                or ""
            ).strip()
            if not question:
                return {
                    "status": "error",
                    "message": "discovery requires a question",
                    "error": {"type": "invalid_arguments", "recoverable": True},
                }
            scope = params.get("scope") or params.get("path")
            scope = str(scope).strip() if scope else None
            budget_params = {
                k: v
                for k, v in params.items()
                if k in _DISCOVERY_BUDGET_KEYS and isinstance(v, int)
            }
            return self._run_discovery(
                question, scope, budget_params, budget, metrics, asked
            )

        if tool in ("planner", "planning"):
            if action in ("add_task", "submit_task"):
                return self._add_task(params, self.tasks, budget)
            if action in ("finalize", "complete"):
                payload = (
                    params.get("plan")
                    if isinstance(params.get("plan"), dict)
                    else params
                )
                plan, accepted, status = self._finalize(payload)
                return {
                    "status": "finalized" if accepted else "plan_rejected",
                    "accepted": accepted,
                    "plan": plan,
                    "plan_status": status,
                    "issues": plan.get("issues", []),
                    "warnings": plan.get("warnings", []),
                }
            return {
                "status": "error",
                "message": f"unknown planner action {action!r}",
            }

        if tool in self.readonly_registry.list_tools():
            result = self.readonly_registry.dispatch(tool, action, **params)
            return {
                "status": "analysis",
                "tool": tool,
                "action": action,
                "result": result,
            }

        return {
            "status": "error",
            "message": (
                f"unknown tool {tool!r}; this planning agent only exposes "
                "discovery, planner and read-only analysis tools."
            ),
        }

    # -- loop feedback -------------------------------------------------------

    def _feedback(self, result: dict[str, Any], budget: PlanningBudget) -> str:
        status = result.get("status")
        if status == "task_added":
            parts = [
                f"Task {result.get('task_id')} added to the plan"
                + (" (replaced an existing task)" if result.get("replaced") else "")
                + "."
            ]
            missing = result.get("missing") or []
            if missing:
                parts.append(
                    "It is NOT ready for execution: missing "
                    + "; ".join(missing)
                    + ". Run a focused discovery to fill the gaps, then "
                    "re-submit the task as READY_FOR_EXECUTION, or keep it as "
                    "DISCOVERY_REQUIRED."
                )
            parts.append("\nTASKS SO FAR:\n" + self._tasks_summary(self.tasks))
            parts.append(
                "Continue: run a discovery, add more tasks, or emit the final plan JSON."
            )
            return "\n".join(parts)
        if status == "discovery_done":
            question = result.get("question", "")
            report = str(result.get("result", {}).get("report") or "")[
                : budget.max_context_chars
            ]
            return (
                f"DISCOVERY EVIDENCE for question: {question}\n\n{report}\n\n"
                "Use this evidence to build or adjust tasks (each task keeps "
                "only its own relevant context, not this whole report). If you "
                "still lack information, ask another focused discovery "
                "question; otherwise add tasks or emit the final plan JSON."
            )
        if status == "discovery_error":
            return (
                f"Discovery failed for {result.get('question')!r}: "
                f"{result.get('message')}. Reformulate the question or plan "
                "with the evidence you have, marking uncertain tasks "
                "DISCOVERY_REQUIRED."
            )
        if status == "analysis":
            return f"Analysis result:\n{result.get('result')}\n\n" "Continue planning."
        if status == "plan_rejected":
            return (
                "Final plan rejected:\n"
                + "\n".join(f"- {issue}" for issue in result.get("issues", []))
                + "\n\nFix the issues and emit the final plan again, or "
                "continue with discovery/add_task."
            )
        return (
            f"Planning step result:\n{result}\n\n"
            "Continue planning or emit the final plan JSON."
        )

    # -- main loop -----------------------------------------------------------

    def _plan_loop(
        self,
        request: str,
        budget: PlanningBudget,
        metrics: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        self.memory.clear()
        if self._mode != "replan":
            self.tasks = []
        asked: set[str] = set()
        start = time.monotonic()
        self.memory.add_user(request)
        current = (
            f"USER REQUEST\n{request}\n\n"
            "Plan the work: determine the needed changes, run focused "
            "discovery when information is missing, and produce atomic tasks. "
            "Emit tool calls or, when done, the final plan JSON."
        )
        plan: dict[str, Any] | None = None
        status = "partial"
        reason = ""
        finalize_failures = 0

        with _working_dir(self.root):
            for index in range(budget.max_iterations):
                metrics["planning_iterations"] = index + 1
                config = self._config(request, budget)
                config_tokens = estimate_tokens(config)
                metrics["estimated_input_tokens"] += config_tokens
                if config_tokens > budget.max_context_tokens:
                    reason = (
                        f"context exceeded {budget.max_context_tokens} tokens "
                        f"({config_tokens} estimated)"
                    )
                    break

                response = self.provider.infer(current, config)
                metrics["estimated_output_tokens"] += estimate_tokens(response)

                kind, payload = self._interpret(response)
                if kind == "plan":
                    plan, accepted, status = self._finalize(payload)
                    self.memory.add_assistant(response)
                    if accepted:
                        break
                    finalize_failures += 1
                    if finalize_failures >= self.max_finalize_retries:
                        reason = "final plan rejected repeatedly"
                        break
                    current = (
                        "Final plan rejected:\n"
                        + "\n".join(f"- {issue}" for issue in plan.get("issues", []))
                        + "\n\nFix the issues and emit the final plan again, or "
                        "continue with discovery/add_task."
                    )
                    continue

                if kind == "call":
                    result = self._dispatch_call(payload, budget, metrics, asked)
                    if result.get("accepted"):
                        plan = result.get("plan") or {}
                        status = result.get("plan_status") or "partial"
                        break
                    self.memory.add_tool(
                        payload.get("tool", "?"),
                        payload.get("action", ""),
                        payload.get("params") or {},
                        self._storage(result),
                    )
                    self.memory.add_assistant(response)
                    current = self._feedback(result, budget)
                    continue

                if kind == "fragment":
                    result = self._add_task(payload, self.tasks, budget)
                    self.memory.add_assistant(response)
                    current = self._feedback(result, budget)
                    continue

                self.memory.add_assistant(response)
                excerpt = (
                    str(payload)[:500] if kind == "plan_bad" else str(response)[:500]
                )
                current = (
                    "Your reply was not a usable planning step. It must be a "
                    "discovery run, a planner add_task/finalize, or the final "
                    'plan JSON ({"goal": ..., "summary": ..., "tasks": '
                    "[...]}). Your reply was:\n"
                    f"{excerpt}"
                )

        if plan is None:
            plan, status, reason = self._assemble_from_tasks(request, budget, reason)

        metrics["tokens_used"] = (
            metrics["estimated_input_tokens"] + metrics["estimated_output_tokens"]
        )
        metrics["final_plan_tokens"] = estimate_tokens(
            json.dumps(plan.get("tasks", []))
        )
        metrics["planning_time"] = round(time.monotonic() - start, 3)
        self._update_task_metrics(metrics, plan.get("tasks", []))
        plan["metrics"] = metrics
        if reason:
            plan["reason"] = reason
        return plan, status

    def _assemble_from_tasks(
        self,
        request: str,
        budget: PlanningBudget,
        loop_reason: str = "",
    ) -> tuple[dict[str, Any], str, str]:
        tasks = list(self.tasks)
        plan: dict[str, Any] = {
            "goal": request,
            "summary": (
                f"Plan assembled from {len(tasks)} submitted task(s); the "
                "planner reached its budget before emitting a final plan."
            ),
            "tasks": tasks,
        }
        issues, warnings = validate_plan(plan)
        plan["issues"] = issues
        plan["warnings"] = warnings
        plan["valid"] = not issues
        if not tasks:
            issues.append("no tasks were planned; the request needs a discovery pass")
            plan["summary"] = (
                "No tasks were produced. Re-run with a larger budget or a more "
                "concrete request."
            )
        status = (
            "success" if plan["valid"] and plan_all_ready(plan["tasks"]) else "partial"
        )
        if loop_reason:
            reason = loop_reason
        else:
            reason = (
                f"budget exhausted; assembled {len(tasks)} task(s) from submissions"
                if tasks
                else "no tasks planned before the budget ended"
            )
        return plan, status, reason

    @staticmethod
    def _update_task_metrics(
        metrics: dict[str, Any], tasks: list[dict[str, Any]]
    ) -> None:
        metrics["tasks_created"] = len(tasks)
        metrics["tasks_ready"] = sum(
            1
            for task in tasks
            if task.get("status") in ("READY_FOR_EXECUTION", "COMPLETED")
        )
        metrics["tasks_blocked"] = sum(
            1 for task in tasks if task.get("status") == "BLOCKED"
        )
        metrics["tasks_discovery_required"] = sum(
            1 for task in tasks if task.get("status") == "DISCOVERY_REQUIRED"
        )
        metrics["tasks_completed"] = sum(
            1 for task in tasks if task.get("status") == "COMPLETED"
        )

    # -- public API ----------------------------------------------------------

    def run(self, request: str, **overrides: Any) -> dict[str, Any]:
        request = str(request or "").strip()
        if not request:
            return {
                "status": "error",
                "error": {
                    "type": "invalid_arguments",
                    "message": "request is required",
                    "recoverable": True,
                },
            }
        budget = self.budget.apply_overrides(**self._int_overrides(overrides))
        metrics = self._empty_metrics()
        self._request = request
        self.last_request = request
        plan, status = self._plan_loop(request, budget, metrics)
        self.last_plan = plan
        return {
            "status": status,
            "request": request,
            "workspace": self.root,
            "plan": plan,
            "metrics": metrics,
        }

    def discover(
        self, question: str, scope: str | None = None, **overrides: Any
    ) -> dict[str, Any]:
        question = str(question or "").strip()
        if not question:
            return {
                "status": "error",
                "error": {
                    "type": "invalid_arguments",
                    "message": "question is required",
                    "recoverable": True,
                },
            }
        budget = self.budget.apply_overrides(**self._int_overrides(overrides))
        metrics = {"discovery_calls": 0, "discovery_revisits": 0}
        asked: set[str] = set()
        if _is_generic_discovery_question(question):
            return {
                "status": "error",
                "message": (
                    "discovery question is too generic; ask ONE specific "
                    "question about a concrete artifact, optionally with a "
                    "scope path."
                ),
                "error": {"type": "invalid_arguments", "recoverable": True},
                "question": question,
            }
        result = self._run_discovery(question, scope, {}, budget, metrics, asked)
        return {
            "status": result.get("status"),
            "question": question,
            "scope": scope or "",
            "result": result,
            "metrics": metrics,
        }

    def pending(self) -> dict[str, Any]:
        if self.last_plan is None:
            return {"status": "no_plan"}
        return {
            "status": "pending_plan",
            "request": self.last_request,
            "plan": self.last_plan,
        }

    def replan(
        self,
        request: str,
        previous_plan: dict[str, Any] | None = None,
        task_results: Any = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        request = str(request or "").strip()
        if not request:
            return {
                "status": "error",
                "error": {
                    "type": "invalid_arguments",
                    "message": "request is required",
                    "recoverable": True,
                },
            }
        budget = self.budget.apply_overrides(**self._int_overrides(overrides))
        metrics = self._empty_metrics()
        results = self._normalize_results(task_results)
        self._mode = "replan"
        self.tasks = self._seed_tasks(previous_plan, results)
        self._replan_context = self._build_replan_context(previous_plan, results)
        self._request = request
        self.last_request = request
        try:
            plan, status = self._plan_loop(request, budget, metrics)
        finally:
            self._mode = "plan"
            self._replan_context = None
        self.last_plan = plan
        return {
            "status": status,
            "request": request,
            "workspace": self.root,
            "replan": True,
            "plan": plan,
            "metrics": metrics,
        }

    # -- replan helpers ------------------------------------------------------

    @classmethod
    def _normalize_results(cls, task_results: Any) -> dict[str, dict[str, str]]:
        normalized: dict[str, dict[str, str]] = {}
        if isinstance(task_results, dict):
            items: Any = task_results.items()
        elif isinstance(task_results, (list, tuple)):
            items = (
                (str(entry.get("id") or entry.get("task_id") or ""), entry)
                for entry in task_results
                if isinstance(entry, dict)
            )
        else:
            return normalized
        for task_id, raw in items:
            task_id = str(task_id).strip()
            if not task_id:
                continue
            if isinstance(raw, str):
                normalized[task_id] = {"status": raw.lower(), "notes": ""}
            elif isinstance(raw, dict):
                normalized[task_id] = {
                    "status": str(raw.get("status") or "").lower(),
                    "notes": str(raw.get("notes") or ""),
                }
                cls._attach_executor_discovery(normalized[task_id], raw)
        return normalized

    @staticmethod
    def _attach_executor_discovery(entry: dict[str, str], raw: dict[str, Any]) -> None:
        """Attach the discovery questions the Executor had to ask back into the
        replan notes, so the planner learns exactly what information its plan
        was missing (the key planning-quality signal).
        """
        calls = raw.get("discovery_calls")
        if not calls:
            return
        if isinstance(calls, dict):
            calls = calls.values()
        questions = [
            str(call.get("question") or "")
            for call in calls
            if isinstance(call, dict) and call.get("question")
        ]
        if not questions:
            return
        missing = " | ".join(f"executor discovery: {q}" for q in questions)
        entry["notes"] = (entry.get("notes", "") + " | " + missing).strip(" |")

    def _seed_tasks(
        self,
        previous_plan: dict[str, Any] | None,
        results: dict[str, dict[str, str]],
    ) -> list[dict[str, Any]]:
        seeded: list[dict[str, Any]] = []
        if not previous_plan or not isinstance(previous_plan, dict):
            return seeded
        for index, raw in enumerate(previous_plan.get("tasks", [])):
            try:
                task = normalize_task(raw, index + 1)
            except ValueError:
                continue
            outcome = results.get(task["id"])
            if outcome and outcome.get("status") in _COMPLETED_STATUSES:
                task["status"] = "COMPLETED"
            elif outcome:
                detail = outcome.get("notes") or outcome.get("status") or "failed"
                note = f"execution: {detail}"
                task["notes"] = (str(task.get("notes") or "") + " | " + note).strip(
                    " |"
                )
            seeded.append(task)
        return seeded

    def _build_replan_context(
        self,
        previous_plan: dict[str, Any] | None,
        results: dict[str, dict[str, str]],
    ) -> str:
        lines = [
            "You are replanning after execution. Completed tasks keep status "
            "COMPLETED; failed/revised tasks must be re-planned or fixed. "
            "Emit the revised FULL plan (existing + new tasks)."
        ]
        if not previous_plan or not isinstance(previous_plan, dict):
            return "\n".join(lines)
        lines.append(f"GOAL: {previous_plan.get('goal') or ''}")
        for task in previous_plan.get("tasks", []):
            task_id = task.get("id")
            title = task.get("title")
            outcome = results.get(task_id)
            line = f"- {task_id} [{task.get('status')}] {title}"
            if outcome:
                line += f"  -> executed: {outcome['status']}"
                if outcome.get("notes"):
                    line += f" ({outcome['notes']})"
            lines.append(line)
        return "\n".join(lines)


def build_discovery_registry(root: str | Path = ".") -> ToolRegistry:
    """Private read-only registry for the planning loop: only analysis tools
    (base read-only tools + the ``code`` symbol toolset + aliases). No mutating
    tool is ever registered, so the planner cannot modify the project."""
    from src.discovery.tools import build_discovery_registry as _build

    return _build(root)


__all__ = [
    "PLANNING_INSTRUCTIONS",
    "PlanningAgent",
    "build_discovery_registry",
]
