from __future__ import annotations

import re
from typing import Any

#: Allowed task statuses. COMPLETED is only meaningful during replanning.
VALID_STATUSES = {
    "DISCOVERY_REQUIRED",
    "READY_FOR_EXECUTION",
    "BLOCKED",
    "COMPLETED",
}

#: Statuses that imply the task is fully specified and ready to run (or already
#: ran). Everything else means investigation is still needed.
READY_STATUSES = {"READY_FOR_EXECUTION", "BLOCKED", "COMPLETED"}

#: Optional per-task fields, kept when present.
OPTIONAL_FIELDS = ("risks", "constraints", "notes", "confidence")

#: Placeholder-looking file entries that never make a task "ready".
_PLACEHOLDER_FILES = {"", "unknown", "tbd", "todo", "none", "n/a", "na", "?"}

#: Acceptance criteria that give the Executor no way to check success.
#: Detected as warnings (never blocking) so the planner is nudged to be
#: verifiable without being rejected.
_VAGUE_CRITERIA_PATTERNS = (
    "deixar funcionando",
    "deixar funcionar",
    "deve funcionar",
    "funcionar corretamente",
    "funciona corretamente",
    "corrigir o problema",
    "corrigir o erro",
    "melhorar o código",
    "melhorar o codigo",
    "deixar pronto",
    "ficar pronto",
    "tudo certo",
    "make it work",
    "fix the issue",
    "fix the problem",
    "improve the code",
    "works correctly",
    "as expected",
    "ok",
    "okay",
)
_VAGUE_CRITERIA_RE = re.compile(
    "|".join(re.escape(pattern) for pattern in _VAGUE_CRITERIA_PATTERNS),
    re.IGNORECASE,
)

#: Canonical task fields produced by :func:`normalize_task`.
CANONICAL_FIELDS = (
    "id",
    "title",
    "objective",
    "status",
    "dependencies",
    "files",
    "context",
    "expected_changes",
    "acceptance_criteria",
    "evidence",
) + OPTIONAL_FIELDS


def as_list(value: Any) -> list[str]:
    """Normalize a value into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def files_real(files: Any) -> bool:
    """True when the task names at least one concrete target file.

    A task listing only placeholders ("frontend", "the styles", "unknown") is
    not ready for execution.
    """
    for entry in as_list(files):
        normalized = entry.strip().lower().rstrip(".")
        if normalized in _PLACEHOLDER_FILES:
            continue
        if len(normalized) < 2:
            continue
        if any(token in normalized for token in ("/", "\\", ".")):
            return True
    return False


def normalize_task(raw: Any, index: int = 1) -> dict[str, Any]:
    """Normalize a raw task object into the canonical task schema.

    Accepts the loose spellings models emit (``changes`` for
    ``expected_changes``, ``acceptance`` for ``acceptance_criteria``, a
    ``context`` as text or dict, ``files`` as string or list). Raises
    ``ValueError`` for a structurally invalid task (not an object / bad status).
    """
    if not isinstance(raw, dict):
        raise ValueError("task must be an object")

    title = str(raw.get("title") or raw.get("name") or "").strip()
    objective = str(raw.get("objective") or raw.get("description") or "").strip()
    status = str(raw.get("status") or "READY_FOR_EXECUTION").strip().upper()
    if status not in VALID_STATUSES:
        raise ValueError(
            f"invalid status {status!r}; expected one of {sorted(VALID_STATUSES)}"
        )

    dependencies = as_list(raw.get("dependencies"))
    files = as_list(raw.get("files") or raw.get("file") or raw.get("targets"))
    context = _as_text(raw.get("context"))
    expected_changes = as_list(
        raw.get("expected_changes") or raw.get("changes") or raw.get("result")
    )
    acceptance_criteria = as_list(
        raw.get("acceptance_criteria") or raw.get("acceptance") or raw.get("criteria")
    )
    evidence = _as_text(raw.get("evidence") or raw.get("facts"))

    task: dict[str, Any] = {
        "id": str(raw.get("id") or f"TASK-{index:03d}"),
        "title": title,
        "objective": objective,
        "status": status,
        "dependencies": dependencies,
        "files": files,
        "context": context,
        "expected_changes": expected_changes,
        "acceptance_criteria": acceptance_criteria,
        "evidence": evidence,
    }
    for field in OPTIONAL_FIELDS:
        if field in raw and raw[field] is not None:
            value = raw[field]
            task[field] = (
                as_list(value) if field in ("risks", "constraints") else _as_text(value)
            )
    return task


def task_structure_issues(task: dict[str, Any]) -> list[str]:
    """Structural problems that always block a task (title/objective/status)."""
    issues: list[str] = []
    if not str(task.get("title") or "").strip():
        issues.append("missing title")
    if not str(task.get("objective") or "").strip():
        issues.append("missing objective")
    status = task.get("status")
    if status not in VALID_STATUSES:
        issues.append(f"invalid status {status!r}")
    return issues


def task_readiness_issues(task: dict[str, Any]) -> list[str]:
    """Missing information that prevents a task from being READY_FOR_EXECUTION."""
    issues: list[str] = []
    if not files_real(task.get("files")):
        issues.append("no concrete target files")
    if not str(task.get("context") or "").strip():
        issues.append("missing context")
    if not as_list(task.get("expected_changes")):
        issues.append("missing expected_changes")
    if not as_list(task.get("acceptance_criteria")):
        issues.append("missing acceptance_criteria")
    if not str(task.get("evidence") or "").strip():
        issues.append("missing evidence")
    return issues


def resolve_dependencies(tasks: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Resolve ``dependencies`` entries to real task ids.

    Entries may be ids ("TASK-001") or titles (mapped via the id-by-title
    index). Returns a map of unknown dependency -> task ids that reference it.
    """
    by_id = {task["id"]: task for task in tasks}
    by_title: dict[str, str] = {}
    for task in tasks:
        title = str(task.get("title") or "").strip().lower()
        if title:
            by_title[title] = task["id"]

    unknown: dict[str, list[str]] = {}
    for task in tasks:
        resolved: list[str] = []
        for entry in task.get("dependencies") or []:
            raw = str(entry).strip()
            if raw in by_id:
                resolved.append(raw)
            elif raw.lower() in by_title:
                resolved.append(by_title[raw.lower()])
            else:
                unknown.setdefault(raw, []).append(task["id"])
        task["dependencies"] = resolved
    return unknown


def _dependency_cycle(tasks: list[dict[str, Any]]) -> list[str] | None:
    by_id = {task["id"]: task for task in tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(tid: str, stack: list[str]) -> list[str] | None:
        if tid in visiting:
            cycle = stack[stack.index(tid) :] + [tid]
            return cycle
        if tid in visited:
            return None
        visiting.add(tid)
        stack.append(tid)
        for dep in by_id[tid].get("dependencies") or []:
            if dep in by_id:
                found = visit(dep, stack)
                if found is not None:
                    return found
        stack.pop()
        visiting.discard(tid)
        visited.add(tid)
        return None

    for task in tasks:
        cycle = visit(task["id"], [])
        if cycle is not None:
            return cycle
    return None


def vague_criteria(criteria: Any) -> list[str]:
    """Return the acceptance criteria that give the Executor no way to check
    success (subjective outcome words instead of observable checks)."""
    items = [str(item).strip() for item in as_list(criteria)]
    return [item for item in items if _VAGUE_CRITERIA_RE.search(item)]


def validate_plan(plan: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Validate a plan and correct non-ready tasks in place.

    Returns ``(issues, warnings)``. ``issues`` are structural problems that
    make the plan invalid (empty tasks, missing title/objective, bad status,
    duplicate ids, unknown dependencies, dependency cycles). ``warnings``
    record tasks that claimed READY_FOR_EXECUTION / BLOCKED but lack the
    required information: those tasks are downgraded to DISCOVERY_REQUIRED so
    the planner never fakes readiness.
    """
    issues: list[str] = []
    warnings: list[str] = []

    tasks = plan.get("tasks") or []
    if not isinstance(tasks, list) or not tasks:
        return ["plan must contain a non-empty 'tasks' list"], warnings
    if not str(plan.get("goal") or "").strip():
        issues.append("missing plan 'goal'")

    ids = [task.get("id") for task in tasks]
    duplicates = sorted({tid for tid in ids if ids.count(tid) > 1})
    if duplicates:
        issues.append(f"duplicate task ids: {duplicates}")

    for task in tasks:
        task_id = task.get("id")
        structural = task_structure_issues(task)
        if structural:
            issues.extend(f"{task_id}: {problem}" for problem in structural)
            continue
        if task["status"] in ("READY_FOR_EXECUTION", "BLOCKED"):
            readiness = task_readiness_issues(task)
            if readiness:
                warnings.append(
                    f"{task_id} marked {task['status']} but is not ready: "
                    f"{'; '.join(readiness)}; downgraded to DISCOVERY_REQUIRED"
                )
                task["status"] = "DISCOVERY_REQUIRED"
            else:
                vague = vague_criteria(task.get("acceptance_criteria"))
                if vague:
                    warnings.append(
                        f"{task_id} has vague acceptance criteria that may be "
                        "hard to verify: "
                        f"{'; '.join(vague)}"
                    )
        elif task["status"] == "DISCOVERY_REQUIRED":
            missing = task_readiness_issues(task)
            if missing:
                warnings.append(
                    f"{task_id} still requires discovery: {'; '.join(missing)}"
                )

    unknown = resolve_dependencies(tasks)
    if unknown:
        details = ", ".join(
            f"{dep} (used by {', '.join(ids)})" for dep, ids in unknown.items()
        )
        issues.append(f"unknown dependencies: {details}")

    cycle = _dependency_cycle(tasks)
    if cycle:
        issues.append(f"dependency cycle: {' -> '.join(cycle)}")

    return issues, warnings


def plan_all_ready(tasks: list[dict[str, Any]]) -> bool:
    """True when every task is ready to run (or already completed)."""
    return bool(tasks) and all(task.get("status") in READY_STATUSES for task in tasks)


__all__ = [
    "CANONICAL_FIELDS",
    "OPTIONAL_FIELDS",
    "READY_STATUSES",
    "VALID_STATUSES",
    "as_list",
    "files_real",
    "normalize_task",
    "plan_all_ready",
    "resolve_dependencies",
    "task_readiness_issues",
    "task_structure_issues",
    "validate_plan",
    "vague_criteria",
]
