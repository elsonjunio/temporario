from __future__ import annotations

from typing import Any

from src.executor.risk import RiskLevel, classify_task, risk_description, top_risk
from src.planning.plan import as_list

#: One-liners shown when the user asks for plan details (verbosity > 0).
_VERBOSE = 2


def format_plan_preview(preview: dict[str, Any], verbosity: int = 0) -> str:
    """Render the plan-confirmation block shown before any change.

    Example layout::

        Plano:
        TASK-001 Atualizar tokens [LOW]
        TASK-002 Atualizar tipografia [LOW]
        TASK-003 Atualizar Header [MEDIUM]

        Arquivos potencialmente afetados:
        - src/tokens.css

        Deseja executar este plano?
        [SIM] [NAO] [DETALHES]
    """
    summary = preview.get("summary") or []
    lines = ["Plano:"]
    files: set[str] = set()
    dependencies: list[str] = []
    risk_labels = [RiskLevel.LOW]
    for item in summary:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id") or item.get("step") or "?"
        title = item.get("title") or item.get("description") or ""
        risk = item.get("risk") or RiskLevel.LOW
        risk_labels.append(risk)
        lines.append(f"{task_id} {title} [{risk}]")
        for f in as_list(item.get("files")):
            files.add(str(f))
        for dep in as_list(item.get("dependencies")):
            dependencies.append(str(dep))

    if files:
        lines.append("\nArquivos potencialmente afetados:")
        lines.extend(f"- {f}" for f in sorted(files))
    if dependencies:
        lines.append("\nDependências entre tasks:")
        lines.extend(f"- {dep}" for dep in dependencies)

    impact = preview.get("impact")
    if impact:
        lines.append(f"\n{impact}")

    if verbosity >= _VERBOSE:
        details = preview.get("details")
        if details:
            lines.append("\nDetalhes do plano:")
            lines.append(str(details))

    lines.append("\nDeseja executar este plano?\n[SIM]\n[NÃO]\n[DETALHES]")
    return "\n".join(lines)


def format_task_header(
    task: dict[str, Any], index: int, total: int, risk: str | None = None
) -> str:
    """One-line header for a task during execution."""
    risk = risk or classify_task(task)
    task_id = task.get("id") or "?"
    title = task.get("title") or ""
    return f"{task_id} {title} ({index}/{total}) — Risco: {risk}"


def format_confirmation(
    task_id: str,
    level: str,
    message: str,
    affected: list[str] | None = None,
) -> str:
    """Render the critical-action confirmation prompt ([SIM]/[NAO])."""
    lines = [f"{task_id}", f"Risco: {level}"]
    description = risk_description(level)
    if description:
        lines.append(description)
    lines.append("")
    lines.append(message)
    for path in affected or []:
        lines.append(f"- {path}")
    lines.append("\nA operação pode ser irreversível.\n")
    lines.append("Deseja continuar?\n[SIM]\n[NÃO]")
    return "\n".join(lines)


def format_question(task_id: str, question: str, options: list[str] | None) -> str:
    """Render a user question, with numbered options when available."""
    lines = [f"{task_id}", "", question]
    if options:
        lines.append("")
        lines.extend(f"{index}. {option}" for index, option in enumerate(options, 1))
    return "\n".join(lines)


def format_validation_result(validation: dict[str, Any], verbosity: int = 0) -> str:
    """Render a single task's ``validation_result`` block."""
    status = validation.get("status", "?")
    lines = [f"Validação: {status}"]
    summary = validation.get("summary")
    if summary:
        lines.append(f"  {summary}")
    for entry in validation.get("acceptance_results") or []:
        criterion = entry.get("criterion")
        result = entry.get("result", "?")
        checked_by = entry.get("checked_by")
        suffix = f" [{checked_by}]" if checked_by else ""
        lines.append(f"  - {criterion}: {result}{suffix}")
        for evidence in (entry.get("evidence") or [])[:3]:
            lines.append(f"      • {evidence}")
    for test in validation.get("tests") or []:
        lines.append(f"  teste: {test.get('command')} [{test.get('status')}]")
    for command in validation.get("commands") or []:
        lines.append(f"  comando: {command.get('command')} [{command.get('status')}]")
    if verbosity >= _VERBOSE:
        for warning in validation.get("warnings") or []:
            lines.append(f"  aviso: {warning}")
        for error in validation.get("errors") or []:
            lines.append(f"  erro: {error}")
    return "\n".join(lines)


def _aggregate_validation(validations: dict[str, Any]) -> dict[str, str]:
    """Aggregate per-task validation results into tests/build/browser status."""
    outcome: dict[str, str] = {}
    for kind, label in (
        ("tests", "testes"),
        ("build", "build"),
        ("browser", "browser"),
    ):
        passed = 0
        failed = 0
        ran = 0
        for task_validation in validations.values():
            commands = task_validation.get("commands") or []
            for command in commands:
                command_kind = command.get("kind")
                if command_kind != kind:
                    continue
                ran += 1
                if command.get("status") == "success":
                    passed += 1
                else:
                    failed += 1
            tests = task_validation.get("tests") or []
            if kind == "tests" and tests:
                ran += len(tests)
                passed += sum(1 for t in tests if t.get("status") == "success")
                failed += len(tests) - sum(
                    1 for t in tests if t.get("status") == "success"
                )
        if ran == 0:
            outcome[label] = "não executado"
        elif failed == 0:
            outcome[label] = "PASS"
        else:
            outcome[label] = f"FAIL ({failed} falha(s))"
    return outcome


def format_execution_summary(result: dict[str, Any], verbosity: int = 0) -> str:
    """Render the final EXECUTION SUMMARY block: per-task outcome buckets,
    changed files, aggregated validation, warnings and replanning requests."""
    lines = ["EXECUTION SUMMARY"]
    plan_status = result.get("plan_status") or result.get("status")
    lines.append(f"Plano: {str(plan_status or '?').upper()}")

    for label, key in (
        ("Completadas", "completed_tasks"),
        ("Falharam", "failed_tasks"),
        ("Bloqueadas", "blocked_tasks"),
        ("Canceladas", "cancelled_tasks"),
    ):
        tasks = result.get(key) or []
        lines.append(f"{label}: {', '.join(tasks) if tasks else '(nenhuma)'}")

    changed = result.get("changed_files") or []
    if changed:
        lines.append("Arquivos modificados:")
        lines.extend(f"- {path}" for path in changed[:20])
        if len(changed) > 20:
            lines.append(f"  ... (+{len(changed) - 20} arquivos)")

    validations = result.get("validations") or {}
    if validations:
        lines.append("Validação:")
        aggregated = _aggregate_validation(validations)
        for label, state in aggregated.items():
            lines.append(f"  {label}: {state}")
    repairs = result.get("metrics", {}).get("retries_total") or 0
    if repairs:
        lines.append(f"  reparos aplicados: {repairs}")
    executor_discovery = result.get("metrics", {}).get("executor_discovery_total") or 0
    if executor_discovery:
        lines.append(
            f"  discovery pelo executor: {executor_discovery} "
            "(planejamento insuficiente)"
        )
    for task_id, task_validation in validations.items():
        questions = task_validation.get("discovery_calls") or []
        if questions:
            lines.append(
                f"  {task_id}: discovery necessário - "
                + "; ".join(str(q.get("question") or "?") for q in questions[:5])
            )

    for task_id, task_validation in validations.items():
        status = task_validation.get("status") or "?"
        attempts = task_validation.get("attempts") or 0
        repairs = task_validation.get("repairs") or 0
        if attempts > 1:
            lines.append(
                f"  {task_id}: {status} ({attempts} tentativa(s), "
                f"{repairs} reparo(s))"
            )

    warnings = result.get("warnings") or []
    if warnings:
        lines.append("Avisos:")
        lines.extend(f"- {warning}" for warning in warnings[:10])

    replans = result.get("replanning_requests") or []
    if replans:
        lines.append("Replanejamento necessário:")
        for request in replans[:10]:
            lines.append(f"- {request.get('task_id')}: {request.get('problem')}")

    if verbosity >= _VERBOSE:
        for request in replans:
            evidence = request.get("evidence") or []
            if evidence:
                lines.append("  evidências:")
                lines.extend(f"    - {entry}" for entry in evidence[:10])
    return "\n".join(lines)


def format_result(result: dict[str, Any], verbosity: int = 0) -> str:
    """Render a finished run result without dumping huge tool payloads.

    Shows the overall status, per-task one-liners (state + reason + changed
    files) and the error list. Only when ``verbosity`` is high are tool calls
    included, and even then capped.
    """
    status = result.get("status", "?")
    lines = [f"Executor: {status.upper()}"]
    goal = result.get("goal")
    if goal:
        lines.append(f"Objetivo: {goal}")

    for row in result.get("tasks") or []:
        task_id = row.get("task_id", "?")
        title = row.get("title", "")
        state = row.get("status", "")
        lines.append(f"\n{task_id} {title} — {state}")
        reason = row.get("reason")
        if reason:
            lines.append(f"  motivo: {reason}")
        res = row.get("result") or {}
        changed = res.get("changed_files") or []
        if changed:
            lines.append("  alterados:")
            lines.extend(f"    - {path}" for path in changed[:10])
            if len(changed) > 10:
                lines.append(f"    ... (+{len(changed) - 10} arquivos)")
        discovery = res.get("discovery_calls") or []
        if discovery:
            lines.append(f"  discovery: {len(discovery)} consulta(s) focado(s)")
        inputs = res.get("user_inputs") or []
        if inputs:
            lines.append(f"  perguntas ao usuário: {len(inputs)}")
        replan = res.get("replan")
        if replan:
            lines.append(f"  replanning: {replan.get('reason', '')}")

    for error in result.get("errors") or []:
        lines.append(f"\nerro: {error}")
    for warning in result.get("warnings") or []:
        lines.append(f"\naviso: {warning}")

    metrics = result.get("metrics") or {}
    if metrics:
        lines.append(
            "\nMétricas: "
            f"tasks={metrics.get('tasks_executed', 0)}/"
            f"{metrics.get('tasks_total', 0)}, "
            f"tool_calls={metrics.get('tool_calls', 0)}, "
            f"elapsed={metrics.get('elapsed', 0)}s"
        )

    if verbosity >= _VERBOSE:
        for row in result.get("tasks") or []:
            res = row.get("result") or {}
            for call in res.get("tool_calls") or []:
                tool = call.get("tool", "?")
                action = call.get("action", "")
                params = call.get("params") or {}
                lines.append(
                    f"\n[debug] {row.get('task_id')}: {tool}:{action} params={params}"
                )
    return "\n".join(lines)


def format_pause(pause: dict[str, Any], verbosity: int = 0) -> str:
    """Render a mid-task pause (confirmation or question) for relay/terminal."""
    kind = pause.get("kind", "")
    task_id = pause.get("task_id", "")
    if kind == "confirmation":
        return format_confirmation(
            task_id,
            pause.get("level") or RiskLevel.HIGH,
            pause.get("message") or "",
            pause.get("files"),
        )
    return format_question(task_id, pause.get("message") or "", pause.get("options"))


def summarize_risks(tasks: list[dict[str, Any]]) -> str:
    """Short risk overview for a plan (top risk + counts per level)."""
    counts = {level: 0 for level in RiskLevel.all()}
    for task in tasks:
        counts[classify_task(task)] += 1
    parts = [f"risco máximo: {top_risk(tasks)}"]
    parts.extend(f"{level}={count}" for level, count in counts.items())
    return "; ".join(parts)


__all__ = [
    "format_confirmation",
    "format_execution_summary",
    "format_pause",
    "format_plan_preview",
    "format_question",
    "format_result",
    "format_task_header",
    "format_validation_result",
    "summarize_risks",
]
