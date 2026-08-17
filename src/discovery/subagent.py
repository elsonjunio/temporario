from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.context import estimate_tokens
from src.discovery.budget import DiscoveryBudget
from src.discovery.tools import build_discovery_registry
from src.memory import Memory
from src.tools.registry import ToolRegistry
from src.utils import build_config_prompt, build_environment_info, parse_tool_call

DISCOVERY_INSTRUCTIONS = (
    "You are a temporary Discovery Agent. Your only job is to investigate a "
    "task in the workspace and produce a compact, evidence-based report that a "
    "separate Executor Agent will use to implement the solution. You DO NOT "
    "implement, edit, delete or move anything, and you DO NOT run build/test "
    "commands or scripts: every tool you have only reads the project.\n"
    "Your goal is to find the smallest amount of information the executor "
    "needs to act, maximizing useful information per token used. Reduce "
    "uncertainty progressively; do NOT try to understand the whole project.\n"
    "Investigate iteratively, never as a fixed sequence of tools. Each step is "
    "a cycle: request -> hypothesis -> tool -> evidence -> new hypothesis. "
    "Before every call ask yourself:\n"
    "- What do I already know?\n"
    "- What do I still need to know?\n"
    "- Which piece of information would reduce my uncertainty the most?\n"
    "- Which tool obtains it at the lowest cost?\n"
    "Funnel strategy (cheap and shallow first, deepen only relevant "
    "candidates): project structure -> file names -> text search -> symbols -> "
    "definitions -> references -> imports/dependencies -> code snippets -> "
    "full file. Prefer a query that eliminates many possibilities over reading "
    "a file that may be irrelevant. Read a full file only when it is central "
    "and the relevant part is unclear; prefer start_line/end_line windows.\n"
    "UI/STYLE REQUESTS: if the task mentions a user-facing surface or visual "
    "style (catálogo/catalog, tela/page, component/componente, estilo/style, "
    "design, cor/color, tema/theme, ui, css, layout, visual), the "
    "IMPLEMENTATION AREA must name the REAL source/style files "
    "(.component.ts, .ts, .css, .scss, .html, .vue, .svelte) - never conclude "
    "that the 'implementation' is a markdown SPEC or DESIGN doc. Inspect at "
    "least one actual code/style file (grep/seek the entity name under "
    "src/, app/, components/, pages/) before reporting, and prefer "
    "accent-stripped search terms (search 'catalog' when the request says "
    "'catálogo'). Do not stop at specification or design documents.\n"
    "Form hypotheses about where the relevant implementation lives and use the "
    "tools to confirm or reject them. Prioritize investigations that rule out "
    "several possibilities at once. Register the hypotheses you actually "
    "investigated in the HYPOTHESES section of the report.\n"
    "Classify candidates by utility: relevant / likely relevant / uncertain / "
    "irrelevant. Discard irrelevant information quickly and do not keep full "
    "contents of discarded files; when a discard is notable, record only a "
    "compact justification in the DISCARDED section.\n"
    "Reuse results: do not re-run a search already performed and do not "
    "re-read files already inspected without a reason. Rely on earlier results "
    "to choose the next action and keep only what the reasoning still needs.\n"
    "Identify relationships, not just files: who calls whom, who imports whom, "
    "who implements a symbol, who references a symbol, which APIs are used, "
    "which components depend on which. Build a small mental graph of the "
    "relevant area.\n"
    "Be precise: report exact paths, symbols when possible, and line numbers "
    "when available. Distinguish facts found (evidence) from hypotheses (not "
    "yet confirmed). Never invent relationships or claim something is relevant "
    "without evidence.\n"
    "Stop as soon as you can answer all of these:\n"
    "- Where is the relevant implementation?\n"
    "- Which components are involved, and how do they relate (call sites with "
    "file:line)?\n"
    "- Which files will likely need to be changed?\n"
    "- What does the executor need to know to start?\n"
    "- Are there dependencies or risks that matter?\n"
    "- Is there any uncertainty that could block the implementation?\n"
    "Do not solve the task: answer WHAT the executor needs to know, not HOW to "
    "implement the solution. You may point out files to modify, possible "
    "causes and hypotheses, but never implement the fix.\n"
    "Keep tool parameters narrow (path=, include=, line ranges). Never paste "
    "large code blocks; reference file:line. Prefer precise statements like "
    '"AuthenticationService.authenticate() is called by src/api/auth.py:42".\n'
    "Rules:\n"
    "- Each tool call must be a single JSON block:\n"
    '  {"tool": "<tool>", "action": "<action>", "params": {<arguments>}}\n'
    "- Only use the tools listed below; they are read-only. All paths are "
    "relative to the workspace or absolute.\n"
    "- When you have enough evidence, STOP and write the final report below as "
    "plain text (no JSON block).\n"
    "FINAL REPORT (exact structure, plain text):\n"
    "\n"
    "DISCOVERY RESULT\n"
    "\n"
    "TASK\n"
    "<original request>\n"
    "\n"
    "SUMMARY\n"
    "<short summary of what you found>\n"
    "\n"
    "RELEVANT FILES\n"
    "1. path/to/file.py\n"
    "   Purpose: ...\n"
    "   Relevant symbols: ...\n"
    "   Relevance: ...\n"
    "   Important details: ...\n"
    "\n"
    "RELATIONSHIPS\n"
    "<how the components connect, with file:line references>\n"
    "\n"
    "IMPLEMENTATION AREA\n"
    "<files likely to be modified>\n"
    "\n"
    "HYPOTHESES\n"
    "<hypotheses investigated and whether each was confirmed or rejected>\n"
    "\n"
    "EVIDENCE\n"
    "<key evidence found (facts, distinct from hypotheses)>\n"
    "\n"
    "DISCARDED\n"
    "<candidates investigated but ruled out, with a compact justification>\n"
    "\n"
    "RISKS\n"
    "<risks, dependencies, ambiguities>\n"
    "\n"
    "NOT INVESTIGATED\n"
    "<points that remain uncertain>\n"
    "\n"
    "CONFIDENCE\n"
    "high | medium | low"
    " - if not high, add one brief line explaining why."
)

_STORAGE_RESULT_CHARS = 4096


def _collect_paths(value: Any, acc: set[str], depth: int = 0) -> None:
    """Collect strings stored under 'path'/'file' keys, recursively."""
    if depth > 4:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("path", "file") and isinstance(item, str) and item.strip():
                acc.add(item.strip())
            else:
                _collect_paths(item, acc, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _collect_paths(item, acc, depth + 1)


def _looks_uninspected(report: str) -> bool:
    lowered = report.lower()
    return any(marker in lowered for marker in _UNINSPECTED_MARKERS)


_UNINSPECTED_MARKERS = (
    "not yet inspected",
    "not been inspected",
    "not inspected",
    "needs inspection",
    "needs to be inspected",
    "não foram inspecionados",
    "não inspecionados",
    "não foram lidos",
    "ainda não foi",
    "content needs",
)


def _result_lines(result: Any) -> int:
    if not isinstance(result, dict):
        return 0
    start = result.get("start_line")
    end = result.get("end_line")
    if isinstance(start, int) and isinstance(end, int):
        return max(0, end - start + 1)
    for key in ("content", "current_page_content"):
        content = result.get(key)
        if isinstance(content, str):
            return len(content.splitlines())
    return 0


def _call_params_key(params: Any) -> str:
    """A stable string key for deduplicating identical tool calls."""
    if not isinstance(params, dict):
        return repr(params)
    return "|".join(
        f"{k}={repr(v)}" for k, v in sorted(params.items(), key=lambda kv: str(kv[0]))
    )


def _storage_result(result: Any) -> Any:
    text = str(result)
    if len(text) <= _STORAGE_RESULT_CHARS:
        return result
    return {
        "truncated_for_history": True,
        "size": len(text),
        "preview": text[:_STORAGE_RESULT_CHARS],
    }


@contextmanager
def _working_dir(path: str) -> Iterator[None]:
    """chdir into ``path`` for the investigation, restoring cwd afterwards."""
    original = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


class DiscoveryAgent:
    """Ephemeral subagent that investigates a task using only read-only
    discovery tools and produces a compact report plus telemetry.

    Modeled after ``NavigationSubAgent``: it owns a private read-only
    ``ToolRegistry`` and its own ``Memory``, runs a short loop of
    config-prompt -> infer -> parse -> dispatch, and stops either when the
    model emits a plain-text report (no tool call) or when a budget limit is
    reached (status ``partial``).
    """

    def __init__(
        self,
        provider: Any,
        root: str = ".",
        *,
        budget: DiscoveryBudget | None = None,
        registry: ToolRegistry | None = None,
        environment: str | None = None,
    ) -> None:
        self.provider = provider
        self.root = str(Path(root).resolve())
        self.budget = (budget or DiscoveryBudget()).with_env()
        self.registry = registry or build_discovery_registry(self.root)
        self.environment = environment or build_environment_info(workspace=self.root)
        self.memory = Memory()

    def _config(self, current: str, budget: DiscoveryBudget) -> str:
        extra = (
            f"\nBudget (stop before reaching these): {budget.to_dict()}\n"
            f"Tools are READ-ONLY: never modify files or run commands."
        )
        return build_config_prompt(
            tool_manuals=self.registry.get_manual(),
            memory=self.memory,
            instructions=DISCOVERY_INSTRUCTIONS,
            max_history_entries=30,
            environment=self.environment + extra,
        )

    def _fallback_report(self, request: str, reason: str) -> str:
        return (
            "DISCOVERY RESULT\n\n"
            f"TASK\n{request}\n\n"
            "SUMMARY\n"
            f"Investigation stopped before a final report was produced: {reason}.\n"
            "The executor should re-run discovery or provide more explicit "
            "terms/paths.\n\n"
            "RELEVANT FILES\n(none confirmed)\n\n"
            "RELATIONSHIPS\n(unknown)\n\n"
            "IMPLEMENTATION AREA\n(unknown)\n\n"
            "HYPOTHESES\n(none investigated)\n\n"
            "EVIDENCE\n(no evidence gathered)\n\n"
            "DISCARDED\n(none)\n\n"
            "RISKS\n(unknown)\n\n"
            "NOT INVESTIGATED\n(most of the workspace)\n\n"
            "CONFIDENCE\nlow"
        )

    def run(
        self,
        request: str,
        *,
        max_iterations: int | None = None,
        max_tool_calls: int | None = None,
        max_context_tokens: int | None = None,
        max_files_read: int | None = None,
        max_lines_read: int | None = None,
    ) -> dict[str, Any]:
        if not request or not str(request).strip():
            return {
                "status": "error",
                "error": {
                    "type": "invalid_arguments",
                    "message": "request is required",
                    "recoverable": True,
                },
            }

        budget = self.budget.apply_overrides(
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            max_context_tokens=max_context_tokens,
            max_files_read=max_files_read,
            max_lines_read=max_lines_read,
        )

        metrics: dict[str, Any] = {
            "tool_calls": 0,
            "iterations": 0,
            "files_inspected": 0,
            "files_read": 0,
            "lines_read": 0,
            "candidates_found": 0,
            "candidates_discarded": 0,
            "repeated_calls": 0,
            "estimated_input_tokens": 0,
            "estimated_output_tokens": 0,
            "final_context_tokens": 0,
            "elapsed_time": 0.0,
        }
        read_paths: set[str] = set()
        candidates: set[str] = set()
        inspected: set[str] = set()
        executed_calls: dict[tuple[str, str, str], int] = {}
        repeat_streak = 0

        start = time.monotonic()
        self.memory.clear()
        current = str(request).strip()
        self.memory.add_user(current)

        report: str | None = None
        status = "partial"
        reason = ""

        with _working_dir(self.root):
            for index in range(budget.max_iterations):
                metrics["iterations"] = index + 1
                config = self._config(current, budget)
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

                call = parse_tool_call(response)
                if call is None:
                    candidate = response.strip()
                    # Weak models often "report" after the first cheap query
                    # without reading anything. If no file has been read yet and
                    # the candidate admits it, feed it back once to investigate
                    # instead of accepting a thin report.
                    if (
                        len(read_paths) == 0
                        and metrics["tool_calls"] < 2
                        and _looks_uninspected(candidate)
                    ):
                        current = (
                            f"Their interim answer:\n{candidate}\n\n"
                            "That answer says the key contents were NOT "
                            "inspected. You must inspect them (read_file the "
                            "central files) before reporting. Continue with the "
                            "discovery tools."
                        )
                        continue
                    report = candidate
                    status = "success"
                    break

                # Reuse results: skip re-reading a file already fully inspected
                # (a narrow line-range read of an inspected file is new
                # evidence and is allowed).
                if call["tool"] == "read_file":
                    params = call.get("params", {})
                    fp = str(params.get("file_path", ""))
                    full_read = not params.get("start_line")
                    if fp and fp in read_paths and full_read:
                        metrics["repeated_calls"] += 1
                        repeat_streak += 1
                        if repeat_streak >= 3:
                            reason = "repeated identical calls without progress"
                            break
                        current = (
                            f"You already inspected {fp}; its content is still "
                            "in the conversation. Do NOT re-read it. Reuse the "
                            "earlier result to pick the next action, or write "
                            "the final report."
                        )
                        continue

                # Reuse results: do not re-run an identical search/list/grep.
                params_key = _call_params_key(call.get("params", {}))
                call_key = (call["tool"], call["action"], params_key)
                if call_key in executed_calls:
                    metrics["repeated_calls"] += 1
                    repeat_streak += 1
                    if repeat_streak >= 3:
                        reason = "repeated identical calls without progress"
                        break
                    current = (
                        f"That exact call (tool={call['tool']}, "
                        f"action={call['action']}, params={params_key}) was "
                        "already executed and its result is still in the "
                        "conversation. Do NOT repeat it: reuse the earlier "
                        "result to decide the next action, or write the final "
                        "report."
                    )
                    continue
                executed_calls[call_key] = 1
                repeat_streak = 0

                result = self.registry.dispatch(
                    call["tool"], call["action"], **call.get("params", {})
                )
                metrics["tool_calls"] += 1

                if call["tool"] == "read_file":
                    path = str(
                        result.get("path") or call["params"].get("file_path", "")
                    )
                    if path:
                        read_paths.add(path)
                    metrics["lines_read"] += _result_lines(result)

                _collect_paths(result, inspected)
                if call["tool"] in ("list_dir", "search_files", "grep_files"):
                    _collect_paths(result, candidates)
                self.memory.add_tool(
                    call["tool"],
                    call["action"],
                    call.get("params", {}),
                    _storage_result(result),
                )
                self.memory.add_assistant(response)

                if metrics["tool_calls"] >= budget.max_tool_calls:
                    reason = f"tool call limit reached ({budget.max_tool_calls})"
                    break
                if len(read_paths) >= budget.max_files_read:
                    reason = f"files read limit reached ({budget.max_files_read})"
                    break
                if metrics["lines_read"] >= budget.max_lines_read:
                    reason = f"lines read limit reached ({budget.max_lines_read})"
                    break

                inspected_list = ", ".join(sorted(read_paths)) or "(none)"
                current = (
                    f"Tool result:\n{result}\n\n"
                    "Before your next action, ask yourself: what do I already "
                    "know? What do I still need to know? Which piece of "
                    "information would reduce my uncertainty the most? Which "
                    "tool obtains it at the lowest cost?\n"
                    f"Already inspected: {inspected_list}\n"
                    "If you now have enough evidence (see the stop checklist in "
                    "your instructions), give your final DISCOVERY RESULT "
                    "report now (plain text, no JSON block). Otherwise keep "
                    "investigating with the discovery tools."
                )

        if report is None:
            reason = reason or f"iteration limit reached ({budget.max_iterations})"
            report = self._fallback_report(str(request).strip(), reason)

        metrics["files_inspected"] = len(inspected)
        metrics["files_read"] = len(read_paths)
        metrics["candidates_found"] = len(candidates)
        metrics["candidates_discarded"] = max(0, len(candidates) - len(read_paths))
        metrics["final_context_tokens"] = estimate_tokens(report)
        metrics["efficiency"] = round(
            metrics["final_context_tokens"]
            / max(
                1,
                metrics["estimated_input_tokens"] + metrics["estimated_output_tokens"],
            ),
            4,
        )
        metrics["elapsed_time"] = round(time.monotonic() - start, 3)

        return {
            "status": status,
            "task": str(request).strip(),
            "workspace": self.root,
            "report": report,
            "metrics": metrics,
            **({"reason": reason} if reason else {}),
        }


__all__ = ["DISCOVERY_INSTRUCTIONS", "DiscoveryAgent"]
