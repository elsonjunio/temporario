from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from src.navigation import create_navigation_tool
from src.tools.registry import ToolRegistry
from src.utils import AGENT_MODES

MODE_NAMES = ("fast", "balanced", "precision")

#: Subagents are disabled - kept empty for compatibility with apply_mode.
SUBAGENT_NAMES: tuple[str, ...] = ()

#: Tools registered per execution mode (beyond the base tools + navigation).
#: Subagents (orchestrator/planner/discovery/executor) are not registered, so
#: every mode uses only the base toolset.
MODE_TOOLS: dict[str, tuple[str, ...]] = {
    "fast": (),
    "balanced": (),
    "precision": (),
}

MODE_DESCRIPTIONS: dict[str, str] = {
    "fast": (
        "apenas ferramentas base + navigation; sem subagents de "
        "orquestração/descoberta/planejamento."
    ),
    "balanced": (
        "apenas ferramentas base + navigation; sem subagents de "
        "orquestração/descoberta/planejamento."
    ),
    "precision": (
        "apenas ferramentas base + navigation; sem subagents de "
        "orquestração/descoberta/planejamento."
    ),
}

_SECTION_RE = re.compile(r"^[A-Z][A-Z0-9 ]+$")
_ENTRY_RE = re.compile(r"^\s*\d+\.\s+(\S+)\s*$")
_PATH_LIKE_RE = re.compile(r"^[\w./\\-]+$")


def _report_candidates(report: str, root: str) -> list[dict[str, Any]]:
    """Extract candidates from a DISCOVERY RESULT report.

    Parses the numbered list under RELEVANT FILES (path plus the indented
    Purpose/Relevance lines as snippet). Falls back to path-like lines under
    IMPLEMENTATION AREA when RELEVANT FILES is empty.
    """
    root_path = Path(root).resolve()
    lines = report.splitlines()
    entries: list[tuple[str, str]] = []
    section: str | None = None
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if _SECTION_RE.match(stripped):
            section = stripped.lower()
            i += 1
            continue
        if section == "relevant files":
            match = _ENTRY_RE.match(lines[i])
            if match:
                path = match.group(1).rstrip(".,;")
                snippet_lines: list[str] = []
                j = i + 1
                while j < n:
                    nxt = lines[j].strip()
                    if not nxt or _SECTION_RE.match(nxt) or _ENTRY_RE.match(lines[j]):
                        break
                    snippet_lines.append(nxt)
                    j += 1
                entries.append((path, " | ".join(snippet_lines)))
        i += 1

    if not entries:
        section = None
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if _SECTION_RE.match(stripped):
                section = stripped.lower()
                continue
            if section == "implementation area" and _PATH_LIKE_RE.match(stripped):
                entries.append((stripped, ""))

    seen: dict[str, dict[str, Any]] = {}
    for raw, snippet in entries:
        resolved = (root_path / raw).resolve()
        key = str(resolved)
        if key in seen:
            continue
        if resolved.exists():
            ctype = "dir" if resolved.is_dir() else "file"
        else:
            ctype = "new_file"
        seen[key] = {
            "name": resolved.name,
            "path": key,
            "type": ctype,
            "snippet": snippet,
        }
    return list(seen.values())


def make_discovery_fallback(
    registry: ToolRegistry, root: str = "."
) -> Callable[..., dict[str, Any]]:
    """Build the orchestrator discovery fallback (balanced mode).

    Invoked automatically when the orchestrator's internal keyword discovery
    returns weak evidence. Dispatches the discovery subagent and converts its
    DISCOVERY RESULT report into an evidence dict compatible with the internal
    one (candidates keyed by path). No-ops (returns poor evidence) when the
    discovery subagent is not registered, so fast mode behaves as before.
    """

    def fallback(
        request: str,
        terms: list[str] | None,
        paths: list[str] | None,
    ) -> dict[str, Any]:
        if "discovery" not in registry.list_tools():
            return {
                "status": "poor",
                "request": request,
                "terms": terms or [],
                "reason": "discovery subagent not registered",
                "count": 0,
                "candidates": [],
            }
        result = registry.dispatch("discovery", "run", request=request)
        if result.get("status") not in ("success", "partial"):
            error = result.get("error", {})
            message = error.get("message") if isinstance(error, dict) else str(error)
            return {
                "status": "poor",
                "request": request,
                "terms": terms or [],
                "reason": message or "discovery subagent failed",
                "count": 0,
                "candidates": [],
            }
        report = str(result.get("report", ""))
        candidates = _report_candidates(report, root)
        if not candidates:
            return {
                "status": "poor",
                "request": request,
                "terms": terms or [],
                "reason": "discovery subagent found no relevant files",
                "count": 0,
                "candidates": [],
            }
        return {
            "status": "ok",
            "request": request,
            "terms": terms or [],
            "reason": "evidence filled by the discovery subagent",
            "count": len(candidates),
            "candidates": candidates,
            "report": report,
            "source": "discovery_subagent",
        }

    return fallback


def build_agent_specs(
    registry: ToolRegistry,
    provider: Any,
    memory: Any,
    *,
    root: str,
    max_plan_steps: int = 8,
    verbose: bool = False,
    interactive: bool = True,
) -> dict[str, Any]:
    """Build the special tool specs once (none registered yet).

    The orchestrator always gets the discovery fallback closure; it no-ops in
    fast mode (subagent not registered) and the orchestrator is not registered
    at all in precision mode.
    """
    specs: dict[str, Any] = {
        "navigation": create_navigation_tool(provider),
    }
    return specs


def apply_mode(
    registry: ToolRegistry,
    mode: str,
    specs: dict[str, Any],
) -> list[str]:
    """Switch the registry to an execution mode.

    Unregisters every subagent (idempotent) and registers the ones the mode
    uses. The base tools and the browser stay registered in every mode.
    Returns the resulting list of registered tools.
    """
    for name in SUBAGENT_NAMES:
        registry.unregister(name)
    if "navigation" in specs:
        registry.register(specs["navigation"].name, specs["navigation"])
    for name in MODE_TOOLS[mode]:
        registry.register(specs[name].name, specs[name])
    return registry.list_tools()


def discard_pending(registry: ToolRegistry) -> None:
    """Discard any pending orchestrator plan/snapshots before unregistering it."""
    if "orchestrator" in registry.list_tools():
        registry.dispatch("orchestrator", "abort")


def switch_mode(
    registry: ToolRegistry,
    specs: dict[str, Any],
    agent: Any,
    current: str,
    raw: str,
    instructions: dict[str, str] | None = None,
) -> str:
    """Handle the ``/mode`` REPL command; returns the active mode name."""
    instruction_texts = instructions if instructions is not None else AGENT_MODES
    parts = raw.split()
    if len(parts) == 1:
        print(f"Modo atual: {current} — {MODE_DESCRIPTIONS[current]}")
        print(f"Tools ativas: {', '.join(sorted(registry.list_tools()))}")
        return current
    new_mode = parts[1].lower()
    if new_mode not in MODE_NAMES:
        print(f"Modo inválido: {new_mode}. Opções: {', '.join(MODE_NAMES)}")
        return current
    if new_mode == current:
        print(f"Já está no modo {new_mode}.")
        return current
    if agent._pending_call is not None:
        agent._pending_call = None
    discard_pending(registry)
    tools = apply_mode(registry, new_mode, specs)
    agent.instructions = instruction_texts[new_mode]
    print(f"Modo: {new_mode} — {MODE_DESCRIPTIONS[new_mode]}")
    print(f"Tools ativas: {', '.join(sorted(tools))}")
    return new_mode
