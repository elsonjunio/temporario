from __future__ import annotations

import os
from typing import Any

from src.memory import Memory
from src.navigation.tools import build_browser_spec
from src.tools.registry import ToolRegistry
from src.utils import build_config_prompt, build_environment_info, parse_tool_call

SUBAGENT_INSTRUCTIONS = (
    "You are a web navigation agent. You browse live pages with the browser "
    "tool and report findings back to the main agent.\n"
    "Workflow:\n"
    "1. Discovery: open the target page, then snapshot to see the interactive "
    "elements and their refs.\n"
    "2. Planning: decide the steps needed to satisfy the request before acting.\n"
    "3. Execution: use refs from the latest snapshot (e.g. click e3, fill e2); "
    "prefer refs over CSS selectors.\n"
    "4. Diagnosis: use inspect (dom/console/errors/network) and assert to "
    "verify outcomes before concluding. inspect include=['trace'] captures a "
    "trace with extracted screenshots; include=['video'] reports the recording.\n"
    "5. When done, reply with a concise plain-text summary (no JSON block).\n"
    "Rules:\n"
    "- Reuse browser.snapshot_map when the page hasn't changed (it returns the\n"
    "  last snapshot without re-crawling); call snapshot only when the map is\n"
    "  stale, mutated or not found, or the page changed significantly.\n"
    "- If a ref action returns invalid_reference (stale) or status shows\n"
    "  mutated: the page changed since the snapshot — take a new snapshot\n"
    "  before continuing.\n"
    "- Take a new snapshot whenever the page changes significantly; refs are "
    "valid only until the next snapshot.\n"
    "- Never dump full HTML; use snapshot and inspect.\n"
    "- Use browser.assert to validate expected outcomes.\n"
    "- If the page is unreachable or a tool returns an error, report it "
    "clearly and suggest the next step.\n"
    "- Each browser call must be a single JSON block:\n"
    '  {"tool": "browser", "action": "<action>", "params": {<arguments>}}'
)


class NavigationSubAgent:
    """Autonomous browser agent: takes a natural-language navigation task,
    discovers the page, plans and executes steps with the browser toolset and
    returns a compact structured summary. It owns a private registry that only
    contains the ``browser`` spec (no recursive delegation)."""

    def __init__(
        self,
        provider: Any,
        controller: Any,
        *,
        max_steps: int = 12,
        environment: str | None = None,
    ) -> None:
        self.provider = provider
        self.controller = controller
        self.max_steps = max_steps
        self.environment = environment or build_environment_info(
            workspace=str(os.getcwd())
        )
        self.memory = Memory()
        self.registry = ToolRegistry()
        spec = build_browser_spec(controller, name="browser", run_handler=None)
        self.registry.register(spec.name, spec)

    def reset(self) -> None:
        self.memory.clear()

    def _page_summary(self) -> dict[str, Any]:
        try:
            status = self.controller.status()
        except Exception:
            return {}
        summary = {
            "open": status.get("open"),
            "url": status.get("url") or "",
            "ref_count": status.get("ref_count", 0),
            "latest_snapshot": (status.get("latest_snapshot") or "")[:2000],
        }
        try:
            snapshot_map = self.controller.snapshot_map()
        except Exception:
            snapshot_map = {}
        if snapshot_map and snapshot_map.get("found"):
            content = str(snapshot_map.get("content") or "")[:3000]
            if content:
                summary["page_map"] = content
        return summary

    def run(
        self,
        request: str,
        *,
        url: str | None = None,
        max_steps: int | None = None,
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

        initial = str(request).strip()
        if url:
            initial = (
                f"Task: {initial}\nStart by opening this URL with browser.open: {url}"
            )
        self.memory.add_user(initial)

        iterations = max_steps or self.max_steps
        current_prompt = initial
        result: dict[str, Any] = {}

        for index in range(iterations):
            config = build_config_prompt(
                tool_manuals=self.registry.get_manual(),
                memory=self.memory,
                instructions=SUBAGENT_INSTRUCTIONS,
                max_history_entries=40,
                environment=self.environment,
            )
            response = self.provider.infer(current_prompt, config)
            call = parse_tool_call(response)
            if call is None:
                return {
                    "status": "success",
                    "message": response,
                    "iterations": index + 1,
                    **_page_info_summary(self._page_summary()),
                }
            result = self.registry.dispatch(
                call["tool"], call["action"], **call["params"]
            )
            self.memory.add_tool(call["tool"], call["action"], call["params"], result)
            self.memory.add_assistant(response)
            current_prompt = (
                f"Tool result:\n{result}\n\n"
                "If that answers the request, give your final plain-text "
                "summary now (no JSON block). Otherwise keep working with the "
                "browser tools."
            )

        return {
            "status": "partial",
            "message": (
                f"Reached the maximum of {iterations} browser steps without a "
                f"final answer. Last tool result: {result}"
            ),
            "iterations": iterations,
            **_page_info_summary(self._page_summary()),
        }


def _page_info_summary(page_summary: dict[str, Any]) -> dict[str, Any]:
    out = {}
    if page_summary.get("open"):
        out["url"] = page_summary.get("url", "")
        out["ref_count"] = page_summary.get("ref_count", 0)
        snapshot = page_summary.get("latest_snapshot")
        if snapshot:
            out["snapshot"] = snapshot
    return out
