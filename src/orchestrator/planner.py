from __future__ import annotations

import json
import time
from typing import Any

from src.tools.registry import ToolRegistry
from src.utils import extract_json_object

PLANNING_SYSTEM = (
    "You are a planning engine for a filesystem agent. You only produce JSON "
    "plans that use the provided tools; you never execute anything."
)

#: Pause between attempts when the provider itself fails (transient 5xx /
#: network errors), so one flaky response does not abort the whole flow.
_PROVIDER_RETRY_SLEEP = 4


class Planner:
    """Turns a request plus discovery evidence into a validated JSON plan of
    tool steps, using the injected provider."""

    def __init__(
        self,
        provider: Any,
        registry: ToolRegistry,
        max_plan_steps: int = 8,
        max_plan_retries: int = 2,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.max_plan_steps = max_plan_steps
        self.max_plan_retries = max_plan_retries

    def build_prompt(
        self, request: str, evidence: dict[str, Any], feedback: str | None = None
    ) -> str:
        manual = self.registry.get_manual()
        evidence_block = json.dumps(evidence, ensure_ascii=False, indent=2)
        feedback_block = (
            f"\n\nREJECTION FEEDBACK for your previous plan (fix ALL of it):\n{feedback}\n"
            if feedback
            else ""
        )
        return (
            "Plan a filesystem modification.\n\n"
            f"Request:\n{request}\n\n"
            f"Discovery evidence (candidate targets, including impact):\n{evidence_block}\n\n"
            f"Available tools:\n{manual}\n\n"
            f"{feedback_block}"
            'Return ONLY a JSON object with a "steps" list. Each step is:\n'
            '{"tool": "<tool>", "action": "<action>", "params": {...}, '
            '"validate_after": true, "expect": "<optional substring to verify", '
            '"description": "..."}\n\n'
            "The response must be STRICT, VALID JSON: never put a raw newline "
            "or tab inside a string value -- escape them (\\\\n); keep every "
            "command string single-line.\n"
            "expect semantics: for write_file/patch_file it is a substring "
            "expected in the written file content, or a result keyword "
            '("created"/"overwritten"/"unchanged"/"applied"). For run_command '
            "it is a substring of stdout/stderr. Set validate_after=true only "
            "when a validation matters; the step fails and rolls back if "
            "expect is set but not found.\n"
            "For write_file/patch_file, set expect to a SHORT string you are "
            "certain appears VERBATIM in the file (an import line, a class or "
            "function name, a router prefix literal). NEVER set expect to a "
            "value assembled from separate literals, e.g. a full endpoint "
            "path like /api/auth/register when the file only holds prefix "
            "=/api/auth and route /register separately; that check will not "
            "match and the step rolls back.\n"
            "Reading a file that does not exist no longer fails the plan: "
            "the step logs a read_warn and execution continues. Before "
            "reading a file, list its directory first (list_dir) and use "
            "exact paths from that listing; never invent a filename.\n"
            "Rules:\n"
            "- Use only the available tools listed above.\n"
            "- Every step must be ATOMIC and self-contained: its params fully "
            "describe the change on their own and must never rely on results "
            "(or memory) from earlier steps.\n"
            "- Use concrete file paths from the discovery evidence.\n"
            "- If the workspace has a docs/SPEC.md (or docs/PLAYBOOK.md), add a "
            "read_file step for it first and follow its contract for the files "
            "you create or modify.\n"
            "- Before writing a module that depends on existing project files "
            "(models, schemas, services, routes), add read_file steps for those "
            "files and REUSE their exact class/enum/field/endpoint names and "
            "values; never invent new ones. When the SPEC conflicts with "
            "existing code, follow the SPEC and call out the deviation.\n"
            "- When planning a consumer of an existing backend API (frontend "
            "services, components, guards), add read_file steps for the actual "
            "backend router/schema files and derive the EXACT url paths, "
            "request payloads and response field names from that committed "
            "code. The committed backend is the source of truth for what the "
            "API returns (auth/login returns e.g. access_token, not token). "
            "NEVER invent endpoints, payloads or response shapes. If a "
            "capability the UI needs is missing from the backend (e.g. no "
            "endpoint listing customers), add a step that CREATES it in the "
            "backend (new endpoint + registration) instead of faking it "
            "client-side.\n"
            "- run_command integration/smoke checks that call an HTTP API must "
            "use paths verified against the backend source: for each endpoint "
            "called, add a read_file step for its router module and build the "
            "EXACT path as prefix + route decorator (e.g. an APIRouter with "
            "prefix '/products' and @router.get('/') means GET /products, NOT "
            "/api/products; an auth router with prefix '/api/auth' means "
            "POST /api/auth/login). The router prefix overrides any path "
            "written in the prompt, SPEC or PLAYBOOK: when those documents "
            "name a different URL, the committed code wins. When unsure, curl "
            "the app's /openapi.json first and use the exact paths listed "
            "there. Never write a smoke-test URL from memory.\n"
            "- run_command smoke checks that authenticate must NOT guess "
            "credentials or payload fields: add read_file steps for "
            "backend/app/seed.py and backend/app/config.py to copy the exact "
            "admin credentials (e.g. SEED_ADMIN_EMAIL/SEED_ADMIN_PASSWORD), "
            "and read the auth router + its request schema to use the exact "
            "login payload field names (e.g. email, not username). A 4xx/5xx "
            "response means the check itself is wrong -- fix it, do not weaken "
            "the assertion.\n"
            "- A run_command smoke/integration step that calls the HTTP API "
            "MUST derive URLs and credentials AT RUNTIME inside the same "
            "command, never hardcode them: (1) use python3 to fetch "
            "http://127.0.0.1:8000/openapi.json and pick the exact login path "
            "(e.g. the key that ends with '/auth/login') and the products "
            "list path (the key that has no '/api' segment and is exactly "
            "'/products'); (2) read backend/app/config.py and "
            "backend/app/seed.py with python3 and extract the owner "
            "email/password values that seed() actually uses (the real seed "
            "credentials are admin@volupia.com/admin123); (3) only then curl "
            "those exact paths with those exact fields. If the server answers "
            "non-200, print the response body. Never put a guessed email, "
            "password, URL or field name in the command.\n"
            "- EXISTING targets (change_mode 'modify') must be edited with "
            "patch_file (replace or apply). Use write_file ONLY to create NEW "
            "files (change_mode 'new'), or for an intentional full rewrite in "
            'which case set "rewrite": true on the step.\n'
            "- patch_file steps must NOT fabricate anchors: set params to "
            '{"file_path": <path>, "instruction": "<precise description of '
            'the change>"} (+ optional "replace_all": true) and let the '
            "runtime patch generator produce old/new from the CURRENT file "
            "content. The instruction must name the exact symbols/lines to "
            "change and what the new content should be, so a minimal unique "
            "anchor can be derived. NEVER invent 'old', 'new' or 'diff' "
            "values; never add read_file steps just to copy patch anchors "
            "(read_file is only for deriving names/values you need elsewhere, "
            "e.g. inside write_file content).\n"
            "- GREENFIELD: targets seeded with type 'new_file' do not exist yet "
            "and must be created with write_file using the exact path from the "
            "evidence. Create parent directories implicitly via write_file "
            "(create_dirs defaults to true); there is no separate mkdir tool.\n"
            '- GREENFIELD WORKSPACE (evidence has "greenfield": true and a '
            "'workspace_root' candidate): the workspace is EMPTY, so there are "
            "no existing files to read or patch — every step must create new "
            "files with write_file at exact relative-ish paths under the "
            "workspace root (backend/, frontend/, ...). Decompose the whole "
            "project into concrete, COMPLETE file contents (real working code, "
            "never placeholders). Prefer writing files directly over scaffolding "
            "commands; only use run_command for dependency installation/builds "
            "(pip install, npm install) with cwd=<workspace root> and a generous "
            "timeout (600+).\n"
            "- run_command steps that scaffold or install (pip install, npm "
            "install, ng new, git init) must set cwd to the project root and a "
            "generous timeout (600+) so they do not time out; set "
            "validate_after=true and expect a success marker when possible.\n"
            "- Registered tool modules (is_registered_tool=true) must keep "
            "their module contract: MANUAL, SPEC (name + handlers), "
            "get_manual() and dispatch(action, **params). A rewrite that drops "
            "these attributes fails validation and is rolled back.\n"
            "- Targets work in ANY language (python, typescript, javascript, "
            "java, kotlin, c, cpp, go, rust, ruby, php, csharp, swift, ...). "
            "The impact block reports the language of each target. Follow the "
            "language's idioms; never assume Python conventions.\n"
            "- Tests for changed modules are NOT mandatory (best effort): a "
            "failing test step must NEVER roll back the implemented changes.\n"
            "  * If the impact block lists test_files with a test_command for a "
            "changed module, add a final run_command step running that "
            "test_command verbatim, with validate_after=true, "
            'expect=<the impact block\'s success marker> and "soft": true.\n'
            "  * If test_files is empty but test_framework_configured=true and "
            "a test_suggestion exists, add final steps that (1) write_file the "
            "suggested new_test_path with a unit test covering the change "
            "(change_mode 'new', \"soft\": true), then (2) run_command the "
            "suggested command with validate_after=true, expect=<success "
            'marker> and "soft": true.\n'
            "  * If test_framework_configured=false, the impact block carries a "
            "test_recommendation. Do NOT auto-create a test harness and do NOT "
            "run any test command. The execution result will tell the user the "
            "change could not be tested and ask whether to keep the changes or "
            "apply a custom test.\n"
            "- Prefer patch_file/write_file for edits; delete_file and "
            "move_file only when required.\n"
            f"- At most {self.max_plan_steps} steps.\n"
            "- No markdown, no prose: pure JSON only."
        )

    def _validate_steps(self, steps: Any) -> str | None:
        if not isinstance(steps, list) or not steps:
            return "plan must contain a non-empty 'steps' list"
        if len(steps) > self.max_plan_steps:
            return f"plan exceeds the {self.max_plan_steps} step limit"
        available = set(self.registry.list_tools())
        for i, step in enumerate(steps, 1):
            if not isinstance(step, dict):
                return f"step {i} is not an object"
            tool = step.get("tool")
            action = step.get("action")
            params = step.get("params", {})
            if not isinstance(tool, str) or tool not in available:
                return f"step {i}: unknown tool {tool!r}"
            if not isinstance(action, str) or not action:
                return f"step {i}: missing action"
            if not isinstance(params, dict):
                return f"step {i}: params must be an object"
        return None

    def plan(
        self,
        request: str,
        evidence: dict[str, Any],
        feedback: str | None = None,
    ) -> dict[str, Any]:
        prompt = self.build_prompt(request, evidence, feedback=feedback)
        attempts = 0
        while True:
            attempts += 1
            try:
                raw = self.provider.infer(prompt, PLANNING_SYSTEM)
            except Exception as exc:
                # Provider/network failure (transient 5xx, timeouts...):
                # retry instead of aborting the whole orchestrator run.
                if attempts < self.max_plan_retries + 1:
                    time.sleep(_PROVIDER_RETRY_SLEEP * attempts)
                    continue
                return {
                    "status": "error",
                    "message": f"provider failed after {attempts} attempts: {exc}",
                }

            data = extract_json_object(raw)
            if isinstance(data, dict) and "steps" in data:
                steps = data["steps"]
                problem = self._validate_steps(steps)
                if problem is None:
                    return {"status": "ok", "steps": steps}
                if attempts < self.max_plan_retries:
                    continue
                return {"status": "error", "message": problem, "raw": raw[:500]}

            if attempts < self.max_plan_retries:
                continue
            return {
                "status": "error",
                "message": "provider did not return a JSON plan with 'steps'",
                "raw": raw[:500],
            }
