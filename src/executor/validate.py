from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.context import estimate_tokens
from src.executor.budget import ExecutorBudget
from src.executor.result import TaskResult
from src.orchestrator.languages import detect_language, runner_command
from src.planning.plan import as_list
from src.tools._fs import is_binary, resolve_path
from src.utils import extract_json_object

#: Tool actions whose ``params`` carry the primary target path.
_PATH_PARAM_KEYS: dict[str, str] = {
    "write_file": "file_path",
    "patch_file": "file_path",
    "move_file": "destination",
    "delete_file": "path",
}


def changed_targets(tool_calls: list[dict[str, Any]]) -> list[str]:
    """Collect the paths mutating tools reported touching during a task."""
    changed: list[str] = []
    for call in tool_calls:
        tool = call.get("tool")
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        path = result.get("path")
        if tool in _PATH_PARAM_KEYS and isinstance(path, str) and path.strip():
            changed.append(path.strip())
        elif tool == "move_file":
            destination = result.get("destination") or call.get("params", {}).get(
                "destination"
            )
            if isinstance(destination, str) and destination.strip():
                changed.append(destination.strip())
    return changed


# -- criterion classification -------------------------------------------------

_HEX_COLOR_RE = re.compile(r"#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")

_NEGATIVE_MARKERS = (
    "não",
    "nao",
    "removid",
    "ausent",
    "deleted",
    "no longer",
    "not present",
    "inexistent",
    "exclu",
)
_EXISTENCE_MARKERS = (
    "existe",
    "existir",
    "existam",
    "presente",
    "criad",
    "gerad",
    "contém",
    "contem",
    "criar",
)
_BUILD_MARKERS = ("build", "compil", "builda")
_TEST_MARKERS = ("test", "teste")
_LINT_MARKERS = ("lint", "flake8", "eslint", "pylint", "ruff")
_TYPE_MARKERS = ("type", "tipo", "tipagem", "typing")
_BROWSER_MARKERS = (
    "browser",
    "navegador",
    "pagina",
    "página",
    "abrir",
    "site",
    "url",
    "clique",
    "click",
    "dom",
    "console",
    "network",
    "screenshot",
    "renderiz",
    "viewport",
)
_VISUAL_MARKERS = _BROWSER_MARKERS + (
    "visual",
    "visualmente",
    "aparec",
    "exib",
    "tela",
    "interface",
    "cor ",
    " cores",
    "fundo",
    "background",
    "display",
    "hover",
    "cabeçalho",
    "cabecalho",
    "navbar",
    "menu",
    "header",
)

_COLOR_JS_TEMPLATE = (
    "(() => {"
    "const t=document.createElement('div');"
    "t.style.backgroundColor='{target}';"
    "const target=t.style.backgroundColor;"
    "const out=[];"
    "for (const el of document.querySelectorAll('*')){"
    "const bg=getComputedStyle(el).backgroundColor;"
    "if(bg===target){"
    "out.push({tag:el.tagName,id:el.id||'',cls:typeof el.className==='string'?el.className:''});"
    "}}"
    "return out.slice(0,10);})()"
)


def _hex_colors(text: str) -> list[str]:
    return re.findall(_HEX_COLOR_RE, text)


def _categorize(criterion: str) -> set[str]:
    """Classify a criterion into the heuristic families that can resolve it."""
    text = criterion.lower()
    cats: set[str] = set()
    if _hex_colors(text):
        cats.add("color")
    if any(marker in text for marker in _NEGATIVE_MARKERS):
        cats.add("negative")
    if any(marker in text for marker in _EXISTENCE_MARKERS):
        cats.add("existence")
    if any(marker in text for marker in _BUILD_MARKERS):
        cats.add("build")
    if any(marker in text for marker in _TEST_MARKERS):
        cats.add("test")
    if any(marker in text for marker in _LINT_MARKERS):
        cats.add("lint")
    if any(marker in text for marker in _TYPE_MARKERS):
        cats.add("type")
    if any(marker in text for marker in _BROWSER_MARKERS):
        cats.add("browser")
    if any(marker in text for marker in _VISUAL_MARKERS):
        cats.add("visual")
    return cats


def _distinctive_tokens(text: str) -> list[str]:
    """Quoted tokens and CSS variables a heuristic can grep for."""
    tokens = re.findall(r'["\']([^"\']{2,80})["\']', text)
    tokens += re.findall(r"--[a-zA-Z][\w-]*", text)
    seen: set[str] = set()
    unique: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


#: Keywords that identify a recorded command as build/test/lint/type work.
_COMMAND_KINDS: dict[str, tuple[str, ...]] = {
    "build": ("build", "compile", "compileall", "webpack", "tsc", "vite", "make"),
    "test": (
        "pytest",
        "unittest",
        "jest",
        "vitest",
        "mocha",
        "go test",
        "cargo test",
        "npm test",
        "yarn test",
        "mvn test",
        "gradle test",
        "phpunit",
        "rails test",
        "dotnet test",
        "python -m test",
    ),
    "lint": ("lint", "flake8", "eslint", "pylint", "ruff"),
    "type": ("mypy", "pyright", "tsc --noEmit", "typecheck", "typing"),
}


def _command_kinds(command: str) -> set[str]:
    text = command.lower()
    kinds: set[str] = set()
    for kind, keywords in _COMMAND_KINDS.items():
        if any(keyword in text for keyword in keywords):
            kinds.add(kind)
    return kinds


# -- helpers ------------------------------------------------------------------


def _env_flag(name: str, default: bool) -> bool:
    import os

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _derive_url(task: dict[str, Any], result: TaskResult) -> str | None:
    """Discover a local URL to validate from task context or tool results."""
    texts: list[str] = []
    context = str(task.get("context") or "")
    if context:
        texts.append(context)
    for call in result.tool_calls:
        texts.append(str(call.get("params") or ""))
        res = call.get("result")
        if res:
            texts.append(str(res))
    blob = "\n".join(texts)
    for match in re.findall(r"(?:https?://[\w.\-:/?#[\]@!$&'()*+,;=%]+)", blob):
        if "localhost" in match or "127.0.0.1" in match:
            return match
    for match in re.findall(r"(?:localhost|127\.0\.0\.1):\d+", blob):
        return match
    for match in re.findall(r"(?:https?://[\w.\-:/?#[\]@!$&'()*+,;=%]+)", blob):
        return match
    return None


def _compileall_command(py_targets: list[Path]) -> str | None:
    if not py_targets:
        return None
    python = sys.executable or "python"
    return " ".join([python, "-m", "compileall", "-q"] + [str(p) for p in py_targets])


def _npm_build_command(root: Path) -> str | None:
    package = root / "package.json"
    if not package.exists():
        return None
    try:
        scripts = json.loads(package.read_text(encoding="utf-8")).get("scripts", {})
    except (OSError, ValueError):
        return None
    if isinstance(scripts, dict) and scripts.get("build"):
        return "npm run build"
    return None


def _line_matches(target: Path, needle: str) -> list[str]:
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if is_binary(target):
        return []
    needle = needle.lower()
    return [
        f"{target.name}:{index}"
        for index, line in enumerate(content.splitlines(), 1)
        if needle in line.lower()
    ]


@dataclass
class _Verdict:
    result: str
    evidence: str = ""


def _parse_verdicts(response: str) -> dict[str, _Verdict]:
    """Tolerantly parse the judge's JSON (or fallback bullet lines)."""
    verdicts: dict[str, _Verdict] = {}
    data: Any = None
    try:
        parsed = extract_json_object(response)
        if isinstance(parsed, dict):
            data = parsed.get("criteria", parsed)
        elif isinstance(parsed, list):
            data = parsed
    except Exception:  # noqa: BLE001
        data = None
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            criterion = str(item.get("criterion") or "").strip()
            verdict = str(item.get("result") or "").upper()
            if not criterion or verdict not in ("PASS", "FAIL", "UNVERIFIED"):
                continue
            evidence = item.get("evidence")
            verdicts[criterion] = _Verdict(verdict, str(evidence or "").strip())
    if verdicts:
        return verdicts
    line_re = re.compile(
        r"^\s*(PASS|FAIL|UNVERIFIED)\b\s*[:\-–—]?\s+(.+)$", re.IGNORECASE
    )
    for line in response.splitlines():
        match = line_re.match(line)
        if not match:
            continue
        result, rest = match.group(1).upper(), match.group(2).strip()
        criterion, sep, evidence = (
            rest.partition("—") if "—" in rest else (rest, "", "")
        )
        if not sep:
            criterion, sep, evidence = rest.partition(" - ")
        verdicts.setdefault(criterion.strip(), _Verdict(result, evidence.strip()))
    return verdicts


def _evidence_digest(
    task: dict[str, Any], result: TaskResult, root: str, cap: int = 4000
) -> str:
    """Compact, evidence-based digest for the LLM judge."""
    lines: list[str] = []
    objective = str(task.get("objective") or "")
    if objective:
        lines.append(f"OBJETIVO: {objective}")
    declared = [f for f in as_list(task.get("files")) if f]
    if declared:
        lines.append("ARQUIVOS DECLARADOS: " + ", ".join(declared))
    if result.changed_files:
        lines.append("ARQUIVOS ALTERADOS: " + ", ".join(result.changed_files))
    for call in result.tool_calls:
        tool = call.get("tool")
        action = call.get("action")
        res = call.get("result")
        if isinstance(res, dict):
            status = res.get("status", "?")
            body = str(res)[:400]
        else:
            status = "?"
            body = str(res)[:200]
        lines.append(f"- {tool}:{action} [{status}] {body}")
    for entry in result.changed_files[:8]:
        path = (
            resolve_path(entry)
            if Path(entry).is_absolute()
            else resolve_path(str(Path(root) / entry))
        )
        if not path.exists() or is_binary(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        excerpt = "\n".join(text.splitlines()[:40])
        lines.append(f"\n--- {entry} (trecho) ---\n{excerpt}")
    body = "\n".join(lines)
    if len(body) > cap:
        body = body[:cap] + "\n[... truncado]"
    return body


# -- validators ---------------------------------------------------------------

TaskValidatorResult = dict[str, Any]


class TaskValidator(Protocol):
    """Interface implemented by validation strategies.

    ``validate(task, result, budget=None)`` runs in the VALIDATING state and
    returns a structured ``validation_result`` with at least ``ok`` (bool) and
    ``checks`` (list of per-check dicts). The Executor transitions the task to
    COMPLETED only when ``ok`` is true, otherwise it may repair and retry up to
    ``max_retries`` times before classifying the failure.
    """

    def validate(
        self,
        task: dict[str, Any],
        result: TaskResult,
        budget: ExecutorBudget | None = None,
    ) -> TaskValidatorResult: ...


class _ValidationContext:
    """Shared state across a single validation pass (budget + recorders)."""

    def __init__(self, root: str, budget: ExecutorBudget) -> None:
        self.root = root
        self.budget = budget
        self.commands_run: list[dict[str, Any]] = []
        self.tests: list[dict[str, Any]] = []
        self.browser_steps = 0
        self.tokens_used = 0


class AcceptanceValidator:
    """Deterministic validator: structural checks + criterion heuristics.

    Checks ``tool_success`` (at least one successful tool call) and
    ``target_files_exist`` (every declared file exists or was deleted), then
    resolves each ``acceptance_criteria`` entry against recorded evidence and
    disk state:

    * hex colors / CSS tokens grepped in the changed files (positive and
      negative criteria);
    * existence / negative-existence markers ("X existe", "arquivo não existe
      mais", "removido");
    * build/test/lint/type markers matched against recorded ``run_command``
      calls;
    * quoted tokens and CSS variables grepped in the changed files.

    A criterion the heuristics cannot resolve stays UNVERIFIED; the
    ``ValidationManager`` feeds those to the targeted-command / browser / LLM
    layers. Every criterion resolves to PASS, FAIL or UNVERIFIED with concrete
    evidence and the layer that checked it.
    """

    def __init__(self, root: str = ".") -> None:
        self.root = str(Path(root).resolve())

    def _resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(self.root) / candidate
        return candidate.resolve()

    def _targets(
        self, task: dict[str, Any], result: TaskResult
    ) -> tuple[list[Path], list[Path]]:
        """Return ``(declared_paths, inspect_paths)``.

        ``declared_paths`` are the resolved declared files (existence checks);
        ``inspect_paths`` are the existing, non-binary files among declared and
        changed (content greps).
        """
        declared: list[Path] = []
        inspect: list[Path] = []
        seen: set[str] = set()
        declared_seen: set[str] = set()
        entries = [str(f) for f in as_list(task.get("files")) if f]
        for entry in entries:
            if entry in declared_seen:
                continue
            declared_seen.add(entry)
            resolved = self._resolve(entry)
            declared.append(resolved)
            if resolved.exists() and resolved.is_file() and not is_binary(resolved):
                if str(resolved) not in seen:
                    seen.add(str(resolved))
                    inspect.append(resolved)
        for entry in result.changed_files:
            if not entry or entry in declared_seen:
                continue
            declared_seen.add(entry)
            resolved = self._resolve(entry)
            if resolved.exists() and resolved.is_file() and not is_binary(resolved):
                if str(resolved) not in seen:
                    seen.add(str(resolved))
                    inspect.append(resolved)
        return declared, inspect

    @staticmethod
    def _entry(
        criterion: str, verdict: str, evidence: list[str], checked_by: str
    ) -> dict[str, Any]:
        return {
            "criterion": criterion,
            "result": verdict,
            "evidence": evidence,
            "checked_by": checked_by,
        }

    def _resolve_criterion(
        self,
        ctx: _ValidationContext,
        task: dict[str, Any],
        result: TaskResult,
        criterion: str,
        declared: list[Path],
        inspect: list[Path],
    ) -> dict[str, Any]:
        text = criterion.strip()
        cats = _categorize(criterion)
        if not text:
            return self._entry(text, "UNVERIFIED", ["criterion vazio"], "heuristics")

        colors = _hex_colors(criterion)
        if "color" in cats and colors:
            target = colors[0]
            matches = [m for path in inspect for m in _line_matches(path, target)]
            if "negative" in cats:
                if matches:
                    return self._entry(criterion, "FAIL", matches[:5], "heuristics")
                return self._entry(
                    criterion,
                    "PASS",
                    [f"cor {target} ausente dos arquivos alterados"],
                    "heuristics",
                )
            if matches:
                return self._entry(criterion, "PASS", matches[:5], "heuristics")
            return self._entry(
                criterion,
                "FAIL",
                [f"cor {target} não encontrada em nenhum arquivo alterado"],
                "heuristics",
            )

        if "existence" in cats and "negative" in cats:
            missing = [str(p) for p in declared if not p.exists()]
            if missing:
                return self._entry(
                    criterion,
                    "PASS",
                    ["arquivos ausentes: " + ", ".join(missing)],
                    "heuristics",
                )
            return self._entry(
                criterion,
                "FAIL",
                ["todos os arquivos declarados ainda existem"],
                "heuristics",
            )

        if "existence" in cats:
            missing = [str(p) for p in declared if not p.exists()]
            if missing:
                return self._entry(
                    criterion,
                    "FAIL",
                    ["arquivos ausentes: " + ", ".join(missing)],
                    "heuristics",
                )
            return self._entry(
                criterion,
                "PASS",
                [
                    "arquivos presentes: " + ", ".join(str(p) for p in declared)
                    or "(nenhum)"
                ],
                "heuristics",
            )

        if "negative" in cats and not (
            cats & {"color", "build", "test", "lint", "type"}
        ):
            present = [str(p) for p in declared if p.exists()]
            if not declared:
                return self._entry(criterion, "UNVERIFIED", [], "heuristics")
            if present:
                return self._entry(
                    criterion,
                    "FAIL",
                    ["arquivos ainda existem: " + ", ".join(present)],
                    "heuristics",
                )
            return self._entry(
                criterion, "PASS", ["nenhum arquivo declarado existe"], "heuristics"
            )

        kinds = cats & {"build", "test", "lint", "type"}
        if kinds:
            recorded = [
                call
                for call in result.tool_calls
                if call.get("tool") == "run_command"
                and isinstance(call.get("result"), dict)
            ]
            for kind in ("build", "test", "lint", "type"):
                if kind not in kinds:
                    continue
                matched = [
                    call
                    for call in recorded
                    if kind
                    in _command_kinds(str(call.get("params", {}).get("command") or ""))
                ]
                if any(
                    call["result"].get("status") == "success"
                    and call["result"].get("returncode") == 0
                    for call in matched
                ):
                    return self._entry(
                        criterion,
                        "PASS",
                        [f"{kind} registrado com sucesso no histórico"],
                        "heuristics",
                    )
                if any(call["result"].get("status") != "success" for call in matched):
                    return self._entry(
                        criterion,
                        "FAIL",
                        [f"{kind} registrado com falha no histórico"],
                        "heuristics",
                    )
            return self._entry(criterion, "UNVERIFIED", [], "heuristics")

        tokens = _distinctive_tokens(criterion)
        if tokens:
            matches = [
                f"{token} em {path.name}:{line}"
                for path in inspect
                for token in tokens
                for line in _line_matches(path, token)
            ]
            if matches:
                return self._entry(criterion, "PASS", matches[:5], "heuristics")
            if inspect:
                return self._entry(
                    criterion,
                    "FAIL",
                    ["token(s) não encontrado(s) nos arquivos alterados"],
                    "heuristics",
                )
            return self._entry(criterion, "UNVERIFIED", [], "heuristics")

        return self._entry(criterion, "UNVERIFIED", [], "heuristics")

    def validate(
        self,
        task: dict[str, Any],
        result: TaskResult,
        budget: ExecutorBudget | None = None,
    ) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []

        successful = [
            call
            for call in result.tool_calls
            if isinstance(call.get("result"), dict)
            and call["result"].get("status") == "success"
        ]
        checks.append(
            {
                "check": "tool_success",
                "ok": len(successful) > 0,
                "detail": f"{len(successful)} successful tool call(s)",
            }
        )

        declared, inspect = self._targets(task, result)
        deleted = {
            str(self._resolve(path))
            for call in result.tool_calls
            if call.get("tool") == "delete_file"
            for path in [call.get("params", {}).get("path", "")]
            if path
        }
        missing = [
            path for path in declared if str(path) not in deleted and not path.exists()
        ]
        checks.append(
            {
                "check": "target_files_exist",
                "ok": not missing,
                "detail": (
                    ", ".join(str(p) for p in missing)
                    if missing
                    else "all declared files exist"
                ),
            }
        )

        ctx = _ValidationContext(self.root, budget or ExecutorBudget())
        entries = [
            self._resolve_criterion(ctx, task, result, criterion, declared, inspect)
            for criterion in as_list(task.get("acceptance_criteria"))
        ]

        ok = all(check["ok"] for check in checks) and all(
            entry["result"] == "PASS" for entry in entries
        )
        fail_parts = [check["check"] for check in checks if not check["ok"]]
        fail_parts += [
            f"{entry['criterion']} ({entry['result']})"
            for entry in entries
            if entry["result"] != "PASS"
        ]
        summary = (
            "acceptance checks passed"
            if ok
            else "acceptance validation failed: " + "; ".join(fail_parts)
        )
        return {
            "ok": ok,
            "checks": checks,
            "summary": summary,
            "acceptance_results": entries,
            "tests": ctx.tests,
            "commands": ctx.commands_run,
            "evidence": [],
            "errors": [],
            "warnings": [],
            "status": "PASS" if ok else "FAIL",
            "tokens_used": ctx.tokens_used,
        }


class LLMCriterionJudge:
    """Final, provider-driven judge for criteria the heuristics left open.

    Only ever invoked for UNVERIFIED criteria, with a compact evidence digest
    (objective, declared/changed files, recorded tool results, file excerpts).
    The verdicts are parsed tolerantly (JSON, or PASS/FAIL bullets); a malformed
    or missing verdict keeps the criterion UNVERIFIED, which the aggregator
    counts as FAIL (conservative).
    """

    def __init__(self, provider: Any, root: str = ".", max_chars: int = 4000) -> None:
        self.provider = provider
        self.root = root
        self.max_chars = max_chars

    @staticmethod
    def _config() -> str:
        return (
            "Você é um avaliador criterioso de resultados de implementação. "
            "Classifique exclusivamente os critérios listados, usando SOMENTE "
            "as evidências fornecidas. Responda APENAS com JSON válido:\n"
            '{"criteria":[{"criterion":"<texto exato>","result":"PASS|FAIL",'
            '"evidence":"<síntese>", "reasoning":"<1 linha>"}]}'
        )

    def judge(
        self,
        ctx: _ValidationContext,
        task: dict[str, Any],
        result: TaskResult,
        entries: list[dict[str, Any]],
    ) -> int:
        open_entries = [e for e in entries if e["result"] == "UNVERIFIED"]
        if not open_entries:
            return 0
        digest = _evidence_digest(task, result, self.root, self.max_chars)
        criteria = [e["criterion"] for e in open_entries]
        prompt = (
            "EVIDÊNCIAS:\n"
            f"{digest}\n\n"
            "CRITÉRIOS A CLASSIFICAR:\n"
            + "\n".join(f"- {c}" for c in criteria)
            + "\n\nResponda apenas o JSON."
        )
        config = self._config()
        try:
            response = self.provider.infer(prompt, config)
        except Exception as exc:  # noqa: BLE001
            ctx.tokens_used += estimate_tokens(prompt + config)
            for entry in open_entries:
                entry["warnings"] = entry.get("warnings", []) + [
                    f"juiz LLM indisponível: {exc}"
                ]
            return 0
        ctx.tokens_used += estimate_tokens(prompt + config)

        verdicts = _parse_verdicts(str(response))
        applied = 0
        by_index = {e["criterion"]: e for e in open_entries}
        for criterion, verdict in verdicts.items():
            target = by_index.get(criterion)
            if target is None:
                continue
            if verdict.result == "UNVERIFIED":
                continue
            target["result"] = "PASS" if verdict.result == "PASS" else "FAIL"
            target["checked_by"] = "llm"
            if verdict.evidence:
                target["evidence"].append(f"llm: {verdict.evidence}")
            applied += 1
        if not verdicts or applied < len(open_entries):
            for entry in open_entries:
                if entry["result"] == "UNVERIFIED":
                    entry["warnings"] = entry.get("warnings", []) + [
                        "juiz LLM não resolveu este critério; conta como FAIL"
                    ]
        return applied


class FunctionalBrowserValidator:
    """Functional (browser) validation for visual criteria.

    Reuses recorded ``browser:*`` evidence first; a bounded live pass
    (open -> evaluate/snapshot/network, capped by the budget) is launched only
    for criteria the recorded evidence did not resolve. The URL is discovered
    from the task context / tool results. Browser unavailable or no URL -> the
    criteria stay UNVERIFIED with a warning (conservative FAIL downstream).
    """

    def __init__(self, root: str = ".", registry: Any = None) -> None:
        self.root = root
        self.registry = registry

    @property
    def available(self) -> bool:
        try:
            return self.registry is not None and "browser" in self.registry.list_tools()
        except Exception:  # noqa: BLE001
            return False

    def _call(
        self, ctx: _ValidationContext, action: str, **params: Any
    ) -> dict[str, Any]:
        try:
            res = self.registry.dispatch("browser", action, **params)
        except Exception as exc:  # noqa: BLE001
            res = {"status": "failed", "message": str(exc)}
        ctx.browser_steps += 1
        ctx.commands_run.append(
            {
                "kind": "browser",
                "command": f"browser:{action}",
                "status": res.get("status") if isinstance(res, dict) else "failed",
                "returncode": None,
                "stdout": str(res)[:600],
                "stderr": "",
            }
        )
        return res if isinstance(res, dict) else {"status": "failed", "result": res}

    def phase(
        self,
        ctx: _ValidationContext,
        task: dict[str, Any],
        result: TaskResult,
        entries: list[dict[str, Any]],
    ) -> None:
        visual = [
            e
            for e in entries
            if e["result"] == "UNVERIFIED"
            and bool(_categorize(e["criterion"]) & {"visual", "browser"})
        ]
        if not visual or not self.available:
            return
        self._reuse_recorded(result, visual)
        open_entries = [e for e in visual if e["result"] == "UNVERIFIED"]
        if not open_entries:
            return
        url = _derive_url(task, result)
        if not url:
            for entry in open_entries:
                entry["warnings"] = entry.get("warnings", []) + [
                    "nenhuma URL encontrada para validação funcional"
                ]
            return
        steps = ctx.budget.max_browser_validation_steps
        opened = self._call(ctx, "open", url=url)
        if opened.get("status") != "success":
            for entry in open_entries:
                entry["warnings"] = entry.get("warnings", []) + [
                    "navegador indisponível para validação funcional"
                ]
            return
        for entry in open_entries:
            if ctx.browser_steps >= steps:
                break
            colors = _hex_colors(entry["criterion"])
            if colors:
                expression = _COLOR_JS_TEMPLATE.format(target=colors[0])
                res = self._call(ctx, "evaluate", expression=expression)
                if _eval_has_matches(res):
                    entry["result"] = "PASS"
                    entry["checked_by"] = "browser"
                    entry["evidence"].append(
                        f"cor {colors[0]} presente na página renderizada"
                    )
                else:
                    entry["result"] = "FAIL"
                    entry["checked_by"] = "browser"
                    entry["evidence"].append(
                        f"cor {colors[0]} não encontrada na página renderizada"
                    )
            else:
                res = self._call(ctx, "snapshot")
                body = str(res)
                tokens = _distinctive_tokens(entry["criterion"])
                if tokens and any(token.lower() in body.lower() for token in tokens):
                    entry["result"] = "PASS"
                    entry["checked_by"] = "browser"
                    entry["evidence"].append(
                        "conteúdo esperado presente no snapshot da página"
                    )
                else:
                    entry["result"] = "FAIL"
                    entry["checked_by"] = "browser"
                    entry["evidence"].append(
                        "conteúdo esperado ausente do snapshot da página"
                    )

    @staticmethod
    def _reuse_recorded(result: TaskResult, entries: list[dict[str, Any]]) -> None:
        recorded = [
            call
            for call in result.tool_calls
            if call.get("tool") == "browser" and isinstance(call.get("result"), dict)
        ]
        if not recorded:
            return
        for entry in entries:
            colors = _hex_colors(entry["criterion"])
            tokens = _distinctive_tokens(entry["criterion"])
            if colors:
                target = colors[0]
                if any(
                    target.lower() in str(call["result"]).lower() for call in recorded
                ):
                    entry["result"] = "PASS"
                    entry["checked_by"] = "browser"
                    entry["evidence"].append(
                        f"cor {target} presente em evidência de navegador registrada"
                    )
            elif tokens and any(
                token.lower() in str(call["result"]).lower()
                for call in recorded
                for token in tokens
            ):
                entry["result"] = "PASS"
                entry["checked_by"] = "browser"
                entry["evidence"].append(
                    "conteúdo esperado presente em evidência de navegador registrada"
                )


def _eval_has_matches(res: dict[str, Any]) -> bool:
    if res.get("status") != "success":
        return False
    value = res.get("result")
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        return bool(value.get("matches") or value.get("found"))
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return "match" in value.lower() or "encontrado" in value.lower()
    return False


class ValidationManager:
    """Full stage-3 validator: heuristics -> commands -> browser -> LLM judge.

    Orchestrates the layered validation for a task and assembles the single
    structured ``validation_result`` the Executor uses to decide COMPLETED /
    FAILED / repair. Deterministic layers run first and record evidence; the
    expensive ones (targeted commands, browser pass, LLM judge) only run for
    criteria still UNVERIFIED and are bounded by the budget.
    """

    def __init__(
        self,
        *,
        root: str = ".",
        registry: Any = None,
        provider: Any = None,
        llm: bool = True,
        max_chars: int = 4000,
    ) -> None:
        self.root = str(Path(root).resolve())
        self.registry = registry
        self.validator = AcceptanceValidator(root=self.root)
        use_llm = bool(
            llm and provider is not None and _env_flag("EXECUTOR_LLM_VALIDATION", True)
        )
        self.judge = (
            LLMCriterionJudge(provider, root=self.root, max_chars=max_chars)
            if use_llm
            else None
        )
        self.browser = FunctionalBrowserValidator(self.root, registry)

    def _commands_phase(
        self,
        ctx: _ValidationContext,
        task: dict[str, Any],
        result: TaskResult,
        entries: list[dict[str, Any]],
        inspect: list[Path],
    ) -> None:
        if self.registry is None:
            return
        unresolved = [e for e in entries if e["result"] == "UNVERIFIED"]
        needed: set[str] = set()
        for entry in unresolved:
            needed |= _categorize(entry["criterion"]) & {
                "build",
                "test",
                "lint",
                "type",
            }
        if not needed:
            return
        root = Path(self.root)
        py_targets = [p for p in inspect if p.suffix == ".py"]
        ran: set[str] = set()
        for kind in ("build", "test", "lint", "type"):
            if kind not in needed or kind in ran:
                continue
            if len(ctx.commands_run) >= ctx.budget.max_validation_commands:
                break
            command = None
            if kind == "build":
                command = _compileall_command(py_targets) or _npm_build_command(root)
            elif kind == "test":
                profile = detect_language(py_targets[0]) if py_targets else None
                if profile is not None:
                    command = runner_command(profile, root)
            if not command:
                continue
            ran.add(kind)
            record = self._run_command(ctx, command)
            record["kind"] = kind
            ctx.commands_run.append(record)
            if kind == "test":
                ctx.tests.append(record)
            ok = record["status"] == "success"
            for entry in unresolved:
                if kind not in _categorize(entry["criterion"]):
                    continue
                entry["result"] = "PASS" if ok else "FAIL"
                entry["checked_by"] = f"command:{kind}"
                entry["evidence"].append(
                    f"{kind}: {command} -> {record['status']} "
                    f"(returncode {record['returncode']})"
                )

    def _run_command(self, ctx: _ValidationContext, command: str) -> dict[str, Any]:
        timeout = ctx.budget.validation_timeout
        try:
            res = self.registry.dispatch(
                "run_command", "run", command=command, cwd=self.root, timeout=timeout
            )
        except Exception as exc:  # noqa: BLE001
            res = {
                "status": "failed",
                "returncode": None,
                "stdout": "",
                "stderr": str(exc),
            }
        return {
            "command": command,
            "status": res.get("status") if isinstance(res, dict) else "failed",
            "returncode": res.get("returncode") if isinstance(res, dict) else None,
            "stdout": str(res.get("stdout", ""))[:800],
            "stderr": str(res.get("stderr", ""))[:800],
        }

    def validate(
        self,
        task: dict[str, Any],
        result: TaskResult,
        budget: ExecutorBudget | None = None,
    ) -> dict[str, Any]:
        budget = budget or ExecutorBudget()
        ctx = _ValidationContext(self.root, budget)

        outcome = self.validator.validate(task, result, budget=budget)
        entries: list[dict[str, Any]] = outcome["acceptance_results"]
        _, inspect = self.validator._targets(task, result)

        self._commands_phase(ctx, task, result, entries, inspect)
        if self.browser is not None and self.browser.available:
            self.browser.phase(ctx, task, result, entries)

        unresolved = [e for e in entries if e["result"] == "UNVERIFIED"]
        if unresolved and self.judge is not None:
            self.judge.judge(ctx, task, result, entries)

        structural_ok = all(bool(check.get("ok")) for check in outcome["checks"])
        failing = [e for e in entries if e["result"] != "PASS"]
        ok = structural_ok and not failing
        if any(e["result"] == "UNVERIFIED" for e in entries):
            ctx_warning = "critérios UNVERIFIED contam como FAIL"
            if ctx_warning not in outcome["warnings"]:
                outcome["warnings"].append(ctx_warning)

        reason_parts = [c["check"] for c in outcome["checks"] if not c["ok"]]
        reason_parts += [f"{e['criterion']} ({e['result']})" for e in failing]
        summary = (
            "acceptance validation passed"
            if ok
            else "acceptance validation failed: " + "; ".join(reason_parts)
        )
        return {
            "status": "PASS" if ok else "FAIL",
            "ok": ok,
            "summary": summary,
            "acceptance_results": entries,
            "checks": outcome["checks"],
            "tests": ctx.tests,
            "commands": ctx.commands_run,
            "evidence": outcome["evidence"],
            "errors": outcome["errors"],
            "warnings": outcome["warnings"],
            "tokens_used": ctx.tokens_used,
        }


__all__ = [
    "AcceptanceValidator",
    "FunctionalBrowserValidator",
    "LLMCriterionJudge",
    "TaskValidator",
    "ValidationManager",
    "changed_targets",
]
