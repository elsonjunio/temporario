from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from typing import Any

from src.orchestrator.languages import (
    build_command,
    detect_language,
    find_test_files,
    recommend_framework,
    runner_command,
    success_marker_for,
    suggest_test,
)
from src.tools._fs import resolve_path
from src.tools.patch_file import _find_block, _hunk_sides, _parse_unified_diff
from src.tools.registry import ToolRegistry

TOOL_CONTRACT = ("MANUAL", "SPEC", "get_manual", "dispatch")


def _is_generated_patch(step: dict[str, Any]) -> bool:
    """True when the patch payload is generated at execution time from the
    current disk content (step carries an ``instruction`` and no explicit
    anchors), so plan-time anchor checks do not apply."""
    params = step.get("params") or {}
    return (
        "old" not in params and "diff" not in params and bool(params.get("instruction"))
    )


class ImpactAssessor:
    """Deterministic target profiling and impact analysis.

    Tells the planner whether a target should be created, patched or rewritten,
    and what surface (module contract, cross-references, unit tests) must not
    break. Also guards plans: a ``write_file`` over an existing registered tool
    module is rejected unless the step opts into a full rewrite with
    ``rewrite: true``.
    """

    def __init__(self, registry: ToolRegistry, root: str = ".") -> None:
        self.registry = registry
        self.root = str(Path(root).resolve())

    # -- module identity ----------------------------------------------------

    def _module_name(self, path: Path) -> str | None:
        if path.suffix != ".py":
            return None
        try:
            rel = path.resolve().relative_to(self.root)
        except ValueError:
            return None
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]
        if not parts:
            return None
        return ".".join(parts)

    def _is_registered_tool(self, path: Path) -> bool:
        if path.suffix != ".py":
            return False
        try:
            resolved = path.resolve()
        except OSError:
            return False
        for name in self.registry.list_tools():
            module = self.registry.get(name)
            module_file = getattr(module, "__file__", None)
            if module_file and Path(module_file).resolve() == resolved:
                return True
        return False

    # -- static structure ---------------------------------------------------

    @staticmethod
    def _top_level_defs(path: Path) -> list[str]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            return []
        names: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(f"def {node.name}")
            elif isinstance(node, ast.ClassDef):
                names.append(f"class {node.name}")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.append(f"{target.id} = ...")
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.append(f"{node.target.id} = ...")
        return names

    def _contract(self, module_name: str) -> dict[str, Any]:
        try:
            module = importlib.import_module(module_name)
            importlib.reload(module)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            return {"import_error": f"{type(exc).__name__}: {exc}"}
        return {attr: hasattr(module, attr) for attr in TOOL_CONTRACT}

    # -- profile ------------------------------------------------------------

    def profile(self, path: str) -> dict[str, Any]:
        target = resolve_path(path)
        exists = target.exists()
        registered = self._is_registered_tool(target) if exists else False
        is_python = target.suffix == ".py"
        language = detect_language(target)

        profile: dict[str, Any] = {
            "path": str(target),
            "name": target.name,
            "exists": exists,
            "type": "directory" if target.is_dir() else "file",
            "is_python": is_python,
            "language": language.name if language else None,
            "is_registered_tool": registered,
            "change_mode": "new" if not exists else "modify",
            "recommended_tool": "write_file" if not exists else "patch_file",
        }

        if not exists or target.is_dir():
            profile["test_files"] = []
            return profile

        try:
            with target.open("r", encoding="utf-8", errors="replace") as fh:
                profile["lines"] = sum(1 for _ in fh)
        except OSError:
            profile["lines"] = None

        profile["test_files"] = self.map_tests(target)
        profile["test_command"] = self._test_command(
            target, profile.get("test_files", [])
        )
        self._attach_test_plan(target, profile)

        if is_python:
            profile["top_level"] = self._top_level_defs(target)
            module_name = self._module_name(target)
            if registered and module_name:
                contract = self._contract(module_name)
                profile["contract"] = contract
                missing = [
                    attr for attr in TOOL_CONTRACT if contract.get(attr) is False
                ]
                profile["contract_missing"] = missing
                profile["contract_preserved"] = not missing and not contract.get(
                    "import_error"
                )

        return profile

    # -- tests --------------------------------------------------------------

    def map_tests(self, path: str | Path) -> list[str]:
        target = Path(path).resolve()
        language = detect_language(target)
        if language is None:
            return []
        return [str(f) for f in find_test_files(language, target, Path(self.root))]

    def _test_command(self, path: str | Path, test_files: list[str]) -> str | None:
        if not test_files:
            return None
        target = Path(path).resolve()
        language = detect_language(target)
        if language is None:
            return None
        runner = runner_command(language, Path(self.root))
        if runner is None:
            return None
        return build_command(
            language,
            Path(self.root),
            [Path(f) for f in test_files],
            target,
        )

    def _attach_test_plan(self, target: Path, profile: dict[str, Any]) -> None:
        """Fill in test_framework_configured plus a suggestion (configured but
        uncovered) or a recommendation (no toolchain at all)."""
        language = detect_language(target)
        if language is None:
            profile["test_framework_configured"] = False
            profile["test_recommendation"] = (
                "No recognized language or test setup. Consider a lightweight "
                "test for this change or one suggested by the user."
            )
            return
        root = Path(self.root)
        runner = runner_command(language, root)
        configured = bool(profile.get("test_files")) or runner is not None
        profile["test_framework_configured"] = configured
        if profile.get("test_files"):
            profile["test_success_marker"] = success_marker_for(language, runner)
            return
        if configured and runner:
            profile["test_suggestion"] = suggest_test(language, target, root, runner)
        else:
            profile["test_recommendation"] = recommend_framework(language, root)

    # -- impact -------------------------------------------------------------

    def _referenced_by(self, path: str | Path) -> list[str]:
        target = Path(path).resolve()
        language = detect_language(target)
        if language is None:
            return []
        extensions = language.reference_extensions or language.extensions
        src = Path(self.root) / "src"
        if not src.is_dir():
            return []
        stem = target.stem
        refs: list[str] = []
        for candidate in src.rglob("*"):
            if not candidate.is_file() or candidate.suffix.lower() not in extensions:
                continue
            if candidate.resolve() == target:
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if re.search(rf"\b{re.escape(stem)}\b", text):
                try:
                    refs.append(str(candidate.relative_to(self.root)))
                except ValueError:
                    refs.append(str(candidate))
        return refs

    def risks(self, profile: dict[str, Any]) -> list[str]:
        risks: list[str] = []
        if profile.get("is_registered_tool"):
            missing = profile.get("contract_missing", [])
            if missing:
                risks.append(f"would drop module contract attrs: {', '.join(missing)}")
            else:
                risks.append("must keep MANUAL/SPEC/get_manual/dispatch intact")
        referenced = self._referenced_by(profile.get("path", ""))
        if referenced:
            risks.append(f"imported by {', '.join(referenced)}")
        tests = profile.get("test_files", [])
        if tests:
            risks.append(f"covered by {len(tests)} test file(s)")
        return risks

    def assess(self, request: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        targets: list[dict[str, Any]] = []
        for candidate in candidates:
            target_path = candidate.get("path")
            if not target_path:
                continue
            prof = self.profile(target_path)
            if candidate.get("new_project") or candidate.get("type") == "workspace_root":
                # Greenfield root: the directory exists but the PROJECT does
                # not; everything under it will be created from scratch.
                prof["change_mode"] = "create_project"
                prof["recommended_tool"] = "write_file"
                prof["greenfield"] = True
            prof["risks"] = self.risks(prof)
            targets.append(prof)
        return {"request": request, "targets": targets}

    # -- plan guard ---------------------------------------------------------

    def _patch_anchor_issues(
        self,
        step_index: int,
        action: str,
        file_path: str,
        params: dict[str, Any],
    ) -> list[str]:
        """Deterministic check that a patch_file step's anchors actually exist
        in the current file content. Only called when the target exists on disk
        and was not mutated earlier in the plan; line numbers in ``@@`` hunks
        are tolerated (the apply logic matches by content), but an invented
        old/context block is rejected before the plan is confirmed."""
        issues: list[str] = []
        if action not in ("replace", "apply"):
            return issues
        target = resolve_path(file_path)
        if not target.is_file():
            return issues
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return issues
        if action == "replace":
            old = params.get("old")
            if not old:
                return issues
            occurrences = content.count(old)
            if occurrences == 0:
                issues.append(
                    f"step {step_index}: patch_file replace 'old' text not found "
                    f"in {file_path}. Copy the 'old' text VERBATIM from the "
                    "current file (add a read_file step and re-read it); the "
                    "step would fail and roll back at execution."
                )
            elif occurrences > 1 and not params.get("replace_all"):
                issues.append(
                    f"step {step_index}: patch_file replace 'old' text appears "
                    f"{occurrences} times in {file_path}; set replace_all=true "
                    "or provide more context, else execution would fail."
                )
            return issues
        diff = params.get("diff")
        if not diff:
            return issues
        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        for hunk in _parse_unified_diff(str(diff)):
            old_side, _new_side = _hunk_sides(hunk)
            if not old_side:
                continue
            if _find_block(lines, old_side, hunk["old_start"]) is None:
                issues.append(
                    f"step {step_index}: patch_file apply hunk "
                    f"@@ -{hunk['old_start']},{hunk['old_count']} "
                    f"+{hunk['new_start']},{hunk['new_count']} @@ not found in "
                    f"{file_path}. Copy the context/removal lines VERBATIM from "
                    "the current file (add a read_file step and re-read it); the "
                    "step would fail and roll back at execution."
                )
        return issues

    def check_plan_steps(self, steps: list[dict[str, Any]]) -> list[str]:
        issues: list[str] = []
        known: set[str] = set()
        mutated: set[str] = set()
        for index, step in enumerate(steps, start=1):
            tool = step.get("tool")
            params = step.get("params") or {}
            file_path = params.get("file_path")
            if not file_path:
                continue
            norm = str(resolve_path(file_path))
            if tool == "read_file":
                known.add(norm)
                continue
            if tool == "write_file":
                rewrite = step.get("rewrite", params.get("rewrite"))
                prof = self.profile(file_path)
                if prof.get("exists"):
                    if rewrite is False:
                        issues.append(
                            f"step {index}: write_file over existing file "
                            f"{prof['name']} with 'rewrite' explicitly false. The "
                            "write tool refuses to touch an existing file then. "
                            "Use patch_file for a targeted edit, or set "
                            "'rewrite': true to overwrite the whole file."
                        )
                    elif prof.get("is_registered_tool") and not rewrite:
                        issues.append(
                            f"step {index}: write_file over existing registered "
                            f"tool module {prof['name']} requires an explicit "
                            "rewrite. Use patch_file instead, or set "
                            "'rewrite': true and preserve "
                            "MANUAL/SPEC/get_manual/dispatch."
                        )
                known.add(norm)
                mutated.add(norm)
                continue
            if tool == "patch_file" and norm not in known:
                # Steps that carry an instruction get their concrete payload
                # generated at execution time from the CURRENT disk content
                # (dedicated patch generator), so no prior read_file is
                # required and there are no hand-written anchors to check.
                if not _is_generated_patch(step):
                    issues.append(
                        f"step {index}: patch_file {params.get('action')} on "
                        f"{file_path} without a prior read_file (or write_file) of "
                        "that exact path in the plan. Patch anchors must be copied "
                        "verbatim from the file, so add a read_file step for it "
                        "first."
                    )
            if tool == "patch_file":
                if (
                    norm in known
                    and norm not in mutated
                    and not _is_generated_patch(step)
                ):
                    issues.extend(
                        self._patch_anchor_issues(
                            index, str(step.get("action")), file_path, params
                        )
                    )
                mutated.add(norm)
                continue
            if tool in ("move_file", "delete_file"):
                mutated.add(norm)
        return issues
