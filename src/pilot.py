from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.context import ContextCompressor
from src.memory import Memory
from src.orchestrator.orchestrator import Orchestrator
from src.providers.opencode import OpenCodeProvider, ProviderError
from src.tools.registry import build_default_registry
from src.utils import build_environment_info


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive the agent through the phases of docs/phases.json."
    )
    parser.add_argument(
        "--root", required=True, help="project root (orchestrator root)"
    )
    parser.add_argument("--phases", required=True, help="path to the phases.json file")
    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        help="start at this phase index (0-based) to resume after a failure",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="max tool iterations per agent turn",
    )
    parser.add_argument(
        "--max-plan-steps",
        type=int,
        default=15,
        help="max steps per orchestrator plan",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="append a structured run log to this file",
    )
    return parser.parse_args(argv)


def load_phases(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["phases"]


def _fmt_plan(res: dict[str, Any]) -> str:
    lines = []
    for step in res.get("summary", []):
        lines.append(f"  {step.get('step')}. {step.get('description')}")
        params = step.get("params")
        if params:
            lines.append(f"      params: {params}")
    impact = res.get("impact")
    if impact:
        lines.append(impact)
    return "\n".join(lines)


def _run_once(
    orch: Any,
    prompt: str,
    paths: list[str] | None,
    note: Any,
    sticky: bool = False,
) -> tuple[bool, dict[str, Any]]:
    try:
        plan = orch.run(prompt, paths=paths)
    except ProviderError as exc:
        note(f"[FAIL] provider error: {exc}")
        return False, {}

    if plan.get("status") != "awaiting_confirmation":
        note(
            f"[STOP] plan status = {plan.get('status')}: {plan.get('reason') or plan.get('message')}"
        )
        note(json.dumps(plan, ensure_ascii=False, default=str)[:2000])
        return False, {}

    note("PLAN (awaiting confirmation):")
    note(_fmt_plan(plan))
    note(f"PLAN: {len(plan.get('steps', []))} steps; approving automatically.")

    result = orch.execute(confirm=True, rollback_on_failure=not sticky)
    if result.get("status") == "failed":
        note(
            f"[FAIL] execution failed on step {json.dumps(result.get('failed_step'), default=str)}"
        )
        failure_sres: dict[str, Any] = {}
        for step in result.get("trace", []):
            if step.get("ok"):
                continue
            sres = step.get("result", {})
            failure_sres = sres
            note(
                f"[FAILED-STEP] {step.get('tool')} {step.get('action')} :: {step.get('description')}"
            )
            for key in (
                "stdout",
                "stderr",
                "error",
                "message",
                "returncode",
                "mode",
                "reason",
            ):
                if sres.get(key) is not None:
                    val = str(sres.get(key))
                    note(f"    {key}: {val[:500]}")
        note(json.dumps(result, ensure_ascii=False, default=str)[:3000])
        result["_failure"] = failure_sres
        return False, result

    trace = result.get("trace", [])
    for step in trace:
        status = "ok" if step.get("ok") else "NOT-OK"
        note(
            f"  [{status}] {step.get('tool')} {step.get('action')} :: {step.get('description')}"
        )
        validated = step.get("validated")
        if validated is not None:
            note(f"      validated: {validated}")
        sres = step.get("result", {})
        if step.get("tool") == "run_command":
            note(
                f"      rc={sres.get('returncode')} :: {(sres.get('stdout') or '')[:300]!r}"
            )
        elif "error" in sres:
            note(f"      error: {sres.get('error')} {sres.get('message', '')}")

    ok = all(step.get("ok") for step in trace)
    note(f"RESULT: {'SUCCESS' if ok else 'FAIL'} ({len(trace)} steps)")
    return ok, result


_REPAIR_FILE_RE = re.compile(
    r"(?:^|[\s\`\'\"])((?:frontend|backend|docs|tests|src)/[\w./-]+\.(?:ts|html|css|py|json))"
)


def _repair_paths(root: str, result: dict[str, Any]) -> list[str]:
    failed = result.get("failed_step") or {}
    params = failed.get("params") or {}
    target = params.get("file_path")
    if target:
        return [str(Path(target))]
    sres = result.get("_failure") or {}
    text = ""
    for key in ("stdout", "stderr", "error", "message"):
        text += " " + str(sres.get(key) or "")
    if failed.get("tool") == "run_command":
        for step in result.get("trace", []):
            r = step.get("result", {})
            for key in ("stdout", "stderr", "error", "message"):
                text += " " + str(r.get(key) or "")
    found: list[str] = []
    for m in _REPAIR_FILE_RE.finditer(text):
        found.append(str(Path(root) / m.group(1)))
    return list(dict.fromkeys(found))


def run_phase(
    orch: Any,
    phase: dict[str, Any],
    index: int,
    logf: Any,
    max_repairs: int = 3,
) -> bool:
    name = phase.get("name", f"phase{index}")
    base_prompt = phase.get("prompt", "")
    base_paths = phase.get("paths")
    root = (
        str(Path(orch.impact.root).resolve()) if getattr(orch, "impact", None) else "."
    )

    def note(text: str) -> None:
        print(text)
        if logf is not None:
            logf.write(text + "\n")
            logf.flush()

    note(f"\n{'=' * 70}\n=== {name} ===\n{'=' * 70}")

    prompt = base_prompt
    paths = base_paths
    repairs_left = max_repairs
    for attempt in range(1 + max_repairs):
        sticky = attempt > 0
        ok, result = _run_once(orch, prompt, paths, note, sticky=sticky)
        if ok:
            orch.abort()
            return True
        if not result:
            return False
        if repairs_left == 0:
            note(
                ">>> Repairs exhausted; keeping the partial fixes made during "
                "repair attempts (no rollback) so the next run builds on them."
            )
            return False
        repairs_left -= 1
        orch.abort()
        failed = result.get("failed_step") or {}
        fparams = failed.get("params") or {}
        detail = {
            "tool": failed.get("tool"),
            "action": failed.get("action"),
            "description": failed.get("description"),
            "params": fparams,
        }
        sres = result.get("_failure") or {}
        evidence = ""
        for key in ("stdout", "stderr", "error", "message"):
            if sres.get(key):
                evidence += f"\n--- {key} ---\n{str(sres.get(key))[:2000]}"
        note(
            f"\n>>> Repairing phase {index} ({name}) - repair #{attempt} "
            "(mutations kept on failure)"
        )
        prompt = (
            "A execucao da fase abaixo falhou. Diagnostique a CAUSA RAIZ "
            "lendo os arquivos envolvidos (use read_file nos arquivos "
            "apontados pelas mensagens de erro) e planeje a correcao minima "
            "necessaria. IMPORTANTE: depois de corrigir, o seu plano deve "
            "COMPLETAR TODA A FASE, reexecutando todas as verificacoes do "
            "prompt original (testes, subir o backend, smoke test, build do "
            "frontend, atualizacao de README) e, ao final, se houver "
            "mudancas no repositorio, rodar 'git add -A && git commit -m "
            '"<nome da fase>"\' com cwd na raiz do repositorio. NAO pare '
            "apos corrigir um unico passo.\n\n"
            "PROMPT ORIGINAL DA FASE:\n"
            + base_prompt
            + "\n\nPASSO QUE FALHOU:\n"
            + json.dumps(detail, ensure_ascii=False)[:2000]
            + evidence
        )
        paths = _repair_paths(root, result) or paths
    return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()
    # Pilot drives the orchestrator programmatically (run -> execute
    # confirm=True); never let the interactive confirmation prompt take over.
    os.environ.setdefault("ORCH_CONFIRM_MODE", "agent")
    root = str(Path(args.root).resolve())
    phases = load_phases(args.phases)

    provider = OpenCodeProvider(model="big-pickle")
    if not provider.api_key:
        print("OPENCODE_API_KEY is not set.")
        return 2

    registry = build_default_registry()
    memory = Memory(compressor=ContextCompressor(provider=provider))
    orchestrator = Orchestrator(
        registry=registry,
        provider=provider,
        memory=memory,
        root=root,
        max_plan_steps=args.max_plan_steps,
    )

    logf = None
    if args.log:
        logf = open(args.log, "a", encoding="utf-8")

    start = max(0, min(args.phase, len(phases) - 1))
    for index in range(start, len(phases)):
        ok = run_phase(orchestrator, phases[index], index, logf)
        if not ok:
            if logf:
                logf.close()
            print(
                f"\nPilot stopped at phase {index} ({phases[index].get('name')}). "
                f"Resume later with --phase {index}."
            )
            return 1

    if logf:
        logf.close()
    print("\nAll phases completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
