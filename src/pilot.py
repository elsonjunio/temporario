from __future__ import annotations

import argparse
import json
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


def run_phase(
    orch: Any,
    phase: dict[str, Any],
    index: int,
    logf: Any,
) -> bool:
    name = phase.get("name", f"phase{index}")
    prompt = phase.get("prompt", "")
    paths = phase.get("paths")

    def note(text: str) -> None:
        print(text)
        if logf is not None:
            logf.write(text + "\n")
            logf.flush()

    note(f"\n{'=' * 70}\n=== {name} ===\n{'=' * 70}")
    try:
        plan = orch.run(prompt, paths=paths)
    except ProviderError as exc:
        note(f"[FAIL] provider error: {exc}")
        return False

    if plan.get("status") != "awaiting_confirmation":
        note(
            f"[STOP] plan status = {plan.get('status')}: {plan.get('reason') or plan.get('message')}"
        )
        note(json.dumps(plan, ensure_ascii=False, default=str)[:2000])
        return False

    note("PLAN (awaiting confirmation):")
    note(_fmt_plan(plan))
    note(f"PLAN: {len(plan.get('steps', []))} steps; approving automatically.")

    result = orch.execute(confirm=True)
    if result.get("status") == "failed":
        note(
            f"[FAIL] execution failed on step {json.dumps(result.get('failed_step'), default=str)}"
        )
        for step in result.get("trace", []):
            if step.get("ok"):
                continue
            sres = step.get("result", {})
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
        return False

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
    orch.abort()
    return ok


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()
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
