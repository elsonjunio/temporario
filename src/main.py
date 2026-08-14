import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import Agent
from src.context import ContextCompressor
from src.memory import Memory
from src.navigation import create_navigation_tool
from src.orchestrator import create_orchestrator_tool

from src.providers.base import ProviderError
from src.providers.opencode import OpenCodeProvider
from src.providers.lmstudio import LMStudioProvider

from src.tools.registry import build_default_registry
from src.utils import build_environment_info


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCode agent demo / driver")
    parser.add_argument(
        "--root",
        default=os.getcwd(),
        help="workspace root for the agent (default: current directory). "
        "The process chdirs into it so run_command lands in the project.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="max tool iterations per agent turn (default: 5).",
    )
    parser.add_argument(
        "--max-plan-steps",
        type=int,
        default=8,
        help="max steps per orchestrator plan (default: 8).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="run a single non-interactive prompt and exit (useful for tests "
        "and scripting phases).",
    )
    return parser.parse_args(argv)


def build_provider():
    """Instantiate the provider named by ``AGENT_PROVIDER`` (from ``.env``).

    Mirrors ``src/main.py``; duplicated here so this entry point never imports
    the readline-based REPL (which is not available on Windows).
    """
    name = os.getenv("AGENT_PROVIDER", "opencode").strip().lower()
    if name == "opencode":
        return OpenCodeProvider()
    if name == "lmstudio":
        return LMStudioProvider()
    raise ValueError(f"Unknown AGENT_PROVIDER: {name!r}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    load_dotenv()

    root = os.path.abspath(args.root)
    os.chdir(root)
    workspace = root
    print(f"Workspace: {workspace}")

    provider = build_provider()

    registry = build_default_registry()

    if not provider.api_key:
        print(
            "Provedor sem chave/modelo configurado; exibindo manuais das ferramentas."
        )
        return

    memory = Memory(compressor=ContextCompressor(provider=provider))

    orchestrator = create_orchestrator_tool(
        registry,
        provider,
        memory,
        root=workspace,
        max_plan_steps=args.max_plan_steps,
    )
    registry.register(orchestrator.name, orchestrator)

    navigation = create_navigation_tool(provider)
    registry.register(navigation.name, navigation)

    agent = Agent(
        provider=provider,
        memory=memory,
        registry=registry,
        max_iterations=args.max_iterations,
        environment=build_environment_info(workspace=workspace),
    )

    if args.prompt:
        try:
            print("Agent:", agent.run(args.prompt, max_iterations=args.max_iterations))
        except ProviderError as exc:
            print(f"Provider error: {exc}")
            raise SystemExit(1)
        return

    print(f"--- Agent chat (model={provider.model}) ---")
    print("Type a question, or /quit to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            print()
            break
        if not question:
            continue
        if question.lower() in {"/quit", "/exit", "/bye"}:
            break

        try:
            answer = agent.run(question, max_iterations=args.max_iterations)
        except ProviderError as exc:
            print(f"Provider error: {exc}")
            continue

        print("Agent:", answer)


if __name__ == "__main__":
    main()
