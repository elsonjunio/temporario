import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import Agent
from src.context import ContextCompressor
from src.memory import Memory
from src.orchestrator import create_orchestrator_tool
from src.providers.lmstudio import LMStudioProvider
from src.providers.opencode import OpenCodeProvider, ProviderError
from src.repl import read_turn, setup_readline
from src.tools.registry import build_default_registry
from src.utils import build_environment_info


def build_provider():
    """Instantiate the provider named by ``AGENT_PROVIDER`` (from ``.env``).

    Supported values: ``opencode`` (default) or ``lmstudio``.
    """
    name = os.getenv("AGENT_PROVIDER", "opencode").strip().lower()
    if name == "lmstudio":
        return LMStudioProvider()
    if name == "opencode":
        return OpenCodeProvider()
    raise ValueError(f"Unknown AGENT_PROVIDER: {name!r}")


def main() -> None:
    load_dotenv()

    workspace = os.path.abspath(os.getcwd())
    print(f"Workspace: {workspace}")

    provider = build_provider()

    registry = build_default_registry()

    if isinstance(provider, OpenCodeProvider) and not provider.api_key:
        print("OPENCODE_API_KEY is not set. Skipping live agent demo.")
        print("Tool manuals available via the registry get_manual():")
        print(registry.get_manual())
        return
    if isinstance(provider, LMStudioProvider) and not provider.model:
        print("LMSTUDIO_MODEL is not set. Skipping live agent demo.")
        print("Tool manuals available via the registry get_manual():")
        print(registry.get_manual())
        return

    memory = Memory(compressor=ContextCompressor(provider=provider))

    orchestrator = create_orchestrator_tool(
        registry,
        provider,
        memory,
        root=workspace,
    )
    registry.register(orchestrator.name, orchestrator)

    agent = Agent(
        provider=provider,
        memory=memory,
        registry=registry,
        environment=build_environment_info(workspace=workspace),
    )

    print(f"--- Agent chat (model={provider.model}) ---")
    print("Type a question; a blank line sends it. /quit to exit.")
    print("Arrow keys edit the current line; up/down recalls past prompts.\n")

    setup_readline()

    while True:
        try:
            question = read_turn()
        except KeyboardInterrupt:
            print()
            continue
        if question is None:
            print()
            break
        question = question.strip()
        if not question:
            continue
        if question.lower() in {"/quit", "/exit", "/bye"}:
            break

        try:
            answer = agent.run(question)
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue
        except ProviderError as exc:
            print(f"Provider error: {exc}")
            continue

        print("Agent:", answer)


if __name__ == "__main__":
    main()
