import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import Agent
from src.context import ContextCompressor
from src.memory import Memory
from src.orchestrator import create_orchestrator_tool
from src.providers.opencode import OpenCodeProvider, ProviderError
from src.tools.registry import build_default_registry
from src.utils import build_environment_info


def main() -> None:
    load_dotenv()

    workspace = os.path.abspath(os.getcwd())
    print(f"Workspace: {workspace}")

    provider = OpenCodeProvider(model="big-pickle")

    registry = build_default_registry()

    if not provider.api_key:
        print("OPENCODE_API_KEY is not set. Skipping live agent demo.")
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
            answer = agent.run(question)
        except ProviderError as exc:
            print(f"Provider error: {exc}")
            continue

        print("Agent:", answer)


if __name__ == "__main__":
    main()
