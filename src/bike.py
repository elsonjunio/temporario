import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import Agent
from src.context import ContextCompressor, estimate_tokens
from src.memory import Memory
from src.modes import apply_mode, build_agent_specs, switch_mode

from src.providers.base import ProviderError
from src.providers.opencode import OpenCodeProvider
from src.providers.lmstudio import LMStudioProvider

from src.tools.registry import build_default_registry
from src.ui import ChatUI
from src.utils import AGENT_MODES, DEFAULT_MODE, build_environment_info

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


def main() -> None:
    load_dotenv()

    workspace = os.path.abspath(os.getcwd())
    provider = build_provider()

    ui = ChatUI(model=provider.model, workspace=workspace, mode=DEFAULT_MODE)
    ui.banner()

    registry = build_default_registry()

    if (isinstance(provider, OpenCodeProvider) and not provider.api_key) or (
        isinstance(provider, LMStudioProvider) and not provider.api_key
    ):
        ui.show_info(
            "Provedor sem chave/modelo configurado; exibindo manuais das ferramentas."
        )
        return

    memory = Memory(compressor=ContextCompressor(provider=provider))

    specs = build_agent_specs(
        registry,
        provider,
        memory,
        root=workspace,
        verbose=os.getenv("EXECUTOR_VERBOSE") == "1",
    )
    apply_mode(registry, DEFAULT_MODE, specs)



    agent = Agent(
        provider=provider,
        memory=memory,
        registry=registry,
        environment=build_environment_info(workspace=workspace),
        instructions=AGENT_MODES[DEFAULT_MODE],
    )

    ui.show_info(
        "Digite uma pergunta; linha em branco envia a mensagem. "
        "/mode troca o modo de execução (fast|balanced|precision), "
        "/quit encerra, /help lista os comandos."
    )

    current_mode = DEFAULT_MODE

    while True:
        try:
            question = ui.read_turn()
        except KeyboardInterrupt:
            ui.newline()
            continue
        if question is None:
            ui.newline()
            break
        question = question.strip()
        if not question:
            continue
        if question.lower() in {"/quit", "/exit", "/bye"}:
            break
        if question.lower().startswith("/mode"):
            current_mode = switch_mode(registry, specs, agent, current_mode, question)
            ui.mode = current_mode
            continue
        if question.lower() == "/help":
            ui.show_help()
            continue
        if question.lower() == "/clear":
            ui.clear()
            continue
        if question.lower() == "/context":
            entries = memory.get_entries()
            estimate = (
                memory.compressor.estimate
                if memory.compressor is not None
                else estimate_tokens
            )
            ui.show_context(
                context=memory.get_context(),
                token_count=sum(estimate(e.content) for e in entries),
                entries=len(entries),
                threshold=(
                    memory.compressor.threshold
                    if memory.compressor is not None
                    else None
                ),
            )
            continue

        try:
            with ui.working():
                answer = agent.run(question)
        except KeyboardInterrupt:
            ui.show_info("(interrompido)")
            continue
        except ProviderError as exc:
            ui.show_error(str(exc))
            continue

        ui.show_assistant(answer)


if __name__ == "__main__":
    main()
