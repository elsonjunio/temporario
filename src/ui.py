from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Iterator

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

DEFAULT_HISTORY = os.path.join(os.path.expanduser("~"), ".agent_bike_history")

SLASH_COMMANDS = [
    ("/quit", "sai do agente"),
    ("/exit", "sai do agente"),
    ("/bye", "sai do agente"),
    ("/help", "mostra esta ajuda"),
    ("/clear", "limpa a tela"),
    ("/context", "mostra o contexto e a estimativa de tokens"),
    ("/mode", "troca o modo de execução (fast|balanced|precision)"),
]

COMMAND_NAMES = [name for name, _ in SLASH_COMMANDS]

STYLE = Style.from_dict(
    {
        "prompt": "bold cyan",
        "continuation": "cyan",
        "toolbar": "bg:#005f87 fg:#ffffff",
    }
)


def _key_bindings() -> KeyBindings:
    """Enter submits on an empty line (matching ``repl.py``); otherwise it
    inserts a newline so a turn can span multiple lines."""
    kb = KeyBindings()

    @kb.add("enter")
    def _enter(event: Any) -> None:
        buffer = event.app.current_buffer
        if not buffer.text or buffer.text.endswith("\n"):
            buffer.validate_and_handle()
        else:
            buffer.insert_text("\n")

    return kb


class ChatUI:
    """Visual REPL built on prompt_toolkit (input) and Rich (output).

    Only the presentation layer lives here: the agent loop and tools are
    untouched. prompt_toolkit and Rich are cross-platform, so this works on
    Linux and Windows (unlike the readline-based ``src/repl.py``).
    """

    def __init__(
        self,
        *,
        console: Console | None = None,
        session: Any = None,
        history: str | None = DEFAULT_HISTORY,
        model: str = "",
        workspace: str = "",
        mode: str = "",
        interactive: bool | None = None,
    ) -> None:
        self.console = console or Console()
        self.model = model
        self.workspace = workspace
        self.mode = mode
        self.session: Any = None
        self._interactive = sys.stdin.isatty() if interactive is None else interactive
        if session is not None:
            self.session = session
        elif self._interactive:
            self.session = self._build_session(history)

    def _build_session(self, history: str | None) -> PromptSession[Any]:
        file_history = FileHistory(history) if history else None
        return PromptSession(
            history=file_history,
            key_bindings=_key_bindings(),
            style=STYLE,
            completer=WordCompleter(COMMAND_NAMES, ignore_case=True),
            complete_while_typing=False,
            enable_history_search=True,
        )

    def read_turn(
        self, prompt: str = "Você: ", continuation: str = "... "
    ) -> str | None:
        """Read one user turn; returns None on EOF (Ctrl+D).

        KeyboardInterrupt propagates to the caller, mirroring ``repl.py``.
        Without a TTY a single line is read per turn to preserve scripted
        usage.
        """
        if self.session is None:
            try:
                return input(prompt)
            except EOFError:
                return None
        try:
            return self.session.prompt(
                prompt,
                multiline=True,
                prompt_continuation=(
                    lambda width, line_number, is_soft_wrap: continuation
                ),
                bottom_toolbar=self._toolbar,
            )
        except EOFError:
            return None

    def _toolbar(self) -> list[tuple[str, str]]:
        info = f" modelo: {self.model or '?'}"
        if self.mode:
            info += f"  |  modo: {self.mode}"
        info += f"  |  workspace: {self.workspace}"
        return [("class:toolbar", info)]

    def banner(self) -> None:
        header = Text()
        header.append("Bicicleta com Rodinhas", style="bold cyan")
        header.append(f"\nmodelo: {self.model or '?'}", style="dim")
        if self.mode:
            header.append(f"\nmodo: {self.mode}", style="dim")
        header.append(f"\nworkspace: {self.workspace}", style="dim")
        self.console.print(Panel(header, border_style="cyan", padding=(1, 2)))

    def show_help(self) -> None:
        lines = [f"  {name:<10} {desc}" for name, desc in SLASH_COMMANDS]
        self.console.print(
            Panel(Text("\n".join(lines)), title="Ajuda", border_style="blue")
        )

    def show_context(
        self,
        context: str,
        *,
        token_count: int,
        entries: int,
        threshold: int | None = None,
    ) -> None:
        """Render the conversation context with a token/entry summary."""
        body = Text()
        summary = f"entradas: {entries}  |  tokens estimados: {token_count}"
        if threshold:
            summary += f"  |  limiar de compressão: {threshold}"
        body.append(summary, style="bold")
        body.append("\n\n")
        body.append(context)
        self.console.print(Panel(body, title="Contexto", border_style="magenta"))

    def show_manual(self, text: str) -> None:
        self.console.print(Panel(Text(text), title="Manuais", border_style="blue"))

    def show_assistant(self, message: str) -> None:
        self.console.print(Panel(Text(message), title="Agente", border_style="green"))

    def show_error(self, message: str) -> None:
        self.console.print(Text(f"Erro: {message}", style="bold red"))

    def show_info(self, message: str) -> None:
        self.console.print(Text(message, style="dim"))

    def newline(self) -> None:
        self.console.print()

    def clear(self) -> None:
        try:
            self.console.clear()
        except Exception:
            pass

    @contextmanager
    def working(self, message: str = "Pensando...") -> Iterator[None]:
        with self.console.status(message, spinner="dots"):
            yield
