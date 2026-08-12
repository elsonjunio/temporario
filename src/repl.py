from __future__ import annotations

import atexit
import os
import readline
import sys
from typing import Callable

DEFAULT_HISTORY = os.path.join(os.path.expanduser("~"), ".agent_history")
HISTORY_LIMIT = 1000


def setup_readline(
    history: str = DEFAULT_HISTORY, register_atexit: bool = True
) -> None:
    """Enable GNU readline line editing and persistent prompt history.

    ``readline`` must be imported for ``input()`` to support cursor movement
    (arrows, Home/End, Ctrl+A/E) and history recall while typing.
    """
    readline.set_history_length(HISTORY_LIMIT)
    try:
        readline.read_history_file(history)
    except OSError:
        pass
    if register_atexit:
        atexit.register(readline.write_history_file, history)


def _safe_read(reader: Callable[[str], str], prompt: str) -> str:
    try:
        return reader(prompt)
    except EOFError:
        return ""


def _read_lines(prompt: str, reader: Callable[[str], str]) -> str | None:
    """Read lines from ``reader`` until an empty line (TTY mode).

    On a non-TTY stdin a single line is read per turn to preserve scripted
    usage. Returns the joined prompt, or None on EOF before any text.
    """
    if not sys.stdin.isatty():
        line = _safe_read(reader, prompt)
        if line == "":
            return None
        return line

    first = _safe_read(reader, prompt)
    if first == "":
        return None
    lines = [first]
    while True:
        line = _safe_read(reader, "... ")
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def _history_items(length: int) -> list[str]:
    items: list[str] = []
    for index in range(1, length + 1):
        item = readline.get_history_item(index)
        if item:
            items.append(item)
    return items


def _commit_history(history: str, entries: list[str]) -> None:
    """Rebuild readline history so it holds whole prompts only (each turn
    auto-adds its partial lines to readline's history; drop them)."""
    readline.clear_history()
    for entry in entries:
        readline.add_history(entry)
    readline.write_history_file(history)


def read_turn(
    prompt: str = "You: ",
    *,
    history: str = DEFAULT_HISTORY,
    reader: Callable[[str], str] | None = None,
) -> str | None:
    """Read one user turn and return the prompt, or None on EOF (Ctrl+D).

    On a TTY the turn spans multiple lines and is sent with an empty line.
    After reading, the readline history is rebuilt holding only whole prompts.
    Passing a custom ``reader`` (tests) disables history management entirely,
    so the user's real history file is never touched.
    """
    manage_history = reader is None and sys.stdin.isatty()
    line_reader = reader or input
    start = readline.get_current_history_length()

    text = _read_lines(prompt, line_reader)
    if not text:
        return text

    if manage_history:
        try:
            _commit_history(history, _history_items(start) + [text])
        except Exception:
            pass
    return text
