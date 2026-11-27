"""Workspace-aware approval policy.

agent-mini only knows ``tools.confirm``: a list of tool names that always ask
for approval. This wrapper keeps that list, but auto-approves calls whose target
stays inside the workspace:

- file tools: the ``path`` argument resolves inside the workspace;
- shell: the command has no absolute path, ``~``, escaping ``..``, symlink
  escape, ``$``/backtick expansion or ``cd`` without a target.

Anything else falls through to agent-mini's normal approval prompt. This is a
heuristic, not a security boundary: a command can still reach outside without a
path token (e.g. ``pip install`` writing to the global site-packages).

Disable with ``AGENT_MINI_CONFIRM_POLICY=off``.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any

# `/dev/null` and friends are redirections, not workspace escapes.
_HARMLESS = {"/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr"}
_OPERATORS = {"|", "||", "&&", ";", "&", "(", ")"}
_PATH_TOOLS = {"write_file", "edit_file", "read_file"}
_DEFAULT_PATH_TOOLS = {"list_files", "search_files"}
_CD = {"cd", "pushd"}
# Paths after a delimiter: /abs, ~/home, or ../parent.
_ABS_PATH = re.compile(r"""(?<![\w:/.`])(~[^\s"'`|&;<>()]*|/[^\s"'`|&;<>()]*)""")
_REDIRECT_PREFIX = re.compile(r"^\d*[<>]+")


def _inside(raw: str, workspace: Path) -> bool:
    """Whether *raw* resolves inside *workspace* (following symlinks)."""
    try:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = workspace / path
        return path.resolve().is_relative_to(workspace.resolve())
    except (OSError, RuntimeError, ValueError):
        return False


def _looks_like_path(token: str) -> bool:
    return token in (".", "..") or token.startswith(("/", "~", "./", "../")) or "/" in token


def _escapes_via_symlink(token: str, workspace: Path) -> bool:
    """A bare filename that is a symlink pointing outside the workspace."""
    if token.startswith("-") or "=" in token:
        return False
    try:
        candidate = workspace / token
        return candidate.is_symlink() and not _inside(token, workspace)
    except OSError:
        return False


def shell_stays_inside(command: str, workspace: Path) -> bool:
    """Heuristic: can *command* only touch the workspace?"""
    if not command.strip():
        return True
    # Expansions and command substitution can hide any path.
    if "$" in command or "`" in command:
        return False

    for match in _ABS_PATH.finditer(command):
        raw = match.group(0)
        if raw in _HARMLESS:
            continue
        if not _inside(raw, workspace):
            return False

    try:
        tokens = shlex.split(command)
    except ValueError:
        return False

    previous = ""
    for raw in tokens:
        token = _REDIRECT_PREFIX.sub("", raw)
        if previous in _CD and (not token or token == "-" or token in _OPERATORS):
            return False  # `cd`, `cd -`: jumps to $HOME or the previous directory
        if not token or token in _OPERATORS:
            previous = token
            continue
        if token.startswith("-"):
            value = token.split("=", 1)[1] if "=" in token else ""
            if not value:
                match = re.search(r"/[^\s]*", token)  # e.g. -I/usr/include
                value = match.group(0) if match else ""
            if value and value not in _HARMLESS and not _inside(value, workspace):
                return False
            if token == "--global":
                return False  # git config --global writes to $HOME
        elif "=" in token and "/" in token.split("=", 1)[1]:
            value = token.split("=", 1)[1]
            if value not in _HARMLESS and not _inside(value, workspace):
                return False
        elif _looks_like_path(token):
            if token not in _HARMLESS and not _inside(token, workspace):
                return False
        elif _escapes_via_symlink(token, workspace):
            return False
        previous = token
    if previous in _CD:
        return False  # `cd` with no target goes to $HOME
    return True


def auto_approve_ok(name: str, arguments: dict[str, Any], tools: Any) -> bool:
    """Whether *name* can run without asking, given the workspace."""
    workspace = Path(tools.workspace)
    if name == "shell":
        command = arguments.get("command")
        return isinstance(command, str) and shell_stays_inside(command, workspace)
    if name in _PATH_TOOLS:
        path = arguments.get("path")
        return isinstance(path, str) and bool(path) and _inside(path, workspace)
    if name in _DEFAULT_PATH_TOOLS:
        path = arguments.get("path")
        return not path or (isinstance(path, str) and _inside(path, workspace))
    return False


def install(cli_module: Any) -> None:
    """Wrap agent-mini's approval prompt with the workspace policy."""
    if os.getenv("AGENT_MINI_CONFIRM_POLICY", "workspace").strip().lower() in {"off", "0", "false", "no"}:
        return
    original = cli_module._ask_approval
    if getattr(original, "_workspace_policy", False):
        return

    def policy(name: str, arguments: dict[str, Any], tools: Any, always: set[str]) -> bool:
        if auto_approve_ok(name, arguments, tools):
            return True
        return original(name, arguments, tools, always)

    policy._workspace_policy = True  # type: ignore[attr-defined]
    cli_module._ask_approval = policy
