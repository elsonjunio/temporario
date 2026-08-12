from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any

from src.tools.base import ToolSpec

DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_OUTPUT = 20000
_ENCODING = "utf-8"
_ERRORS = "replace"


def _decode(data: bytes | str) -> str:
    if isinstance(data, str):
        return data
    return data.decode(_ENCODING, errors=_ERRORS)


def _trim(output: str, max_output: int) -> tuple[str, bool]:
    if len(output) <= max_output:
        return output, False
    head = max_output // 2
    tail = max_output - head
    return f"{output[:head]}\n...[output truncated]...\n{output[-tail:]}", True


def handle_run_command(
    command: str,
    cwd: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    env: dict[str, str] | None = None,
    input: str | None = None,
    shell: bool = True,
    max_output: int = DEFAULT_MAX_OUTPUT,
) -> dict[str, Any]:
    """Execute an external command through the system shell.

    Cross-platform: on Windows the command runs via ``cmd.exe``, on POSIX via
    ``/bin/sh``. Captures stdout/stderr separately and caps the captured size.
    """
    if not command or not command.strip():
        return {"error": "invalid_arguments", "message": "command must not be empty"}

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=run_env,
            shell=shell,
            input=input,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding=_ENCODING,
            errors=_ERRORS,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "command": command,
            "timeout_seconds": timeout,
            "message": f"Command timed out after {timeout}s.",
        }
    except FileNotFoundError as exc:
        return {"error": "command_not_found", "command": command, "message": str(exc)}
    except OSError as exc:
        return {"error": "execution_failed", "command": command, "message": str(exc)}

    elapsed = time.monotonic() - started

    stdout, stdout_truncated = _trim(_decode(proc.stdout), max_output)
    stderr, stderr_truncated = _trim(_decode(proc.stderr), max_output)

    return {
        "status": "success" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "command": command,
        "cwd": cwd,
        "platform": sys.platform,
        "duration_seconds": round(elapsed, 3),
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }


MANUAL = (
    "run_command: execute an external command on the system shell.\n"
    "Cross-platform: Windows runs the command via cmd.exe, POSIX via /bin/sh,\n"
    "so shell syntax must match the running OS. Do not use this to read or\n"
    "write files -- prefer read_file/write_file. Prefer direct filesystem\n"
    "tools over shell pipelines whenever possible.\n"
    "Actions:\n"
    "  - run\n"
    "    Params:\n"
    "      command (str, required): command line to execute.\n"
    "      cwd (str, optional): working directory for the command.\n"
    "      timeout (float, default 120): max seconds before aborting.\n"
    "      env (dict, optional): extra environment variables to set.\n"
    "      input (str, optional): text to feed to the command's stdin.\n"
    "      shell (bool, default true): run through the system shell.\n"
    "      max_output (int, default 20000): cap on captured output per stream.\n"
    "    Returns: returncode, stdout, stderr (each capped), platform, and\n"
    "    duration. A non-zero exit code is reported via status 'failed', a\n"
    "    timeout via status 'timeout'."
)


SPEC = ToolSpec(
    name="run_command",
    handlers={"run": handle_run_command},
    manual=MANUAL,
)


def get_manual() -> str:
    return SPEC.get_manual()


def dispatch(action: str, **params: object) -> dict:
    return SPEC.dispatch(action, **params)
