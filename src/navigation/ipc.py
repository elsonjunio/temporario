from __future__ import annotations

"""RPC client for the browser worker subprocess.

``BrowserClient`` exposes the same surface as ``BrowserController`` by
forwarding any attribute access to a method call over a JSON-lines pipe to
``src/navigation/worker``. The browser (Chromium + Playwright) runs in a
separate process: a crash or a hung browser never takes down the agent.

  - ``client.open(url=...)``  ->  ``{"id": n, "method": "open", ...}``
  - responses are parsed back into the same structured dicts the in-process
    controller returns (status/error payloads, snapshot text, base64 data...).
  - ``close()`` shuts the browser down and reaps the child process.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.navigation.errors import BrowserError, SessionClosed

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_MODULE = "src.navigation.worker"


class BrowserClient:
    """JSON-lines RPC client driving a ``src.navigation.worker`` subprocess.

    Duck-types ``BrowserController`` for the tool registry and the
    ``NavigationSubAgent``: unknown attribute access forwards to a worker
    method call of the same name.
    """

    def __init__(
        self,
        *,
        request_timeout: float | None = None,
        stderr: Any = None,
        **controller_config: Any,
    ) -> None:
        self.request_timeout = request_timeout
        self._seq = 0
        self._closed = False
        self._proc = subprocess.Popen(
            [sys.executable, "-m", _WORKER_MODULE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr if stderr is not None else subprocess.DEVNULL,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        try:
            self._call("__init__", controller_config)
            self._call("__ping__")
        except Exception:
            self.close()
            raise

    @property
    def is_open(self) -> bool:
        return self._proc.poll() is None

    # -- RPC ---------------------------------------------------------------

    def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if not self.is_open:
            raise SessionClosed(
                "browser session is closed; call browser.open or browser.goto first",
                recoverable=True,
            )
        self._seq += 1
        rid = self._seq
        line = json.dumps({"id": rid, "method": method, "params": params or {}}) + "\n"
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise SessionClosed(
                f"browser worker pipe closed: {exc}", recoverable=True
            ) from exc
        raw = self._read_line()
        if raw is None:
            raise SessionClosed("browser worker exited unexpectedly", recoverable=True)
        try:
            message = json.loads(raw)
        except ValueError as exc:
            raise BrowserError(
                "browser worker sent an invalid response", recoverable=False
            ) from exc
        if message.get("id") != rid:
            raise BrowserError("browser worker protocol mismatch", recoverable=False)
        if "error" in message:
            error = message["error"] or {}
            raise BrowserError(
                error.get("message", "browser worker error"),
                recoverable=bool(error.get("recoverable", True)),
            )
        return message.get("result")

    def _read_line(self) -> str | None:
        assert self._proc.stdout is not None
        if self.request_timeout is not None and sys.platform != "win32":
            import select

            ready, _, _ = select.select(
                [self._proc.stdout], [], [], self.request_timeout
            )
            if not ready:
                self.terminate()
                raise BrowserError(
                    f"browser worker did not answer within " f"{self.request_timeout}s",
                    recoverable=False,
                )
        line = self._proc.stdout.readline()
        if line == "":
            return None
        return line.strip()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def _forward(*args: Any, **kwargs: Any) -> Any:
            if args:
                return {
                    "status": "error",
                    "error": {
                        "type": "invalid_arguments",
                        "message": (
                            f"{name}() over IPC accepts keyword arguments only"
                        ),
                        "recoverable": True,
                    },
                }
            try:
                return self._call(name, kwargs)
            except BrowserError as exc:
                return {"status": "error", "error": exc.to_dict()}

        return _forward

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> dict[str, Any]:
        try:
            result = self._call("close")
        except BrowserError:
            result = {"status": "success", "operation": "close"}
        self._closed = True
        self._reap()
        return result

    def _reap(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
        for pipe in (self._proc.stdin, self._proc.stdout):
            if pipe is not None and not pipe.closed:
                try:
                    pipe.close()
                except OSError:
                    pass

    def terminate(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.terminate()
            except OSError:
                pass

    def __enter__(self) -> "BrowserClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
