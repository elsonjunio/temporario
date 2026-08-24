from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from src.tools._fs import ensure_parent, resolve_path

MUTATING_TOOLS = ("write_file", "patch_file", "move_file", "delete_file")


class UndoLog:
    """Snapshots files before every mutation so a failed run can be rolled back.

    Each entry records how to restore the original state. Backups live in a
    temp directory (configurable via ``backup_root``) and are restored in
    reverse order on ``rollback()``.
    """

    def __init__(self, backup_root: str | None = None) -> None:
        self.backup_root = backup_root or tempfile.mkdtemp(prefix="orchestrator_undo_")
        self._entries: list[dict] = []
        self._backup_id = 0

    def _new_backup_dir(self) -> Path:
        self._backup_id += 1
        backup = Path(self.backup_root) / f"b{self._backup_id}"
        backup.mkdir(parents=True, exist_ok=True)
        return backup

    @staticmethod
    def _snapshot_path(path: Path, backup: Path) -> dict:
        """Snapshot a single path; returns its pre-mutation state."""
        if path.is_symlink() or not path.exists():
            return {"type": "none", "path": path, "existed": False}
        if path.is_file():
            target = backup / "file"
            shutil.copy2(path, target)
            return {"type": "file", "path": path, "backup": target, "existed": True}
        target = backup / "tree"
        shutil.copytree(path, target)
        return {"type": "dir", "path": path, "backup": target, "existed": True}

    def snapshot(self, tool: str, action: str, params: dict) -> None:
        if tool not in MUTATING_TOOLS:
            return

        if tool in ("write_file", "patch_file"):
            path = resolve_path(params.get("file_path", ""))
            state = self._snapshot_path(path, self._new_backup_dir())
            self._entries.append(
                {"tool": tool, "action": action, "kind": "path", "state": state}
            )
            return

        if tool == "move_file":
            src = resolve_path(params.get("source", ""))
            dst = resolve_path(params.get("destination", ""))
            backup = self._new_backup_dir()
            self._entries.append(
                {
                    "tool": tool,
                    "action": action,
                    "kind": "move",
                    "src": src,
                    "dst": dst,
                    "src_state": self._snapshot_path(src, backup / "src"),
                    "dst_state": self._snapshot_path(dst, backup / "dst"),
                }
            )
            return

        if tool == "delete_file":
            path = resolve_path(params.get("path", ""))
            state = self._snapshot_path(path, self._new_backup_dir())
            self._entries.append(
                {"tool": tool, "action": action, "kind": "path", "state": state}
            )

    @staticmethod
    def _restore_state(state: dict) -> None:
        path: Path = state["path"]
        kind = state["type"]
        ensure_parent(path)
        if kind == "none":
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        elif kind == "file":
            if path.is_dir():
                shutil.rmtree(path)
            shutil.copy2(state["backup"], path)
        else:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            shutil.copytree(state["backup"], path)

    def _restore(self, entry: dict) -> None:
        if entry["kind"] == "move":
            src, dst = entry["src"], entry["dst"]
            if dst.exists():
                if src.exists():
                    if src.is_dir():
                        shutil.rmtree(src)
                    else:
                        src.unlink()
                shutil.move(str(dst), str(src))
            self._restore_state(entry["dst_state"])
            return
        self._restore_state(entry["state"])

    def rollback(self) -> dict:
        restored = 0
        errors: list[str] = []
        for entry in reversed(self._entries):
            try:
                self._restore(entry)
                restored += 1
            except OSError as exc:
                errors.append(f"{entry['tool']} {entry.get('action', '')}: {exc}")
        self._entries.clear()
        return {
            "status": "rolled_back",
            "restored": restored,
            "errors": errors,
        }

    def discard(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count

    def checkpoint(self) -> int:
        """Current entry count — a mark usable with :meth:`rollback_to`."""
        return len(self._entries)

    def rollback_to(self, mark: int) -> dict:
        """Roll back and drop only the entries appended after ``mark``
        (typically from :meth:`checkpoint`). Entries before ``mark`` — e.g.
        mutations of previously completed parts — stay intact."""
        mark = max(0, min(mark, len(self._entries)))
        tail = self._entries[mark:]
        del self._entries[mark:]
        restored = 0
        errors: list[str] = []
        for entry in reversed(tail):
            try:
                self._restore(entry)
                restored += 1
            except OSError as exc:
                errors.append(f"{entry['tool']} {entry.get('action', '')}: {exc}")
        return {
            "status": "rolled_back",
            "restored": restored,
            "errors": errors,
        }

    def pending(self) -> int:
        return len(self._entries)
