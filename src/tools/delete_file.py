from __future__ import annotations

import shutil

from src.tools._fs import resolve_path
from src.tools.base import ToolSpec


def handle_delete_file(
    path: str,
    recursive: bool = False,
) -> dict:
    """Remove a file or, with recursive=True, an entire directory tree."""
    root = resolve_path(path)

    if not root.exists():
        return {"error": "Path not found", "path": str(root)}

    was_dir = root.is_dir()
    try:
        if was_dir:
            if not recursive:
                return {
                    "status": "error",
                    "message": (
                        "path is a directory; pass recursive=True to delete it."
                    ),
                    "path": str(root),
                }
            shutil.rmtree(root)
            mode = "deleted_directory"
        else:
            root.unlink()
            mode = "deleted_file"
    except OSError as exc:
        return {"status": "error", "path": str(root), "message": str(exc)}

    return {
        "status": "success",
        "path": str(root),
        "mode": mode,
    }


MANUAL = (
    "delete_file: remove a file or directory from disk.\n"
    "Actions:\n"
    "  - delete\n"
    "    Params:\n"
    "      path (str, required): absolute or relative path to remove.\n"
    "      recursive (bool, default false): allow deleting a directory tree;\n"
    "        without it, deleting a directory returns an error.\n"
    "    Returns: status and mode (deleted_file, deleted_directory).\n"
    "    A missing path returns a clean error."
)


SPEC = ToolSpec(
    name="delete_file",
    handlers={"delete": handle_delete_file},
    manual=MANUAL,
)


def get_manual() -> str:
    return SPEC.get_manual()


def dispatch(action: str, **params: object) -> dict:
    return SPEC.dispatch(action, **params)
