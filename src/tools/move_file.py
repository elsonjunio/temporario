from __future__ import annotations

import shutil

from src.tools._fs import ensure_parent, resolve_path
from src.tools.base import ToolSpec


def handle_move_file(
    source: str,
    destination: str,
    create_dirs: bool = True,
    overwrite: bool = False,
) -> dict:
    """Move or rename a file, handling cross-filesystem moves and overwrites."""
    src = resolve_path(source)
    dst = resolve_path(destination)

    if not src.exists():
        return {"error": "Source not found", "source": str(src)}

    if src == dst:
        return {
            "status": "success",
            "source": str(src),
            "destination": str(dst),
            "mode": "unchanged",
        }

    if dst.exists() and not overwrite:
        return {
            "status": "error",
            "message": (
                "destination already exists; pass overwrite=True to replace it."
            ),
            "destination": str(dst),
        }

    if dst.is_dir():
        return {
            "status": "error",
            "message": "destination is a directory; provide a full file path.",
            "destination": str(dst),
        }

    if create_dirs:
        ensure_parent(dst)

    try:
        if overwrite and dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))
    except OSError as exc:
        return {
            "status": "error",
            "message": str(exc),
            "source": str(src),
            "destination": str(dst),
        }

    return {
        "status": "success",
        "source": str(src),
        "destination": str(dst),
        "mode": "moved",
    }


MANUAL = (
    "move_file: move or rename a file (also works across filesystems).\n"
    "Actions:\n"
    "  - move\n"
    "    Params:\n"
    "      source (str, required): current path.\n"
    "      destination (str, required): new path, including the target file name.\n"
    "      create_dirs (bool, default true): create missing destination parents.\n"
    "      overwrite (bool, default false): replace destination if it exists.\n"
    "    Returns: status and mode (moved, unchanged).\n"
    "    Missing sources and existing destinations (without overwrite) return\n"
    "    a clean error."
)


SPEC = ToolSpec(
    name="move_file",
    handlers={"move": handle_move_file},
    manual=MANUAL,
)


def get_manual() -> str:
    return SPEC.get_manual()


def dispatch(action: str, **params: object) -> dict:
    return SPEC.dispatch(action, **params)
