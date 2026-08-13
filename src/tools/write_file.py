from __future__ import annotations

from pathlib import Path

from src.tools._fs import atomic_write, ensure_parent, resolve_path
from src.tools.base import ToolSpec


def handle_write_file(
    file_path: str,
    content: str,
    create_dirs: bool = True,
    encoding: str = "utf-8",
    rewrite: bool = True,
) -> dict:
    """Create or overwrite a file. Skips the write when content is unchanged
    and writes atomically to avoid leaving partial files behind.

    rewrite=False refuses to touch an existing file (use patch_file instead).
    """
    root = resolve_path(file_path)
    existed = root.exists()

    if existed and not rewrite:
        return {
            "status": "error",
            "path": str(root),
            "message": (
                "file already exists; pass rewrite=true to overwrite it or "
                "use patch_file for a targeted edit"
            ),
        }

    if existed:
        try:
            existing = root.read_text(encoding=encoding)
        except OSError as exc:
            return {"status": "error", "path": str(root), "message": str(exc)}
        if existing == content:
            return {
                "status": "success",
                "path": str(root),
                "mode": "unchanged",
                "bytes": len(content.encode(encoding)),
            }
    elif create_dirs:
        ensure_parent(root)

    try:
        atomic_write(root, content, encoding)
    except OSError as exc:
        return {"status": "error", "path": str(root), "message": str(exc)}

    return {
        "status": "success",
        "path": str(root),
        "mode": ("overwritten" if existed else "created"),
        "bytes": len(content.encode(encoding)),
    }


def handle_append_file(
    file_path: str,
    content: str,
    create_dirs: bool = True,
    encoding: str = "utf-8",
) -> dict:
    """Append text to a file, creating it (and parents) if needed."""
    root = resolve_path(file_path)
    existed = root.exists()

    if not existed and create_dirs:
        ensure_parent(root)

    try:
        with open(root, "a", encoding=encoding) as f:
            f.write(content)
    except OSError as exc:
        return {"status": "error", "path": str(root), "message": str(exc)}

    return {
        "status": "success",
        "path": str(root),
        "mode": ("appended" if existed else "created"),
        "bytes": len(content.encode(encoding)),
    }


MANUAL = (
    "write_file: create or update text files on disk.\n"
    "Actions:\n"
    "  - write\n"
    "    Params:\n"
    "      file_path (str, required): absolute or relative path of the file.\n"
    "      content (str, required): full text to write (replaces existing content).\n"
    "      create_dirs (bool, default true): create missing parent directories.\n"
    '      encoding (str, default "utf-8"): text encoding to use.\n'
    "      rewrite (bool, default true): allow overwriting an existing file.\n"
    "        Pass rewrite=false to refuse touching an existing file\n"
    "        (use patch_file for targeted edits of existing files).\n"
    "    Returns: status plus mode (created, overwritten, unchanged) and byte count.\n"
    "    Efficient: identical content returns 'unchanged' without touching the file;\n"
    "    writes are atomic (temp file + rename), so no partial files.\n"
    "  - append\n"
    "    Params:\n"
    "      file_path (str, required): absolute or relative path of the file.\n"
    "      content (str, required): text to append at the end of the file.\n"
    "      create_dirs (bool, default true): create missing parent directories.\n"
    '      encoding (str, default "utf-8"): text encoding to use.\n'
    "    Returns: status plus mode (appended, created) and byte count."
)


SPEC = ToolSpec(
    name="write_file",
    handlers={
        "write": handle_write_file,
        "append": handle_append_file,
    },
    manual=MANUAL,
)


def get_manual() -> str:
    return SPEC.get_manual()


def dispatch(action: str, **params: object) -> dict:
    return SPEC.dispatch(action, **params)
