from __future__ import annotations

import os
import tempfile
from pathlib import Path

#: Dependency / build / cache directories that should never be traversed or
#: listed when inspecting a project (used by list_dir and discovery).
EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "vendor",
    "site-packages",
    ".idea",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".angular",
    "coverage",
}


def is_excluded(path: str | Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in Path(path).parts)


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def ensure_parent(path: Path) -> None:
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def is_binary(path: Path) -> bool:
    with open(path, "rb") as f:
        head = f.read(8192)
    return b"\x00" in head
